"""Quality metrics for DPO prompt/chosen/rejected training pairs.

The scoring in this module is heuristic by design. It uses only local text
features so it can run quickly before training without model or network calls.
"""

import math
import re
import statistics
from collections import Counter

try:
    import numpy as _np
except ImportError:  # pragma: no cover - depends on local environment
    _np = None


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_SENTENCE_RE = re.compile(r"[.!?]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
    "for", "from", "has", "have", "i", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "was",
    "were", "with", "you", "your",
}

_UNCERTAINTY_PHRASES = (
    "i don't know",
    "i do not know",
    "not sure",
    "unsure",
    "unknown",
    "cannot answer",
    "can't answer",
    "no information",
    "maybe",
    "probably",
)

_QUALITY_INDICATORS = (
    "because",
    "therefore",
    "for example",
    "for instance",
    "defined as",
    "depends on",
    "caused by",
    "results in",
    "step",
    "first",
    "second",
    "however",
    "whereas",
)

_DIFFICULTY_TERMS = (
    "analyze",
    "compare",
    "contrast",
    "derive",
    "evaluate",
    "explain",
    "justify",
    "prove",
    "synthesize",
    "tradeoff",
    "tradeoffs",
    "failure mode",
    "regulatory",
    "mechanism",
    "complexity",
    "assumption",
)

_METRIC_KEYS = (
    "semantic_similarity",
    "length_ratio",
    "answer_diversity",
    "preference_margin",
    "difficulty_score",
    "overall_quality",
)


def compute_pair_quality(prompt, chosen, rejected):
    """Compute quality metrics for one DPO preference pair.

    Args:
        prompt: The user prompt or question.
        chosen: The preferred response.
        rejected: The rejected response.

    Returns:
        dict with normalized scores for quality metrics plus the raw
        chosen/rejected length ratio. Higher ``semantic_similarity`` means the
        two answers are more similar, which lowers ``overall_quality``.
    """
    prompt_text = _as_text(prompt)
    chosen_text = _as_text(chosen)
    rejected_text = _as_text(rejected)

    chosen_tokens = _tokens(chosen_text)
    rejected_tokens = _tokens(rejected_text)

    semantic_similarity = _semantic_similarity(chosen_text, rejected_text)
    length_ratio = _length_ratio(len(chosen_tokens), len(rejected_tokens))
    answer_diversity = _lexical_diversity(chosen_tokens)
    difficulty_score = _difficulty_score(prompt_text)

    chosen_features = _answer_feature_score(prompt_text, chosen_text, difficulty_score)
    rejected_features = _answer_feature_score(prompt_text, rejected_text, difficulty_score)
    preference_margin = _preference_margin(chosen_features, rejected_features)

    similarity_quality = 1.0 - semantic_similarity
    length_quality = _length_ratio_quality(length_ratio)
    presence_quality = 1.0 if chosen_tokens and rejected_tokens else 0.0

    overall_quality = _clamp(
        (
            0.35 * preference_margin
            + 0.25 * similarity_quality
            + 0.15 * length_quality
            + 0.10 * answer_diversity
            + 0.10 * difficulty_score
            + 0.05 * presence_quality
        )
    )
    if semantic_similarity > 0.85:
        duplicate_penalty = min((semantic_similarity - 0.85) / 0.15, 1.0)
        overall_quality *= 1.0 - 0.55 * duplicate_penalty

    return {
        "semantic_similarity": float(_clamp(semantic_similarity)),
        "length_ratio": float(length_ratio),
        "answer_diversity": float(_clamp(answer_diversity)),
        "preference_margin": float(_clamp(preference_margin)),
        "difficulty_score": float(_clamp(difficulty_score)),
        "overall_quality": float(_clamp(overall_quality)),
    }


def filter_low_quality(entries, threshold=0.3):
    """Return DPO entries whose overall quality is at least ``threshold``."""
    kept = []
    for entry in entries:
        prompt, chosen, rejected = _entry_text(entry)
        scores = compute_pair_quality(prompt, chosen, rejected)
        if scores["overall_quality"] >= threshold:
            kept.append(entry)
    return kept


def generate_quality_report(entries, threshold=0.3):
    """Generate summary statistics for a list of DPO entries."""
    scores = []
    low_quality_indices = []

    for index, entry in enumerate(entries):
        prompt, chosen, rejected = _entry_text(entry)
        pair_scores = compute_pair_quality(prompt, chosen, rejected)
        scores.append(pair_scores)
        if pair_scores["overall_quality"] < threshold:
            low_quality_indices.append(index)

    if not scores:
        return {
            "total_entries": 0,
            "kept_entries": 0,
            "filtered_entries": 0,
            "threshold": threshold,
            "metrics": {},
            "low_quality_indices": [],
        }

    kept_entries = len(scores) - len(low_quality_indices)
    metrics = {}
    for key in _METRIC_KEYS:
        values = [score[key] for score in scores]
        metrics[key] = _summary_stats(values)

    return {
        "total_entries": len(scores),
        "kept_entries": kept_entries,
        "filtered_entries": len(low_quality_indices),
        "threshold": threshold,
        "metrics": metrics,
        "low_quality_indices": low_quality_indices,
    }


def _as_text(value):
    if value is None:
        return ""
    return str(value)


def _tokens(text):
    return _WORD_RE.findall(_as_text(text).lower())


def _content_tokens(tokens):
    content = [token for token in tokens if token not in _STOPWORDS]
    return content or list(tokens)


def _entry_text(entry):
    if not isinstance(entry, dict):
        return "", "", ""
    return (
        entry.get("prompt") or entry.get("question") or "",
        entry.get("chosen") or "",
        entry.get("rejected") or "",
    )


def _semantic_similarity(chosen, rejected):
    chosen_tokens = _content_tokens(_tokens(chosen))
    rejected_tokens = _content_tokens(_tokens(rejected))

    if not chosen_tokens and not rejected_tokens:
        return 1.0
    if not chosen_tokens or not rejected_tokens:
        return 0.0

    chosen_counts = Counter(chosen_tokens)
    rejected_counts = Counter(rejected_tokens)
    vocab = sorted(set(chosen_counts) | set(rejected_counts))

    if _np is not None:
        chosen_vec = _np.array([chosen_counts[token] for token in vocab], dtype=float)
        rejected_vec = _np.array([rejected_counts[token] for token in vocab], dtype=float)
        denom = float(_np.linalg.norm(chosen_vec) * _np.linalg.norm(rejected_vec))
        cosine = float(_np.dot(chosen_vec, rejected_vec) / denom) if denom else 0.0
    else:
        dot = sum(chosen_counts[token] * rejected_counts[token] for token in vocab)
        chosen_norm = math.sqrt(sum(count * count for count in chosen_counts.values()))
        rejected_norm = math.sqrt(sum(count * count for count in rejected_counts.values()))
        cosine = dot / (chosen_norm * rejected_norm) if chosen_norm and rejected_norm else 0.0

    overlap = len(set(chosen_tokens) & set(rejected_tokens)) / max(
        len(set(chosen_tokens) | set(rejected_tokens)),
        1,
    )
    return _clamp(0.8 * cosine + 0.2 * overlap)


def _length_ratio(chosen_count, rejected_count):
    if chosen_count == 0 and rejected_count == 0:
        return 1.0
    if rejected_count == 0:
        return float(max(chosen_count, 1))
    return float(chosen_count) / float(rejected_count)


def _length_ratio_quality(ratio):
    if ratio <= 0:
        return 0.0
    return _clamp(1.0 - abs(math.log(ratio)) / math.log(10.0))


def _lexical_diversity(tokens):
    if not tokens:
        return 0.0
    content = _content_tokens(tokens)
    return len(set(content)) / max(len(content), 1)


def _difficulty_score(prompt):
    tokens = _tokens(prompt)
    if not tokens:
        return 0.0

    lower_prompt = prompt.lower()
    token_component = min(len(tokens) / 35.0, 1.0)
    question_component = min(
        sum(1 for word in ("why", "how", "compare", "explain", "analyze") if word in tokens)
        / 3.0,
        1.0,
    )
    term_component = min(
        sum(1 for term in _DIFFICULTY_TERMS if term in lower_prompt) / 4.0,
        1.0,
    )
    clause_component = min(
        (lower_prompt.count(",") + lower_prompt.count(";") + lower_prompt.count(" versus ")) / 4.0,
        1.0,
    )
    technical_component = min(
        len([token for token in _content_tokens(tokens) if len(token) >= 9]) / 5.0,
        1.0,
    )

    return _clamp(
        0.30 * token_component
        + 0.25 * question_component
        + 0.20 * term_component
        + 0.15 * clause_component
        + 0.10 * technical_component
    )


def _answer_feature_score(prompt, answer, difficulty_score):
    tokens = _tokens(answer)
    if not tokens:
        return 0.0

    lower_answer = answer.lower()
    content = _content_tokens(tokens)
    word_count = len(tokens)
    expected_words = 5.0 + 24.0 * difficulty_score

    length_score = min(word_count / expected_words, 1.0)
    if word_count > expected_words * 8.0:
        length_score *= max(0.2, (expected_words * 8.0) / word_count)

    specificity_score = min(
        (len(set(content)) + 2.0 * _numeric_count(tokens)) / max(expected_words, 6.0),
        1.0,
    )
    indicator_score = min(
        (_phrase_hits(lower_answer, _QUALITY_INDICATORS) + _numeric_count(tokens)) / 4.0,
        1.0,
    )
    sentence_count = max(1, len([part for part in _SENTENCE_RE.split(answer) if part.strip()]))
    structure_score = min(sentence_count / max(1.0 + 2.0 * difficulty_score, 1.0), 1.0)
    diversity_score = _lexical_diversity(tokens)

    uncertainty_penalty = 0.35 if _phrase_hits(lower_answer, _UNCERTAINTY_PHRASES) else 0.0
    terse_penalty = 0.20 if word_count <= 3 and difficulty_score > 0.15 else 0.0

    return _clamp(
        0.30 * length_score
        + 0.30 * specificity_score
        + 0.20 * indicator_score
        + 0.15 * structure_score
        + 0.05 * diversity_score
        - uncertainty_penalty
        - terse_penalty
    )


def _preference_margin(chosen_score, rejected_score):
    return _clamp(0.05 + 1.8 * max(chosen_score - rejected_score, 0.0))


def _numeric_count(tokens):
    return sum(1 for token in tokens if any(char.isdigit() for char in token))


def _phrase_hits(text, phrases):
    return sum(1 for phrase in phrases if phrase in text)


def _summary_stats(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}

    if _np is not None:
        mean = float(_np.mean(values))
        median = float(_np.median(values))
        min_value = float(_np.min(values))
        max_value = float(_np.max(values))
        stdev = float(_np.std(values))
    else:
        mean = float(statistics.fmean(values))
        median = float(statistics.median(values))
        min_value = float(min(values))
        max_value = float(max(values))
        stdev = float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    return {
        "mean": mean,
        "median": median,
        "min": min_value,
        "max": max_value,
        "stdev": stdev,
    }


def _clamp(value, low=0.0, high=1.0):
    if value < low:
        return low
    if value > high:
        return high
    return value
