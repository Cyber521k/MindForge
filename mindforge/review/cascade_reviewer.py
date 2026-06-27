"""Cascade-based automated review using multiple judge models."""

import logging
import os

from mindforge.hardware.ollama_detector import detect_ollama
from mindforge.probe.adapters import create_adapter
from mindforge.review.auto_reviewer import AutoReviewer

logger = logging.getLogger(__name__)


class CascadeReviewer:
    """Multi-model verification wrapper around AutoReviewer.

    The cascade runs a primary judge first, escalates to a secondary judge when
    confidence is low, and uses a tiebreaker when the first two judges disagree.
    """

    CONFIDENCE_THRESHOLD = 0.7
    JUDGE_ROLES = ("primary", "secondary", "tiebreaker")

    def __init__(self, judge_adapters, web_search_enabled=False, confidence_threshold=None):
        """Initialize the cascade reviewer.

        Args:
            judge_adapters: Ordered adapters for primary, secondary, and tiebreaker.
            web_search_enabled: Stored for compatibility with auto review sessions.
            confidence_threshold: Confidence below which secondary review is used.
        """
        self.judge_adapters = [adapter for adapter in (judge_adapters or []) if adapter][:3]
        self.web_search_enabled = web_search_enabled
        self.confidence_threshold = (
            self.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        self.reviewers = [
            AutoReviewer(judge_adapter=adapter, web_search_enabled=web_search_enabled)
            for adapter in self.judge_adapters
        ]
        self.judge_model_name = self._build_judge_model_name()

    def _build_judge_model_name(self):
        """Return a readable label for the configured cascade."""
        if not self.judge_adapters:
            return "cascade:none"
        models = [
            getattr(adapter, "model_name", "unknown")
            for adapter in self.judge_adapters
        ]
        return "cascade:" + ",".join(models)

    def review_entry(self, entry):
        """Review a single training entry with the configured judge cascade."""
        if not self.reviewers:
            consensus = self._combine_verdicts([])
            return self._build_review_result(consensus, [], False, False)

        verdicts = []
        escalated = False
        tiebreaker_used = False

        primary = self._run_judge(0, entry)
        verdicts.append(primary)

        if (
            primary.get("confidence", 0.0) < self.confidence_threshold
            and len(self.reviewers) > 1
        ):
            escalated = True
            secondary = self._run_judge(1, entry)
            verdicts.append(secondary)

            if (
                primary.get("correct") != secondary.get("correct")
                and len(self.reviewers) > 2
            ):
                tiebreaker_used = True
                verdicts.append(self._run_judge(2, entry))

        consensus = self._combine_verdicts(verdicts)
        return self._build_review_result(consensus, verdicts, escalated, tiebreaker_used)

    def review_batch(self, entries, on_progress=None):
        """Review multiple training entries with optional progress callbacks."""
        results = []
        total = len(entries)

        for i, entry in enumerate(entries):
            result = self.review_entry(entry)
            results.append(result)
            if on_progress:
                on_progress(i + 1, total, result)

        return results

    def _run_judge(self, judge_index, entry):
        """Run one AutoReviewer judge and annotate its verdict."""
        reviewer = self.reviewers[judge_index]
        adapter = self.judge_adapters[judge_index]
        role = self.JUDGE_ROLES[judge_index]
        question = entry.get("question") or entry.get("prompt", "")
        chosen = entry.get("chosen", "")
        claimed_correct = entry.get("correct_answer")

        verdict = reviewer._judge_answer(question, chosen, claimed_correct)
        confidence = self._coerce_confidence(verdict.get("confidence", 0.5))
        correct = bool(verdict.get("correct", False))

        return {
            "role": role,
            "model": getattr(adapter, "model_name", "unknown"),
            "correct": correct,
            "confidence": confidence,
            "action": "accept" if correct else "reject",
            "explanation": verdict.get("explanation", ""),
        }

    def _combine_verdicts(self, verdicts):
        """Combine multiple judge verdicts into a consensus result."""
        if not verdicts:
            return {
                "correct": None,
                "confidence": 0.0,
                "agreement_score": 0.0,
                "final_action": "skip",
                "explanation": "No judge verdicts available.",
            }

        normalized = [
            {
                **verdict,
                "correct": bool(verdict.get("correct", False)),
                "confidence": self._coerce_confidence(verdict.get("confidence", 0.5)),
            }
            for verdict in verdicts
        ]

        correct_count = sum(1 for verdict in normalized if verdict["correct"])
        incorrect_count = len(normalized) - correct_count

        if correct_count == incorrect_count:
            consensus_correct = max(
                normalized,
                key=lambda verdict: verdict["confidence"],
            )["correct"]
        else:
            consensus_correct = correct_count > incorrect_count

        agreeing = [
            verdict for verdict in normalized
            if verdict["correct"] == consensus_correct
        ]
        agreement_score = len(agreeing) / len(normalized)
        confidence = (
            sum(verdict["confidence"] for verdict in agreeing) / len(agreeing)
            if agreeing else 0.0
        )
        final_action = "accept" if consensus_correct else "reject"

        return {
            "correct": consensus_correct,
            "confidence": confidence,
            "agreement_score": agreement_score,
            "final_action": final_action,
            "explanation": (
                f"{len(agreeing)}/{len(normalized)} judges agreed on {final_action}."
            ),
        }

    def _build_review_result(self, consensus, verdicts, escalated, tiebreaker_used):
        """Build the public result dictionary for a cascade review."""
        return {
            "action": consensus["final_action"],
            "final_action": consensus["final_action"],
            "confidence": consensus["confidence"],
            "agreement_score": consensus["agreement_score"],
            "model_verdicts": verdicts,
            "individual_model_verdicts": verdicts,
            "consensus": consensus,
            "escalated": escalated,
            "tiebreaker_used": tiebreaker_used,
            "explanation": consensus["explanation"],
            "edited_chosen": None,
            "edited_rejected": None,
            "web_source": None,
        }

    def close(self):
        """Clean up all underlying reviewer adapters."""
        for reviewer in self.reviewers:
            reviewer.close()

    @staticmethod
    def _coerce_confidence(value):
        """Convert confidence to a bounded float."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.5
        return max(0.0, min(1.0, confidence))


def create_cascade_reviewer(web_search_enabled=False, confidence_threshold=None):
    """Create a cascade reviewer from available models in priority order."""
    adapters = []

    if os.environ.get("OPENAI_API_KEY"):
        _append_adapter(adapters, "gpt-4o", "OpenAI")

    if len(adapters) < 3 and os.environ.get("OPENROUTER_API_KEY"):
        _append_adapter(
            adapters,
            "openrouter/anthropic/claude-3.5-sonnet",
            "OpenRouter",
        )

    if len(adapters) < 3:
        _append_ollama_adapter(adapters)

    if len(adapters) < 3:
        _append_adapter(
            adapters,
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "MLX",
        )

    return CascadeReviewer(
        adapters,
        web_search_enabled=web_search_enabled,
        confidence_threshold=confidence_threshold,
    )


def _append_adapter(adapters, model_name, provider):
    """Create and append an adapter if the provider can initialize."""
    try:
        adapters.append(create_adapter(model_name))
        logger.info("Auto-detected %s cascade judge (%s)", provider, model_name)
    except Exception as exc:
        logger.debug("%s cascade adapter creation failed: %s", provider, exc)


def _append_ollama_adapter(adapters):
    """Append the first available Ollama model to the cascade."""
    try:
        ollama_info = detect_ollama()
        models = ollama_info.get("models") or []
        if not ollama_info.get("running") or not models:
            return

        model_name = models[0]
        if isinstance(model_name, dict):
            model_name = model_name.get("name") or model_name.get("model")
        if not model_name:
            return

        _append_adapter(adapters, f"ollama/{model_name}", "Ollama")
    except Exception as exc:
        logger.debug("Ollama cascade detection failed: %s", exc)
