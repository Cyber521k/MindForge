"""Question generation for probing using MMLU dataset.

Phase 2 additions:
    - Tier 2 (depth): generate_tier2_followups(question, model_answer, subject)
    - Tier 3 (edge cases): generate_tier3_edge_cases(subject)
    - ProbeEngine integration for multi-tier probing
"""

import os
import yaml
import logging
import re

logger = logging.getLogger(__name__)

# Path to taxonomy YAML
TAXONOMY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "taxonomy",
    "subjects.yaml",
)

_LETTERS = ["A", "B", "C", "D"]


def load_taxonomy():
    """Load the subject taxonomy YAML file."""
    with open(TAXONOMY_PATH, "r") as f:
        return yaml.safe_load(f)


def resolve_subject(subject_input):
    """Resolve a user-provided subject name to an MMLU subject key.

    If the input is already a valid MMLU subject (in the taxonomy categories),
    return it as-is. Otherwise, try the subject_mapping.
    """
    taxonomy = load_taxonomy()

    # Collect all valid MMLU subjects
    all_subjects = set()
    for cat_subjects in taxonomy.get("categories", {}).values():
        all_subjects.update(cat_subjects)

    # If it's already a valid MMLU subject, return it
    if subject_input in all_subjects:
        return subject_input

    # Try the mapping
    mapping = taxonomy.get("subject_mapping", {})
    if subject_input in mapping:
        resolved = mapping[subject_input]
        if resolved in all_subjects:
            return resolved

    # Try replacing spaces with underscores
    candidate = subject_input.replace(" ", "_").lower()
    if candidate in all_subjects:
        return candidate

    # Try the mapping with underscores
    if candidate in mapping:
        resolved = mapping[candidate]
        if resolved in all_subjects:
            return resolved

    return None


def format_mcq_prompt(question, choices, subject=None):
    """Format a multiple-choice question prompt for the model.

    Args:
        question: The question text
        choices: List of 4 answer choices
        subject: Optional subject name for context

    Returns:
        Formatted prompt string
    """
    lines = []
    if subject:
        lines.append(f"Subject: {subject.replace('_', ' ').title()}")
        lines.append("")

    lines.append(question)
    lines.append("")

    for i, choice in enumerate(choices):
        lines.append(f"{_LETTERS[i]}) {choice}")

    lines.append("")
    lines.append("Answer with a single letter (A, B, C, or D).")

    return "\n".join(lines)


def load_mmlu_questions(subject, limit=25, split="dev"):
    """Load MMLU questions for a subject from HuggingFace datasets.

    Args:
        subject: MMLU subject key (e.g., 'high_school_mathematics')
        limit: Maximum number of questions to return
        split: Dataset split to use ('dev', 'validation', 'test', 'auxiliary_train')

    Returns:
        List of dicts with keys: question, choices, answer (int 0-3), subject
    """
    from datasets import load_dataset

    logger.info(f"Loading MMLU dataset (config='all', split='{split}')...")

    try:
        ds = load_dataset("cais/mmlu", "all", split=split)
    except Exception as e:
        logger.warning(f"Failed to load split '{split}': {e}")
        # Fall back to 'dev' split which is small and always available
        if split != "dev":
            logger.info("Falling back to 'dev' split...")
            ds = load_dataset("cais/mmlu", "all", split="dev")
        else:
            raise

    # Filter by subject
    subject_data = ds.filter(lambda x: x["subject"] == subject)

    questions = []
    for item in subject_data:
        questions.append({
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
            "subject": item["subject"],
        })
        if len(questions) >= limit:
            break

    logger.info(f"Loaded {len(questions)} questions for subject '{subject}'")

    if len(questions) < limit:
        # If we don't have enough in this split, try the test split
        logger.info(f"Only {len(questions)} questions in '{split}' split. Trying 'test' split...")
        try:
            ds_test = load_dataset("cais/mmlu", "all", split="test")
            subject_test = ds_test.filter(lambda x: x["subject"] == subject)

            for item in subject_test:
                questions.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "subject": item["subject"],
                })
                if len(questions) >= limit:
                    break
        except Exception as e:
            logger.warning(f"Failed to load test split: {e}")

    logger.info(f"Total questions loaded: {len(questions)}")

    return questions[:limit]


# ─── Tier 2: Depth Probing ──────────────────────────────────────────

def generate_tier2_followups(question, model_answer, subject):
    """Generate Tier 2 (depth) follow-up questions.

    Tier 2 drills deeper into the model's answer to test depth of
    understanding. Follow-ups ask the model to explain, justify,
    or elaborate on its Tier 1 answer.

    Args:
        question: The original Tier 1 question text
        model_answer: The model's Tier 1 answer text
        subject: The subject area (e.g., 'high_school_mathematics')

    Returns:
        list[str]: Follow-up questions for Tier 2 probing
    """
    subject_display = subject.replace("_", " ").title()

    followups = [
        f"Regarding your answer to the previous question about {subject_display}, "
        f"can you explain the reasoning behind your answer? Walk through your thought process step by step.",

        f"You answered: \"{model_answer[:200]}\". "
        f"Why is that the correct answer? What principles or concepts support this?",

        f"If someone disagreed with your answer, what counterargument might they make? "
        f"How would you defend your answer against that objection?",

        f"What are the key assumptions underlying your answer to the question about {subject_display}? "
        f"Are there cases where those assumptions might not hold?",

        f"Can you provide a specific example or illustration that demonstrates why your answer is correct?",
    ]

    return followups


# ─── Tier 3: Edge Cases & Adversarial ──────────────────────────────

# Subject-specific common misconceptions for adversarial questions
_MISCONCEPTIONS = {
    "high_school_mathematics": [
        "A common misconception is that (a+b)^2 = a^2 + b^2. Explain why this is incorrect and provide the correct expansion.",
        "Some students believe that the derivative of a product is the product of derivatives. Is this correct? Explain why or why not.",
        "Is 0.999... equal to 1? Many people believe they are different. Explain the correct mathematical reasoning.",
    ],
    "high_school_physics": [
        "A common misconception is that heavier objects fall faster than lighter ones in a vacuum. Explain why this is incorrect.",
        "Some people believe that a force is always needed to keep an object moving. Is this true? Explain using Newton's laws.",
        "Is it true that energy is always conserved in every process? What about in inelastic collisions?",
    ],
    "high_school_chemistry": [
        "A common misconception is that atoms are the smallest particles that exist. Is this accurate? Explain.",
        "Some students believe that chemical bonds store energy. Explain why this is incorrect and what actually happens.",
        "Is it true that all reactions release energy? Distinguish between exothermic and endothermic reactions.",
    ],
    "high_school_biology": [
        "A common misconception is that evolution is a random process. Explain the role of natural selection.",
        "Some people believe that DNA directly produces proteins. What step is missing? Explain.",
        "Is it true that all bacteria are harmful? Explain the role of beneficial bacteria.",
    ],
    "high_school_us_history": [
        "A common misconception is that the Civil War was primarily about states' rights. Explain the central role of slavery.",
        "Some believe the American Revolution was universally popular. Explain the role of loyalists.",
        "Is it true that the Great Depression was caused solely by the stock market crash? Explain other contributing factors.",
    ],
    "high_school_world_history": [
        "A common misconception is that the Dark Ages were a period of no progress. Explain what was actually happening.",
        "Some believe the fall of Rome was sudden. Explain the gradual nature of the collapse.",
        "Is it accurate to say the Renaissance was a complete break from medieval thinking? Explain the continuity.",
    ],
}

# Generic edge-case questions that work for any subject
_GENERIC_EDGE_CASES = [
    "What is a common misconception about this topic that many people hold? Explain why it is wrong.",
    "What is an edge case or exception to the general rule you just described? When does the typical answer not apply?",
    "If the question conditions were slightly different (e.g., changed assumptions), how would the answer change?",
    "What would happen in the extreme case or boundary condition of this problem?",
    "What is a tricky or deceptive version of this question that might fool a student?",
]


def generate_tier3_edge_cases(subject):
    """Generate Tier 3 (edge cases) adversarial questions.

    Tier 3 tests the model's ability to handle trick questions,
    common misconceptions, and boundary conditions.

    Args:
        subject: The subject area (e.g., 'high_school_mathematics')

    Returns:
        list[str]: Adversarial/edge-case questions for Tier 3 probing
    """
    # Try subject-specific misconceptions first
    specific = _MISCONCEPTIONS.get(subject, [])
    result = list(specific)

    # Add generic edge cases
    result.extend(_GENERIC_EDGE_CASES)

    return result
