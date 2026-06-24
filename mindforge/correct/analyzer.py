"""Error analysis for incorrect model responses.

Enhanced in Phase 2 with specific error types:
    - factual_error: Model's answer contains a factual inaccuracy
    - reasoning_error: Model's logic/reasoning is flawed
    - calculation_error: Model made a computational mistake
    - incomplete_answer: Model's answer is partially correct but incomplete
    - no_answer: Model did not provide a clear answer
"""

import re
import logging

logger = logging.getLogger(__name__)

_LETTERS = ["A", "B", "C", "D"]


def analyze_error(question, choices, model_answer_letter, correct_answer_idx,
                  model_response=None):
    """Analyze why a model got an answer wrong.

    Args:
        question: The question text
        choices: List of answer choices
        model_answer_letter: The letter the model chose (A/B/C/D), or None
        correct_answer_idx: Index of the correct answer (0-3)
        model_response: Optional full model response text for deeper analysis

    Returns:
        dict with keys:
            - error_type: One of factual_error, reasoning_error, calculation_error,
                         incomplete_answer, no_answer, close_confusion, distant_confusion
            - analysis: Human-readable analysis text
            - model_choice: The text of the model's chosen answer
            - correct_choice: The text of the correct answer
            - model_letter: The letter the model chose
            - correct_letter: The correct letter
    """
    correct_letter = _LETTERS[correct_answer_idx]
    correct_choice = choices[correct_answer_idx]

    if model_answer_letter is None:
        return {
            "error_type": "no_answer",
            "analysis": "The model did not provide a clear answer letter (A/B/C/D).",
            "model_choice": None,
            "correct_choice": correct_choice,
            "model_letter": None,
            "correct_letter": correct_letter,
        }

    if model_answer_letter not in _LETTERS:
        return {
            "error_type": "no_answer",
            "analysis": f"Model gave an invalid answer letter: {model_answer_letter}",
            "model_choice": None,
            "correct_choice": correct_choice,
            "model_letter": model_answer_letter,
            "correct_letter": correct_letter,
        }

    model_idx = _LETTERS.index(model_answer_letter)
    model_choice = choices[model_idx]

    # If model got it right
    if model_idx == correct_answer_idx:
        return {
            "error_type": "none",
            "analysis": "The model answered correctly.",
            "model_choice": model_choice,
            "correct_choice": correct_choice,
            "model_letter": model_answer_letter,
            "correct_letter": correct_letter,
        }

    # Determine specific error type
    error_type = _classify_error_type(
        question, choices, model_idx, correct_answer_idx, model_response
    )

    # Build analysis text
    analysis = (
        f"Model chose {model_answer_letter}) '{model_choice}', "
        f"but the correct answer is {correct_letter}) '{correct_choice}'. "
    )

    if error_type == "factual_error":
        analysis += "The model's answer contains a factual inaccuracy."
    elif error_type == "reasoning_error":
        analysis += "The model's reasoning is flawed -- the logical steps don't lead to the correct conclusion."
    elif error_type == "calculation_error":
        analysis += "The model made a computational or arithmetic error."
    elif error_type == "incomplete_answer":
        analysis += "The model's answer is partially correct but incomplete."
    elif error_type == "close_confusion":
        analysis += "The model confused a closely related answer."
    elif error_type == "distant_confusion":
        analysis += "The model's answer was significantly different from the correct one."

    return {
        "error_type": error_type,
        "analysis": analysis,
        "model_choice": model_choice,
        "correct_choice": correct_choice,
        "model_letter": model_answer_letter,
        "correct_letter": correct_letter,
    }


def _classify_error_type(question, choices, model_idx, correct_idx, model_response):
    """Classify the specific type of error based on context.

    Args:
        question: The question text
        choices: List of answer choices
        model_idx: Index the model chose
        correct_idx: Index of the correct answer
        model_response: Optional model response text

    Returns:
        str: Error type from the enhanced taxonomy
    """
    # If we have the model response, try to classify more precisely
    if model_response:
        response_lower = model_response.lower()

        # Check for calculation errors (math-related)
        calc_indicators = [
            r"\d+\s*[+\-*/]\s*\d+",  # arithmetic operations
            r"\bcalculation\b", r"\bcompute\b", r"\bequals\b",
            r"\bsum\b", r"\bproduct\b", r"\bdifference\b",
        ]
        math_question = any(
            kw in question.lower()
            for kw in ["calculate", "compute", "solve", "equation", "sum",
                       "product", "derivative", "integral", "evaluate"]
        )
        if math_question and any(re.search(p, response_lower) for p in calc_indicators):
            return "calculation_error"

        # Check for reasoning errors
        reasoning_indicators = [
            r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bso\b",
            r"\bbecause\b", r"\bsince\b", r"\bimplies\b",
        ]
        if any(re.search(p, response_lower) for p in reasoning_indicators):
            return "reasoning_error"

        # Check for incomplete answer
        incomplete_indicators = [
            r"\bnot sure\b", r"\bpartial\b", r"\bmaybe\b", r"\bpossibly\b",
            r"\bmight be\b", r"\bcould be\b", r"\bi think\b",
        ]
        if any(re.search(p, response_lower) for p in incomplete_indicators):
            return "incomplete_answer"

        # Default to factual error if there's a clear wrong answer
        return "factual_error"

    # Without model response, use positional heuristics (Phase 1 compatible)
    if abs(model_idx - correct_idx) == 1:
        return "close_confusion"
    else:
        return "distant_confusion"
