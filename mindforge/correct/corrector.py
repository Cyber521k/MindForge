"""Corrected answer formulation.

Enhanced in Phase 2 to generate proper corrected responses with
explanations, not just the answer letter.
"""

_LETTERS = ["A", "B", "C", "D"]


def formulate_correction(correct_answer_idx, choices, question=None, model_response=None):
    """Formulate a corrected answer response.

    For Phase 1, this simply produces a clear, correct answer statement.
    For Phase 2, it also generates an explanation when context is available.

    Args:
        correct_answer_idx: Index of correct answer (0-3)
        choices: List of answer choices
        question: Optional question text for context
        model_response: Optional original model response for reference

    Returns:
        str: The corrected answer text
    """
    correct_letter = _LETTERS[correct_answer_idx]
    correct_choice = choices[correct_answer_idx]

    return f"The answer is {correct_letter}) {correct_choice}."


def formulate_correction_full(correct_answer_idx, choices, question=None,
                               model_response=None, error_analysis=None):
    """Formulate a full corrected answer with explanation.

    Generates a proper corrected response that includes:
    - The correct answer
    - Why it's correct
    - What the model got wrong (if applicable)

    Args:
        correct_answer_idx: Index of correct answer (0-3)
        choices: List of answer choices
        question: The question text
        model_response: The original model response (what it got wrong)
        error_analysis: Optional dict from analyze_error() with error_type, analysis, etc.

    Returns:
        str: The full corrected answer text with explanation
    """
    correct_letter = _LETTERS[correct_answer_idx]
    correct_choice = choices[correct_answer_idx]

    lines = [f"The correct answer is {correct_letter}) {correct_choice}."]

    # Add explanation if we have context
    if question:
        # Provide a brief explanation of why this answer is correct
        lines.append("")
        lines.append(f"Explanation: {correct_choice} is the correct answer to this question.")

    if error_analysis and error_analysis.get("error_type") != "none":
        error_type = error_analysis.get("error_type", "unknown")
        analysis_text = error_analysis.get("analysis", "")

        lines.append("")

        # Map error types to helpful descriptions
        error_descriptions = {
            "factual_error": "The original response contained a factual inaccuracy.",
            "reasoning_error": "The original response had flawed reasoning.",
            "calculation_error": "The original response contained a calculation error.",
            "incomplete_answer": "The original response was incomplete.",
            "no_answer": "The original response did not provide a clear answer.",
            "close_confusion": "The original response confused a closely related option.",
            "distant_confusion": "The original response was significantly different from the correct answer.",
        }

        desc = error_descriptions.get(error_type, analysis_text)
        lines.append(f"Error type: {error_type}.")
        lines.append(desc)

    if model_response:
        lines.append("")
        lines.append(f"The model's original response was: \"{model_response[:200]}\"")

    return " ".join(lines)


def generate_corrected_answer(question, choices, correct_answer_idx,
                               model_response=None, error_analysis=None):
    """Generate a full corrected answer text.

    This is the main Phase 2 function for producing corrected responses.
    It combines the correct answer with an explanation of the error.

    Args:
        question: The question text
        choices: List of answer choices
        correct_answer_idx: Index of correct answer (0-3)
        model_response: The original (incorrect) model response
        error_analysis: Optional dict from analyze_error()

    Returns:
        str: The full corrected answer text
    """
    return formulate_correction_full(
        correct_answer_idx=correct_answer_idx,
        choices=choices,
        question=question,
        model_response=model_response,
        error_analysis=error_analysis,
    )


def formulate_rejection(model_answer_letter, choices, question=None):
    """Formulate the rejected (incorrect) answer response.

    Args:
        model_answer_letter: The letter the model chose
        choices: List of answer choices
        question: Optional question text

    Returns:
        str: The rejected answer text
    """
    if model_answer_letter is None or model_answer_letter not in _LETTERS:
        return "I'm not sure about the answer to this question."

    idx = _LETTERS.index(model_answer_letter)
    choice_text = choices[idx] if idx < len(choices) else "Unknown"

    return f"The answer is {model_answer_letter}) {choice_text}."
