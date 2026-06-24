"""MMLU answer key loader and scoring."""

_LETTERS = ["A", "B", "C", "D"]


def score_answer(model_answer_letter, correct_answer_letter):
    """Score a model's answer against the correct answer.

    Args:
        model_answer_letter: The letter the model chose (A/B/C/D), or None
        correct_answer_letter: The correct letter (A/B/C/D)

    Returns:
        bool: True if correct, False otherwise
    """
    if model_answer_letter is None:
        return False
    return model_answer_letter.upper() == correct_answer_letter.upper()


def get_answer_letter(answer_idx):
    """Convert answer index (0-3) to letter (A-D)."""
    if 0 <= answer_idx < len(_LETTERS):
        return _LETTERS[answer_idx]
    return None


def get_answer_idx(answer_letter):
    """Convert answer letter (A-D) to index (0-3)."""
    letter = answer_letter.upper()
    if letter in _LETTERS:
        return _LETTERS.index(letter)
    return None
