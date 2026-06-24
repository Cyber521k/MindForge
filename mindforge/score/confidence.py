"""Confidence scoring for model responses.

Confidence scale (from design doc):
    1.0 = Answer key match (ground truth)
    0.9 = LLM judge correct + RAG verified
    0.7 = LLM judge correct, no RAG
    0.5 = LLM judge uncertain
    0.3 = LLM judge wrong, RAG agrees
    0.0 = Answer key wrong

Auto-approve threshold: 0.7
    >= 0.7 -> auto_approved (goes to Correct Vault)
    < 0.7 and > 0.0 -> needs_review (goes to Review Queue)
    == 0.0 -> rejected
"""

import re

# Confidence constants matching the design doc scale
CONFIDENCE_ANSWER_KEY_MATCH = 1.0
CONFIDENCE_JUDGE_RAG = 0.9
CONFIDENCE_JUDGE_NO_RAG = 0.7
CONFIDENCE_JUDGE_UNCERTAIN = 0.5
CONFIDENCE_JUDGE_WRONG_RAG_AGREES = 0.3
CONFIDENCE_ANSWER_KEY_WRONG = 0.0

# Auto-approve threshold
AUTO_APPROVE_THRESHOLD = 0.7


def compute_confidence(is_correct, model_response, model_answer_letter):
    """Compute a confidence score (0.0-1.0) for a model response.

    Heuristics:
    - If answer is None (couldn't extract), confidence 0.0
    - If correct: base 0.8, +0.2 if response contains hedging language
    - If incorrect: base 0.2, +0.3 if response seems very confident

    Args:
        is_correct: Whether the answer was correct
        model_response: The full model response text
        model_answer_letter: The extracted answer letter (or None)

    Returns:
        float between 0.0 and 1.0
    """
    if model_answer_letter is None:
        return 0.0

    response_lower = model_response.lower()

    # Hedging language patterns
    hedging_patterns = [
        r"\bi think\b", r"\bi believe\b", r"\bmaybe\b", r"\bperhaps\b",
        r"\bnot sure\b", r"\buncertain\b", r"\bcould be\b", r"\bmight be\b",
        r"\bprobably\b", r"\blikely\b",
    ]

    # Confident language patterns
    confident_patterns = [
        r"\bdefinitely\b", r"\bcertainly\b", r"\babsolutely\b",
        r"\bclearly\b", r"\bobviously\b", r"\bthe answer is\b",
    ]

    has_hedging = any(re.search(p, response_lower) for p in hedging_patterns)
    has_confident = any(re.search(p, response_lower) for p in confident_patterns)

    if is_correct:
        base = 0.8
        if has_hedging:
            base += 0.15
        if has_confident:
            base += 0.05
    else:
        base = 0.2
        if has_confident:
            base += 0.3
        if has_hedging:
            base -= 0.1

    return max(0.0, min(1.0, base))


def compute_confidence_with_judge(
    is_correct_answer_key,
    judge_verdict=None,
    rag_verified=False,
):
    """Compute confidence using the full design-doc confidence scale.

    This integrates Layer 1 (answer key) and Layer 2 (LLM judge) results
    to produce the final confidence score.

    Args:
        is_correct_answer_key: True/False/None
            - True if answer key confirms correctness
            - False if answer key confirms incorrectness
            - None if no answer key available
        judge_verdict: dict from LLMJudge.judge() with keys:
            - correct (bool or None)
            - confidence (float)
            - explanation (str)
            None if judge was not used.
        rag_verified: Whether RAG verification confirmed the answer.

    Returns:
        float between 0.0 and 1.0
    """
    # Layer 1: Answer key (ground truth)
    if is_correct_answer_key is True:
        return CONFIDENCE_ANSWER_KEY_MATCH  # 1.0

    if is_correct_answer_key is False:
        return CONFIDENCE_ANSWER_KEY_WRONG  # 0.0

    # Layer 2: LLM Judge (no answer key available)
    if judge_verdict is not None:
        judge_correct = judge_verdict.get("correct")
        judge_confidence = judge_verdict.get("confidence", 0.5)

        if judge_correct is True:
            if rag_verified:
                return CONFIDENCE_JUDGE_RAG  # 0.9
            else:
                return CONFIDENCE_JUDGE_NO_RAG  # 0.7

        elif judge_correct is False:
            if rag_verified:
                return CONFIDENCE_JUDGE_WRONG_RAG_AGREES  # 0.3
            else:
                # Judge says wrong but no RAG -- moderate low confidence
                return 0.2

        else:
            # Judge is uncertain
            return CONFIDENCE_JUDGE_UNCERTAIN  # 0.5

    # No answer key, no judge -- very uncertain
    return CONFIDENCE_JUDGE_UNCERTAIN  # 0.5


def should_auto_approve(confidence):
    """Determine if a response should be auto-approved based on confidence.

    Auto-approve threshold is 0.7. Responses with confidence >= 0.7
    go directly to the Correct Vault.

    Args:
        confidence: Float confidence score (0.0-1.0)

    Returns:
        bool: True if the response should be auto-approved
    """
    return confidence >= AUTO_APPROVE_THRESHOLD


def classify_result(confidence):
    """Classify a result based on its confidence score.

    Args:
        confidence: Float confidence score (0.0-1.0)

    Returns:
        str: One of "auto_approved", "needs_review", or "rejected"
    """
    if confidence >= AUTO_APPROVE_THRESHOLD:
        return "auto_approved"
    elif confidence <= 0.0:
        return "rejected"
    else:
        return "needs_review"
