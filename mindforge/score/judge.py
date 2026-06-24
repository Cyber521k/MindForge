"""LLM-as-Judge scoring (used as fallback when no answer key exists)."""

import re
import json
import logging

logger = logging.getLogger(__name__)

_LETTERS = ["A", "B", "C", "D"]


class LLMJudge:
    """LLM-as-Judge for scoring model responses.

    When no MMLU answer key exists, a stronger model can be used to
    judge the correctness of the probed model's answer.

    The judge receives: question, model's answer, correct answer (if available).
    Returns: {correct: bool, confidence: float, explanation: str}
    """

    def __init__(self, model_adapter=None, model_name=None):
        """Initialize the LLM Judge.

        Args:
            model_adapter: A ModelAdapter instance (MLXAdapter, OpenAIAdapter, etc.)
                           used to make the judge LLM call. If None, falls back to
                           rule-based judging.
            model_name: Name of the judge model (for logging/metadata).
        """
        self.adapter = model_adapter
        self.model_name = model_name or "rule-based"

    def judge(self, question, model_answer, correct_answer=None):
        """Judge a model's answer to a question.

        This is the primary entry point for LLM-as-Judge scoring.
        When no MMLU answer key exists, use this to evaluate correctness.

        Args:
            question: The question text (string)
            model_answer: The model's answer text (string)
            correct_answer: The correct answer text if available (string or None)

        Returns:
            dict with keys:
                - correct (bool): whether the model's answer is correct
                - confidence (float): 0.0-1.0 confidence in the judgment
                - explanation (str): why the judge reached this verdict
        """
        if self.adapter is not None:
            return self._llm_judge(question, model_answer, correct_answer)
        return self._rule_based_judge(question, model_answer, correct_answer)

    def _llm_judge(self, question, model_answer, correct_answer):
        """Use the LLM adapter to judge correctness."""
        prompt = self._build_judge_prompt(question, model_answer, correct_answer)
        try:
            response = self.adapter.ask(prompt, max_tokens=512)
            return self._parse_judge_response(response)
        except Exception as e:
            logger.warning(f"LLM judge call failed: {e}. Falling back to rule-based.")
            return self._rule_based_judge(question, model_answer, correct_answer)

    def _build_judge_prompt(self, question, model_answer, correct_answer):
        """Build the prompt sent to the judge model."""
        lines = [
            "You are an expert judge evaluating a model's answer to a question.",
            "Analyze the model's answer for correctness and provide your verdict.",
            "",
            f"Question: {question}",
            "",
            f"Model's Answer: {model_answer}",
            "",
        ]
        if correct_answer:
            lines.append(f"Correct Answer: {correct_answer}")
            lines.append("")
        lines.extend([
            "Evaluate whether the model's answer is correct.",
            "Respond in the following JSON format only:",
            '{"correct": true/false, "confidence": 0.0-1.0, "explanation": "your reasoning"}',
        ])
        return "\n".join(lines)

    def _parse_judge_response(self, response):
        """Parse the LLM judge response into a structured verdict."""
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return {
                    "correct": bool(data.get("correct", False)),
                    "confidence": float(data.get("confidence", 0.5)),
                    "explanation": str(data.get("explanation", "No explanation provided.")),
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse judge response as JSON: {e}")

        # Fallback: try to infer from text
        response_lower = response.lower()
        if "correct" in response_lower and "incorrect" not in response_lower:
            correct = True
            confidence = 0.7
        elif "incorrect" in response_lower or "wrong" in response_lower:
            correct = False
            confidence = 0.3
        else:
            correct = False
            confidence = 0.5

        return {
            "correct": correct,
            "confidence": confidence,
            "explanation": response.strip()[:500],
        }

    def _rule_based_judge(self, question, model_answer, correct_answer):
        """Fallback rule-based judging when no LLM adapter is available."""
        from mindforge.probe.adapters import extract_answer_letter

        model_letter = extract_answer_letter(model_answer)

        if model_letter is None:
            return {
                "correct": False,
                "confidence": 0.0,
                "explanation": "Could not extract a valid answer from the model's response.",
            }

        if correct_answer is None:
            # Without a correct answer and without an LLM, we can't judge
            return {
                "correct": False,
                "confidence": 0.5,
                "explanation": "No correct answer provided and no LLM judge available. Unable to determine correctness.",
            }

        correct_letter = extract_answer_letter(correct_answer)
        if correct_letter:
            is_correct = model_letter.upper() == correct_letter.upper()
        else:
            # Compare text directly
            is_correct = model_answer.strip().lower() == correct_answer.strip().lower()

        if is_correct:
            return {
                "correct": True,
                "confidence": 0.9,
                "explanation": f"Model's answer matches the correct answer.",
            }
        else:
            return {
                "correct": False,
                "confidence": 0.3,
                "explanation": f"Model's answer does not match the correct answer.",
            }

    def judge_response(self, question, choices, model_response, correct_answer_idx):
        """Judge a model response (Phase 1 compatible API).

        Args:
            question: The question text
            choices: List of answer choices
            model_response: The model's response text
            correct_answer_idx: Index of correct answer (0-3)

        Returns:
            dict with keys: is_correct, reasoning, confidence
        """
        correct_letter = _LETTERS[correct_answer_idx]
        correct_choice = choices[correct_answer_idx]
        correct_answer_text = f"{correct_letter}) {correct_choice}"

        verdict = self.judge(question, model_response, correct_answer_text)

        return {
            "is_correct": verdict["correct"],
            "reasoning": verdict["explanation"],
            "confidence": verdict["confidence"],
        }
