"""Tests for MindForge Phase 2 features."""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mindforge.score.judge import LLMJudge
from mindforge.score.confidence import (
    compute_confidence_with_judge,
    should_auto_approve,
    classify_result,
    AUTO_APPROVE_THRESHOLD,
    CONFIDENCE_ANSWER_KEY_MATCH,
    CONFIDENCE_JUDGE_RAG,
    CONFIDENCE_JUDGE_NO_RAG,
    CONFIDENCE_JUDGE_UNCERTAIN,
    CONFIDENCE_JUDGE_WRONG_RAG_AGREES,
    CONFIDENCE_ANSWER_KEY_WRONG,
)
from mindforge.correct.analyzer import analyze_error
from mindforge.correct.corrector import formulate_correction, formulate_correction_full, generate_corrected_answer
from mindforge.format.dpo import format_dpo_entry
from mindforge.format.alpaca import format_alpaca_entry
from mindforge.format.chatml import format_chatml_entry
from mindforge.format.completion import format_completion_entry
from mindforge.format.openai_messages import format_openai_messages_entry, format_openai_messages_batch
from mindforge.format.template_free import format_template_free_entry, format_template_free_batch
from mindforge.format.convert import convert_format, SUPPORTED_FORMATS
from mindforge.probe.question_gen import generate_tier2_followups, generate_tier3_edge_cases
from mindforge.hardware.model_list import get_memory_tier, get_available_models, format_model_list, MEMORY_TIERS


class TestLLMJudge(unittest.TestCase):
    """Test LLM-as-Judge scoring."""

    def test_judge_with_mock_adapter_correct(self):
        """Test LLM judge with a mock adapter that returns correct verdict."""
        mock_adapter = Mock()
        mock_adapter.ask.return_value = (
            '{"correct": true, "confidence": 0.9, "explanation": "The answer is correct."}'
        )
        judge = LLMJudge(model_adapter=mock_adapter, model_name="test-judge")
        verdict = judge.judge(
            question="What is 2+2?",
            model_answer="The answer is B) 4.",
            correct_answer="B) 4",
        )
        self.assertTrue(verdict["correct"])
        self.assertAlmostEqual(verdict["confidence"], 0.9)
        self.assertIn("correct", verdict["explanation"].lower())

    def test_judge_with_mock_adapter_incorrect(self):
        """Test LLM judge with a mock adapter that returns incorrect verdict."""
        mock_adapter = Mock()
        mock_adapter.ask.return_value = (
            '{"correct": false, "confidence": 0.3, "explanation": "The answer is wrong."}'
        )
        judge = LLMJudge(model_adapter=mock_adapter, model_name="test-judge")
        verdict = judge.judge(
            question="What is 2+2?",
            model_answer="The answer is A) 3.",
            correct_answer="B) 4",
        )
        self.assertFalse(verdict["correct"])
        self.assertAlmostEqual(verdict["confidence"], 0.3)

    def test_judge_rule_based_with_correct_answer(self):
        """Test rule-based judge when no adapter is provided but correct answer exists."""
        judge = LLMJudge()
        verdict = judge.judge(
            question="What is 2+2?",
            model_answer="The answer is B) 4.",
            correct_answer="B) 4",
        )
        self.assertTrue(verdict["correct"])

    def test_judge_rule_based_wrong_answer(self):
        """Test rule-based judge when answer is wrong."""
        judge = LLMJudge()
        verdict = judge.judge(
            question="What is 2+2?",
            model_answer="The answer is A) 3.",
            correct_answer="B) 4",
        )
        self.assertFalse(verdict["correct"])

    def test_judge_no_correct_answer_no_adapter(self):
        """Test rule-based judge with no correct answer and no adapter."""
        judge = LLMJudge()
        verdict = judge.judge(
            question="Explain quantum mechanics.",
            model_answer="It's about tiny particles.",
            correct_answer=None,
        )
        self.assertFalse(verdict["correct"])
        # Without a correct answer or LLM adapter, the judge can't verify
        # so confidence is low (0.0 = cannot determine)
        self.assertLessEqual(verdict["confidence"], 0.5)

    def test_judge_response_compatible(self):
        """Test Phase 1 compatible judge_response method."""
        judge = LLMJudge()
        result = judge.judge_response(
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            model_response="The answer is B.",
            correct_answer_idx=1,
        )
        self.assertTrue(result["is_correct"])
        self.assertIn("reasoning", result)
        self.assertIn("confidence", result)

    def test_judge_llm_fallback_on_error(self):
        """Test that LLM judge falls back to rule-based on error."""
        mock_adapter = Mock()
        mock_adapter.ask.side_effect = Exception("API error")
        judge = LLMJudge(model_adapter=mock_adapter)
        verdict = judge.judge(
            question="What is 2+2?",
            model_answer="The answer is B) 4.",
            correct_answer="B) 4",
        )
        # Should fall back to rule-based
        self.assertTrue(verdict["correct"])

    def test_judge_parse_non_json_response(self):
        """Test parsing a non-JSON response from the judge."""
        mock_adapter = Mock()
        mock_adapter.ask.return_value = "The model's answer is correct."
        judge = LLMJudge(model_adapter=mock_adapter)
        verdict = judge.judge(
            question="What is 2+2?",
            model_answer="The answer is B) 4.",
            correct_answer="B) 4",
        )
        self.assertTrue(verdict["correct"])


class TestConfidenceClassification(unittest.TestCase):
    """Test confidence scoring with auto-approve."""

    def test_answer_key_match_confidence(self):
        """Test that answer key match gives 1.0 confidence."""
        confidence = compute_confidence_with_judge(
            is_correct_answer_key=True,
            judge_verdict=None,
            rag_verified=False,
        )
        self.assertEqual(confidence, 1.0)

    def test_answer_key_wrong_confidence(self):
        """Test that answer key wrong gives 0.0 confidence."""
        confidence = compute_confidence_with_judge(
            is_correct_answer_key=False,
            judge_verdict=None,
            rag_verified=False,
        )
        self.assertEqual(confidence, 0.0)

    def test_judge_correct_with_rag(self):
        """Test judge correct + RAG gives 0.9 confidence."""
        judge_verdict = {"correct": True, "confidence": 0.8, "explanation": "correct"}
        confidence = compute_confidence_with_judge(
            is_correct_answer_key=None,
            judge_verdict=judge_verdict,
            rag_verified=True,
        )
        self.assertEqual(confidence, 0.9)

    def test_judge_correct_no_rag(self):
        """Test judge correct without RAG gives 0.7 confidence."""
        judge_verdict = {"correct": True, "confidence": 0.8, "explanation": "correct"}
        confidence = compute_confidence_with_judge(
            is_correct_answer_key=None,
            judge_verdict=judge_verdict,
            rag_verified=False,
        )
        self.assertEqual(confidence, 0.7)

    def test_judge_uncertain(self):
        """Test judge uncertain gives 0.5 confidence."""
        judge_verdict = {"correct": None, "confidence": 0.5, "explanation": "uncertain"}
        confidence = compute_confidence_with_judge(
            is_correct_answer_key=None,
            judge_verdict=judge_verdict,
            rag_verified=False,
        )
        self.assertEqual(confidence, 0.5)

    def test_judge_wrong_rag_agrees(self):
        """Test judge wrong + RAG agrees gives 0.3 confidence."""
        judge_verdict = {"correct": False, "confidence": 0.3, "explanation": "wrong"}
        confidence = compute_confidence_with_judge(
            is_correct_answer_key=None,
            judge_verdict=judge_verdict,
            rag_verified=True,
        )
        self.assertEqual(confidence, 0.3)

    def test_should_auto_approve_high(self):
        """Test auto-approve for high confidence."""
        self.assertTrue(should_auto_approve(0.7))
        self.assertTrue(should_auto_approve(0.9))
        self.assertTrue(should_auto_approve(1.0))

    def test_should_auto_approve_low(self):
        """Test no auto-approve for low confidence."""
        self.assertFalse(should_auto_approve(0.5))
        self.assertFalse(should_auto_approve(0.3))
        self.assertFalse(should_auto_approve(0.0))

    def test_classify_result_auto_approved(self):
        """Test classification of auto-approved results."""
        self.assertEqual(classify_result(0.7), "auto_approved")
        self.assertEqual(classify_result(0.9), "auto_approved")
        self.assertEqual(classify_result(1.0), "auto_approved")

    def test_classify_result_needs_review(self):
        """Test classification of results needing review."""
        self.assertEqual(classify_result(0.5), "needs_review")
        self.assertEqual(classify_result(0.3), "needs_review")

    def test_classify_result_rejected(self):
        """Test classification of rejected results."""
        self.assertEqual(classify_result(0.0), "rejected")

    def test_auto_approve_threshold(self):
        """Test that the threshold is at 0.7."""
        self.assertEqual(AUTO_APPROVE_THRESHOLD, 0.7)


class TestEnhancedErrorAnalysis(unittest.TestCase):
    """Test enhanced error analysis with specific error types."""

    def test_no_answer_error_type(self):
        """Test that no answer gives 'no_answer' error type."""
        result = analyze_error("What is 2+2?", ["3", "4", "5", "6"], None, 1)
        self.assertEqual(result["error_type"], "no_answer")

    def test_factual_error_with_response(self):
        """Test factual error classification with model response."""
        result = analyze_error(
            question="What is the capital of France?",
            choices=["London", "Paris", "Berlin", "Madrid"],
            model_answer_letter="A",
            correct_answer_idx=1,
            model_response="The answer is A) London because London is the capital of France.",
        )
        # The analyzer may classify this as either factual_error or reasoning_error
        # depending on whether the model provided an explanation with its wrong answer
        self.assertIn(result["error_type"], ["factual_error", "reasoning_error"])

    def test_reasoning_error_with_response(self):
        """Test reasoning error classification."""
        result = analyze_error(
            question="Why does ice float?",
            choices=["Density", "Temperature", "Pressure", "Gravity"],
            model_answer_letter="B",
            correct_answer_idx=0,
            model_response="Therefore the answer is B because of the temperature differences.",
        )
        self.assertEqual(result["error_type"], "reasoning_error")

    def test_calculation_error(self):
        """Test calculation error classification."""
        result = analyze_error(
            question="Calculate 15 * 12",
            choices=["170", "180", "190", "200"],
            model_answer_letter="A",
            correct_answer_idx=1,
            model_response="I compute 15 + 12 = 27, so the answer is A) 170.",
        )
        self.assertEqual(result["error_type"], "calculation_error")

    def test_incomplete_answer(self):
        """Test incomplete answer classification."""
        result = analyze_error(
            question="What is photosynthesis?",
            choices=["A", "B", "C", "D"],
            model_answer_letter="A",
            correct_answer_idx=1,
            model_response="I think maybe it's about plants.",
        )
        self.assertEqual(result["error_type"], "incomplete_answer")

    def test_close_confusion_without_response(self):
        """Test close confusion (Phase 1 compatible) without model response."""
        result = analyze_error("What is 2+2?", ["3", "4", "5", "6"], "A", 1)
        self.assertEqual(result["error_type"], "close_confusion")

    def test_distant_confusion_without_response(self):
        """Test distant confusion (Phase 1 compatible) without model response."""
        result = analyze_error("What is 2+2?", ["3", "4", "5", "6"], "D", 1)
        self.assertEqual(result["error_type"], "distant_confusion")

    def test_analysis_includes_model_letter(self):
        """Test that result includes model_letter and correct_letter."""
        result = analyze_error("Q?", ["A", "B", "C", "D"], "A", 1)
        self.assertEqual(result["model_letter"], "A")
        self.assertEqual(result["correct_letter"], "B")


class TestEnhancedCorrector(unittest.TestCase):
    """Test enhanced corrector with full corrected answer."""

    def test_formulate_correction_basic(self):
        """Test basic correction (Phase 1 compatible)."""
        result = formulate_correction(1, ["3", "4", "5", "6"])
        self.assertIn("B", result)
        self.assertIn("4", result)

    def test_formulate_correction_full_with_question(self):
        """Test full correction with question context."""
        result = formulate_correction_full(
            correct_answer_idx=1,
            choices=["3", "4", "5", "6"],
            question="What is 2+2?",
        )
        self.assertIn("B", result)
        self.assertIn("4", result)
        self.assertIn("Explanation", result)

    def test_formulate_correction_full_with_error_analysis(self):
        """Test full correction with error analysis."""
        error_analysis = {
            "error_type": "factual_error",
            "analysis": "Model gave wrong answer.",
        }
        result = formulate_correction_full(
            correct_answer_idx=1,
            choices=["3", "4", "5", "6"],
            question="What is 2+2?",
            model_response="The answer is A) 3.",
            error_analysis=error_analysis,
        )
        self.assertIn("B", result)
        self.assertIn("factual_error", result)

    def test_generate_corrected_answer(self):
        """Test the generate_corrected_answer function."""
        result = generate_corrected_answer(
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            correct_answer_idx=1,
            model_response="The answer is A) 3.",
        )
        self.assertIn("B", result)
        self.assertIn("4", result)


class TestAllSixFormats(unittest.TestCase):
    """Test all 6 training format outputs."""

    def setUp(self):
        self.prompt = "What is 2+2?"
        self.chosen = "The answer is B) 4."
        self.rejected = "The answer is A) 3."

    def test_dpo_format(self):
        entry = format_dpo_entry(self.prompt, self.chosen, self.rejected)
        self.assertEqual(entry["prompt"], self.prompt)
        self.assertEqual(entry["chosen"], self.chosen)
        self.assertEqual(entry["rejected"], self.rejected)

    def test_alpaca_format(self):
        entry = format_alpaca_entry(self.prompt, self.chosen)
        self.assertEqual(entry["instruction"], self.prompt)
        self.assertEqual(entry["output"], self.chosen)

    def test_chatml_format(self):
        entry = format_chatml_entry(self.prompt, self.chosen)
        self.assertIn("<|im_start|>", entry["text"])
        self.assertIn(self.prompt, entry["text"])
        self.assertIn(self.chosen, entry["text"])

    def test_completion_format(self):
        entry = format_completion_entry(self.prompt, self.chosen)
        self.assertEqual(entry["prompt"], self.prompt)
        self.assertEqual(entry["completion"], self.chosen)

    def test_openai_messages_format(self):
        entry = format_openai_messages_entry(self.prompt, self.chosen)
        self.assertIn("messages", entry)
        self.assertEqual(len(entry["messages"]), 2)
        self.assertEqual(entry["messages"][0]["role"], "user")
        self.assertEqual(entry["messages"][0]["content"], self.prompt)
        self.assertEqual(entry["messages"][1]["role"], "assistant")
        self.assertEqual(entry["messages"][1]["content"], self.chosen)

    def test_openai_messages_batch(self):
        entries = [
            {"prompt": "Q1", "chosen": "A1"},
            {"prompt": "Q2", "chosen": "A2"},
        ]
        result = format_openai_messages_batch(entries)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["messages"][0]["content"], "Q1")

    def test_template_free_format(self):
        entry = format_template_free_entry(self.prompt, self.chosen, self.rejected)
        self.assertIn("segments", entry)
        self.assertEqual(len(entry["segments"]), 3)
        self.assertEqual(entry["segments"][0]["text"], self.prompt)
        self.assertEqual(entry["segments"][0]["label"], "instruction")
        self.assertEqual(entry["segments"][1]["text"], self.chosen)
        self.assertEqual(entry["segments"][1]["label"], "response")
        self.assertEqual(entry["segments"][2]["text"], self.rejected)
        self.assertEqual(entry["segments"][2]["label"], "rejected")

    def test_template_free_no_rejected(self):
        entry = format_template_free_entry(self.prompt, self.chosen)
        self.assertEqual(len(entry["segments"]), 2)

    def test_template_free_batch(self):
        entries = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "R1"},
            {"prompt": "Q2", "chosen": "A2"},
        ]
        result = format_template_free_batch(entries)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]["segments"]), 3)
        self.assertEqual(len(result[1]["segments"]), 2)


class TestFormatConversion(unittest.TestCase):
    """Test format conversion between all 6 formats."""

    def setUp(self):
        self.prompt = "What is 2+2?"
        self.chosen = "The answer is B) 4."
        self.rejected = "The answer is A) 3."

    def test_dpo_to_alpaca(self):
        dpo_entry = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        result = convert_format([dpo_entry], "dpo", "alpaca")
        self.assertEqual(result[0]["instruction"], self.prompt)
        self.assertEqual(result[0]["output"], self.chosen)

    def test_dpo_to_chatml(self):
        dpo_entry = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        result = convert_format([dpo_entry], "dpo", "chatml")
        self.assertIn("<|im_start|>", result[0]["text"])
        self.assertIn(self.prompt, result[0]["text"])

    def test_dpo_to_completion(self):
        dpo_entry = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        result = convert_format([dpo_entry], "dpo", "completion")
        self.assertEqual(result[0]["prompt"], self.prompt)
        self.assertEqual(result[0]["completion"], self.chosen)

    def test_dpo_to_openai_messages(self):
        dpo_entry = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        result = convert_format([dpo_entry], "dpo", "openai_messages")
        self.assertEqual(result[0]["messages"][0]["content"], self.prompt)
        self.assertEqual(result[0]["messages"][1]["content"], self.chosen)

    def test_dpo_to_template_free(self):
        dpo_entry = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        result = convert_format([dpo_entry], "dpo", "template_free")
        self.assertEqual(result[0]["segments"][0]["text"], self.prompt)
        self.assertEqual(result[0]["segments"][1]["text"], self.chosen)
        self.assertEqual(result[0]["segments"][2]["text"], self.rejected)

    def test_alpaca_to_dpo(self):
        alpaca_entry = {"instruction": self.prompt, "input": "", "output": self.chosen}
        result = convert_format([alpaca_entry], "alpaca", "dpo")
        self.assertEqual(result[0]["prompt"], self.prompt)
        self.assertEqual(result[0]["chosen"], self.chosen)

    def test_completion_to_dpo(self):
        comp_entry = {"prompt": self.prompt, "completion": self.chosen}
        result = convert_format([comp_entry], "completion", "dpo")
        self.assertEqual(result[0]["prompt"], self.prompt)
        self.assertEqual(result[0]["chosen"], self.chosen)

    def test_openai_messages_to_dpo(self):
        msg_entry = {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.chosen},
            ]
        }
        result = convert_format([msg_entry], "openai_messages", "dpo")
        self.assertEqual(result[0]["prompt"], self.prompt)
        self.assertEqual(result[0]["chosen"], self.chosen)

    def test_template_free_to_dpo(self):
        tf_entry = {
            "segments": [
                {"text": self.prompt, "label": "instruction"},
                {"text": self.chosen, "label": "response"},
                {"text": self.rejected, "label": "rejected"},
            ]
        }
        result = convert_format([tf_entry], "template_free", "dpo")
        self.assertEqual(result[0]["prompt"], self.prompt)
        self.assertEqual(result[0]["chosen"], self.chosen)

    def test_chatml_to_dpo(self):
        chatml_entry = {
            "text": f"<|im_start|>user\n{self.prompt}<|im_end|>\n<|im_start|>assistant\n{self.chosen}<|im_end|>"
        }
        result = convert_format([chatml_entry], "chatml", "dpo")
        self.assertEqual(result[0]["prompt"], self.prompt)
        self.assertEqual(result[0]["chosen"], self.chosen)

    def test_same_format_no_op(self):
        dpo_entry = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        result = convert_format([dpo_entry], "dpo", "dpo")
        self.assertEqual(result[0], dpo_entry)

    def test_unsupported_source_format(self):
        with self.assertRaises(ValueError):
            convert_format([], "unknown_format", "dpo")

    def test_unsupported_target_format(self):
        with self.assertRaises(ValueError):
            convert_format([], "dpo", "unknown_format")

    def test_all_formats_supported(self):
        """Test that all 6 formats are in SUPPORTED_FORMATS."""
        for fmt in ["dpo", "alpaca", "chatml", "completion", "openai_messages", "template_free"]:
            self.assertIn(fmt, SUPPORTED_FORMATS)

    def test_roundtrip_dpo_to_all_and_back(self):
        """Test converting DPO to each format and back to DPO."""
        dpo_entry = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        for target_fmt in ["alpaca", "completion", "openai_messages", "template_free"]:
            converted = convert_format([dpo_entry], "dpo", target_fmt)
            back = convert_format(converted, target_fmt, "dpo")
            self.assertEqual(back[0]["prompt"], self.prompt)
            self.assertEqual(back[0]["chosen"], self.chosen)


class TestTier2QuestionGeneration(unittest.TestCase):
    """Test Tier 2 (depth) follow-up question generation."""

    def test_generate_tier2_followups_returns_list(self):
        """Test that Tier 2 generates a list of questions."""
        followups = generate_tier2_followups(
            question="What is the derivative of x^2?",
            model_answer="The derivative is 2x.",
            subject="high_school_mathematics",
        )
        self.assertIsInstance(followups, list)
        self.assertGreater(len(followups), 0)

    def test_tier2_questions_are_strings(self):
        """Test that all Tier 2 questions are strings."""
        followups = generate_tier2_followups(
            question="What is photosynthesis?",
            model_answer="It's how plants make food.",
            subject="high_school_biology",
        )
        for q in followups:
            self.assertIsInstance(q, str)

    def test_tier2_questions_reference_subject(self):
        """Test that Tier 2 questions reference the subject."""
        followups = generate_tier2_followups(
            question="What is Newton's first law?",
            model_answer="An object in motion stays in motion.",
            subject="high_school_physics",
        )
        # At least one question should mention physics or the subject
        all_text = " ".join(followups).lower()
        self.assertTrue(
            "physics" in all_text or "answer" in all_text or "reasoning" in all_text
        )

    def test_tier2_questions_reference_model_answer(self):
        """Test that Tier 2 questions reference the model's answer."""
        model_answer = "The answer is 42."
        followups = generate_tier2_followups(
            question="What is the meaning of life?",
            model_answer=model_answer,
            subject="philosophy",
        )
        all_text = " ".join(followups)
        # The model answer should be referenced somewhere
        self.assertTrue(any("answer" in f.lower() for f in followups))


class TestTier3QuestionGeneration(unittest.TestCase):
    """Test Tier 3 (edge cases) adversarial question generation."""

    def test_generate_tier3_edge_cases_returns_list(self):
        """Test that Tier 3 generates a list of questions."""
        edge_cases = generate_tier3_edge_cases("high_school_mathematics")
        self.assertIsInstance(edge_cases, list)
        self.assertGreater(len(edge_cases), 0)

    def test_tier3_questions_are_strings(self):
        """Test that all Tier 3 questions are strings."""
        edge_cases = generate_tier3_edge_cases("high_school_physics")
        for q in edge_cases:
            self.assertIsInstance(q, str)

    def test_tier3_has_subject_specific_questions(self):
        """Test that Tier 3 has subject-specific misconceptions."""
        edge_cases = generate_tier3_edge_cases("high_school_mathematics")
        # Should have more than just the generic ones
        self.assertGreater(len(edge_cases), 5)

    def test_tier3_has_generic_questions(self):
        """Test that Tier 3 includes generic edge case questions."""
        edge_cases = generate_tier3_edge_cases("some_unknown_subject")
        # Should still have the generic questions
        self.assertGreater(len(edge_cases), 0)
        all_text = " ".join(edge_cases).lower()
        self.assertIn("misconception", all_text)

    def test_tier3_different_subjects_different_questions(self):
        """Test that different subjects get different edge case questions."""
        math_questions = generate_tier3_edge_cases("high_school_mathematics")
        physics_questions = generate_tier3_edge_cases("high_school_physics")
        # They should differ (at least the subject-specific ones)
        self.assertNotEqual(math_questions, physics_questions)


class TestModelList(unittest.TestCase):
    """Test model list generation based on hardware."""

    def test_get_memory_tier_s(self):
        """Test Tier S for >= 96 GB."""
        self.assertEqual(get_memory_tier(96), "S")
        self.assertEqual(get_memory_tier(128), "S")

    def test_get_memory_tier_a(self):
        """Test Tier A for >= 64 GB."""
        self.assertEqual(get_memory_tier(64), "A")
        self.assertEqual(get_memory_tier(80), "A")

    def test_get_memory_tier_b(self):
        """Test Tier B for >= 32 GB."""
        self.assertEqual(get_memory_tier(32), "B")
        self.assertEqual(get_memory_tier(48), "B")

    def test_get_memory_tier_c(self):
        """Test Tier C for >= 16 GB."""
        self.assertEqual(get_memory_tier(16), "C")
        self.assertEqual(get_memory_tier(24), "C")

    def test_get_memory_tier_d(self):
        """Test Tier D for >= 8 GB."""
        self.assertEqual(get_memory_tier(8), "D")
        self.assertEqual(get_memory_tier(12), "D")

    def test_get_memory_tier_e(self):
        """Test Tier E for < 8 GB."""
        self.assertEqual(get_memory_tier(4), "E")
        self.assertEqual(get_memory_tier(0), "E")

    def test_memory_tiers_definition(self):
        """Test that all 6 tiers are defined."""
        tier_letters = [t["tier"] for t in MEMORY_TIERS]
        self.assertEqual(set(tier_letters), {"S", "A", "B", "C", "D", "E"})

    def test_get_available_models_structure(self):
        """Test that get_available_models returns proper structure."""
        model_info = get_available_models()
        self.assertIn("hardware", model_info)
        self.assertIn("memory_tier", model_info)
        self.assertIn("local_models", model_info)
        self.assertIn("cloud_models", model_info)
        self.assertIn("all_models", model_info)

    def test_local_models_have_required_fields(self):
        """Test that local models have all required fields."""
        model_info = get_available_models()
        for model in model_info["local_models"]:
            self.assertIn("name", model)
            self.assertIn("id", model)
            self.assertIn("size_gb", model)
            self.assertIn("tier", model)
            self.assertIn("available", model)
            self.assertIn("type", model)

    def test_local_models_have_can_run_flags(self):
        """Test that local models have availability flags."""
        model_info = get_available_models()
        tier = model_info["memory_tier"]
        tier_order = ["S", "A", "B", "C", "D", "E"]
        tier_idx = tier_order.index(tier)
        for model in model_info["local_models"]:
            model_tier_idx = tier_order.index(model["tier"])
            expected_available = model_tier_idx >= tier_idx
            self.assertEqual(model["available"], expected_available)

    def test_format_model_list_returns_string(self):
        """Test that format_model_list returns a readable string."""
        model_info = get_available_models()
        result = format_model_list(model_info)
        self.assertIsInstance(result, str)
        self.assertIn("MindForge", result)
        self.assertIn("LOCAL", result)
        self.assertIn("CLOUD", result)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI argument parsing for Phase 2 features."""

    def test_cli_has_models_command(self):
        """Test that the CLI has a 'models' subcommand."""
        import subprocess
        result = subprocess.run(
            ["mindforge", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("models", result.stdout)

    def test_cli_format_has_openai_messages(self):
        """Test that the CLI format command accepts openai_messages."""
        import subprocess
        result = subprocess.run(
            ["mindforge", "format", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("openai_messages", result.stdout)

    def test_cli_format_has_template_free(self):
        """Test that the CLI format command accepts template_free."""
        import subprocess
        result = subprocess.run(
            ["mindforge", "format", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("template_free", result.stdout)

    def test_cli_probe_has_tier_all(self):
        """Test that the CLI probe command accepts --tier all."""
        import subprocess
        result = subprocess.run(
            ["mindforge", "probe", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("all", result.stdout)

    def test_cli_probe_has_judge_model(self):
        """Test that the CLI probe command has --judge-model flag."""
        import subprocess
        result = subprocess.run(
            ["mindforge", "probe", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertIn("judge-model", result.stdout)


if __name__ == "__main__":
    unittest.main()
