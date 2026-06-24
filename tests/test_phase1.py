"""Basic tests for MindForge Phase 1."""

import os
import sys
import json
import tempfile
import unittest

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mindforge.probe.adapters import extract_answer_letter
from mindforge.probe.question_gen import format_mcq_prompt, resolve_subject, load_taxonomy
from mindforge.score.answer_key import score_answer, get_answer_letter, get_answer_idx
from mindforge.score.confidence import compute_confidence
from mindforge.correct.analyzer import analyze_error
from mindforge.correct.corrector import formulate_correction, formulate_rejection
from mindforge.format.dpo import format_dpo_entry, write_dpo_jsonl
from mindforge.format.alpaca import format_alpaca_entry
from mindforge.format.chatml import format_chatml_entry
from mindforge.format.completion import format_completion_entry
from mindforge.vault.database import Database


class TestAnswerExtraction(unittest.TestCase):
    """Test answer letter extraction from model responses."""

    def test_extract_simple_letter(self):
        self.assertEqual(extract_answer_letter("B"), "B")

    def test_extract_answer_is(self):
        self.assertEqual(extract_answer_letter("The answer is C."), "C")

    def test_extract_answer_colon(self):
        self.assertEqual(extract_answer_letter("Answer: D"), "D")

    def test_extract_paren_letter(self):
        self.assertEqual(extract_answer_letter("I think (A) is correct."), "A")

    def test_extract_none(self):
        self.assertIsNone(extract_answer_letter("I don't know"))

    def test_extract_lowercase(self):
        self.assertEqual(extract_answer_letter("the answer is b"), "B")


class TestScoring(unittest.TestCase):
    """Test answer scoring."""

    def test_correct_score(self):
        self.assertTrue(score_answer("A", "A"))

    def test_incorrect_score(self):
        self.assertFalse(score_answer("B", "A"))

    def test_none_answer(self):
        self.assertFalse(score_answer(None, "A"))

    def test_case_insensitive(self):
        self.assertTrue(score_answer("a", "A"))

    def test_answer_letter_conversion(self):
        self.assertEqual(get_answer_letter(0), "A")
        self.assertEqual(get_answer_letter(1), "B")
        self.assertEqual(get_answer_letter(3), "D")

    def test_answer_idx_conversion(self):
        self.assertEqual(get_answer_idx("A"), 0)
        self.assertEqual(get_answer_idx("D"), 3)


class TestConfidence(unittest.TestCase):
    """Test confidence scoring."""

    def test_correct_confident(self):
        c = compute_confidence(True, "The answer is B.", "B")
        self.assertGreater(c, 0.5)
        self.assertLessEqual(c, 1.0)

    def test_incorrect(self):
        c = compute_confidence(False, "The answer is A.", "A")
        self.assertLessEqual(c, 0.5)

    def test_no_answer(self):
        c = compute_confidence(False, "I don't know", None)
        self.assertEqual(c, 0.0)


class TestFormatMCQ(unittest.TestCase):
    """Test multiple-choice question formatting."""

    def test_format_basic(self):
        prompt = format_mcq_prompt("What is 2+2?", ["3", "4", "5", "6"])
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("A) 3", prompt)
        self.assertIn("B) 4", prompt)
        self.assertIn("C) 5", prompt)
        self.assertIn("D) 6", prompt)

    def test_format_with_subject(self):
        prompt = format_mcq_prompt("What is 2+2?", ["3", "4", "5", "6"], "high_school_mathematics")
        self.assertIn("Mathematics", prompt)


class TestSubjectResolution(unittest.TestCase):
    """Test subject name resolution."""

    def test_load_taxonomy(self):
        tax = load_taxonomy()
        self.assertIn("categories", tax)
        self.assertIn("subject_mapping", tax)
        self.assertIn("STEM", tax["categories"])
        self.assertIn("high_school_mathematics", tax["categories"]["STEM"])

    def test_resolve_mathematics(self):
        self.assertEqual(resolve_subject("mathematics"), "high_school_mathematics")

    def test_resolve_physics(self):
        self.assertEqual(resolve_subject("physics"), "high_school_physics")

    def test_resolve_already_valid(self):
        self.assertEqual(resolve_subject("high_school_mathematics"), "high_school_mathematics")

    def test_resolve_invalid(self):
        self.assertIsNone(resolve_subject("nonexistent_subject_12345"))


class TestFormatters(unittest.TestCase):
    """Test output formatters."""

    def test_dpo_format(self):
        entry = format_dpo_entry(
            prompt="What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6",
            chosen="The answer is B) 4.",
            rejected="The answer is A) 3.",
        )
        self.assertEqual(entry["prompt"], "What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6")
        self.assertEqual(entry["chosen"], "The answer is B) 4.")
        self.assertEqual(entry["rejected"], "The answer is A) 3.")

    def test_alpaca_format(self):
        entry = format_alpaca_entry("prompt", "response")
        self.assertEqual(entry["instruction"], "prompt")
        self.assertEqual(entry["output"], "response")

    def test_chatml_format(self):
        entry = format_chatml_entry("prompt", "response")
        self.assertIn("<|im_start|>", entry["text"])
        self.assertIn("prompt", entry["text"])
        self.assertIn("response", entry["text"])

    def test_completion_format(self):
        entry = format_completion_entry("prompt", "response")
        self.assertEqual(entry["prompt"], "prompt")
        self.assertEqual(entry["completion"], "response")

    def test_write_dpo_jsonl(self):
        entries = [
            format_dpo_entry("q1", "a1", "r1"),
            format_dpo_entry("q2", "a2", "r2"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            count = write_dpo_jsonl(entries, path)
            self.assertEqual(count, 2)
            with open(path, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            entry = json.loads(lines[0])
            self.assertIn("prompt", entry)
            self.assertIn("chosen", entry)
            self.assertIn("rejected", entry)
        finally:
            os.unlink(path)


class TestErrorAnalysis(unittest.TestCase):
    """Test error analysis."""

    def test_analyze_wrong_answer(self):
        choices = ["3", "4", "5", "6"]
        result = analyze_error("What is 2+2?", choices, "A", 1)
        self.assertEqual(result["error_type"], "close_confusion")
        self.assertEqual(result["model_choice"], "3")
        self.assertEqual(result["correct_choice"], "4")

    def test_analyze_no_answer(self):
        choices = ["3", "4", "5", "6"]
        result = analyze_error("What is 2+2?", choices, None, 1)
        self.assertIn(result["error_type"], ["no_answer_extracted", "no_answer"])


class TestCorrector(unittest.TestCase):
    """Test corrected answer formulation."""

    def test_formulate_correction(self):
        choices = ["3", "4", "5", "6"]
        result = formulate_correction(1, choices)
        self.assertIn("B", result)
        self.assertIn("4", result)

    def test_formulate_rejection(self):
        choices = ["3", "4", "5", "6"]
        result = formulate_rejection("A", choices)
        self.assertIn("A", result)
        self.assertIn("3", result)

    def test_formulate_rejection_none(self):
        result = formulate_rejection(None, ["a", "b", "c", "d"])
        self.assertIn("not sure", result.lower())


class TestDatabase(unittest.TestCase):
    """Test SQLite database operations."""

    def setUp(self):
        self.db = Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_store_response(self):
        result = {
            "question_idx": 0,
            "prompt": "Test prompt",
            "question": "Test question",
            "choices": ["A", "B", "C", "D"],
            "correct_answer_idx": 1,
            "correct_answer_letter": "B",
            "model_response": "The answer is A.",
            "model_answer_letter": "A",
            "is_correct": False,
            "confidence": 0.3,
            "subject": "test_subject",
            "model": "test-model",
        }
        rid = self.db.store_response(result)
        self.assertIsNotNone(rid)
        self.assertEqual(result["db_id"], rid)

    def test_store_training_entry(self):
        result = {
            "question_idx": 0,
            "prompt": "Test prompt",
            "question": "Test question",
            "choices": ["A", "B", "C", "D"],
            "correct_answer_idx": 1,
            "correct_answer_letter": "B",
            "model_response": "The answer is A.",
            "model_answer_letter": "A",
            "is_correct": False,
            "confidence": 0.3,
            "subject": "test_subject",
            "model": "test-model",
        }
        rid = self.db.store_response(result)
        tid = self.db.store_training_entry(rid, "prompt", "chosen", "rejected", "dpo", "test")
        self.assertIsNotNone(tid)

    def test_get_pending_entries(self):
        result = {
            "question_idx": 0,
            "prompt": "Test prompt",
            "question": "Test question",
            "choices": ["A", "B", "C", "D"],
            "correct_answer_idx": 1,
            "correct_answer_letter": "B",
            "model_response": "The answer is A.",
            "model_answer_letter": "A",
            "is_correct": False,
            "confidence": 0.3,
            "subject": "test_subject",
            "model": "test-model",
        }
        rid = self.db.store_response(result)
        self.db.store_training_entry(rid, "prompt", "chosen", "rejected", "dpo", "test")
        entries = self.db.get_pending_entries()
        # store_training_entry creates one pending entry; the response itself
        # doesn't create training entries, so there should be exactly 1.
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0]["prompt"], "prompt")


class TestEndToEnd(unittest.TestCase):
    """Test the full pipeline with mock data."""

    def test_full_pipeline_mock(self):
        """Test the full pipeline with mock data (no model required)."""
        choices = ["3", "4", "5", "6"]
        prompt = format_mcq_prompt("What is 2+2?", choices, "high_school_mathematics")

        model_response = "The answer is A) 3."
        model_letter = extract_answer_letter(model_response)
        correct_letter = "B"

        is_correct = score_answer(model_letter, correct_letter)
        self.assertFalse(is_correct)

        # Analyze error
        analysis = analyze_error("What is 2+2?", choices, model_letter, 1)
        self.assertEqual(analysis["model_choice"], "3")

        # Formulate correction
        chosen = formulate_correction(1, choices)
        rejected = formulate_rejection("A", choices)

        # Format as DPO
        dpo = format_dpo_entry(prompt, chosen, rejected)
        self.assertIn("prompt", dpo)
        self.assertIn("chosen", dpo)
        self.assertIn("rejected", dpo)

        # Verify DPO structure is valid JSON
        json_str = json.dumps(dpo)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["chosen"], chosen)
        self.assertEqual(parsed["rejected"], rejected)


if __name__ == "__main__":
    unittest.main()
