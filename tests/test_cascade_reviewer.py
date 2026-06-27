"""Tests for cascade-based automated review."""

import os
import sys
import unittest
from unittest.mock import Mock, patch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestCascadeReviewer(unittest.TestCase):
    """Tests for multi-model cascade review."""

    def _adapter(self, model_name, response):
        adapter = Mock()
        adapter.model_name = model_name
        adapter.ask.return_value = response
        return adapter

    def test_confident_primary_returns_without_escalation(self):
        """A confident primary verdict should not call secondary or tiebreaker."""
        from mindforge.review.cascade_reviewer import CascadeReviewer

        primary = self._adapter(
            "primary-model",
            '{"correct": true, "confidence": 0.92, "explanation": "correct"}',
        )
        secondary = self._adapter(
            "secondary-model",
            '{"correct": false, "confidence": 0.99, "explanation": "wrong"}',
        )
        tiebreaker = self._adapter(
            "tiebreaker-model",
            '{"correct": false, "confidence": 0.99, "explanation": "wrong"}',
        )

        reviewer = CascadeReviewer([primary, secondary, tiebreaker])
        result = reviewer.review_entry({"prompt": "What is 2+2?", "chosen": "4"})

        self.assertEqual(result["final_action"], "accept")
        self.assertEqual(result["action"], "accept")
        self.assertEqual(result["agreement_score"], 1.0)
        self.assertEqual([v["role"] for v in result["model_verdicts"]], ["primary"])
        primary.ask.assert_called_once()
        secondary.ask.assert_not_called()
        tiebreaker.ask.assert_not_called()

    def test_low_confidence_primary_escalates_to_secondary(self):
        """A low-confidence primary verdict should call the secondary judge."""
        from mindforge.review.cascade_reviewer import CascadeReviewer

        primary = self._adapter(
            "primary-model",
            '{"correct": true, "confidence": 0.62, "explanation": "probably"}',
        )
        secondary = self._adapter(
            "secondary-model",
            '{"correct": true, "confidence": 0.86, "explanation": "correct"}',
        )
        tiebreaker = self._adapter(
            "tiebreaker-model",
            '{"correct": false, "confidence": 0.99, "explanation": "wrong"}',
        )

        reviewer = CascadeReviewer([primary, secondary, tiebreaker])
        result = reviewer.review_entry({"prompt": "What is 2+2?", "chosen": "4"})

        self.assertEqual(result["final_action"], "accept")
        self.assertEqual(result["agreement_score"], 1.0)
        self.assertTrue(result["escalated"])
        self.assertFalse(result["tiebreaker_used"])
        self.assertEqual([v["role"] for v in result["model_verdicts"]], ["primary", "secondary"])
        primary.ask.assert_called_once()
        secondary.ask.assert_called_once()
        tiebreaker.ask.assert_not_called()

    def test_disagreement_uses_tiebreaker_for_consensus(self):
        """Primary-secondary disagreement should include a tiebreaker verdict."""
        from mindforge.review.cascade_reviewer import CascadeReviewer

        primary = self._adapter(
            "primary-model",
            '{"correct": true, "confidence": 0.55, "explanation": "probably"}',
        )
        secondary = self._adapter(
            "secondary-model",
            '{"correct": false, "confidence": 0.82, "explanation": "wrong"}',
        )
        tiebreaker = self._adapter(
            "tiebreaker-model",
            '{"correct": false, "confidence": 0.90, "explanation": "also wrong"}',
        )

        reviewer = CascadeReviewer([primary, secondary, tiebreaker])
        result = reviewer.review_entry({"prompt": "What is 2+2?", "chosen": "5"})

        self.assertEqual(result["final_action"], "reject")
        self.assertEqual(result["action"], "reject")
        self.assertAlmostEqual(result["agreement_score"], 2 / 3)
        self.assertTrue(result["tiebreaker_used"])
        self.assertEqual(
            [verdict["model"] for verdict in result["model_verdicts"]],
            ["primary-model", "secondary-model", "tiebreaker-model"],
        )

    def test_combine_verdicts_returns_majority_consensus(self):
        """_combine_verdicts should summarize majority action and confidence."""
        from mindforge.review.cascade_reviewer import CascadeReviewer

        reviewer = CascadeReviewer([])
        consensus = reviewer._combine_verdicts([
            {"correct": True, "confidence": 0.60, "model": "a"},
            {"correct": False, "confidence": 0.90, "model": "b"},
            {"correct": False, "confidence": 0.70, "model": "c"},
        ])

        self.assertFalse(consensus["correct"])
        self.assertEqual(consensus["final_action"], "reject")
        self.assertAlmostEqual(consensus["agreement_score"], 2 / 3)
        self.assertAlmostEqual(consensus["confidence"], 0.80)

    def test_factory_detects_adapters_in_priority_order(self):
        """create_cascade_reviewer should prefer OpenAI, OpenRouter, Ollama, then MLX."""
        from mindforge.review import cascade_reviewer

        created = []

        def fake_create_adapter(model_name):
            adapter = Mock()
            adapter.model_name = model_name
            created.append(model_name)
            return adapter

        env = {
            "OPENAI_API_KEY": "openai-key",
            "OPENROUTER_API_KEY": "openrouter-key",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(cascade_reviewer, "create_adapter", side_effect=fake_create_adapter):
                with patch.object(cascade_reviewer, "detect_ollama", return_value={
                    "running": True,
                    "models": ["llama3.1:latest"],
                }):
                    reviewer = cascade_reviewer.create_cascade_reviewer()

        self.assertEqual(created, [
            "gpt-4o",
            "openrouter/anthropic/claude-3.5-sonnet",
            "ollama/llama3.1:latest",
        ])
        self.assertEqual(
            [adapter.model_name for adapter in reviewer.judge_adapters],
            created,
        )


if __name__ == "__main__":
    unittest.main()
