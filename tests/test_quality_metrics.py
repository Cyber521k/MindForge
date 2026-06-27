"""Tests for DPO quality metrics and CLI analysis."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

PYTHON = sys.executable


def run_cli(*args):
    """Run a mindforge CLI command and return the CompletedProcess."""
    cmd = [PYTHON, "-m", "mindforge.cli"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_project_root,
        timeout=60,
    )


class TestComputePairQuality(unittest.TestCase):
    """Unit tests for compute_pair_quality."""

    def test_returns_expected_metric_keys(self):
        from mindforge.review.quality_metrics import compute_pair_quality

        scores = compute_pair_quality(
            "Explain why water expands when it freezes.",
            "Water expands when it freezes because hydrogen bonds form an open crystalline lattice.",
            "Water gets colder and therefore takes up more space.",
        )

        for key in [
            "semantic_similarity",
            "length_ratio",
            "answer_diversity",
            "preference_margin",
            "difficulty_score",
            "overall_quality",
        ]:
            self.assertIn(key, scores)
            self.assertIsInstance(scores[key], float)

    def test_near_duplicate_answers_score_lower_than_clear_preference(self):
        from mindforge.review.quality_metrics import compute_pair_quality

        duplicate = compute_pair_quality(
            "What is photosynthesis?",
            "Photosynthesis is how plants use sunlight to make food.",
            "Photosynthesis is how plants use sunlight to make food.",
        )
        clear = compute_pair_quality(
            "What is photosynthesis?",
            "Photosynthesis converts light, carbon dioxide, and water into glucose and oxygen in chloroplasts.",
            "I do not know.",
        )

        self.assertGreater(duplicate["semantic_similarity"], clear["semantic_similarity"])
        self.assertLess(duplicate["overall_quality"], clear["overall_quality"])

    def test_difficulty_increases_with_complex_prompt(self):
        from mindforge.review.quality_metrics import compute_pair_quality

        easy = compute_pair_quality("What is 2+2?", "4", "5")
        hard = compute_pair_quality(
            "Compare the economic incentives, regulatory tradeoffs, and failure modes of cap-and-trade versus a carbon tax.",
            "A carbon tax sets a price while cap-and-trade fixes quantity; each shifts uncertainty differently.",
            "They are the same policy.",
        )

        self.assertGreater(hard["difficulty_score"], easy["difficulty_score"])


class TestQualityFilteringAndReport(unittest.TestCase):
    """Tests for filtering and report generation."""

    def setUp(self):
        self.good_entry = {
            "prompt": "Why does a convex lens focus parallel light rays?",
            "chosen": (
                "A convex lens refracts incoming parallel rays toward its optical axis. "
                "Because the surfaces bend light inward, the rays meet near the focal point."
            ),
            "rejected": "It focuses light because it is made of glass.",
        }
        self.bad_entry = {
            "prompt": "What is inertia?",
            "chosen": "Inertia is resistance to changes in motion.",
            "rejected": "Inertia is resistance to changes in motion.",
        }

    def test_filter_low_quality_keeps_only_entries_at_threshold(self):
        from mindforge.review.quality_metrics import filter_low_quality

        kept = filter_low_quality([self.good_entry, self.bad_entry], threshold=0.3)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["prompt"], self.good_entry["prompt"])

    def test_generate_quality_report_returns_summary_statistics(self):
        from mindforge.review.quality_metrics import generate_quality_report

        report = generate_quality_report([self.good_entry, self.bad_entry])

        self.assertEqual(report["total_entries"], 2)
        self.assertEqual(report["kept_entries"], 1)
        self.assertEqual(report["filtered_entries"], 1)
        self.assertIn("overall_quality", report["metrics"])
        self.assertIn("mean", report["metrics"]["overall_quality"])
        self.assertIn("low_quality_indices", report)
        self.assertEqual(report["low_quality_indices"], [1])

    def test_empty_report_has_zero_counts(self):
        from mindforge.review.quality_metrics import generate_quality_report

        report = generate_quality_report([])

        self.assertEqual(report["total_entries"], 0)
        self.assertEqual(report["kept_entries"], 0)
        self.assertEqual(report["filtered_entries"], 0)
        self.assertEqual(report["metrics"], {})


class TestQualityCLI(unittest.TestCase):
    """Functional tests for the quality CLI command."""

    def test_quality_cli_analyzes_jsonl_input(self):
        entries = [
            {
                "prompt": "Why does ice float?",
                "chosen": "Ice floats because its hydrogen-bonded lattice is less dense than liquid water.",
                "rejected": "Ice floats because it is cold.",
            },
            {
                "prompt": "What is velocity?",
                "chosen": "Velocity is speed with direction.",
                "rejected": "Velocity is speed with direction.",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "train.jsonl")
            with open(input_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")

            result = run_cli("quality", "--input", input_path)

        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        self.assertIn("Quality Analysis", result.stdout)
        self.assertIn("Total entries: 2", result.stdout)
        self.assertIn("Kept entries:", result.stdout)
        self.assertIn("Filtered entries:", result.stdout)

    def test_quality_cli_requires_input_file(self):
        result = run_cli("quality", "--input", "/nonexistent/train.jsonl")

        self.assertEqual(result.returncode, 1)
        self.assertIn("Input file not found", result.stdout)
