"""Tests for Phase 8: Hermes Skill Wrapper."""

import unittest
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestHermesSkill(unittest.TestCase):
    """Test that the MindForge Hermes skill is properly installed."""

    def setUp(self):
        self.skill_path = os.path.expanduser("~/.hermes/skills/mlops/mindforge/SKILL.md")

    def test_skill_file_exists(self):
        self.assertTrue(os.path.exists(self.skill_path), 
                        f"Skill file not found at {self.skill_path}")

    def test_skill_has_frontmatter(self):
        if not os.path.exists(self.skill_path):
            self.skipTest("Skill file not found")
        with open(self.skill_path, "r") as f:
            content = f.read()
        self.assertTrue(content.startswith("---"), "Skill file should start with YAML frontmatter")

    def test_skill_has_name(self):
        if not os.path.exists(self.skill_path):
            self.skipTest("Skill file not found")
        with open(self.skill_path, "r") as f:
            content = f.read()
        self.assertIn("name: mindforge", content)

    def test_skill_has_description(self):
        if not os.path.exists(self.skill_path):
            self.skipTest("Skill file not found")
        with open(self.skill_path, "r") as f:
            content = f.read()
        self.assertIn("description:", content)

    def test_skill_documents_cli_commands(self):
        if not os.path.exists(self.skill_path):
            self.skipTest("Skill file not found")
        with open(self.skill_path, "r") as f:
            content = f.read()
        # Check that key CLI commands are documented
        self.assertIn("mindforge probe", content)
        self.assertIn("mindforge train", content)
        self.assertIn("mindforge evaluate", content)
        self.assertIn("mindforge detect", content)

    def test_skill_documents_dpo_format(self):
        if not os.path.exists(self.skill_path):
            self.skipTest("Skill file not found")
        with open(self.skill_path, "r") as f:
            content = f.read()
        self.assertIn("chosen", content)
        self.assertIn("rejected", content)

    def test_skill_documents_subjects(self):
        if not os.path.exists(self.skill_path):
            self.skipTest("Skill file not found")
        with open(self.skill_path, "r") as f:
            content = f.read()
        self.assertIn("STEM", content)
        self.assertIn("Humanities", content)
        self.assertIn("MMLU", content)


class TestCLIStillWorks(unittest.TestCase):
    """Verify the CLI still works after all phases."""

    def test_cli_help_shows_all_commands(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        # All 11 commands should be present
        for cmd in ["detect", "models", "probe", "review", "format", 
                     "convert", "quantize", "train", "evaluate", 
                     "ingest-pdf", "ingest-web"]:
            self.assertIn(cmd, result.stdout, f"Command '{cmd}' missing from CLI help")


if __name__ == "__main__":
    unittest.main()
