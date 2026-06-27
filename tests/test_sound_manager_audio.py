"""Regression tests for Xbox dashboard-inspired Web Audio settings."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "components" / "SoundManager.tsx"


class SoundManagerAudioTests(unittest.TestCase):
    def test_sound_manager_uses_requested_xbox_audio_profile(self):
        source = SOURCE.read_text()

        self.assertIn('type SoundType = "sweep" | "select" | "scroll" | "back" | "boot" | "whoosh";', source)
        self.assertIn('localStorage.getItem("mindforge-muted") === "true"', source)

        self.assertIn("const duration = 0.35;", source)
        self.assertIn("filter.frequency.setValueAtTime(400, now);", source)
        self.assertIn("filter.frequency.exponentialRampToValueAtTime(2000, now + duration);", source)

        self.assertIn('osc.type = "sine";', source)
        self.assertIn("osc.frequency.value = 800;", source)
        self.assertIn("gain.gain.exponentialRampToValueAtTime(0.001, now + 0.05);", source)

        self.assertIn('osc.type = "square";', source)
        self.assertIn("osc.frequency.value = 1200;", source)
        self.assertIn("osc.stop(now + 0.02);", source)

        self.assertIn("filter.frequency.setValueAtTime(2000, now);", source)
        self.assertIn("filter.frequency.exponentialRampToValueAtTime(200, now + duration);", source)

        self.assertIn("osc.frequency.setValueAtTime(82, now);", source)
        self.assertIn("osc.frequency.exponentialRampToValueAtTime(196, now + duration * 0.65);", source)
        self.assertIn("shimmer.frequency.setValueAtTime(330, now + 0.15);", source)
        self.assertIn("shimmer.frequency.exponentialRampToValueAtTime(660, now + duration);", source)

        self.assertIn("osc.frequency.value = 55;", source)
        self.assertIn("gain.gain.value = 0.02;", source)

    def test_sound_manager_exposes_whoosh_method(self):
        source = SOURCE.read_text()

        self.assertRegex(source, r"\n\s+whoosh\(\)\s*\{")

    def test_play_switch_handles_whoosh(self):
        source = SOURCE.read_text()

        self.assertRegex(
            source,
            r'(?s)play\(type: SoundType\)\s*\{\s*switch \(type\)\s*\{.*case "whoosh": this\.whoosh\(\); break;',
        )


if __name__ == "__main__":
    unittest.main()
