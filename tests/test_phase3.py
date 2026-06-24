"""Tests for Phase 3: Model Conversion & Quantization."""

import unittest
import os
import sys
import tempfile
import inspect

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mindforge.vault.database import Database
from mindforge.convert.converter import convert_model
from mindforge.convert.quantizer import quantize_model


class TestDatabaseConversionTables(unittest.TestCase):
    """Test that the database has conversion/quantization tables."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        self.db.close()

    def test_converted_models_table_exists(self):
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        self.assertIn("converted_models", tables)

    def test_quantized_models_table_exists(self):
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        self.assertIn("quantized_models", tables)

    def test_store_and_get_converted_model(self):
        model_info = {
            "source_repo": "mistralai/Mistral-7B-Instruct-v0.3",
            "local_path": "/tmp/models/Mistral-7B-MLX-4bit",
            "quantization": "4bit",
            "model_size_gb": 5.2,
            "uploaded_to_hf": False,
            "hf_repo": None,
        }
        self.db.store_converted_model(model_info)
        models = self.db.get_converted_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["source_repo"], "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(models[0]["quantization"], "4bit")

    def test_store_and_get_quantized_model(self):
        q_info = {
            "source_model_id": 1,
            "source_path": "/tmp/models/Mistral-7B-MLX",
            "output_path": "/tmp/models/Mistral-7B-4bit",
            "bit_depth": 4,
            "group_size": 64,
            "model_size_gb": 5.2,
        }
        self.db.store_quantized_model(q_info)
        qmodels = self.db.get_quantized_models()
        self.assertEqual(len(qmodels), 1)
        self.assertEqual(qmodels[0]["bit_depth"], 4)
        self.assertEqual(qmodels[0]["group_size"], 64)


class TestConverterFunction(unittest.TestCase):
    """Test the convert_model function interface."""

    def test_convert_model_signature(self):
        sig = inspect.signature(convert_model)
        params = list(sig.parameters.keys())
        self.assertIn("source_repo", params)
        self.assertIn("quantize", params)
        self.assertIn("q_bits", params)
        self.assertIn("q_group_size", params)
        self.assertIn("output_dir", params)

    def test_convert_model_defaults(self):
        sig = inspect.signature(convert_model)
        self.assertEqual(sig.parameters["q_bits"].default, 4)
        self.assertEqual(sig.parameters["q_group_size"].default, 64)
        self.assertTrue(sig.parameters["quantize"].default)


class TestQuantizerFunction(unittest.TestCase):
    """Test the quantize_model function interface."""

    def test_quantize_model_signature(self):
        sig = inspect.signature(quantize_model)
        params = list(sig.parameters.keys())
        self.assertIn("source_path", params)
        self.assertIn("bits", params)
        self.assertIn("group_size", params)
        self.assertIn("output_dir", params)

    def test_quantize_model_defaults(self):
        sig = inspect.signature(quantize_model)
        self.assertEqual(sig.parameters["bits"].default, 4)
        self.assertEqual(sig.parameters["group_size"].default, 64)


class TestCLICommands(unittest.TestCase):
    """Test that CLI has convert and quantize commands."""

    def test_cli_has_convert_command(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("convert", result.stdout)

    def test_cli_has_quantize_command(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("quantize", result.stdout)

    def test_convert_help_has_source_flag(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "convert", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("--source", result.stdout)
        self.assertIn("--quantize", result.stdout)

    def test_quantize_help_has_model_flag(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "quantize", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("--model", result.stdout)
        self.assertIn("--bits", result.stdout)


class TestModelListWithConverted(unittest.TestCase):
    """Test that converted models appear in model list."""

    def test_model_list_function_exists(self):
        try:
            from mindforge.hardware.model_list import get_available_models
            models = get_available_models()
            self.assertIsInstance(models, dict)
        except ImportError:
            # May be in detector.py instead
            from mindforge.hardware.detector import detect_hardware
            hw = detect_hardware()
            self.assertIn("memory_gb", hw)


if __name__ == "__main__":
    unittest.main()
