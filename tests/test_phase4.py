"""Tests for Phase 4: Fine-Tuning & Evaluation."""

import os
import sys
import json
import tempfile
import unittest
import inspect
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mindforge.vault.database import Database
from mindforge.train.trainer import train_model, _check_mlx_lm_lora_installed
from mindforge.evaluate.evaluator import (
    evaluate_model,
    compare_models,
    _check_lm_eval_available,
    _check_mlx_lm_evaluate_available,
)


class TestTrainingRunsDatabase(unittest.TestCase):
    """Test training_runs database operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        self.db.close()

    def test_training_runs_table_exists(self):
        """Test that the training_runs table exists."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        self.assertIn("training_runs", tables)

    def test_store_and_get_training_run(self):
        """Test storing and retrieving a training run."""
        run_info = {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mode": "dpo",
            "data_path": "/data/training-data/dpo/",
            "adapter_path": "/data/adapters/adapters.safetensors",
            "iters": 1000,
            "batch_size": 4,
            "learning_rate": 1e-5,
            "beta": 0.1,
            "status": "completed",
            "loss": 0.4523,
            "iters_completed": 1000,
            "started_at": 1000000.0,
            "finished_at": 1000300.0,
        }
        run_id = self.db.store_training_run(run_info)
        self.assertIsNotNone(run_id)

        runs = self.db.get_training_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["model"], "mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertEqual(runs[0]["mode"], "dpo")
        self.assertEqual(runs[0]["iters"], 1000)
        self.assertEqual(runs[0]["batch_size"], 4)
        self.assertEqual(runs[0]["status"], "completed")
        self.assertAlmostEqual(runs[0]["loss"], 0.4523)

    def test_update_training_run(self):
        """Test updating a training run."""
        run_info = {
            "model": "test-model",
            "mode": "sft",
            "data_path": "/data/",
            "adapter_path": "/adapters/",
            "iters": 500,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "beta": 0.0,
            "status": "running",
            "loss": None,
            "iters_completed": 0,
            "started_at": 1000000.0,
            "finished_at": None,
        }
        run_id = self.db.store_training_run(run_info)

        # Update the run
        self.db.update_training_run(run_id, {
            "status": "completed",
            "loss": 0.32,
            "iters_completed": 500,
            "finished_at": 1000200.0,
        })

        runs = self.db.get_training_runs()
        self.assertEqual(runs[0]["status"], "completed")
        self.assertAlmostEqual(runs[0]["loss"], 0.32)
        self.assertEqual(runs[0]["iters_completed"], 500)

    def test_multiple_training_runs_ordered(self):
        """Test that multiple runs are returned in order."""
        for i in range(3):
            self.db.store_training_run({
                "model": f"model-{i}",
                "mode": "sft",
                "data_path": "/data/",
                "adapter_path": f"/adapters/{i}",
                "iters": 100,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "beta": 0.1,
                "status": "completed",
                "loss": 0.1 * i,
                "iters_completed": 100,
                "started_at": 1000000.0 + i,
                "finished_at": 1000100.0 + i,
            })

        runs = self.db.get_training_runs()
        self.assertEqual(len(runs), 3)
        # Should be ordered by started_at DESC
        self.assertEqual(runs[0]["model"], "model-2")
        self.assertEqual(runs[2]["model"], "model-0")


class TestEvaluationResultsDatabase(unittest.TestCase):
    """Test evaluation_results database operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        self.db.close()

    def test_evaluation_results_table_exists(self):
        """Test that the evaluation_results table exists."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        self.assertIn("evaluation_results", tables)

    def test_store_and_get_evaluation_result(self):
        """Test storing and retrieving an evaluation result."""
        eval_info = {
            "training_run_id": None,
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "task": "mmlu_stem",
            "score": 0.72,
            "metric": "accuracy",
            "details": json.dumps({"correct": 18, "total": 25}),
        }
        eval_id = self.db.store_evaluation_result(eval_info)
        self.assertIsNotNone(eval_id)

        results = self.db.get_evaluation_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model"], "mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertEqual(results[0]["task"], "mmlu_stem")
        self.assertAlmostEqual(results[0]["score"], 0.72)
        self.assertEqual(results[0]["metric"], "accuracy")

    def test_store_evaluation_result_with_training_run(self):
        """Test storing an evaluation result linked to a training run."""
        # First create a training run
        run_id = self.db.store_training_run({
            "model": "test-model",
            "mode": "dpo",
            "data_path": "/data/",
            "adapter_path": "/adapters/",
            "iters": 100,
            "batch_size": 4,
            "learning_rate": 1e-5,
            "beta": 0.1,
            "status": "completed",
            "loss": 0.3,
            "iters_completed": 100,
            "started_at": 1000000.0,
            "finished_at": 1000100.0,
        })

        # Then create an evaluation result linked to it
        eval_id = self.db.store_evaluation_result({
            "training_run_id": run_id,
            "model": "test-model",
            "task": "mmlu_stem",
            "score": 0.85,
            "metric": "accuracy",
            "details": json.dumps({"improvement": True}),
        })

        results = self.db.get_evaluation_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["training_run_id"], run_id)
        self.assertAlmostEqual(results[0]["score"], 0.85)

    def test_multiple_evaluation_results(self):
        """Test storing multiple evaluation results."""
        for i in range(3):
            self.db.store_evaluation_result({
                "training_run_id": None,
                "model": f"model-{i}",
                "task": "mmlu_stem",
                "score": 0.5 + i * 0.1,
                "metric": "accuracy",
                "details": "{}",
            })

        results = self.db.get_evaluation_results()
        self.assertEqual(len(results), 3)


class TestTrainModelFunction(unittest.TestCase):
    """Test the train_model function interface."""

    def test_train_model_signature(self):
        """Test that train_model has the correct function signature."""
        sig = inspect.signature(train_model)
        params = list(sig.parameters.keys())
        self.assertIn("model", params)
        self.assertIn("data_path", params)
        self.assertIn("mode", params)
        self.assertIn("iters", params)
        self.assertIn("batch_size", params)
        self.assertIn("learning_rate", params)
        self.assertIn("beta", params)
        self.assertIn("adapter_path", params)

    def test_train_model_defaults(self):
        """Test that train_model has correct default values."""
        sig = inspect.signature(train_model)
        self.assertEqual(sig.parameters["mode"].default, "dpo")
        self.assertEqual(sig.parameters["iters"].default, 1000)
        self.assertEqual(sig.parameters["batch_size"].default, 4)
        self.assertEqual(sig.parameters["learning_rate"].default, 1e-5)
        self.assertEqual(sig.parameters["beta"].default, 0.1)
        self.assertIsNone(sig.parameters["adapter_path"].default)

    def test_train_model_invalid_mode(self):
        """Test that train_model raises ValueError for invalid mode."""
        with self.assertRaises(ValueError):
            train_model(
                model="test-model",
                data_path="/tmp/data/",
                mode="invalid_mode",
            )

    def test_train_model_missing_data_path(self):
        """Test that train_model raises FileNotFoundError for missing data."""
        with self.assertRaises(FileNotFoundError):
            train_model(
                model="test-model",
                data_path="/nonexistent/path/that/does/not/exist",
                mode="sft",
            )

    def test_check_mlx_lm_lora_installed(self):
        """Test the mlx-lm-lora availability checker returns bool."""
        result = _check_mlx_lm_lora_installed()
        self.assertIsInstance(result, bool)

    def test_train_model_returns_dict_on_failure(self):
        """Test that train_model returns a dict with status='failed' on error."""
        # Create a temporary data file so the path exists but model is invalid
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "train.jsonl")
            with open(data_path, "w") as f:
                f.write('{"prompt": "test", "chosen": "a", "rejected": "b"}\n')

            # This will fail because the model doesn't exist
            # But it should return a dict, not raise
            with patch("mindforge.train.trainer._get_db") as mock_db:
                mock_db_instance = Mock()
                mock_db_instance.store_training_run.return_value = 1
                mock_db.return_value = mock_db_instance

                results = train_model(
                    model="nonexistent/model",
                    data_path=data_path,
                    mode="sft",
                    iters=1,
                )

                self.assertIsInstance(results, dict)
                self.assertIn("status", results)
                self.assertIn("adapter_path", results)
                self.assertIn("mode", results)


class TestEvaluateModelFunction(unittest.TestCase):
    """Test the evaluate_model function interface."""

    def test_evaluate_model_signature(self):
        """Test that evaluate_model has the correct function signature."""
        sig = inspect.signature(evaluate_model)
        params = list(sig.parameters.keys())
        self.assertIn("model", params)
        self.assertIn("tasks", params)
        self.assertIn("num_fewshot", params)
        self.assertIn("adapter_path", params)

    def test_evaluate_model_defaults(self):
        """Test that evaluate_model has correct default values."""
        sig = inspect.signature(evaluate_model)
        self.assertEqual(sig.parameters["tasks"].default, "mmlu_stem")
        self.assertEqual(sig.parameters["num_fewshot"].default, 5)
        self.assertIsNone(sig.parameters["adapter_path"].default)

    def test_check_lm_eval_available(self):
        """Test the lm-eval availability checker returns bool."""
        result = _check_lm_eval_available()
        self.assertIsInstance(result, bool)

    def test_check_mlx_lm_evaluate_available(self):
        """Test the mlx_lm.evaluate availability checker returns bool."""
        result = _check_mlx_lm_evaluate_available()
        self.assertIsInstance(result, bool)

    def test_evaluate_model_returns_dict(self):
        """Test that evaluate_model returns a dict with required keys."""
        with patch("mindforge.evaluate.evaluator._get_db") as mock_db:
            mock_db_instance = Mock()
            mock_db_instance.store_evaluation_result.return_value = 1
            mock_db.return_value = mock_db_instance

            # Mock all backends to fail, so we get the fallback
            with patch("mindforge.evaluate.evaluator._check_lm_eval_available", return_value=False):
                with patch("mindforge.evaluate.evaluator._check_mlx_lm_evaluate_available", return_value=False):
                    with patch("mindforge.evaluate.evaluator._run_simple_mmlu_eval") as mock_simple:
                        mock_simple.return_value = {
                            "score": 0.5,
                            "metric": "accuracy",
                            "details": "{}",
                            "backend": "simple_mmlu",
                        }

                        results = evaluate_model(
                            model="test-model",
                            tasks="mmlu_stem",
                        )

                        self.assertIsInstance(results, dict)
                        self.assertIn("score", results)
                        self.assertIn("metric", results)
                        self.assertIn("backend", results)
                        self.assertIn("status", results)
                        self.assertEqual(results["status"], "completed")
                        self.assertEqual(results["backend"], "simple_mmlu")

    def test_evaluate_model_fallback_logic(self):
        """Test that evaluation falls back from lm_eval to simple_mmlu."""
        with patch("mindforge.evaluate.evaluator._get_db") as mock_db:
            mock_db_instance = Mock()
            mock_db_instance.store_evaluation_result.return_value = 1
            mock_db.return_value = mock_db_instance

            # lm_eval is available but fails, simple_mmlu succeeds
            with patch("mindforge.evaluate.evaluator._check_lm_eval_available", return_value=True):
                with patch("mindforge.evaluate.evaluator._check_mlx_lm_evaluate_available", return_value=False):
                    with patch("mindforge.evaluate.evaluator._run_lm_eval", side_effect=Exception("lm_eval failed")):
                        with patch("mindforge.evaluate.evaluator._run_simple_mmlu_eval") as mock_simple:
                            mock_simple.return_value = {
                                "score": 0.6,
                                "metric": "accuracy",
                                "details": "{}",
                                "backend": "simple_mmlu",
                            }

                            results = evaluate_model(
                                model="test-model",
                                tasks="mmlu_stem",
                            )

                            self.assertEqual(results["backend"], "simple_mmlu")
                            self.assertEqual(results["status"], "completed")


class TestCompareModelsFunction(unittest.TestCase):
    """Test the compare_models function."""

    def test_compare_models_signature(self):
        """Test that compare_models has the correct function signature."""
        sig = inspect.signature(compare_models)
        params = list(sig.parameters.keys())
        self.assertIn("base_model", params)
        self.assertIn("tuned_model", params)
        self.assertIn("tasks", params)
        self.assertIn("num_fewshot", params)
        self.assertIn("adapter_path", params)

    def test_compare_models_defaults(self):
        """Test that compare_models has correct default values."""
        sig = inspect.signature(compare_models)
        self.assertEqual(sig.parameters["tasks"].default, "mmlu_stem")
        self.assertEqual(sig.parameters["num_fewshot"].default, 5)
        self.assertIsNone(sig.parameters["adapter_path"].default)

    def test_compare_models_returns_dict(self):
        """Test that compare_models returns a dict with comparison results."""
        with patch("mindforge.evaluate.evaluator._get_db") as mock_db:
            mock_db_instance = Mock()
            mock_db_instance.store_evaluation_result.return_value = 1
            mock_db.return_value = mock_db_instance

            with patch("mindforge.evaluate.evaluator._check_lm_eval_available", return_value=False):
                with patch("mindforge.evaluate.evaluator._check_mlx_lm_evaluate_available", return_value=False):
                    with patch("mindforge.evaluate.evaluator._run_simple_mmlu_eval") as mock_simple:

                        def side_effect(model, tasks, num_fewshot, adapter_path):
                            if "base" in model:
                                return {"score": 0.5, "metric": "accuracy", "details": "{}", "backend": "simple_mmlu"}
                            else:
                                return {"score": 0.7, "metric": "accuracy", "details": "{}", "backend": "simple_mmlu"}

                        mock_simple.side_effect = side_effect

                        results = compare_models(
                            base_model="test-base-model",
                            tuned_model="test-tuned-model",
                            tasks="mmlu_stem",
                        )

                        self.assertIsInstance(results, dict)
                        self.assertIn("base_score", results)
                        self.assertIn("tuned_score", results)
                        self.assertIn("improvement", results)
                        self.assertIn("improvement_pct", results)
                        self.assertEqual(results["base_score"], 0.5)
                        self.assertEqual(results["tuned_score"], 0.7)
                        self.assertAlmostEqual(results["improvement"], 0.2)
                        self.assertGreater(results["improvement_pct"], 0)

    def test_compare_models_improvement_negative(self):
        """Test comparison when tuned model is worse than base."""
        with patch("mindforge.evaluate.evaluator._get_db") as mock_db:
            mock_db_instance = Mock()
            mock_db_instance.store_evaluation_result.return_value = 1
            mock_db.return_value = mock_db_instance

            with patch("mindforge.evaluate.evaluator._check_lm_eval_available", return_value=False):
                with patch("mindforge.evaluate.evaluator._check_mlx_lm_evaluate_available", return_value=False):
                    with patch("mindforge.evaluate.evaluator._run_simple_mmlu_eval") as mock_simple:

                        def side_effect(model, tasks, num_fewshot, adapter_path):
                            if "base" in model:
                                return {"score": 0.8, "metric": "accuracy", "details": "{}", "backend": "simple_mmlu"}
                            else:
                                return {"score": 0.6, "metric": "accuracy", "details": "{}", "backend": "simple_mmlu"}

                        mock_simple.side_effect = side_effect

                        results = compare_models(
                            base_model="test-base-model",
                            tuned_model="test-tuned-model",
                        )

                        self.assertLess(results["improvement"], 0)
                        self.assertLess(results["improvement_pct"], 0)


class TestCLICommands(unittest.TestCase):
    """Test that CLI has train and evaluate commands."""

    def test_cli_has_train_command(self):
        """Test that the CLI has a 'train' subcommand."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("train", result.stdout)

    def test_cli_has_evaluate_command(self):
        """Test that the CLI has an 'evaluate' subcommand."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("evaluate", result.stdout)

    def test_train_help_has_model_flag(self):
        """Test that train --help shows --model flag."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "train", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("--model", result.stdout)
        self.assertIn("--data", result.stdout)
        self.assertIn("--mode", result.stdout)

    def test_train_help_has_all_flags(self):
        """Test that train --help shows all expected flags."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "train", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("--iters", result.stdout)
        self.assertIn("--batch-size", result.stdout)
        self.assertIn("--learning-rate", result.stdout)
        self.assertIn("--beta", result.stdout)
        self.assertIn("--adapter-path", result.stdout)

    def test_train_help_shows_modes(self):
        """Test that train --help shows SFT, DPO, ORPO modes."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "train", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("sft", result.stdout)
        self.assertIn("dpo", result.stdout)
        self.assertIn("orpo", result.stdout)

    def test_evaluate_help_has_model_flag(self):
        """Test that evaluate --help shows --model flag."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "evaluate", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("--model", result.stdout)
        self.assertIn("--tasks", result.stdout)

    def test_evaluate_help_has_all_flags(self):
        """Test that evaluate --help shows all expected flags."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "evaluate", "--help"],
            capture_output=True, text=True, cwd=_project_root
        )
        self.assertIn("--num-fewshot", result.stdout)
        self.assertIn("--adapter-path", result.stdout)
        self.assertIn("--compare", result.stdout)


class TestModuleStructure(unittest.TestCase):
    """Test that Phase 4 modules exist and are importable."""

    def test_train_module_importable(self):
        """Test that the train module can be imported."""
        from mindforge.train import trainer
        self.assertTrue(hasattr(trainer, "train_model"))

    def test_evaluate_module_importable(self):
        """Test that the evaluate module can be imported."""
        from mindforge.evaluate import evaluator
        self.assertTrue(hasattr(evaluator, "evaluate_model"))
        self.assertTrue(hasattr(evaluator, "compare_models"))

    def test_train_init_exists(self):
        """Test that the train module has __init__.py."""
        train_init = os.path.join(_project_root, "mindforge", "train", "__init__.py")
        self.assertTrue(os.path.exists(train_init))

    def test_evaluate_init_exists(self):
        """Test that the evaluate module has __init__.py."""
        eval_init = os.path.join(_project_root, "mindforge", "evaluate", "__init__.py")
        self.assertTrue(os.path.exists(eval_init))

    def test_trainer_py_exists(self):
        """Test that trainer.py exists."""
        trainer_path = os.path.join(_project_root, "mindforge", "train", "trainer.py")
        self.assertTrue(os.path.exists(trainer_path))

    def test_evaluator_py_exists(self):
        """Test that evaluator.py exists."""
        evaluator_path = os.path.join(_project_root, "mindforge", "evaluate", "evaluator.py")
        self.assertTrue(os.path.exists(evaluator_path))


class TestDatabaseSchema(unittest.TestCase):
    """Test that the database schema includes Phase 4 tables with correct columns."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        self.db.close()

    def test_training_runs_columns(self):
        """Test that training_runs table has all expected columns."""
        cursor = self.db.conn.cursor()
        cursor.execute("PRAGMA table_info(training_runs)")
        columns = [row[1] for row in cursor.fetchall()]
        expected = [
            "id", "model", "mode", "data_path", "adapter_path",
            "iters", "batch_size", "learning_rate", "beta",
            "status", "loss", "iters_completed", "started_at", "finished_at"
        ]
        for col in expected:
            self.assertIn(col, columns)

    def test_evaluation_results_columns(self):
        """Test that evaluation_results table has all expected columns."""
        cursor = self.db.conn.cursor()
        cursor.execute("PRAGMA table_info(evaluation_results)")
        columns = [row[1] for row in cursor.fetchall()]
        expected = [
            "id", "training_run_id", "model", "task", "score",
            "metric", "details", "created_at"
        ]
        for col in expected:
            self.assertIn(col, columns)


if __name__ == "__main__":
    unittest.main()
