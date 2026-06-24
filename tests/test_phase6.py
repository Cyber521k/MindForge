"""Tests for MindForge Phase 6: Exo Cluster Integration."""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mindforge.hardware.exo_detector import (
    detect_exo,
    get_cluster_info,
    format_cluster_info,
    EXO_API_URL,
)
from mindforge.probe.adapters import ExoAdapter, create_adapter, ModelAdapter
from mindforge.vault.database import Database


class TestExoDetection(unittest.TestCase):
    """Test exo cluster detection."""

    def test_detect_exo_returns_dict_with_expected_keys(self):
        """Test that detect_exo returns a dict with all expected keys."""
        result = detect_exo()
        self.assertIsInstance(result, dict)
        self.assertIn("running", result)
        self.assertIn("installed", result)
        self.assertIn("api_url", result)
        self.assertIn("peers", result)
        self.assertIn("peer_count", result)
        self.assertIn("status", result)

    def test_detect_exo_returns_correct_types(self):
        """Test that detect_exo returns correct types for each field."""
        result = detect_exo()
        self.assertIsInstance(result["running"], bool)
        self.assertIsInstance(result["installed"], bool)
        self.assertIsInstance(result["peers"], list)
        self.assertIsInstance(result["peer_count"], int)
        self.assertIsInstance(result["status"], str)

    def test_detect_exo_not_detected_status(self):
        """When exo is not installed, status should be 'not_detected'."""
        # On this machine, exo is likely not running
        result = detect_exo()
        # If exo is not installed, status should be not_detected
        if not result["installed"]:
            self.assertEqual(result["status"], "not_detected")
            self.assertFalse(result["running"])
            self.assertEqual(result["peer_count"], 0)

    def test_detect_exo_api_url_constant(self):
        """Test that the exo API URL constant is correct."""
        self.assertEqual(EXO_API_URL, "http://localhost:52415")


class TestExoDetectionMocked(unittest.TestCase):
    """Test exo detection with mocked subprocess and HTTP calls."""

    @patch("mindforge.hardware.exo_detector.subprocess.run")
    @patch("mindforge.hardware.exo_detector.requests")
    def test_detect_exo_running_with_peers(self, mock_requests, mock_subprocess):
        """Test detection when exo is running with peers."""
        # Mock HTTP response
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "mac1", "memory_gb": 32},
            {"name": "mac2", "memory_gb": 64},
        ]
        mock_requests.get.return_value = mock_resp

        result = detect_exo()

        self.assertTrue(result["running"])
        self.assertTrue(result["installed"])
        self.assertEqual(result["peer_count"], 2)
        self.assertEqual(result["status"], "running_with_peers")
        self.assertEqual(result["api_url"], EXO_API_URL)

    @patch("mindforge.hardware.exo_detector.subprocess.run")
    @patch("mindforge.hardware.exo_detector.requests")
    def test_detect_exo_running_no_peers(self, mock_requests, mock_subprocess):
        """Test detection when exo is running with no peers."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_requests.get.return_value = mock_resp

        result = detect_exo()

        self.assertTrue(result["running"])
        self.assertEqual(result["peer_count"], 0)
        self.assertEqual(result["status"], "running_no_peers")

    @patch("mindforge.hardware.exo_detector.subprocess.run")
    def test_detect_exo_installed_not_running(self, mock_subprocess):
        """Test detection when exo is installed but not running."""
        # Simulate: pgrep fails, which succeeds
        pgrep_result = Mock()
        pgrep_result.returncode = 1  # pgrep found no process
        pgrep_result.stdout = ""

        which_result = Mock()
        which_result.returncode = 0
        which_result.stdout = "/usr/local/bin/exo\n"

        def side_effect(*args, **kwargs):
            if args[0] == ["pgrep", "-f", "exo"]:
                return pgrep_result
            elif args[0] == ["which", "exo"]:
                return which_result
            return Mock(returncode=1, stdout="")

        mock_subprocess.side_effect = side_effect

        # Also mock requests to fail (connection refused)
        with patch("mindforge.hardware.exo_detector.requests") as mock_requests:
            mock_requests.exceptions.ConnectionError = ConnectionError
            mock_requests.get.side_effect = Exception("Connection refused")

            result = detect_exo()

        self.assertFalse(result["running"])
        self.assertTrue(result["installed"])
        self.assertEqual(result["status"], "installed_not_running")


class TestGetClusterInfo(unittest.TestCase):
    """Test get_cluster_info function."""

    @patch("mindforge.hardware.exo_detector.requests")
    def test_get_cluster_info_returns_dict_with_expected_keys(self, mock_requests):
        """Test that get_cluster_info returns a dict with expected keys."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "mac1", "memory_gb": 32, "model": "M3 Max"},
            {"name": "mac2", "memory_gb": 64, "model": "M2 Ultra"},
        ]
        mock_requests.get.return_value = mock_resp

        info = get_cluster_info()

        self.assertIsInstance(info, dict)
        self.assertIn("total_memory_gb", info)
        self.assertIn("total_usable_gb", info)
        self.assertIn("devices", info)
        self.assertIn("rdma_enabled", info)

    @patch("mindforge.hardware.exo_detector.requests")
    def test_get_cluster_info_calculates_memory(self, mock_requests):
        """Test that cluster info correctly calculates total memory."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "mac1", "memory_gb": 32},
            {"name": "mac2", "memory_gb": 64},
        ]
        mock_requests.get.return_value = mock_resp

        info = get_cluster_info()

        self.assertEqual(info["total_memory_gb"], 96.0)
        # Usable is ~80% of total
        self.assertAlmostEqual(info["total_usable_gb"], 76.8, places=1)
        self.assertEqual(len(info["devices"]), 2)

    @patch("mindforge.hardware.exo_detector.requests")
    def test_get_cluster_info_handles_bytes(self, mock_requests):
        """Test that cluster info handles memory in bytes."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "mac1", "memory_bytes": 34359738368},  # 32 GB
        ]
        mock_requests.get.return_value = mock_resp

        info = get_cluster_info()

        self.assertAlmostEqual(info["total_memory_gb"], 32.0, places=1)
        self.assertEqual(len(info["devices"]), 1)

    @patch("mindforge.hardware.exo_detector.requests")
    def test_get_cluster_info_handles_empty_cluster(self, mock_requests):
        """Test cluster info with no peers."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_requests.get.return_value = mock_resp

        info = get_cluster_info()

        self.assertEqual(info["total_memory_gb"], 0.0)
        self.assertEqual(info["total_usable_gb"], 0.0)
        self.assertEqual(info["devices"], [])

    def test_get_cluster_info_handles_connection_error(self):
        """Test that get_cluster_info returns defaults on connection error."""
        with patch("mindforge.hardware.exo_detector.requests") as mock_requests:
            mock_requests.get.side_effect = Exception("Connection refused")
            info = get_cluster_info()

        self.assertEqual(info["total_memory_gb"], 0.0)
        self.assertEqual(info["total_usable_gb"], 0.0)
        self.assertEqual(info["devices"], [])
        self.assertFalse(info["rdma_enabled"])


class TestFormatClusterInfo(unittest.TestCase):
    """Test format_cluster_info function."""

    def test_format_cluster_info_returns_string(self):
        """Test that format_cluster_info returns a string."""
        info = {
            "total_memory_gb": 96.0,
            "total_usable_gb": 76.8,
            "devices": [
                {"name": "mac1", "memory_gb": 32, "model": "M3 Max"},
                {"name": "mac2", "memory_gb": 64, "model": "M2 Ultra"},
            ],
            "rdma_enabled": True,
        }
        result = format_cluster_info(info)
        self.assertIsInstance(result, str)

    def test_format_cluster_info_contains_key_info(self):
        """Test that formatted info contains key information."""
        info = {
            "total_memory_gb": 96.0,
            "total_usable_gb": 76.8,
            "devices": [
                {"name": "mac1", "memory_gb": 32, "model": "M3 Max"},
            ],
            "rdma_enabled": True,
        }
        result = format_cluster_info(info)
        self.assertIn("Exo Cluster", result)
        self.assertIn("96.0", result)
        self.assertIn("76.8", result)
        self.assertIn("mac1", result)
        self.assertIn("M3 Max", result)
        self.assertIn("Yes", result)  # RDMA enabled

    def test_format_cluster_info_empty_cluster(self):
        """Test formatting with no devices."""
        info = {
            "total_memory_gb": 0.0,
            "total_usable_gb": 0.0,
            "devices": [],
            "rdma_enabled": False,
        }
        result = format_cluster_info(info)
        self.assertIn("Exo Cluster", result)
        self.assertIn("No", result)  # RDMA disabled
        self.assertIn("0", result)  # 0 devices

    def test_format_cluster_info_multiple_devices(self):
        """Test formatting with multiple devices."""
        info = {
            "total_memory_gb": 128.0,
            "total_usable_gb": 102.4,
            "devices": [
                {"name": "node-a", "memory_gb": 64, "model": "M2 Ultra"},
                {"name": "node-b", "memory_gb": 64, "model": "M2 Ultra"},
            ],
            "rdma_enabled": False,
        }
        result = format_cluster_info(info)
        self.assertIn("node-a", result)
        self.assertIn("node-b", result)
        self.assertIn("Cluster Devices", result)


class TestExoAdapter(unittest.TestCase):
    """Test ExoAdapter class."""

    def test_exo_adapter_class_exists(self):
        """Test that ExoAdapter class exists."""
        self.assertTrue(hasattr(__import__("mindforge.probe.adapters", fromlist=["ExoAdapter"]), "ExoAdapter"))

    def test_exo_adapter_is_model_adapter_subclass(self):
        """Test that ExoAdapter is a subclass of ModelAdapter."""
        self.assertTrue(issubclass(ExoAdapter, ModelAdapter))

    def test_exo_adapter_has_correct_interface(self):
        """Test that ExoAdapter has the expected interface."""
        adapter = ExoAdapter("test-model")
        self.assertTrue(hasattr(adapter, "ask"))
        self.assertTrue(hasattr(adapter, "close"))
        self.assertTrue(hasattr(adapter, "model_name"))
        self.assertEqual(adapter.model_name, "test-model")

    def test_exo_adapter_default_base_url(self):
        """Test that ExoAdapter uses the correct default base URL."""
        self.assertEqual(ExoAdapter.EXO_BASE_URL, "http://localhost:52415/v1")

    def test_exo_adapter_default_api_key(self):
        """Test that ExoAdapter uses the correct default API key."""
        self.assertEqual(ExoAdapter.EXO_API_KEY, "exo")

    def test_exo_adapter_init_sets_base_url_and_api_key(self):
        """Test that ExoAdapter stores base_url and api_key."""
        adapter = ExoAdapter("test-model")
        self.assertEqual(adapter.base_url, "http://localhost:52415/v1")
        self.assertEqual(adapter.api_key, "exo")

    def test_exo_adapter_close_resets_client(self):
        """Test that close() resets the client."""
        adapter = ExoAdapter("test-model")
        adapter.client = MagicMock()  # Simulate a loaded client
        adapter.close()
        self.assertIsNone(adapter.client)

    def test_exo_adapter_ask_returns_response(self):
        """Test that ExoAdapter.ask returns the model response."""
        adapter = ExoAdapter("test-model")
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is B."
        mock_client.chat.completions.create.return_value = mock_response
        adapter.client = mock_client

        result = adapter.ask("What is 2+2?", max_tokens=256)
        self.assertEqual(result, "The answer is B.")

        # Verify the call used the right model
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "test-model")
        self.assertEqual(call_args[1]["max_tokens"], 256)


class TestCreateAdapterWithExo(unittest.TestCase):
    """Test that create_adapter returns ExoAdapter when appropriate."""

    def test_create_adapter_exo_prefix(self):
        """Test that 'exo/' prefix returns ExoAdapter."""
        adapter = create_adapter("exo/llama-3.1-70b")
        self.assertIsInstance(adapter, ExoAdapter)
        self.assertEqual(adapter.model_name, "exo/llama-3.1-70b")

    @patch("mindforge.hardware.exo_detector.subprocess.run")
    @patch("mindforge.hardware.exo_detector.requests")
    def test_create_adapter_exo_running_with_peers(self, mock_requests, mock_subprocess):
        """Test that create_adapter returns ExoAdapter when exo is running with peers."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "mac1", "memory_gb": 32},
            {"name": "mac2", "memory_gb": 64},
        ]
        mock_requests.get.return_value = mock_resp

        adapter = create_adapter("mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertIsInstance(adapter, ExoAdapter)

    @patch("mindforge.hardware.exo_detector.subprocess.run")
    @patch("mindforge.hardware.exo_detector.requests")
    def test_create_adapter_exo_running_no_peers_falls_back(self, mock_requests, mock_subprocess):
        """Test that create_adapter falls back to MLXAdapter when exo has no peers."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_requests.get.return_value = mock_resp

        from mindforge.probe.adapters import MLXAdapter
        adapter = create_adapter("mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertIsInstance(adapter, MLXAdapter)

    @patch("mindforge.hardware.exo_detector.subprocess.run")
    @patch("mindforge.hardware.exo_detector.requests")
    def test_create_adapter_exo_not_installed_falls_back(self, mock_requests, mock_subprocess):
        """Test that create_adapter falls back when exo is not installed."""
        # Simulate all detection methods failing
        mock_requests.get.side_effect = Exception("Connection refused")

        pgrep_result = Mock()
        pgrep_result.returncode = 1
        pgrep_result.stdout = ""

        which_result = Mock()
        which_result.returncode = 1
        which_result.stdout = ""

        def side_effect(*args, **kwargs):
            if args[0] == ["pgrep", "-f", "exo"]:
                return pgrep_result
            elif args[0] == ["which", "exo"]:
                return which_result
            return Mock(returncode=1, stdout="")

        mock_subprocess.side_effect = side_effect

        from mindforge.probe.adapters import MLXAdapter
        adapter = create_adapter("mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.assertIsInstance(adapter, MLXAdapter)


class TestDatabaseExoTable(unittest.TestCase):
    """Test exo_cluster database table."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))

    def tearDown(self):
        self.db.close()

    def test_exo_cluster_table_exists(self):
        """Test that exo_cluster table exists in the database."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        self.assertIn("exo_cluster", tables)

    def test_exo_cluster_singleton_row_exists(self):
        """Test that the singleton row exists by default."""
        status = self.db.get_exo_status()
        self.assertIsNotNone(status)
        self.assertFalse(status["running"])
        self.assertFalse(status["installed"])
        self.assertEqual(status["status"], "not_detected")

    def test_store_exo_status_running(self):
        """Test storing exo status when running."""
        exo_info = {
            "running": True,
            "installed": True,
            "api_url": "http://localhost:52415",
            "peer_count": 2,
            "status": "running_with_peers",
            "total_memory_gb": 96.0,
            "total_usable_gb": 76.8,
            "devices": [
                {"name": "mac1", "memory_gb": 32},
                {"name": "mac2", "memory_gb": 64},
            ],
            "rdma_enabled": True,
        }
        self.db.store_exo_status(exo_info)
        status = self.db.get_exo_status()

        self.assertTrue(status["running"])
        self.assertTrue(status["installed"])
        self.assertEqual(status["peer_count"], 2)
        self.assertEqual(status["status"], "running_with_peers")
        self.assertEqual(status["api_url"], "http://localhost:52415")
        self.assertEqual(status["total_memory_gb"], 96.0)
        self.assertEqual(status["total_usable_gb"], 76.8)
        self.assertTrue(status["rdma_enabled"])

    def test_store_exo_status_not_installed(self):
        """Test storing exo status when not installed."""
        exo_info = {
            "running": False,
            "installed": False,
            "status": "not_detected",
            "peer_count": 0,
        }
        self.db.store_exo_status(exo_info)
        status = self.db.get_exo_status()

        self.assertFalse(status["running"])
        self.assertFalse(status["installed"])
        self.assertEqual(status["status"], "not_detected")
        self.assertEqual(status["peer_count"], 0)

    def test_store_exo_status_overwrites(self):
        """Test that storing exo status overwrites the singleton row."""
        # First store
        self.db.store_exo_status({"running": True, "installed": True, "status": "running_with_peers", "peer_count": 3})
        # Second store
        self.db.store_exo_status({"running": False, "installed": False, "status": "not_detected", "peer_count": 0})

        status = self.db.get_exo_status()
        self.assertFalse(status["running"])
        self.assertEqual(status["peer_count"], 0)
        self.assertEqual(status["status"], "not_detected")

    def test_store_exo_status_devices_json(self):
        """Test that devices are stored as JSON and retrieved as a list."""
        devices = [
            {"name": "node-a", "memory_gb": 64},
            {"name": "node-b", "memory_gb": 32},
        ]
        self.db.store_exo_status({
            "running": True,
            "installed": True,
            "devices": devices,
            "peer_count": 2,
        })
        status = self.db.get_exo_status()
        self.assertIsInstance(status["devices"], list)
        self.assertEqual(len(status["devices"]), 2)
        self.assertEqual(status["devices"][0]["name"], "node-a")


class TestCLIDetectExo(unittest.TestCase):
    """Test that CLI detect command shows exo info."""

    def test_cli_detect_imports_exo_detector(self):
        """Test that the detect command can import exo_detector."""
        # This tests that the import works at all
        from mindforge.hardware.exo_detector import detect_exo, get_cluster_info, format_cluster_info
        self.assertTrue(callable(detect_exo))
        self.assertTrue(callable(get_cluster_info))
        self.assertTrue(callable(format_cluster_info))

    def test_cli_detect_runs_without_error(self):
        """Test that 'mindforge detect' runs without crashing even without exo."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "detect"],
            capture_output=True, text=True, timeout=30,
            cwd=_project_root,
        )
        self.assertEqual(result.returncode, 0)
        # Should show hardware info regardless of exo
        self.assertIn("Hardware", result.stdout)

    def test_cli_models_runs_without_error(self):
        """Test that 'mindforge models' runs without crashing even without exo."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "mindforge.cli", "models"],
            capture_output=True, text=True, timeout=30,
            cwd=_project_root,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("MindForge", result.stdout)


if __name__ == "__main__":
    unittest.main()
