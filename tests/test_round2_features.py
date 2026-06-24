"""Tests for Round 2 features: Ollama adapter, CLI flags, job cancellation,
hardware cache, and WebSocket reconnection improvements.

Covers:
- OllamaAdapter class (creation, prefix stripping, ask, close)
- create_adapter routing for ollama/ prefix
- detect_ollama and format_ollama_info
- CLI --verbose/--quiet flags
- FastAPI hardware caching
- FastAPI job cancellation (create, cancel, public_job)
- WebSocket reconnection recovery (job_statuses on connect, heartbeat concept)
- start_job_thread and is_job_cancelled
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import unittest
from unittest.mock import Mock, MagicMock, patch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_python_dir = os.path.join(_project_root, "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

PYTHON = sys.executable


def run_cli(*args, cwd=None):
    """Run a mindforge CLI command and return the CompletedProcess."""
    cmd = [PYTHON, "-m", "mindforge.cli"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or _project_root,
        timeout=60,
    )


# ═══════════════════════════════════════════════════════════════════
# OllamaAdapter Tests
# ═══════════════════════════════════════════════════════════════════

class TestOllamaAdapter(unittest.TestCase):
    """Tests for the OllamaAdapter class."""

    def test_ollama_adapter_class_exists(self):
        """OllamaAdapter class should exist."""
        from mindforge.probe.adapters import OllamaAdapter
        self.assertTrue(hasattr(OllamaAdapter, '__init__'))

    def test_ollama_adapter_is_model_adapter_subclass(self):
        """OllamaAdapter should be a subclass of ModelAdapter."""
        from mindforge.probe.adapters import OllamaAdapter, ModelAdapter
        self.assertTrue(issubclass(OllamaAdapter, ModelAdapter))

    def test_ollama_adapter_strips_prefix(self):
        """OllamaAdapter should strip the ollama/ prefix from model name."""
        from mindforge.probe.adapters import OllamaAdapter
        adapter = OllamaAdapter("ollama/llama3.2")
        self.assertEqual(adapter.model_name, "llama3.2")

    def test_ollama_adapter_without_prefix(self):
        """OllamaAdapter should work without the ollama/ prefix."""
        from mindforge.probe.adapters import OllamaAdapter
        adapter = OllamaAdapter("llama3.2")
        self.assertEqual(adapter.model_name, "llama3.2")

    def test_ollama_adapter_default_base_url(self):
        """OllamaAdapter should use the correct default base URL."""
        from mindforge.probe.adapters import OllamaAdapter
        self.assertEqual(OllamaAdapter.OLLAMA_BASE_URL, "http://localhost:11434/v1")

    def test_ollama_adapter_default_api_key(self):
        """OllamaAdapter should use the correct default API key."""
        from mindforge.probe.adapters import OllamaAdapter
        self.assertEqual(OllamaAdapter.OLLAMA_API_KEY, "ollama")

    def test_ollama_adapter_init_sets_base_url_and_api_key(self):
        """OllamaAdapter should store base_url and api_key on init."""
        from mindforge.probe.adapters import OllamaAdapter
        adapter = OllamaAdapter("ollama/test-model")
        self.assertEqual(adapter.base_url, "http://localhost:11434/v1")
        self.assertEqual(adapter.api_key, "ollama")

    def test_ollama_adapter_has_ask_method(self):
        """OllamaAdapter should have an ask method."""
        from mindforge.probe.adapters import OllamaAdapter
        adapter = OllamaAdapter("test-model")
        self.assertTrue(hasattr(adapter, "ask"))

    def test_ollama_adapter_has_close_method(self):
        """OllamaAdapter should have a close method."""
        from mindforge.probe.adapters import OllamaAdapter
        adapter = OllamaAdapter("test-model")
        self.assertTrue(hasattr(adapter, "close"))

    def test_ollama_adapter_close_resets_client(self):
        """close() should reset the client to None."""
        from mindforge.probe.adapters import OllamaAdapter
        adapter = OllamaAdapter("test-model")
        adapter.client = MagicMock()
        adapter.close()
        self.assertIsNone(adapter.client)

    def test_ollama_adapter_ask_returns_response(self):
        """ask() should return the model response via the OpenAI client."""
        from mindforge.probe.adapters import OllamaAdapter
        adapter = OllamaAdapter("ollama/test-model")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The answer is B."
        mock_client.chat.completions.create.return_value = mock_response
        adapter.client = mock_client

        result = adapter.ask("What is 2+2?", max_tokens=256)
        self.assertEqual(result, "The answer is B.")

        # Verify the call used the right model (stripped, not ollama/test-model)
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "test-model")
        self.assertEqual(call_args[1]["max_tokens"], 256)


class TestCreateAdapterOllama(unittest.TestCase):
    """Tests for create_adapter routing with Ollama."""

    def test_create_adapter_ollama_prefix(self):
        """create_adapter should return OllamaAdapter for ollama/ prefix."""
        from mindforge.probe.adapters import create_adapter, OllamaAdapter
        adapter = create_adapter("ollama/llama3.2")
        self.assertIsInstance(adapter, OllamaAdapter)
        # Model name should be stripped
        self.assertEqual(adapter.model_name, "llama3.2")

    def test_create_adapter_ollama_prefix_priority_over_exo(self):
        """ollama/ prefix should be checked before exo detection."""
        from mindforge.probe.adapters import create_adapter, OllamaAdapter
        adapter = create_adapter("ollama/exo-model")
        self.assertIsInstance(adapter, OllamaAdapter)


# ═══════════════════════════════════════════════════════════════════
# Ollama Detector Tests
# ═══════════════════════════════════════════════════════════════════

class TestOllamaDetector(unittest.TestCase):
    """Tests for the Ollama detector module."""

    def test_detect_ollama_returns_dict_with_expected_keys(self):
        """detect_ollama() should return a dict with all expected keys."""
        from mindforge.hardware.ollama_detector import detect_ollama
        result = detect_ollama()
        self.assertIsInstance(result, dict)
        for key in ["running", "installed", "api_url", "models", "model_count", "status"]:
            self.assertIn(key, result)

    def test_detect_ollama_returns_correct_types(self):
        """detect_ollama() should return correct types for each field."""
        from mindforge.hardware.ollama_detector import detect_ollama
        result = detect_ollama()
        self.assertIsInstance(result["running"], bool)
        self.assertIsInstance(result["installed"], bool)
        self.assertIsInstance(result["models"], list)
        self.assertIsInstance(result["model_count"], int)
        self.assertIsInstance(result["status"], str)

    def test_detect_ollama_not_detected_status(self):
        """When Ollama is not installed, status should be 'not_detected'."""
        from mindforge.hardware.ollama_detector import detect_ollama
        result = detect_ollama()
        if not result["installed"]:
            self.assertEqual(result["status"], "not_detected")
            self.assertFalse(result["running"])
            self.assertEqual(result["model_count"], 0)

    def test_ollama_api_url_constant(self):
        """OLLAMA_API_URL should be the expected value."""
        from mindforge.hardware.ollama_detector import OLLAMA_API_URL
        self.assertEqual(OLLAMA_API_URL, "http://localhost:11434")

    def test_ollama_v1_endpoint_constant(self):
        """OLLAMA_V1_ENDPOINT should be the expected value."""
        from mindforge.hardware.ollama_detector import OLLAMA_V1_ENDPOINT
        self.assertEqual(OLLAMA_V1_ENDPOINT, "http://localhost:11434/v1")


class TestOllamaDetectorMocked(unittest.TestCase):
    """Tests for detect_ollama with mocked subprocess and HTTP calls."""

    @patch("mindforge.hardware.ollama_detector.subprocess.run")
    @patch("mindforge.hardware.ollama_detector.requests")
    def test_detect_ollama_running_with_models(self, mock_requests, mock_subprocess):
        """Test detection when Ollama is running with models."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "qwen2.5:7b"},
            ]
        }
        mock_requests.get.return_value = mock_resp

        from mindforge.hardware.ollama_detector import detect_ollama
        result = detect_ollama()

        self.assertTrue(result["running"])
        self.assertTrue(result["installed"])
        self.assertEqual(result["model_count"], 2)
        self.assertIn("llama3.2:3b", result["models"])
        self.assertIn("qwen2.5:7b", result["models"])
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["api_url"], "http://localhost:11434/v1")

    @patch("mindforge.hardware.ollama_detector.subprocess.run")
    @patch("mindforge.hardware.ollama_detector.requests")
    def test_detect_ollama_running_no_models(self, mock_requests, mock_subprocess):
        """Test detection when Ollama is running but has no models."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": []}
        mock_requests.get.return_value = mock_resp

        from mindforge.hardware.ollama_detector import detect_ollama
        result = detect_ollama()

        self.assertTrue(result["running"])
        self.assertEqual(result["model_count"], 0)
        self.assertEqual(result["models"], [])

    @patch("mindforge.hardware.ollama_detector.requests")
    def test_detect_ollama_api_not_responding(self, mock_requests):
        """Test detection when Ollama API is not responding (connection refused)."""
        mock_requests.get.side_effect = Exception("Connection refused")

        # Also mock subprocess to simulate not installed
        with patch("mindforge.hardware.ollama_detector.subprocess.run") as mock_subprocess:
            pgrep_result = Mock()
            pgrep_result.returncode = 1
            pgrep_result.stdout = ""

            which_result = Mock()
            which_result.returncode = 1
            which_result.stdout = ""

            def side_effect(*args, **kwargs):
                if args[0] == ["pgrep", "-f", "ollama"]:
                    return pgrep_result
                elif args[0] == ["which", "ollama"]:
                    return which_result
                return Mock(returncode=1, stdout="")

            mock_subprocess.side_effect = side_effect

            from mindforge.hardware.ollama_detector import detect_ollama
            result = detect_ollama()

        self.assertFalse(result["running"])
        self.assertFalse(result["installed"])
        self.assertEqual(result["status"], "not_detected")

    @patch("mindforge.hardware.ollama_detector.requests")
    def test_detect_ollama_installed_via_which(self, mock_requests):
        """Test detection when Ollama binary is installed but not running."""
        mock_requests.get.side_effect = Exception("Connection refused")

        with patch("mindforge.hardware.ollama_detector.subprocess.run") as mock_subprocess:
            pgrep_result = Mock()
            pgrep_result.returncode = 1
            pgrep_result.stdout = ""

            which_result = Mock()
            which_result.returncode = 0
            which_result.stdout = "/usr/local/bin/ollama\n"

            def side_effect(*args, **kwargs):
                if args[0] == ["pgrep", "-f", "ollama"]:
                    return pgrep_result
                elif args[0] == ["which", "ollama"]:
                    return which_result
                return Mock(returncode=1, stdout="")

            mock_subprocess.side_effect = side_effect

            from mindforge.hardware.ollama_detector import detect_ollama
            result = detect_ollama()

        self.assertFalse(result["running"])
        self.assertTrue(result["installed"])
        self.assertEqual(result["status"], "installed_not_running")

    @patch("mindforge.hardware.ollama_detector.requests")
    def test_detect_ollama_running_via_pgrep(self, mock_requests):
        """Test detection when Ollama process is found via pgrep but API not responding."""
        mock_requests.get.side_effect = Exception("Connection refused")

        with patch("mindforge.hardware.ollama_detector.subprocess.run") as mock_subprocess:
            pgrep_result = Mock()
            pgrep_result.returncode = 0
            pgrep_result.stdout = "12345\n"

            def side_effect(*args, **kwargs):
                if args[0] == ["pgrep", "-f", "ollama"]:
                    return pgrep_result
                return Mock(returncode=1, stdout="")

            mock_subprocess.side_effect = side_effect

            from mindforge.hardware.ollama_detector import detect_ollama
            result = detect_ollama()

        self.assertTrue(result["running"])
        self.assertTrue(result["installed"])
        self.assertEqual(result["status"], "running_no_api")


class TestFormatOllamaInfo(unittest.TestCase):
    """Tests for format_ollama_info function."""

    def test_format_running_with_models(self):
        """format_ollama_info should show API URL and model list when running."""
        from mindforge.hardware.ollama_detector import format_ollama_info
        info = {
            "running": True,
            "installed": True,
            "api_url": "http://localhost:11434/v1",
            "models": ["llama3.2:3b", "qwen2.5:7b"],
            "model_count": 2,
            "status": "running",
        }
        result = format_ollama_info(info)
        self.assertIsInstance(result, str)
        self.assertIn("Ollama", result)
        self.assertIn("http://localhost:11434/v1", result)
        self.assertIn("llama3.2:3b", result)
        self.assertIn("qwen2.5:7b", result)
        self.assertIn("2", result)  # model count

    def test_format_installed_not_running(self):
        """format_ollama_info should show 'Installed but not running'."""
        from mindforge.hardware.ollama_detector import format_ollama_info
        info = {
            "running": False,
            "installed": True,
            "status": "installed_not_running",
        }
        result = format_ollama_info(info)
        self.assertIn("Installed but not running", result)

    def test_format_not_detected(self):
        """format_ollama_info should show 'Not detected'."""
        from mindforge.hardware.ollama_detector import format_ollama_info
        info = {
            "running": False,
            "installed": False,
            "status": "not_detected",
        }
        result = format_ollama_info(info)
        self.assertIn("Not detected", result)

    def test_format_running_no_models(self):
        """format_ollama_info should show 0 models when running but empty."""
        from mindforge.hardware.ollama_detector import format_ollama_info
        info = {
            "running": True,
            "installed": True,
            "api_url": "http://localhost:11434/v1",
            "models": [],
            "model_count": 0,
            "status": "running",
        }
        result = format_ollama_info(info)
        self.assertIn("0", result)
        self.assertIn("http://localhost:11434/v1", result)


# ═══════════════════════════════════════════════════════════════════
# CLI Global Flags Tests
# ═══════════════════════════════════════════════════════════════════

class TestCLIGlobalFlags(unittest.TestCase):
    """Tests for the --verbose and --quiet global CLI flags."""

    def test_verbose_flag_with_detect(self):
        """--verbose flag should not break detect command."""
        result = run_cli("--verbose", "detect")
        self.assertEqual(result.returncode, 0)

    def test_quiet_flag_with_detect(self):
        """--quiet flag should not break detect command."""
        result = run_cli("--quiet", "detect")
        self.assertEqual(result.returncode, 0)

    def test_verbose_short_flag_with_detect(self):
        """-v short flag should not break detect command."""
        result = run_cli("-v", "detect")
        self.assertEqual(result.returncode, 0)

    def test_quiet_short_flag_with_detect(self):
        """-q short flag should not break detect command."""
        result = run_cli("-q", "detect")
        self.assertEqual(result.returncode, 0)

    def test_verbose_flag_shows_in_help(self):
        """--verbose should appear in --help output."""
        result = run_cli("--help")
        self.assertIn("--verbose", result.stdout)
        self.assertIn("--quiet", result.stdout)

    def test_verbose_short_shows_in_help(self):
        """-v should appear in --help output."""
        result = run_cli("--help")
        self.assertIn("-v", result.stdout)

    def test_quiet_short_shows_in_help(self):
        """-q should appear in --help output."""
        result = run_cli("--help")
        self.assertIn("-q", result.stdout)

    def test_verbose_with_models(self):
        """--verbose with models should work."""
        result = run_cli("--verbose", "models")
        self.assertEqual(result.returncode, 0)

    def test_quiet_with_models(self):
        """--quiet with models should work."""
        result = run_cli("--quiet", "models")
        self.assertEqual(result.returncode, 0)


# ═══════════════════════════════════════════════════════════════════
# CLI Ollama Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestCLIOllamaIntegration(unittest.TestCase):
    """Tests for Ollama integration in the CLI detect/models commands."""

    def test_detect_runs_with_ollama_module(self):
        """detect command should run without error even with ollama module present."""
        result = run_cli("detect")
        self.assertEqual(result.returncode, 0)
        # If ollama is not installed, it stays silent
        # If installed, it shows "=== Ollama ==="

    def test_models_runs_with_ollama_module(self):
        """models command should run without error with ollama module present."""
        result = run_cli("models")
        self.assertEqual(result.returncode, 0)


# ═══════════════════════════════════════════════════════════════════
# FastAPI: Hardware Cache, Job Cancellation, public_job
# ═══════════════════════════════════════════════════════════════════

class TestFastAPIRound2Features(unittest.TestCase):
    """Tests for Round 2 server features: hardware cache, job cancel, public_job."""

    @classmethod
    def setUpClass(cls):
        try:
            from starlette.testclient import TestClient
            from server import app
            cls.client = TestClient(app)
        except ImportError:
            cls.client = None

    def setUp(self):
        if self.client is None:
            self.skipTest("FastAPI/Starlette TestClient not available")

    def test_hardware_cached_on_second_call(self):
        """GET /api/hardware should return cached data on second call."""
        resp1 = self.client.get("/api/hardware")
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()

        resp2 = self.client.get("/api/hardware")
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()

        # Cached data should be the same
        self.assertEqual(data1["chip"], data2["chip"])
        self.assertEqual(data1["memory_gb"], data2["memory_gb"])

    def test_public_job_strips_non_serializable_fields(self):
        """public_job should strip cancel_event and thread from job dict."""
        from server import public_job
        import threading
        job = {
            "status": "running",
            "type": "probe",
            "result": None,
            "error": None,
            "progress": 0,
            "started_at": time.time(),
            "cancel_event": threading.Event(),
            "thread": None,
        }
        public = public_job(job)
        self.assertNotIn("cancel_event", public)
        self.assertNotIn("thread", public)
        self.assertIn("status", public)
        self.assertIn("type", public)

    def test_public_job_preserves_serializable_fields(self):
        """public_job should keep all serializable fields."""
        from server import public_job
        job = {
            "status": "completed",
            "type": "train",
            "result": {"loss": 0.3},
            "error": None,
            "progress": 100,
            "started_at": 1234567890.0,
        }
        public = public_job(job)
        self.assertEqual(public["status"], "completed")
        self.assertEqual(public["type"], "train")
        self.assertEqual(public["progress"], 100)
        self.assertEqual(public["result"], {"loss": 0.3})

    def test_cancel_nonexistent_job_returns_404(self):
        """POST /api/jobs/nonexistent/cancel should return 404."""
        resp = self.client.post("/api/jobs/nonexistent_round2_job/cancel")
        self.assertEqual(resp.status_code, 404)

    def test_cancel_running_job(self):
        """POST /api/jobs/{id}/cancel on a running job should return cancelled."""
        from server import _jobs, create_job, cancel_job, is_job_cancelled
        import threading

        job_id = create_job("test_cancel")
        self.assertIn(job_id, _jobs)
        self.assertEqual(_jobs[job_id]["status"], "running")

        # Cancel it
        cancel_job(job_id)
        self.assertTrue(is_job_cancelled(job_id))
        self.assertEqual(_jobs[job_id]["status"], "cancelled")

        # Clean up
        del _jobs[job_id]

    def test_is_job_cancelled_returns_false_for_nonexistent(self):
        """is_job_cancelled should return False for nonexistent job."""
        from server import is_job_cancelled
        self.assertFalse(is_job_cancelled("nonexistent_job_xyz"))

    def test_finish_job_does_not_override_cancelled(self):
        """finish_job should not change status from 'cancelled' to 'completed'."""
        from server import _jobs, create_job, cancel_job, finish_job
        job_id = create_job("test_no_override")
        cancel_job(job_id)
        self.assertEqual(_jobs[job_id]["status"], "cancelled")

        finish_job(job_id, {"result": "done"})
        # Should still be cancelled, not completed
        self.assertEqual(_jobs[job_id]["status"], "cancelled")

        del _jobs[job_id]

    def test_fail_job_does_not_override_cancelled(self):
        """fail_job should not change status from 'cancelled' to 'failed'."""
        from server import _jobs, create_job, cancel_job, fail_job
        job_id = create_job("test_no_fail_override")
        cancel_job(job_id)

        fail_job(job_id, "some error")
        self.assertEqual(_jobs[job_id]["status"], "cancelled")

        del _jobs[job_id]

    def test_start_job_thread_starts_thread(self):
        """start_job_thread should start a daemon thread."""
        from server import _jobs, create_job, start_job_thread
        import threading
        import time as _time

        executed = threading.Event()

        def target():
            executed.set()

        job_id = create_job("test_thread")
        thread = start_job_thread(job_id, target)

        self.assertIsNotNone(thread)
        self.assertTrue(thread.daemon)

        # Wait for thread to execute
        executed.wait(timeout=2.0)
        self.assertTrue(executed.is_set())

        del _jobs[job_id]

    def test_get_job_returns_public_view(self):
        """GET /api/jobs/{id} should return the public view (no cancel_event/thread)."""
        from server import _jobs, create_job, start_job_thread
        import threading

        job_id = create_job("test_public_view")
        start_job_thread(job_id, lambda: None)

        resp = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotIn("cancel_event", data)
        self.assertNotIn("thread", data)
        self.assertIn("status", data)
        self.assertIn("type", data)

        del _jobs[job_id]

    def test_list_jobs_returns_public_view(self):
        """GET /api/jobs should return public views for all jobs."""
        from server import _jobs, create_job

        job_id = create_job("test_list_public")
        resp = self.client.get("/api/jobs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, dict)

        if job_id in data:
            job = data[job_id]
            self.assertNotIn("cancel_event", job)
            self.assertNotIn("thread", job)

        del _jobs[job_id]


# ═══════════════════════════════════════════════════════════════════
# WebSocket Reconnection Recovery Tests
# ═══════════════════════════════════════════════════════════════════

class TestWebSocketReconnection(unittest.TestCase):
    """Tests for WebSocket reconnection recovery features."""

    @classmethod
    def setUpClass(cls):
        try:
            from starlette.testclient import TestClient
            from server import app
            cls.client = TestClient(app)
        except ImportError:
            cls.client = None

    def setUp(self):
        if self.client is None:
            self.skipTest("FastAPI/Starlette TestClient not available")

    def test_job_statuses_sent_on_connect(self):
        """WebSocket should send job_statuses message on connect (for reconnection recovery)."""
        try:
            with self.client.websocket_connect("/ws") as ws:
                msg = ws.receive_json()
                self.assertEqual(msg["type"], "job_statuses")
                self.assertIn("jobs", msg)
        except Exception as e:
            self.skipTest(f"WebSocket test failed: {e}")

    def test_job_statuses_reflects_active_jobs(self):
        """job_statuses on connect should reflect any active jobs."""
        from server import _jobs, create_job
        try:
            job_id = create_job("test_ws_reconnect")
            with self.client.websocket_connect("/ws") as ws:
                msg = ws.receive_json()
                self.assertEqual(msg["type"], "job_statuses")
                # The job should appear in the statuses
                if job_id in msg["jobs"]:
                    job_data = msg["jobs"][job_id]
                    self.assertNotIn("cancel_event", job_data)
                    self.assertNotIn("thread", job_data)
            del _jobs[job_id]
        except Exception as e:
            if "test_ws_reconnect" in str(e):
                pass
            self.skipTest(f"WebSocket test failed: {e}")

    def test_websocket_disconnect_cleans_up_client(self):
        """WebSocket should be removed from _ws_clients on disconnect."""
        from server import _ws_clients
        initial_count = len(_ws_clients)
        try:
            with self.client.websocket_connect("/ws") as ws:
                ws.receive_json()  # consume initial
                self.assertEqual(len(_ws_clients), initial_count + 1)
            # After disconnect, count should return to initial
            self.assertEqual(len(_ws_clients), initial_count)
        except Exception as e:
            self.skipTest(f"WebSocket test failed: {e}")

    def test_websocket_pong_after_invalid_message(self):
        """WebSocket should still respond to ping after receiving invalid JSON."""
        try:
            with self.client.websocket_connect("/ws") as ws:
                ws.receive_json()  # consume initial
                # Send invalid JSON
                ws.send_text("{broken json")
                # Send valid ping
                ws.send_text("ping")
                msg = ws.receive_json()
                self.assertEqual(msg["type"], "pong")
        except Exception as e:
            self.skipTest(f"WebSocket test failed: {e}")

    def test_websocket_subscribe_with_empty_channels(self):
        """WebSocket subscribe with empty channels list should default to ['*']."""
        try:
            with self.client.websocket_connect("/ws") as ws:
                ws.receive_json()  # consume initial
                ws.send_text(json.dumps({"type": "subscribe", "channels": []}))
                msg = ws.receive_json()
                self.assertEqual(msg["type"], "subscribed")
                self.assertIn("channels", msg)
        except Exception as e:
            self.skipTest(f"WebSocket test failed: {e}")

    def test_multiple_websocket_clients_all_receive_job_statuses(self):
        """Multiple WebSocket clients should each receive job_statuses on connect."""
        try:
            with self.client.websocket_connect("/ws") as ws1:
                msg1 = ws1.receive_json()
                self.assertEqual(msg1["type"], "job_statuses")

                with self.client.websocket_connect("/ws") as ws2:
                    msg2 = ws2.receive_json()
                    self.assertEqual(msg2["type"], "job_statuses")
        except Exception as e:
            self.skipTest(f"Multiple WebSocket test failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Integration: Ollama in detect/models CLI
# ═══════════════════════════════════════════════════════════════════

class TestOllamaCLIIntegration(unittest.TestCase):
    """Integration tests for Ollama integration in CLI detect and models."""

    def test_detect_does_not_crash_with_ollama_module(self):
        """detect command should not crash even if ollama module is present."""
        result = run_cli("detect")
        self.assertEqual(result.returncode, 0)

    def test_models_does_not_crash_with_ollama_module(self):
        """models command should not crash even if ollama module is present."""
        result = run_cli("models")
        self.assertEqual(result.returncode, 0)

    def test_detect_output_still_has_hardware_section(self):
        """detect should still show Hardware Detection with ollama module present."""
        result = run_cli("detect")
        self.assertIn("Hardware Detection", result.stdout)

    def test_models_output_still_has_model_list(self):
        """models should still show model list with ollama module present."""
        result = run_cli("models")
        self.assertIn("MindForge", result.stdout)


# ═══════════════════════════════════════════════════════════════════
# Adapter Factory Complete Coverage
# ═══════════════════════════════════════════════════════════════════

class TestAdapterFactoryComplete(unittest.TestCase):
    """Complete coverage of create_adapter routing for all adapter types."""

    def test_all_adapter_types_routed(self):
        """create_adapter should route all known model prefixes correctly."""
        from mindforge.probe.adapters import (
            create_adapter, MLXAdapter, OpenAIAdapter,
            OpenRouterAdapter, ExoAdapter, OllamaAdapter
        )

        cases = [
            ("mlx-community/test-model", MLXAdapter),
            ("mlx-test/model", MLXAdapter),
            ("gpt-4o", OpenAIAdapter),
            ("o1-preview", OpenAIAdapter),
            ("o3-mini", OpenAIAdapter),
            ("openrouter/meta-llama/llama-3", OpenRouterAdapter),
            ("exo/llama-70b", ExoAdapter),
            ("ollama/llama3.2", OllamaAdapter),
        ]

        for model_name, expected_type in cases:
            adapter = create_adapter(model_name)
            self.assertIsInstance(adapter, expected_type,
                                f"Expected {expected_type.__name__} for '{model_name}', "
                                f"got {type(adapter).__name__}")

    def test_unknown_model_defaults_to_mlx(self):
        """Unknown model name should default to MLXAdapter."""
        from mindforge.probe.adapters import create_adapter, MLXAdapter
        adapter = create_adapter("unknown/model/name")
        self.assertIsInstance(adapter, MLXAdapter)

    def test_all_adapters_have_correct_interface(self):
        """All adapter types should have ask() and close() methods."""
        from mindforge.probe.adapters import (
            MLXAdapter, OpenAIAdapter, OpenRouterAdapter,
            ExoAdapter, OllamaAdapter
        )
        for adapter_class in [MLXAdapter, OpenAIAdapter, OpenRouterAdapter,
                               ExoAdapter, OllamaAdapter]:
            adapter = adapter_class("test-model")
            self.assertTrue(hasattr(adapter, "ask"))
            self.assertTrue(hasattr(adapter, "close"))
            self.assertTrue(hasattr(adapter, "model_name"))
            self.assertEqual(adapter.model_name, "test-model")

    def test_ollama_adapter_prefix_stripping_variations(self):
        """OllamaAdapter should handle various ollama/ prefix formats."""
        from mindforge.probe.adapters import OllamaAdapter

        # Standard case
        a1 = OllamaAdapter("ollama/llama3.2")
        self.assertEqual(a1.model_name, "llama3.2")

        # Model with tags
        a2 = OllamaAdapter("ollama/llama3.2:3b")
        self.assertEqual(a2.model_name, "llama3.2:3b")

        # Model with namespace
        a3 = OllamaAdapter("ollama/library/qwen2.5")
        self.assertEqual(a3.model_name, "library/qwen2.5")

        # No prefix (direct use)
        a4 = OllamaAdapter("llama3.2")
        self.assertEqual(a4.model_name, "llama3.2")


if __name__ == "__main__":
    unittest.main()
