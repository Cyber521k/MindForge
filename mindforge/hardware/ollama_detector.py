"""Ollama local model detection.

Ollama is a local model runner that exposes a REST API at
http://localhost:11434. This module detects whether Ollama is
running, lists available models, and formats information for CLI display.
"""

import subprocess
import logging

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

# Default Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_TAGS_PATH = "/api/tags"
OLLAMA_V1_ENDPOINT = f"{OLLAMA_API_URL}/v1"


def detect_ollama():
    """Detect if Ollama is running, installed, and what models are available.

    Uses three detection methods in order:
      1. HTTP GET to http://localhost:11434/api/tags (2s timeout)
      2. pgrep -f ollama (check for ollama process)
      3. which ollama (check if ollama binary is installed)

    Returns:
        dict with keys:
            - running (bool): whether Ollama API is responding
            - installed (bool): whether Ollama is installed (binary or running)
            - api_url (str or None): the Ollama v1 API base URL if running, else None
            - models (list): list of model name strings available in Ollama
            - model_count (int): number of models available
            - status (str): human-readable status string
    """
    result = {
        "running": False,
        "installed": False,
        "api_url": None,
        "models": [],
        "model_count": 0,
        "status": "not_detected",
    }

    # Method 1: Try HTTP GET to /api/tags endpoint
    try:
        resp = requests.get(
            f"{OLLAMA_API_URL}{OLLAMA_TAGS_PATH}",
            timeout=2,
        )
        if resp.status_code == 200:
            result["running"] = True
            result["installed"] = True
            result["api_url"] = OLLAMA_V1_ENDPOINT
            result["status"] = "running"

            # Parse models from response
            try:
                data = resp.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name", m.get("model", ""))
                    if name:
                        models.append(name)
                result["models"] = models
                result["model_count"] = len(models)
            except Exception:
                pass

            return result
    except Exception:
        pass

    # Method 2: Check for ollama process via pgrep
    try:
        proc_result = subprocess.run(
            ["pgrep", "-f", "ollama"],
            capture_output=True, text=True, timeout=2,
        )
        if proc_result.returncode == 0 and proc_result.stdout.strip():
            result["running"] = True
            result["installed"] = True
            result["status"] = "running_no_api"
            return result
    except Exception:
        pass

    # Method 3: Check if ollama is installed via which
    try:
        which_result = subprocess.run(
            ["which", "ollama"],
            capture_output=True, text=True, timeout=2,
        )
        if which_result.returncode == 0 and which_result.stdout.strip():
            result["installed"] = True
            result["status"] = "installed_not_running"
            return result
    except Exception:
        pass

    return result


def format_ollama_info(ollama_info):
    """Format Ollama info dict as a readable string for CLI display.

    Args:
        ollama_info: dict from detect_ollama()

    Returns:
        str: formatted multi-line string for CLI display
    """
    lines = ["=== Ollama ==="]

    if not ollama_info.get("running"):
        if ollama_info.get("installed"):
            lines.append("  Status: Installed but not running")
            lines.append("  Start ollama to enable local model probing")
        else:
            lines.append("  Status: Not detected")
        return "\n".join(lines)

    lines.append(f"  API URL:   {ollama_info.get('api_url', 'N/A')}")
    lines.append(f"  Models:    {ollama_info.get('model_count', 0)}")

    models = ollama_info.get("models", [])
    if models:
        lines.append("")
        lines.append("  --- Available Models ---")
        for i, name in enumerate(models):
            lines.append(f"    [{i+1}] {name}")

    return "\n".join(lines)
