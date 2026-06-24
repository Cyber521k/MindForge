"""Exo cluster detection and topology information.

Exo is a distributed inference framework that allows multiple Apple Silicon
Macs (or other devices) to form a cluster for running large language models.
This module detects whether exo is running, queries its cluster API, and
formats cluster topology information for CLI display.
"""

import subprocess
import logging

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

# Default exo API endpoint
EXO_API_URL = "http://localhost:52415"
EXO_CLUSTER_PEERS_PATH = "/cluster/peers"
EXO_V1_ENDPOINT = f"{EXO_API_URL}/v1"


def detect_exo():
    """Detect if exo is running, installed, and how many peers are in the cluster.

    Uses three detection methods in order:
      1. HTTP GET to http://localhost:52415/cluster/peers (2s timeout)
      2. pgrep -f exo (check for exo process)
      3. which exo (check if exo binary is installed)

    Returns:
        dict with keys:
            - running (bool): whether exo API is responding
            - installed (bool): whether exo is installed (binary or running)
            - api_url (str or None): the exo API base URL if running, else None
            - peers (list): list of peer info dicts from the cluster API
            - peer_count (int): number of peers in the cluster
            - status (str): human-readable status string
    """
    result = {
        "running": False,
        "installed": False,
        "api_url": None,
        "peers": [],
        "peer_count": 0,
        "status": "not_detected",
    }

    peers = []

    # Method 1: Try HTTP GET to cluster/peers endpoint
    try:
        resp = requests.get(
            f"{EXO_API_URL}{EXO_CLUSTER_PEERS_PATH}",
            timeout=2,
        )
        if resp.status_code == 200:
            result["running"] = True
            result["installed"] = True
            result["api_url"] = EXO_API_URL
            result["status"] = "running"

            # Parse peers from response
            try:
                data = resp.json()
                if isinstance(data, list):
                    peers = data
                elif isinstance(data, dict):
                    # Could be wrapped in a dict with a "peers" key
                    peers = data.get("peers", [])
                    if isinstance(peers, dict):
                        peers = list(peers.values())
            except Exception:
                peers = []

            result["peers"] = peers
            result["peer_count"] = len(peers)

            if len(peers) == 0:
                result["status"] = "running_no_peers"
            else:
                result["status"] = "running_with_peers"

            return result
    except Exception:
        pass

    # Method 2: Check for exo process via pgrep
    try:
        proc_result = subprocess.run(
            ["pgrep", "-f", "exo"],
            capture_output=True, text=True, timeout=2,
        )
        if proc_result.returncode == 0 and proc_result.stdout.strip():
            result["running"] = True
            result["installed"] = True
            result["status"] = "running_no_api"
            return result
    except Exception:
        pass

    # Method 3: Check if exo is installed via which
    try:
        which_result = subprocess.run(
            ["which", "exo"],
            capture_output=True, text=True, timeout=2,
        )
        if which_result.returncode == 0 and which_result.stdout.strip():
            result["installed"] = True
            result["status"] = "installed_not_running"
            return result
    except Exception:
        pass

    return result


def get_cluster_info(exo_api=None):
    """Query the exo cluster API for topology information.

    Args:
        exo_api: The exo API base URL (e.g. http://localhost:52415).
                 If None, uses the default EXO_API_URL.

    Returns:
        dict with keys:
            - total_memory_gb (float): total memory across all peers
            - total_usable_gb (float): estimated usable memory for models
            - devices (list): list of device/peer info dicts
            - rdma_enabled (bool): whether RDMA is enabled
    """
    if exo_api is None:
        exo_api = EXO_API_URL

    info = {
        "total_memory_gb": 0.0,
        "total_usable_gb": 0.0,
        "devices": [],
        "rdma_enabled": False,
    }

    try:
        resp = requests.get(
            f"{exo_api}{EXO_CLUSTER_PEERS_PATH}",
            timeout=2,
        )
        if resp.status_code == 200:
            data = resp.json()

            # Normalize peers into a list
            if isinstance(data, list):
                peers = data
            elif isinstance(data, dict):
                peers = data.get("peers", [])
                if isinstance(peers, dict):
                    peers = list(peers.values())
            else:
                peers = []

            devices = []
            total_mem = 0.0

            for peer in peers:
                if not isinstance(peer, dict):
                    continue

                # Extract memory info - exo peers may report memory in bytes or GB
                mem_bytes = peer.get("memory_bytes", 0)
                mem_gb = peer.get("memory_gb", 0)

                if mem_bytes and not mem_gb:
                    mem_gb = mem_bytes / (1024 ** 3)

                total_mem += mem_gb

                device = {
                    "name": peer.get("name", peer.get("hostname", "unknown")),
                    "memory_gb": round(mem_gb, 2),
                    "model": peer.get("model", peer.get("chip", "unknown")),
                    "ip": peer.get("ip", peer.get("address", "unknown")),
                }
                devices.append(device)

            # Usable memory is typically ~80% of total (leave room for OS)
            usable = total_mem * 0.8

            info["total_memory_gb"] = round(total_mem, 2)
            info["total_usable_gb"] = round(usable, 2)
            info["devices"] = devices
            info["rdma_enabled"] = data.get("rdma_enabled", False) if isinstance(data, dict) else False

    except Exception as e:
        logger.warning(f"Failed to get cluster info from exo: {e}")

    return info


def format_cluster_info(info):
    """Format cluster info dict as a readable string for CLI display.

    Args:
        info: dict from get_cluster_info()

    Returns:
        str: formatted multi-line string for CLI display
    """
    lines = [
        "=== Exo Cluster ===",
        f"  Total Memory:   {info.get('total_memory_gb', 0):.1f} GB",
        f"  Usable Memory:  {info.get('total_usable_gb', 0):.1f} GB",
        f"  RDMA Enabled:   {'Yes' if info.get('rdma_enabled') else 'No'}",
        f"  Devices:        {len(info.get('devices', []))}",
    ]

    devices = info.get("devices", [])
    if devices:
        lines.append("")
        lines.append("  --- Cluster Devices ---")
        for i, dev in enumerate(devices):
            name = dev.get("name", "unknown")
            mem = dev.get("memory_gb", 0)
            model = dev.get("model", "unknown")
            lines.append(f"    [{i+1}] {name:20s} {mem:>6.1f} GB  ({model})")

    return "\n".join(lines)
