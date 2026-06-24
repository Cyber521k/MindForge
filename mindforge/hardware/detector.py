"""Hardware detection for Apple Silicon Macs."""

import subprocess
import re


def detect_hardware():
    """Detect Apple Silicon hardware info via sysctl.

    Returns:
        dict with keys: chip, memory_gb, model_name, cpu_cores, gpu_cores
    """
    info = {
        "chip": "Unknown",
        "memory_gb": 0,
        "model_name": "Unknown",
        "cpu_cores": 0,
        "gpu_cores": 0,
    }

    # Chip name
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            info["chip"] = result.stdout.strip()
    except Exception:
        pass

    # Memory
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            bytes_val = int(result.stdout.strip())
            info["memory_gb"] = round(bytes_val / (1024 ** 3), 1)
    except Exception:
        pass

    # CPU cores
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            info["cpu_cores"] = int(result.stdout.strip())
    except Exception:
        pass

    # Model name
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            info["model_name"] = result.stdout.strip()
    except Exception:
        pass

    # GPU cores - try system_profiler
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            text = result.stdout
            # Extract total number of cores
            cores_match = re.search(r"Total Number of Cores:\s*(\d+)", text)
            if cores_match:
                total = int(cores_match.group(1))
                if info["cpu_cores"] > 0:
                    info["gpu_cores"] = total - info["cpu_cores"]
                else:
                    info["cpu_cores"] = total
                    info["gpu_cores"] = 0
            # Also try to get chip name from system_profiler
            chip_match = re.search(r"Chip:\s*(.+)", text)
            if chip_match and info["chip"] == "Unknown":
                info["chip"] = chip_match.group(1).strip()
            # Model name
            model_match = re.search(r"Model Name:\s*(.+)", text)
            if model_match:
                info["model_name"] = model_match.group(1).strip()
            # Memory from system_profiler too
            mem_match = re.search(r"Memory:\s*(.+)", text)
            if mem_match and info["memory_gb"] == 0:
                mem_str = mem_match.group(1).strip()
                # e.g. "16 GB" or "32 GB"
                num_match = re.search(r"(\d+)", mem_str)
                if num_match:
                    info["memory_gb"] = int(num_match.group(1))
    except Exception:
        pass

    return info


def format_hardware_info(hw):
    """Format hardware info dict as a readable string."""
    lines = [
        "=== Hardware Detection ===",
        f"  Chip:        {hw.get('chip', 'Unknown')}",
        f"  Model:       {hw.get('model_name', 'Unknown')}",
        f"  Memory:      {hw.get('memory_gb', 0)} GB",
        f"  CPU Cores:   {hw.get('cpu_cores', 0)}",
        f"  GPU Cores:   {hw.get('gpu_cores', 0)}",
    ]
    return "\n".join(lines)
