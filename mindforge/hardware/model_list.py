"""Model list generation based on hardware and API key detection."""

import os
import json
import logging

from mindforge.hardware.detector import detect_hardware
from mindforge.hardware.api_keys import detect_available_apis

logger = logging.getLogger(__name__)

# Default directory for converted models
DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), "mindforge-data", "models"
)

# Memory tier definitions from the design doc
# | Memory Tier | Usable Memory | Model Size    | Example Models               |
# |-------------|--------------|---------------|------------------------------|
# | Tier S      | >= 96 GB     | 70B (4-bit)   | Llama 3.1 70B, Qwen 2.5 72B |
# | Tier A      | >= 64 GB     | 13B-32B (4-bit)| Mixtral 8x7B, Qwen 2.5 32B |
# | Tier B      | >= 32 GB     | 8B-9B (4-bit) | Llama 3.1 8B, Gemma 2 9B     |
# | Tier C      | >= 16 GB     | 7B (4-bit)    | Qwen 2.5 7B, Mistral 7B     |
# | Tier D      | >= 8 GB      | 3B-4B (4-bit) | Llama 3.2 3B, Phi-3.5        |
# | Tier E      | < 8 GB       | 1B-2B (4-bit) | Qwen 2.5 1.5B, Phi-3 mini    |

MEMORY_TIERS = [
    {"tier": "S", "min_memory_gb": 96, "label": "70B (4-bit)", "models": [
        {"name": "Llama 3.1 70B", "id": "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit", "size_gb": 40},
        {"name": "Qwen 2.5 72B", "id": "mlx-community/Qwen2.5-72B-Instruct-4bit", "size_gb": 40},
    ]},
    {"tier": "A", "min_memory_gb": 64, "label": "13B-32B (4-bit)", "models": [
        {"name": "Qwen 2.5 32B", "id": "mlx-community/Qwen2.5-32B-Instruct-4bit", "size_gb": 18},
        {"name": "Mixtral 8x7B", "id": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", "size_gb": 24},
    ]},
    {"tier": "B", "min_memory_gb": 32, "label": "8B-9B (4-bit)", "models": [
        {"name": "Llama 3.1 8B", "id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "size_gb": 5},
        {"name": "Gemma 2 9B", "id": "mlx-community/gemma-2-9b-it-4bit", "size_gb": 6},
    ]},
    {"tier": "C", "min_memory_gb": 16, "label": "7B (4-bit)", "models": [
        {"name": "Qwen 2.5 7B", "id": "mlx-community/Qwen2.5-7B-Instruct-4bit", "size_gb": 5},
        {"name": "Mistral 7B", "id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "size_gb": 5},
    ]},
    {"tier": "D", "min_memory_gb": 8, "label": "3B-4B (4-bit)", "models": [
        {"name": "Llama 3.2 3B", "id": "mlx-community/Llama-3.2-3B-Instruct-4bit", "size_gb": 2},
        {"name": "Qwen 2.5 3B", "id": "mlx-community/Qwen2.5-3B-Instruct-4bit", "size_gb": 2},
        {"name": "Phi-3.5 mini", "id": "mlx-community/Phi-3.5-mini-instruct-4bit", "size_gb": 2},
    ]},
    {"tier": "E", "min_memory_gb": 0, "label": "1B-2B (4-bit)", "models": [
        {"name": "Qwen 2.5 1.5B", "id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "size_gb": 1},
        {"name": "Phi-3 mini", "id": "mlx-community/Phi-3-mini-4k-instruct-4bit", "size_gb": 1},
    ]},
]

# Cloud models available per API provider
CLOUD_MODELS = {
    "OpenAI": [
        {"name": "GPT-4o", "id": "gpt-4o"},
        {"name": "GPT-4o mini", "id": "gpt-4o-mini"},
        {"name": "o1", "id": "o1"},
        {"name": "o3", "id": "o3"},
        {"name": "GPT-4 Turbo", "id": "gpt-4-turbo"},
    ],
    "OpenRouter": [
        {"name": "Llama 3.1 405B", "id": "openrouter/meta-llama/llama-3.1-405b-instruct"},
        {"name": "DeepSeek R1", "id": "openrouter/deepseek/deepseek-r1"},
        {"name": "Claude 4 Opus", "id": "openrouter/anthropic/claude-4-opus"},
        {"name": "Qwen 2.5 72B", "id": "openrouter/qwen/qwen-2.5-72b-instruct"},
    ],
    "Anthropic": [
        {"name": "Claude 4 Opus", "id": "anthropic/claude-4-opus"},
        {"name": "Claude 4 Sonnet", "id": "anthropic/claude-4-sonnet"},
        {"name": "Claude 3.5 Haiku", "id": "anthropic/claude-3.5-haiku"},
    ],
    "Groq": [
        {"name": "Llama 3.1 70B", "id": "groq/llama-3.1-70b"},
        {"name": "Llama 3.3 70B", "id": "groq/llama-3.3-70b"},
        {"name": "Mixtral 8x7B", "id": "groq/mistralai/mixtral-8x7b"},
    ],
    "Together": [
        {"name": "Llama 3.x", "id": "together/meta-llama/Llama-3.x"},
        {"name": "Qwen 2.5", "id": "together/Qwen/Qwen2.5"},
        {"name": "DeepSeek", "id": "together/deepseek-ai/deepseek"},
    ],
    "HuggingFace": [
        {"name": "Various HF Models", "id": "hf/various"},
    ],
    "Replicate": [
        {"name": "Various Replicate Models", "id": "replicate/various"},
    ],
}


def get_memory_tier(memory_gb):
    """Determine the memory tier (S, A, B, C, D, E) for a given amount of memory.

    Args:
        memory_gb: Total unified memory in GB

    Returns:
        str: Memory tier letter (S, A, B, C, D, or E)
    """
    for tier_info in MEMORY_TIERS:
        if memory_gb >= tier_info["min_memory_gb"]:
            return tier_info["tier"]
    return "E"


def get_available_models():
    """Get list of available models based on hardware and API keys.

    Returns:
        dict with keys:
            - hardware: hardware info dict
            - memory_tier: str (S, A, B, C, D, E)
            - local_models: list of model dicts with name, id, size_gb, tier, available
            - cloud_models: list of model dicts with name, id, provider
            - all_models: combined list
    """
    hw = detect_hardware()
    memory_gb = hw.get("memory_gb", 0)
    current_tier = get_memory_tier(memory_gb)

    # Build local model list
    tier_order = ["S", "A", "B", "C", "D", "E"]
    current_tier_idx = tier_order.index(current_tier)

    local_models = []
    for tier_info in MEMORY_TIERS:
        tier_idx = tier_order.index(tier_info["tier"])
        can_run = tier_idx >= current_tier_idx

        for model in tier_info["models"]:
            local_models.append({
                "name": model["name"],
                "id": model["id"],
                "size_gb": model["size_gb"],
                "tier": tier_info["tier"],
                "available": can_run,
                "type": "local",
            })

    # Build cloud model list
    apis = detect_available_apis()
    cloud_models = []
    for provider, available in apis.items():
        if available and provider in CLOUD_MODELS:
            for model in CLOUD_MODELS[provider]:
                cloud_models.append({
                    "name": model["name"],
                    "id": model["id"],
                    "provider": provider,
                    "available": True,
                    "type": "cloud",
                })

    # Scan for user-converted models (Phase 3)
    converted_models = scan_converted_models()

    return {
        "hardware": hw,
        "memory_tier": current_tier,
        "local_models": local_models,
        "cloud_models": cloud_models,
        "converted_models": converted_models,
        "all_models": local_models + converted_models + cloud_models,
    }


def format_model_list(model_info):
    """Format the model list as a readable string.

    Args:
        model_info: dict from get_available_models()

    Returns:
        str: Formatted model list
    """
    hw = model_info["hardware"]
    tier = model_info["memory_tier"]

    lines = [
        f"{'='*60}",
        f"  MindForge - Available Models",
        f"{'='*60}",
        f"",
        f"  Hardware:",
        f"    Chip:   {hw.get('chip', 'Unknown')}",
        f"    Memory: {hw.get('memory_gb', 0)} GB",
        f"    Tier:   {tier} (can run up to {_get_tier_label(tier)})",
        f"",
        f"  --- LOCAL (MLX) ---",
    ]

    for model in model_info["local_models"]:
        status = "✓" if model["available"] else "✗ (insufficient)"
        marker = " ← your max" if model["tier"] == tier and model["available"] else ""
        lines.append(
            f"  {status} {model['name']:30s} ~{model['size_gb']} GB   Tier {model['tier']}{marker}"
        )

    # Show converted models (Phase 3)
    converted_models = model_info.get("converted_models", [])
    if converted_models:
        lines.append(f"")
        lines.append(f"  --- CONVERTED (User) ---")
        for model in converted_models:
            quant = model.get("quantization", "full")
            lines.append(
                f"  ✓ {model['name']:30s} ~{model['size_gb']} GB   [Converted: {quant}]"
            )

    lines.append(f"")
    lines.append(f"  --- CLOUD (API) ---")

    if model_info["cloud_models"]:
        for model in model_info["cloud_models"]:
            lines.append(
                f"  ✓ {model['name']:30s} ({model['provider']})"
            )
    else:
        lines.append("  (no API keys detected -- run 'mindforge detect' to check)")

    lines.append(f"")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


def _get_tier_label(tier):
    """Get the label for a memory tier."""
    for tier_info in MEMORY_TIERS:
        if tier_info["tier"] == tier:
            return tier_info["label"]
    return "Unknown"


def scan_converted_models(models_dir=None):
    """Scan the models directory for user-converted MLX models.

    Args:
        models_dir: Directory to scan (default ~/mindforge-data/models/)

    Returns:
        list of dicts with keys:
            - name: display name
            - id: model path (used as model ID)
            - size_gb: estimated size
            - quantization: quantization info (e.g., '4bit', '8bit', 'full')
            - available: True
            - type: 'local'
            - converted: True (badge)
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR

    converted = []
    if not os.path.isdir(models_dir):
        return converted

    for entry in sorted(os.listdir(models_dir)):
        entry_path = os.path.join(models_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Look for config.json to confirm it's a model directory
        config_path = os.path.join(entry_path, "config.json")
        if not os.path.isfile(config_path):
            continue

        # Determine quantization from name
        quantization = "full"
        name_lower = entry.lower()
        for bits in [2, 3, 4, 6, 8]:
            if f"{bits}bit" in name_lower:
                quantization = f"{bits}bit"
                break

        # Calculate size
        total_bytes = 0
        for dirpath, dirnames, filenames in os.walk(entry_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_bytes += os.path.getsize(fp)
        size_gb = round(total_bytes / (1024 ** 3), 2) if total_bytes > 0 else 0

        # Read model name from config if available
        display_name = entry
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                if "model_type" in config:
                    display_name = f"{entry} ({config['model_type']})"
        except Exception:
            pass

        converted.append({
            "name": display_name,
            "id": entry_path,
            "size_gb": size_gb,
            "tier": "Converted",
            "available": True,
            "type": "local",
            "converted": True,
            "quantization": quantization,
        })

    return converted
