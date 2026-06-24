"""Detect available API keys from environment variables."""

import os

# Mapping of provider name to possible env var names
PROVIDER_ENV_VARS = {
    "OpenAI": ["OPENAI_API_KEY"],
    "OpenRouter": ["OPENROUTER_API_KEY"],
    "Anthropic": ["ANTHROPIC_API_KEY"],
    "HuggingFace": ["HUGGINGFACE_API_KEY", "HF_API_KEY", "HUGGING_FACE_HUB_TOKEN"],
    "Together": ["TOGETHER_API_KEY"],
    "Groq": ["GROQ_API_KEY"],
    "Replicate": ["REPLICATE_API_TOKEN"],
}


def detect_available_apis():
    """Scan environment variables for known API provider keys.

    Returns:
        dict mapping provider name to bool (True if key found)
    """
    results = {}
    for provider, env_vars in PROVIDER_ENV_VARS.items():
        found = False
        for var in env_vars:
            val = os.environ.get(var, "")
            if val and len(val) > 5:
                found = True
                break
        results[provider] = found
    return results


def format_api_info(apis):
    """Format API availability info as a readable string."""
    lines = ["", "=== Available APIs ==="]
    for provider, available in apis.items():
        status = "✓ Available" if available else "✗ Not found"
        lines.append(f"  {provider:15s} {status}")
    return "\n".join(lines)
