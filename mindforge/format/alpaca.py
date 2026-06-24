"""Alpaca formatter."""


def format_alpaca_entry(prompt, chosen, rejected=None):
    """Format a single Alpaca training entry.

    Alpaca format: {"instruction": ..., "input": ..., "output": ...}

    Args:
        prompt: The question/prompt text
        chosen: The correct response
        rejected: Ignored in Alpaca format (kept for API compatibility)

    Returns:
        dict with keys: instruction, input, output
    """
    return {
        "instruction": prompt,
        "input": "",
        "output": chosen,
    }


def format_alpaca_batch(entries):
    """Format a batch of Alpaca entries."""
    return [format_alpaca_entry(
        prompt=e["prompt"],
        chosen=e["chosen"],
    ) for e in entries]
