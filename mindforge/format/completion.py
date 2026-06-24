"""Completion formatter."""


def format_completion_entry(prompt, chosen, rejected=None):
    """Format a single completion training entry.

    Completion format: {"prompt": ..., "completion": ...}

    Args:
        prompt: The question/prompt text
        chosen: The correct response
        rejected: Ignored in completion format

    Returns:
        dict with keys: prompt, completion
    """
    return {
        "prompt": prompt,
        "completion": chosen,
    }


def format_completion_batch(entries):
    """Format a batch of completion entries."""
    return [format_completion_entry(
        prompt=e["prompt"],
        chosen=e["chosen"],
    ) for e in entries]
