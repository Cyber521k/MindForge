"""Template-Free formatter.

Format: {segments: [{text, label}, ...]}

Uses labeled segments for custom formatting where the consumer
decides how to render each segment.
"""


def format_template_free_entry(prompt, chosen, rejected=None):
    """Format a single Template-Free training entry.

    Template-Free format uses labeled segments:
    {"segments": [{"text": ..., "label": "instruction"}, {"text": ..., "label": "response"}, ...]}

    Args:
        prompt: The question/prompt text
        chosen: The correct response
        rejected: Optional rejected (incorrect) response, included as a segment

    Returns:
        dict with key: segments (list of {text, label} dicts)
    """
    segments = [
        {"text": prompt, "label": "instruction"},
        {"text": chosen, "label": "response"},
    ]

    if rejected is not None:
        segments.append({"text": rejected, "label": "rejected"})

    return {"segments": segments}


def format_template_free_batch(entries):
    """Format a batch of Template-Free entries.

    Args:
        entries: List of dicts with prompt, chosen, and optional rejected keys

    Returns:
        List of formatted Template-Free dicts
    """
    return [format_template_free_entry(
        prompt=e["prompt"],
        chosen=e["chosen"],
        rejected=e.get("rejected"),
    ) for e in entries]
