"""ChatML formatter."""


def format_chatml_entry(prompt, chosen, rejected=None):
    """Format a single ChatML training entry.

    ChatML format uses <im_start> and <im_end> tokens.

    Args:
        prompt: The question/prompt text
        chosen: The correct response
        rejected: Ignored in ChatML format

    Returns:
        dict with key: text (full ChatML formatted text)
    """
    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{chosen}<|im_end|>"
    )
    return {"text": text}


def format_chatml_batch(entries):
    """Format a batch of ChatML entries."""
    return [format_chatml_entry(
        prompt=e["prompt"],
        chosen=e["chosen"],
    ) for e in entries]
