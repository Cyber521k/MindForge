"""OpenAI Messages formatter.

Format: {messages: [{role, content}, ...]}
"""


def format_openai_messages_entry(prompt, chosen, rejected=None):
    """Format a single OpenAI Messages training entry.

    OpenAI Messages format uses the standard chat messages structure:
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    Args:
        prompt: The question/prompt text
        chosen: The correct response
        rejected: Ignored in OpenAI Messages format

    Returns:
        dict with key: messages (list of {role, content} dicts)
    """
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
    }


def format_openai_messages_batch(entries):
    """Format a batch of OpenAI Messages entries.

    Args:
        entries: List of dicts with prompt, chosen keys

    Returns:
        List of formatted OpenAI Messages dicts
    """
    return [format_openai_messages_entry(
        prompt=e["prompt"],
        chosen=e["chosen"],
    ) for e in entries]
