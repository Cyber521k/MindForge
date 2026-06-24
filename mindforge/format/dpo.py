"""DPO (Direct Preference Optimization) formatter."""


def format_dpo_entry(prompt, chosen, rejected):
    """Format a single DPO training entry.

    Args:
        prompt: The question/prompt text
        chosen: The preferred (correct) response
        rejected: The rejected (incorrect) response

    Returns:
        dict with keys: prompt, chosen, rejected
    """
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def format_dpo_batch(entries):
    """Format a batch of DPO entries.

    Args:
        entries: List of dicts with prompt, chosen, rejected keys

    Returns:
        List of formatted DPO dicts
    """
    return [format_dpo_entry(
        prompt=e["prompt"],
        chosen=e["chosen"],
        rejected=e["rejected"],
    ) for e in entries]


def write_dpo_jsonl(entries, output_path):
    """Write DPO entries to a JSONL file.

    Args:
        entries: List of dicts with prompt, chosen, rejected keys
        output_path: Path to output .jsonl file
    """
    import json
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return len(entries)
