"""Format conversion between training data formats.

Provides convert_format(entries, source_format, target_format) to
convert between any of the 6 supported training formats:
    dpo, alpaca, chatml, completion, openai_messages, template_free

All formats share a common source shape: {prompt, chosen, rejected?}
The conversion extracts prompt and chosen (and rejected if present)
from any format's entries and re-formats them in the target format.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Supported format names
SUPPORTED_FORMATS = ["dpo", "alpaca", "chatml", "completion", "openai_messages", "template_free"]


def _extract_common(entry, source_format):
    """Extract prompt, chosen, rejected from any format's entry.

    Args:
        entry: A dict in one of the supported formats
        source_format: The format name string

    Returns:
        dict with keys: prompt, chosen, rejected (rejected may be None)
    """
    if source_format == "dpo":
        return {
            "prompt": entry.get("prompt", ""),
            "chosen": entry.get("chosen", ""),
            "rejected": entry.get("rejected"),
        }
    elif source_format == "alpaca":
        return {
            "prompt": entry.get("instruction", ""),
            "chosen": entry.get("output", ""),
            "rejected": None,
        }
    elif source_format == "chatml":
        # ChatML stores everything in a single text field
        text = entry.get("text", "")
        # Try to extract prompt and response from ChatML tags
        prompt = ""
        chosen = ""
        if "<|im_start|>user\n" in text and "<|im_end|>" in text:
            try:
                start = text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
                end = text.index("<|im_end|>", start)
                prompt = text[start:end].strip()
            except (ValueError, IndexError):
                prompt = text

            if "<|im_start|>assistant\n" in text:
                try:
                    start = text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
                    end = text.index("<|im_end|>", start)
                    chosen = text[start:end].strip()
                except (ValueError, IndexError):
                    chosen = text
        else:
            prompt = text
            chosen = text
        return {"prompt": prompt, "chosen": chosen, "rejected": None}
    elif source_format == "completion":
        return {
            "prompt": entry.get("prompt", ""),
            "chosen": entry.get("completion", ""),
            "rejected": None,
        }
    elif source_format == "openai_messages":
        messages = entry.get("messages", [])
        prompt = ""
        chosen = ""
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
            elif msg.get("role") == "assistant":
                chosen = msg.get("content", "")
        return {"prompt": prompt, "chosen": chosen, "rejected": None}
    elif source_format == "template_free":
        segments = entry.get("segments", [])
        prompt = ""
        chosen = ""
        rejected = None
        for seg in segments:
            label = seg.get("label", "")
            text = seg.get("text", "")
            if label == "instruction":
                prompt = text
            elif label == "response":
                chosen = text
            elif label == "rejected":
                rejected = text
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    else:
        raise ValueError(f"Unknown source format: {source_format}")


def _format_entry(common, target_format):
    """Format a common entry {prompt, chosen, rejected} into the target format.

    Args:
        common: dict with prompt, chosen, rejected keys
        target_format: The format name string

    Returns:
        dict in the target format
    """
    prompt = common.get("prompt", "")
    chosen = common.get("chosen", "")
    rejected = common.get("rejected")

    if target_format == "dpo":
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected or ""}
    elif target_format == "alpaca":
        return {"instruction": prompt, "input": "", "output": chosen}
    elif target_format == "chatml":
        text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{chosen}<|im_end|>"
        )
        return {"text": text}
    elif target_format == "completion":
        return {"prompt": prompt, "completion": chosen}
    elif target_format == "openai_messages":
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]
        }
    elif target_format == "template_free":
        segments = [
            {"text": prompt, "label": "instruction"},
            {"text": chosen, "label": "response"},
        ]
        if rejected:
            segments.append({"text": rejected, "label": "rejected"})
        return {"segments": segments}
    else:
        raise ValueError(f"Unknown target format: {target_format}")


def convert_format(entries, source_format, target_format):
    """Convert training data from one format to another.

    Args:
        entries: List of dicts in the source format
        source_format: Source format name (one of SUPPORTED_FORMATS)
        target_format: Target format name (one of SUPPORTED_FORMATS)

    Returns:
        List of dicts in the target format

    Raises:
        ValueError: If source or target format is not supported
    """
    if source_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported source format: {source_format}. "
            f"Supported: {SUPPORTED_FORMATS}"
        )
    if target_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported target format: {target_format}. "
            f"Supported: {SUPPORTED_FORMATS}"
        )

    if source_format == target_format:
        # No conversion needed
        return list(entries)

    result = []
    for entry in entries:
        common = _extract_common(entry, source_format)
        formatted = _format_entry(common, target_format)
        result.append(formatted)

    logger.info(f"Converted {len(result)} entries from {source_format} to {target_format}")
    return result
