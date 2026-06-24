"""Anti-prompt-injection sanitization for web-sourced content.

Implements the 3-layer sanitization pipeline from the design doc:
  Layer 1: Remove injection patterns (regex blocklist)
  Layer 2: Unicode normalization (remove zero-width chars, normalize homoglyphs)
  Layer 3: Remove residual markup ([INST], <<SYS>>, etc.)

All web-sourced content is treated as untrusted data until sanitized.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


# ─── Injection Pattern Blocklist (from design doc lines 1422-1479) ──────────

INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"(?i)ignore (all )?(previous |above )?instructions",
    r"(?i)forget (everything |all )?(above|previous)",
    r"(?i)disregard (all )?(previous |above )?instructions",
    r"(?i)override (your |the )?(system |original )?prompt",
    r"(?i)you are now (a |an )?\w+",
    r"(?i)act as (if you are |a |an )",
    r"(?i)pretend (you are |to be )",
    r"(?i)from now on,? you (will|are|must)",

    # Role-play / persona hijacking
    r"(?i)system\s*:",
    r"\[SYSTEM\]",
    r"</system>",
    r"###\s*Human\s*:",
    r"###\s*Assistant\s*:",
    r"###\s*System\s*:",

    # Chat template tokens (should never appear in content)
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"\{\{system\}\}",

    # Capability requests
    r"(?i)reveal (your |the )?(system |hidden )?prompt",
    r"(?i)show (me )?(your |the )?(system |hidden )?prompt",
    r"(?i)what (are |is )your (instructions|rules|system prompt)",

    # Jailbreak patterns
    r"(?i)DAN (mode|prompt)",
    r"(?i)jailbreak",
    r"(?i)do anything now",
    r"(?i)developer mode",
    r"(?i)unrestricted mode",
]


# ─── Zero-width / invisible characters (from design doc) ─────────────────────

ZERO_WIDTH_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\ufeff",  # zero-width no-break space (BOM)
    "\u2060",  # word joiner
    "\u2061",  # function application
    "\u2062",  # invisible times
    "\u2063",  # invisible separator
    "\u2064",  # invisible plus
]

# Compiled zero-width removal regex
_ZERO_WIDTH_RE = re.compile("|".join(re.escape(c) for c in ZERO_WIDTH_CHARS))

# Residual markup patterns (Layer 3)
_RESIDUAL_MARKUP_PATTERNS = [
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"<\|[^|]*\|>",
    r"\{\{[^}]*\}\}",
    r"</?system>",
    r"###\s*(Human|Assistant|System)\s*:",
]
_RESIDUAL_MARKUP_RE = re.compile("|".join(_RESIDUAL_MARKUP_PATTERNS))

# Homoglyph map: common Cyrillic/Latin confusables → ASCII
_HOMOGLYPH_MAP = {
    "\u0430": "a",  # Cyrillic a
    "\u0435": "e",  # Cyrillic e
    "\u043e": "o",  # Cyrillic o
    "\u0440": "p",  # Cyrillic p
    "\u0441": "c",  # Cyrillic c
    "\u0443": "y",  # Cyrillic y
    "\u0445": "x",  # Cyrillic x
    "\u0410": "A",  # Cyrillic A
    "\u0412": "B",  # Cyrillic V → B (visual)
    "\u0415": "E",  # Cyrillic E
    "\u041a": "K",  # Cyrillic K
    "\u041c": "M",  # Cyrillic M
    "\u041d": "H",  # Cyrillic H
    "\u041e": "O",  # Cyrillic O
    "\u0420": "P",  # Cyrillic R → P (visual)
    "\u0421": "C",  # Cyrillic S → C (visual)
    "\u0422": "T",  # Cyrillic T
    "\u0425": "X",  # Cyrillic X
    "\u0432": "b",  # Cyrillic v → b (visual, lowercase)
    "\u043d": "h",  # Cyrillic n → h (visual)
    "\u0442": "t",  # Cyrillic t → t (visual)
    "\u0443": "y",  # Cyrillic u → y (visual)
    "\u0456": "i",  # Cyrillic і → i
    "\u0455": "s",  # Cyrillic ѕ → s
    "\u0458": "j",  # Cyrillic ј → j
}


def _normalize_homoglyphs(text: str) -> str:
    """Replace common homoglyph characters with their ASCII equivalents."""
    for homoglyph, ascii_char in _HOMOGLYPH_MAP.items():
        text = text.replace(homoglyph, ascii_char)
    return text


def sanitize_content(text: str) -> dict:
    """Sanitize content against prompt injection attacks.

    Runs 3 layers of sanitization:
      Layer 1: Remove text matching injection patterns (regex blocklist)
      Layer 2: Unicode normalization (zero-width chars, homoglyphs, whitespace)
      Layer 3: Remove residual markup tokens ([INST], <<SYS>>, etc.)

    Args:
        text: Raw text content from an untrusted source (web page, PDF, etc.)

    Returns:
        dict with keys:
            - clean_text: str, the sanitized text
            - flags: list[str], descriptions of detected injection attempts
            - is_safe: bool, True if no injection patterns were detected
    """
    flags = []
    clean = text

    # ── Layer 1: Injection pattern removal ──────────────────────────────
    for pattern in INJECTION_PATTERNS:
        matches = re.findall(pattern, clean)
        if matches:
            flags.append(f"injection_pattern: {pattern}")
            clean = re.sub(pattern, "", clean)

    # ── Layer 2: Unicode normalization ──────────────────────────────────
    # 2a. Remove zero-width characters
    found_zw = _ZERO_WIDTH_RE.findall(clean)
    if found_zw:
        flags.append(f"zero_width_chars: {len(found_zw)} found")
        clean = _ZERO_WIDTH_RE.sub("", clean)

    # 2b. Normalize homoglyphs
    clean = _normalize_homoglyphs(clean)

    # 2c. Unicode NFKC normalization (compatibility decomposition)
    clean = unicodedata.normalize("NFKC", clean)

    # 2d. Collapse excessive whitespace
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)

    # ── Layer 3: Residual markup removal ────────────────────────────────
    markup_matches = _RESIDUAL_MARKUP_RE.findall(clean)
    if markup_matches:
        flags.append(f"residual_markup: {markup_matches}")
        clean = _RESIDUAL_MARKUP_RE.sub("", clean)

    # Final cleanup: remove any leading/trailing whitespace
    clean = clean.strip()

    is_safe = len(flags) == 0

    return {
        "clean_text": clean,
        "flags": flags,
        "is_safe": is_safe,
    }
