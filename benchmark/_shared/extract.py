"""Generic code-fence + stop-sequence extraction shared across code benchmarks."""
from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

CODE_FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)
DEFAULT_STOP_MARKERS: Tuple[str, ...] = (
    "\nclass ",
    "\nif __name__",
    "<|im_end|>",
    "<|endoftext|>",
)
CHAT_TOKENS: Tuple[str, ...] = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def strip_chat_tokens(text: str) -> str:
    for tok in CHAT_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def extract_code(
    answer_text: str,
    entry_point: Optional[str] = None,
    extra_stops: Iterable[str] = (),
) -> Tuple[str, str]:
    """Return (completion, extraction_method).

    Method is "fence" if a ```python ... ``` block was found, else "raw".
    Truncation: after `def <entry_point>` (or after first 50 chars if not found),
    cut at the first occurrence of any stop marker.
    """
    cleaned = strip_chat_tokens(answer_text)
    method = "fence"
    m = CODE_FENCE.search(cleaned)
    if m:
        body = m.group(1)
    else:
        method = "raw"
        body = cleaned

    head_end = 50
    if entry_point:
        marker = f"def {entry_point}"
        idx = body.find(marker)
        if idx >= 0:
            head_end = idx + len(marker)

    cut = len(body)
    for marker in (*DEFAULT_STOP_MARKERS, *extra_stops):
        idx = body.find(marker, head_end)
        if 0 <= idx < cut:
            cut = idx
    return body[:cut].rstrip(), method


# ── Dream body re-indent ────────────────────────────────────────────────────
# Our decoders .strip() the decoded answer, which eats the leading indent of a
# function body continuing from the prompt's open code fence. Restore it before
# the AST sanitizer runs. (Dream's vanilla path is not stripped and skips this.)
_TOP_LEVEL_PREFIXES = ("def ", "class ", "import ", "from ", "@", "async ")

def _maybe_reindent_first_line(body: str) -> str:
    """Re-indent the first non-empty line by 4 spaces iff it's at column 0
    AND doesn't look like a top-level construct (def/class/import/decorator)."""
    lines = body.split("\n")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if line[:1].isspace():
            return body
        if stripped.startswith(_TOP_LEVEL_PREFIXES):
            return body
        lines[idx] = "    " + line
        return "\n".join(lines)
    return body
