"""Math answer extraction + symbolic equality grading.

Used by MATH-500 (LaTeX answers in \\boxed{}) and GSM8K (last integer in CoT).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from math_verify import parse, verify


_BOXED_RE = re.compile(r"\\boxed\{")
_GSM8K_FINAL_RE = re.compile(r"[-]?\d[\d,]*(?:\.\d+)?")


def extract_boxed(text: str) -> Optional[str]:
    """Return the contents of the LAST \\boxed{...}, balancing braces."""
    matches = []
    for m in _BOXED_RE.finditer(text):
        depth = 1
        i = m.end()
        start = i
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    matches.append(text[start:i])
                    break
            i += 1
    return matches[-1] if matches else None


def extract_last_number(text: str) -> Optional[str]:
    """Return the last numeric substring from text (used for GSM8K)."""
    nums = _GSM8K_FINAL_RE.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def grade_math(predicted: Optional[str], gold: str) -> Tuple[bool, str]:
    """Symbolic equality via math_verify. Returns (passed, reason).

    math_verify's parse() needs the LaTeX wrapped in math delimiters to
    parse compound expressions like tuples and fractions correctly. We
    wrap both sides in \\boxed{...} before parsing — matches the
    standard MATH-500 evaluation convention.
    """
    if predicted is None or predicted.strip() == "":
        return False, "no_prediction"
    gold_wrapped = r"\boxed{" + gold + "}"
    pred_wrapped = r"\boxed{" + predicted + "}"
    try:
        gold_parsed = parse(gold_wrapped)
        pred_parsed = parse(pred_wrapped)
    except Exception as e:  # noqa: BLE001
        return False, f"parse_error: {e}"
    if not gold_parsed or not pred_parsed:
        # Fall back to literal string equality after light normalisation.
        return predicted.strip() == gold.strip(), "literal_fallback"
    try:
        ok = bool(verify(gold_parsed, pred_parsed))
    except Exception as e:  # noqa: BLE001
        return False, f"verify_error: {e}"
    return ok, "verified" if ok else "mismatch"


_GSM8K_HASH_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def grade_gsm8k_boxed(model_output: str, gold_int) -> Tuple[bool, str, Optional[str]]:
    """LLaDA/DAEDAL-style GSM8K extraction.

    Order: last \\boxed{...}  →  <answer>...</answer>  →  last number  →  numeric compare.
    Returns (passed, reason, predicted_str). gold_int may be int or numeric str.
    """
    pred: Optional[str] = None
    source = "none"
    boxed = extract_boxed(model_output)
    if boxed is not None:
        pred = boxed.strip()
        source = "boxed"
    if pred is None:
        m = _ANSWER_TAG_RE.search(model_output)
        if m:
            pred = m.group(1).strip()
            source = "answer_tag"
    if pred is None:
        pred = extract_last_number(model_output)
        source = "lastnum"
    if pred is None:
        return False, "no_prediction", None
    # Normalise — strip non-numeric LaTeX wrappers if present
    pred_clean = pred.replace(",", "").replace("$", "").strip()
    # Try numeric compare first
    try:
        if "." in pred_clean:
            ok = abs(float(pred_clean) - float(gold_int)) < 1e-6
        else:
            ok = int(pred_clean) == int(gold_int)
        return ok, ("matched_" + source) if ok else ("mismatch_" + source), pred_clean
    except ValueError:
        # Fall back to symbolic equality (e.g. \frac{1}{2} vs 0.5)
        ok, reason = grade_math(pred, str(gold_int))
        return ok, reason + "_" + source, pred_clean


def grade_gsm8k(model_output: str, gold_int) -> Tuple[bool, str, Optional[str]]:
    """Extract the predicted final answer from a GSM8K CoT response.

    Strategy (in order):
      1. `#### N` regex (the dataset's canonical end marker — preferred when
         the model emits it, e.g. when fine-tuned on GSM8K canonical format).
      2. Last numeric substring in the output (works for "The answer is N"
         CoT-style endings, which is what the lm-eval gsm8k_cot demonstrations
         elicit).

    Returns (passed, reason, predicted_str).
    """
    m = _GSM8K_HASH_RE.search(model_output)
    if m:
        pred = m.group(1).replace(",", "")
        source = "hash"
    else:
        pred = extract_last_number(model_output)
        source = "lastnum"
    if pred is None:
        return False, "no_number", None
    try:
        if "." in pred:
            ok = abs(float(pred) - float(gold_int)) < 1e-6
        else:
            ok = int(pred) == int(gold_int)
    except ValueError:
        return False, f"parse_error_{source}", pred
    return ok, ("matched_" + source) if ok else ("mismatch_" + source), pred
