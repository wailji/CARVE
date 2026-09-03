"""Prompt helpers for LLaDA-1.5 evaluation.

Same prompt text as llada_prompts.py (DAEDAL prompts) AND now the same
`<reasoning> ` assistant prefix as DAEDAL/scripts/eval_LLaDA_1p5_*.sh. The
prefix primes LLaDA-1.5 (a reasoning-tuned model) to start its reasoning block
immediately, matching DAEDAL's published baseline setup.

Re-exports humaneval/gsm8k/math500/mbpp doc_to_text unchanged.
"""
from __future__ import annotations

from typing import Optional  # noqa: E402

import torch

# Re-export unchanged doc_to_text helpers so runners can import from one place.
from benchmark._shared.llada_prompts import (  # noqa: F401
    humaneval_doc_to_text,
    gsm8k_doc_to_text,
    math500_doc_to_text,
    mbpp_doc_to_text,
)

# DAEDAL's published LLaDA-1.5 prefix (see DAEDAL/scripts/eval_LLaDA_1p5_*.sh)
DEFAULT_ASSISTANT_PREFIX: str = "<reasoning> "


def build_llada_1p5_prompt_ids(
    tokenizer,
    user_text: str,
    *,
    assistant_prefix: Optional[str] = DEFAULT_ASSISTANT_PREFIX,
) -> torch.LongTensor:
    """Wrap user_text in LLaDA-1.5 chat template, prepend `<reasoning> ` prefix by default.

    The prefix matches DAEDAL's published config for LLaDA-1.5
    (assistant_prefix=<reasoning> ). Pass `assistant_prefix=None` to skip
    the prefix (e.g. to reproduce the LLaDA/chat.py "raw" behavior).
    """
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if assistant_prefix:
        chat = chat + assistant_prefix
    enc = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"]
