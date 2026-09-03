"""DAEDAL prompt templates for LLaDA-8B-Instruct evaluation.

Verbatim from DAEDAL/dllm_eval/tasks/{task}/{utils.py, .yaml}.

Used by ALL three LLaDA variants (vanilla, CARVE, DAEDAL) so the prompts are
identical across methods — the only differences across runs are algorithmic.

Convention: every prompt is wrapped in LLaDA's chat template with
`assistant_prefix="<reasoning> "` (DAEDAL's setup); see
benchmark/_shared/llada_prompts.py::build_llada_prompt_ids.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


_MATH_SYSTEM_PROMPT = (
    "You are a math expert. You will be given a question to solve. Solve it step by step. "
    "Wrap the final answer in a \\boxed{}. \n"
    "Respond in the following format:\n"
    "<reasoning>\n"
    "Your reasoning here\n"
    "</reasoning>\n"
    "<answer>\n"
    "\\boxed{...}\n"
    "</answer>"
)


def humaneval_doc_to_text(doc: Dict) -> str:
    """DAEDAL/dllm_eval/tasks/humaneval/humaneval.yaml::doc_to_text"""
    prompt = doc["prompt"]
    entry_point = doc["entry_point"]
    return (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"{prompt}\n\n"
        "First, reason about the solution step-by-step. Then, write the code.\n"
        "Respond in the following format:\n"
        "<reasoning>\n"
        "Your reasoning here\n"
        "</reasoning>\n"
        "<answer>\n"
        "```python\n"
        f"The complete implementation  of the {entry_point} function\n"
        "```\n"
        "</answer>"
    )


def gsm8k_doc_to_text(doc: Dict) -> str:
    """DAEDAL/dllm_eval/tasks/gsm8k/utils.py::gsm_prompt"""
    return f"{_MATH_SYSTEM_PROMPT}\n\n{doc['question']}\n\n"


def math500_doc_to_text(doc: Dict) -> str:
    """DAEDAL/dllm_eval/tasks/math500/utils.py::math500_prompt"""
    return f"{_MATH_SYSTEM_PROMPT}\n\n{doc['problem']}\n\n"


def mbpp_doc_to_text(doc: Dict) -> str:
    """DAEDAL/dllm_eval/tasks/mbpp/mbpp.yaml::doc_to_text"""
    text = doc["text"]
    tests = doc["test_list"]
    return (
        f"\n{text} Your code should pass these tests:\n\n"
        f"{tests[0]}\n{tests[1]}\n{tests[2]} \n\n"
        "First, reason about the solution step-by-step. Then, write the code.\n"
        "Respond in the following format:\n"
        "<reasoning>\n"
        "Your reasoning here\n"
        "</reasoning>\n"
        "<answer>\n"
        "```python\n"
        "The complete implementation of the function\n"
        "```\n"
        "</answer>"
    )


def build_llada_prompt_ids(
    tokenizer,
    user_text: str,
    *,
    assistant_prefix: Optional[str] = "<reasoning> ",
) -> torch.LongTensor:
    """Wrap user_text in LLaDA chat template + DAEDAL's assistant_prefix.

    Mirrors LLaDA_DAEDAL.apply_chat_template (DAEDAL/models/LLaDA_DAEDAL.py:842).
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
