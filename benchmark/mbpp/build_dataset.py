"""Build MBPP-FULL prompts for Dream-Instruct (lm-eval `mbpp_instruct` aligned).

Loads `mbpp` config `full` split `test` (500 problems — note the often-cited
"974" is the *total* MBPP dataset across all splits: train(374) + test(500) +
validation(90) + prompt(10). The standard eval set is the 500-problem test
split, which is what lm-eval-harness `mbpp_instruct.yaml` evaluates.)

4-shot prefix is taken from the `prompt` split (task_ids 1, 2, 3, 4) — these
are the canonical few-shot examples shipped with MBPP. lm-eval uses 3 of
these in `utils.list_fewshot_samples`; we use 4 per the user spec.

Per-shot format (matches lm-eval `mbpp_instruct.yaml.doc_to_text` +
`gen_prefix` + `doc_to_target` for is_fewshot examples):

    You are an expert Python programmer, and here is your task: <text>
    Your code should pass these tests:

    <test_list[0]>
    <test_list[1]>
    <test_list[2]>

    Here is the completed function:
    ```python
    <code>
    [DONE]

The 4-shot prefix concatenates four such blocks separated by "\n\n", then
the test problem ends right after `Your code should pass these tests:`. The
chat template + gen_prefix "Here is the completed function:" followed by an
opening ```python fence
are added at runtime by the driver.

Outputs (under `outputs/benchmarks/mbpp/`):
  prompts_full.jsonl   — all 500 test problems

Each row contains:
  id           — "MBPP_<task_id>"
  task_id      — int (the MBPP task_id)
  prompt       — full 4-shot+test-query string (NOT chat-template-wrapped)
  raw_prompt   — same as prompt (back-compat alias)
  test_imports — list of import lines required by the asserts (extracted from `test_setup_code`)
  test_list    — list of assert statements (graded against)
  code         — canonical solution (sanity reference)
  text         — raw problem description
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "benchmarks" / "mbpp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FULL_PATH = OUT_DIR / "prompts_full.jsonl"

# lm-eval `mbpp_instruct.yaml` doc_to_text:
DOC_TO_TEXT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {text} "
    "Your code should pass these tests:\n\n"
    "{test1}\n{test2}\n{test3}"
)
# Fewshot answer block (lm-eval's gen_prefix + doc_to_target for is_fewshot):
FEWSHOT_ANSWER_TEMPLATE = (
    "Here is the completed function:\n"
    "```python\n"
    "{code}\n"
    "[DONE]"
)


def _shot_text(text: str, test_list: List[str]) -> str:
    """Format the user-side question portion of one shot."""
    tests = list(test_list[:3]) + ["", "", ""]
    return DOC_TO_TEXT_TEMPLATE.format(text=text, test1=tests[0], test2=tests[1], test3=tests[2])


def _build_prompt(fewshot_examples: List[Dict], test_text: str, test_test_list: List[str]) -> str:
    """4-shot prefix + test query, ending right before `Here is the completed function:`."""
    blocks: List[str] = []
    for ex in fewshot_examples:
        q = _shot_text(ex["text"], ex["test_list"])
        a = FEWSHOT_ANSWER_TEMPLATE.format(code=ex["code"])
        blocks.append(q + "\n\n" + a)
    blocks.append(_shot_text(test_text, test_test_list))
    return "\n\n".join(blocks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-fewshot", type=int, default=4)
    args = p.parse_args()

    print(f"Loading mbpp full splits (test + prompt)...", flush=True)
    test_ds = load_dataset("mbpp", "full", split="test")
    prompt_ds = load_dataset("mbpp", "full", split="prompt")
    print(f"  test = {len(test_ds)} | prompt = {len(prompt_ds)}", flush=True)

    fewshot = [dict(prompt_ds[i]) for i in range(args.num_fewshot)]
    print(f"  using fewshot examples task_ids: {[ex['task_id'] for ex in fewshot]}", flush=True)

    rows: List[Dict] = []
    for item in test_ds:
        full_prompt = _build_prompt(fewshot, item["text"], list(item["test_list"]))
        rows.append({
            "id": f"MBPP_{item['task_id']}",
            "task_id": int(item["task_id"]),
            "prompt": full_prompt,
            "raw_prompt": full_prompt,
            "test_imports": [item.get("test_setup_code", "")] if item.get("test_setup_code") else [],
            "test_list": list(item["test_list"]),
            "code": item["code"],
            "text": item["text"],
        })

    with FULL_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {FULL_PATH} ({len(rows)} problems, 4-shot)", flush=True)



if __name__ == "__main__":
    main()
