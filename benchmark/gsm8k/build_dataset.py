"""Build GSM8K plain (non-CoT) 8-shot prompts.

Loads `gsm8k` config `main` split `test` (1319 problems). For each test row we
prepend the first 8 examples from the `train` split, formatted in the lm-eval
plain `gsm8k.yaml` doc_to_text style:

    Question: <shot1.question>
    Answer: <shot1.answer>

    Question: <shot2.question>
    Answer: <shot2.answer>

    ... (8 shots from train[:8])

    Question: <test_question>
    Answer:

The answers in the GSM8K dataset are natural-CoT solutions ending with
`#### N`, so the model still sees CoT-style demonstrations and tends to
emit CoT-style answers — but the *prompt template* is the plain
`Question:/Answer:` format, not the manually-crafted `gsm8k_cot.yaml` Q:/A:
template with hand-written CoT prefixes.

Each row contains:
  id           — "GSM8K_<idx>"
  question     — original question string
  answer_text  — original CoT answer (with `#### N`)
  gold_number  — int or float extracted after `####`
  prompt       — full 8-shot+test-query block (no chat template)
  raw_prompt   — alias for prompt
  gold_answer  — alias for gold_number (back-compat)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "benchmarks" / "gsm8k"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FULL_PATH = OUT_DIR / "prompts_full.jsonl"

_GOLD_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def parse_gold(answer: str):
    m = _GOLD_RE.search(answer)
    if not m:
        raise ValueError(f"no #### in answer: {answer[-80:]}")
    s = m.group(1).replace(",", "")
    return float(s) if "." in s else int(s)


def _build_prompt(question: str, fewshot_examples) -> str:
    blocks = []
    for ex in fewshot_examples:
        blocks.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    blocks.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(blocks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-fewshot", type=int, default=8)
    args = p.parse_args()

    train = load_dataset("gsm8k", "main", split="train")
    test = load_dataset("gsm8k", "main", split="test")
    fewshot_examples = [train[i] for i in range(args.num_fewshot)]

    rows = []
    for i, item in enumerate(test):
        gold = parse_gold(item["answer"])
        prompt = _build_prompt(item["question"], fewshot_examples)
        rows.append({
            "id": f"GSM8K_{i:04d}",
            "question": item["question"],
            "answer_text": item["answer"],
            "gold_number": gold,
            "prompt": prompt,
            "raw_prompt": prompt,
            "gold_answer": gold,
        })

    with FULL_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {FULL_PATH} ({len(rows)} problems, {args.num_fewshot}-shot plain)")


if __name__ == "__main__":
    main()
