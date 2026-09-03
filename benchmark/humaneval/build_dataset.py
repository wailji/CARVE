"""Build HumanEval Pass@1 prompts for Dream-Instruct.

Loads the 164-problem `openai/openai_humaneval` dataset, applies a minimal
chat-style instruction wrapper around each function signature/docstring, and
writes two JSONL files:

  prompts_full.jsonl   — all 164 problems

Each row contains:
  id                  — "HumanEval_<n>" (used by the runner as prompt id)
  task_id             — original "HumanEval/<n>" (used by human-eval scorer)
  prompt              — chat-content fed to dream_csg_experiment.py:prepare_prompt;
                        prepare_prompt then wraps it with the chat template.
  raw_prompt          — original function signature + docstring (no wrapping).
                        Concatenated with extracted completion before scoring.
  entry_point         — function name to test.
  test               — unit-test code from HumanEval.
  canonical_solution — reference solution (for sanity).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "benchmarks" / "humaneval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FULL_PATH = OUT_DIR / "prompts_full.jsonl"

INSTRUCTION_TEMPLATE = (
    "Complete the following Python function. "
    "Provide only the full function as a single Python code block, "
    "with no explanation before or after.\n\n"
    "```python\n{prompt}```"
)


def _index_of(task_id: str) -> int:
    m = re.match(r"HumanEval/(\d+)", task_id)
    return int(m.group(1)) if m else -1


def main() -> None:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for item in ds:
        task_id = item["task_id"]
        rows.append({
            "id": task_id.replace("/", "_"),
            "task_id": task_id,
            "prompt": INSTRUCTION_TEMPLATE.format(prompt=item["prompt"]),
            "raw_prompt": item["prompt"],
            "entry_point": item["entry_point"],
            "test": item["test"],
            "canonical_solution": item["canonical_solution"],
        })
    rows.sort(key=lambda r: _index_of(r["task_id"]))

    with FULL_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {FULL_PATH} ({len(rows)} problems)")


if __name__ == "__main__":
    main()
