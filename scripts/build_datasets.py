#!/usr/bin/env python
"""Download the four benchmarks and write their prompt files.

    python scripts/build_datasets.py            # all four
    python scripts/build_datasets.py gsm8k      # just one

Writes outputs/benchmarks/<task>/prompts_full.jsonl. Run this once before
scripts/run.py.

The per-task builders under benchmark/<task>/build_dataset.py hold the few-shot
prompt construction — MBPP is 4-shot and GSM8K is 8-shot for Dream — so they are
part of the recipe, not just download helpers.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = ["humaneval", "mbpp", "math500", "gsm8k"]


def main() -> None:
    wanted = sys.argv[1:] or TASKS
    unknown = [t for t in wanted if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; expected any of {TASKS}")

    failed = []
    for task in wanted:
        print(f"\n=== building {task} ===", flush=True)
        rc = subprocess.run([sys.executable, f"benchmark/{task}/build_dataset.py"],
                            cwd=REPO).returncode
        if rc != 0:
            failed.append(task)
    if failed:
        raise SystemExit(f"\nFAILED: {failed}")
    print(f"\nprompts written under {REPO / 'outputs' / 'benchmarks'}")


if __name__ == "__main__":
    main()
