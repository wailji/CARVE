#!/usr/bin/env python
"""Collect the finished runs into Table 1. Optional — it only reads results.

    python scripts/table1.py                 # the table
    python scripts/table1.py --latex         # LaTeX source
    python scripts/table1.py --flops         # Figure-2 FLOPs ratios

It walks outputs/benchmarks/<task>/table1/<method>__<model>/summary.json and
fills in whatever it finds. Cells that have not been run yet print as "--", so
a partial sweep is visible instead of being silently averaged away.

Nothing else in the repo depends on this script; deleting it would not affect
reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[1]

MODELS = [("dream", "Dream-v0-Instruct-7B"),
          ("llada-1.5", "LLaDA-1.5"),
          ("llada-8b", "LLaDA-8B-Instruct")]
METHODS = [("vanilla", "Baseline"), ("daedal", "DAEDAL"), ("carve", "CARVE")]
TASKS = [("humaneval", "HumanEval"), ("mbpp", "MBPP"),
         ("math500", "MATH-500"), ("gsm8k", "GSM8K")]


def summary_path(root: Path, method: str, model: str, task: str) -> Path:
    return root / task / "table1" / f"{method}__{model}" / "summary.json"


def load(root: Path, method: str, model: str, task: str) -> Optional[Dict]:
    p = summary_path(root, method, model, task)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def score_of(s: Optional[Dict]) -> Optional[float]:
    """Runners report the headline metric as pass_at_1 for every task."""
    if s is None:
        return None
    for key in ("pass_at_1", "accuracy", "score"):
        if key in s and isinstance(s[key], (int, float)):
            return 100.0 * s[key]
    return None


def flops_of(s: Optional[Dict]) -> Optional[float]:
    if s is None:
        return None
    return (s.get("perf", {}) or {}).get("mean_flops_per_sample") or None


def collect(root: Path, fn):
    return {(me, mo, ta): fn(load(root, me, mo, ta))
            for me, _ in METHODS for mo, _ in MODELS for ta, _ in TASKS}


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def cell(v, delta=None, width=15):
    if v is None:
        return "--".rjust(width)
    s = f"{v:.2f}" + (f" ({delta:+.2f})" if delta is not None else "")
    return s.rjust(width)


def render_text(data, root):
    head = f"{'Model':<22}{'Method':<10}" + "".join(t.rjust(15) for _, t in TASKS) + "Average".rjust(15)
    out = [head, "-" * len(head)]
    for mo, mo_label in MODELS:
        base = {ta: data[("vanilla", mo, ta)] for ta, _ in TASKS}
        base_avg = mean(base.values())
        for i, (me, me_label) in enumerate(METHODS):
            vals = {ta: data[(me, mo, ta)] for ta, _ in TASKS}
            avg = mean(vals.values())
            row = f"{mo_label if i == 0 else '':<22}{me_label:<10}"
            for ta, _ in TASKS:
                d = (vals[ta] - base[ta]
                     if me != "vanilla" and vals[ta] is not None and base[ta] is not None
                     else None)
                row += cell(vals[ta], d)
            d_avg = (avg - base_avg
                     if me != "vanilla" and avg is not None and base_avg is not None else None)
            out.append(row + cell(avg, d_avg))
        out.append("")
    out.append(f"source: {root}")
    return "\n".join(out)


def render_latex(data):
    out = [r"\begin{tabular}{ll" + "r" * (len(TASKS) + 1) + "}", r"\toprule",
           "Model & Method & " + " & ".join(t for _, t in TASKS) + r" & Average \\", r"\midrule"]
    for mo, mo_label in MODELS:
        base = {ta: data[("vanilla", mo, ta)] for ta, _ in TASKS}
        base_avg = mean(base.values())
        out.append(rf"\multicolumn{{{len(TASKS)+2}}}{{l}}{{\textit{{{mo_label}}}}} \\")
        for me, me_label in METHODS:
            vals = {ta: data[(me, mo, ta)] for ta, _ in TASKS}
            avg = mean(vals.values())
            cells = []
            for ta, _ in TASKS:
                v, b = vals[ta], base[ta]
                cells.append("--" if v is None else
                             f"{v:.2f}" if me == "vanilla" or b is None else
                             f"{v:.2f} ({v - b:+.2f})")
            cells.append("--" if avg is None else
                         f"{avg:.2f}" if me == "vanilla" or base_avg is None else
                         f"{avg:.2f} ({avg - base_avg:+.2f})")
            out.append(f"& {me_label} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out)


def render_flops(root):
    data = collect(root, flops_of)
    head = f"{'Model':<22}{'Method':<10}" + "".join(t.rjust(12) for _, t in TASKS)
    out = ["FLOPs relative to the fixed-length baseline (lower is better)", "",
           head, "-" * len(head)]
    for mo, mo_label in MODELS:
        first = True
        for me, me_label in METHODS:
            if me == "vanilla":
                continue
            row = f"{mo_label if first else '':<22}{me_label:<10}"
            first = False
            for ta, _ in TASKS:
                v, b = data[(me, mo, ta)], data[("vanilla", mo, ta)]
                row += ("--" if not v or not b else f"{v / b:.2f}x").rjust(12)
            out.append(row)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO / "outputs" / "benchmarks"),
                   help="outputs/benchmarks directory")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--flops", action="store_true")
    a = p.parse_args()

    root = Path(a.root)
    if not root.exists():
        raise SystemExit(f"no results directory: {root}\nRun:  python scripts/run.py --all")

    if a.flops:
        print(render_flops(root))
        return

    data = collect(root, score_of)
    have = sum(1 for v in data.values() if v is not None)
    if have == 0:
        raise SystemExit(f"no summary.json found under {root}\nRun:  python scripts/run.py --all")
    if have < len(data):
        print(f"# {len(data) - have} of {len(data)} cells not run yet\n")
    print(render_latex(data) if a.latex else render_text(data, root))


if __name__ == "__main__":
    main()
