"""Score MBPP-sanitized Pass@1 for a generation directory.

Reads:
  <config_dir>/vanilla_outputs.jsonl
  <prompts_jsonl>                          dataset (id, test_imports, test_list)

Writes:
  <config_dir>/extracted.jsonl             {id, completion, extraction_method}
  <config_dir>/scored.jsonl                {id, passed, result}
  <config_dir>/summary.md                  Pass@1 + extraction stats

Pass@1 = mean(passed) over the 257 problems (or 20 smoke).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark._shared.extract import extract_code
from benchmark._shared.sandbox import run_program

TIMEOUT_SECONDS = 5.0


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_program(test_imports: List[str], completion: str, test_list: List[str]) -> str:
    parts = []
    if test_imports:
        parts.append("\n".join(test_imports))
    parts.append(completion)
    parts.append("\n".join(test_list))
    return "\n".join(parts)


def score_dir(config_dir: Path, prompts_path: Path, mode: str = "vanilla") -> Dict:
    prompts = {r["id"]: r for r in read_jsonl(prompts_path)}
    out_name = "vanilla_outputs.jsonl" if mode == "vanilla" else "csg_outputs.jsonl"
    outputs = read_jsonl(config_dir / out_name)
    if mode == "csg":
        outputs = [r for r in outputs if r.get("variant_label", "default") == "default"]

    extracted_rows: List[Dict] = []
    scored_rows: List[Dict] = []
    n_passed = 0
    n_total = 0
    fence_count = 0
    raw_count = 0
    n_valid = 0  # non-empty completion

    for row in outputs:
        pid = row["id"]
        if pid not in prompts:
            continue
        meta = prompts[pid]
        completion, method = extract_code(row.get("answer_text", "") or "")
        if completion.strip():
            n_valid += 1
        if method == "fence":
            fence_count += 1
        else:
            raw_count += 1
        extracted_rows.append({
            "id": pid,
            "task_id": meta["task_id"],
            "completion": completion,
            "extraction_method": method,
        })

        program = build_program(meta.get("test_imports", []) or [], completion, meta["test_list"])
        result = run_program(program, timeout=TIMEOUT_SECONDS)
        passed = bool(result["passed"])
        if passed:
            n_passed += 1
        n_total += 1
        scored_rows.append({
            "id": pid,
            "task_id": meta["task_id"],
            "passed": passed,
            "result": result["result"],
            "extraction_method": method,
        })

    write_jsonl(config_dir / "extracted.jsonl", extracted_rows)
    write_jsonl(config_dir / "scored.jsonl", scored_rows)

    pass_at_1 = (n_passed / n_total) if n_total > 0 else 0.0
    valid_pct = (n_valid / n_total) if n_total > 0 else 0.0
    failed = [(r["id"], r["result"]) for r in scored_rows if not r["passed"]]

    lines = [
        f"# {config_dir.name} — MBPP-sanitized Pass@1",
        "",
        f"- mode: **{mode}**",
        f"- problems: **{n_total}**",
        f"- passed: **{n_passed}**",
        f"- **Pass@1: {pass_at_1*100:.2f}%**",
        f"- valid (non-empty completion): {n_valid}/{n_total} ({valid_pct*100:.1f}%)",
        f"- extraction: {fence_count} fenced, {raw_count} raw",
        "",
    ]
    if failed:
        lines.append(f"## Failed ({len(failed)})")
        lines.append("")
        for pid, result in failed[:30]:
            short = (result[:100] + "...") if len(result) > 100 else result
            lines.append(f"- `{pid}` — {short}")
        if len(failed) > 30:
            lines.append(f"- ... and {len(failed) - 30} more (see scored.jsonl)")
    (config_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "config": config_dir.name,
        "mode": mode,
        "n_total": n_total,
        "n_passed": n_passed,
        "pass_at_1": pass_at_1,
        "valid_pct": valid_pct,
        "fence_count": fence_count,
        "raw_count": raw_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_dir")
    ap.add_argument("prompts_path")
    ap.add_argument("--mode", choices=("vanilla", "csg"), default="vanilla")
    args = ap.parse_args()
    summary = score_dir(Path(args.config_dir), Path(args.prompts_path), args.mode)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
