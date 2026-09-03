"""HumanEval driver — DAEDAL on Dream-v0-Instruct-7B.

Uses the existing Dream prompt (DOC_TO_TEXT + GEN_PREFIX from
benchmark.humaneval.run_vanilla_dream) for fair comparison with vanilla Dream
and CARVE Dream. Algorithm: scripts.daedal_dream.run_daedal_dream — DAEDAL's
algorithm on Dream's forward, with all DAEDAL hyperparameters at paper defaults.

Scoring: re-indent function body + raw_prompt prepend (Dream prompt convention)
+ sanitize + check_correctness — same as benchmark.humaneval.run_carve_dream.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.daedal_dream import run_daedal_dream  # noqa: E402
from scripts.dream_backend import load_model_and_tokenizer  # noqa: E402
from benchmark.humaneval.run_vanilla_dream import DOC_TO_TEXT, GEN_PREFIX, sanitize  # noqa: E402
from benchmark.humaneval.run_vanilla_dream import build_input_ids  # noqa: E402
from human_eval.execution import check_correctness  # noqa: E402
from benchmark._shared.results_io import (  # noqa: E402
    shard_output_path,
    merged_output_path,
    write_run_config,
    attach_summary_metadata,
    aggregate_perf_metrics,
    render_config_md,
    read_run_config,
)
from benchmark._shared.paper_config import (  # noqa: E402
    resolve,
    DAEDAL,
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "daedal", "dream", "humaneval"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"
TIMEOUT_SECONDS = 5.0


def run_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts)
    all_prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    shard = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] device={args.device} n={len(shard)}", flush=True)

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, args.device)
    print(
        f"[shard {args.shard_id}] L0={CFG.l0} max_len={CFG.lmax} "
        f"block={CFG.block_length} ef={8} hi={0.9} "
        f"lo={0.1} eos_thr={0.5}",
        flush=True,
    )

    write_run_config(out_dir, args, task="humaneval", method="daedal_dream", num_shots=0, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(shard):
            pid = row["id"]
            try:
                prompt_ids = build_input_ids(tokenizer, row["raw_prompt"])
                t0 = time.time()
                rec = run_daedal_dream(
                    model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
                    seed=CFG.seed,
                    initial_gen_length=CFG.l0,
                    max_gen_length=CFG.lmax,
                    block_length=CFG.block_length,
                    expansion_factor=8,
                    low_conf_threshold=0.1,
                    eos_confidence_threshold=0.5,
                    expand_eos_confidence_threshold=0.9,
                    eos_check_tokens=32,
                    cfg_scale=0.0,
                    temperature=CFG.temperature,
                    max_iterations=CFG.steps,
                )
                elapsed = time.time() - t0
                rec["id"] = pid
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[shard {args.shard_id}] [{idx+1}/{len(shard)}] {pid} done in {elapsed:.1f}s", flush=True)
            except Exception as exc:
                err_trace = traceback.format_exc(limit=5)
                rec = {
                    "id": pid,
                    "method": "daedal_dream",
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": err_trace,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[shard {args.shard_id}] [{idx+1}/{len(shard)}] {pid} FAILED: {exc}", flush=True)
    print(f"[shard {args.shard_id}] DONE -> {out_path}", flush=True)


def merge_and_score(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    prompts_path = Path(args.prompts)
    prompts = {r["id"]: r for r in (json.loads(l) for l in prompts_path.read_text().splitlines() if l.strip())}

    merged: List[Dict] = []
    for sid in range(args.num_shards):
        shard_path = shard_output_path(out_dir, sid, args.num_shards)
        if not shard_path.exists():
            print(f"WARN missing {shard_path}", flush=True)
            continue
        for line in shard_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                merged.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARN corrupt line in {shard_path}: {e}; skipping", flush=True)
    merged.sort(key=lambda r: r["id"])
    merged_path = merged_output_path(out_dir)
    merged_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n", encoding="utf-8")
    print(f"merged {len(merged)} rows -> {merged_path}", flush=True)

    n_pass = n_total = n_error = 0
    extracted_rows: List[Dict] = []
    scored_rows: List[Dict] = []
    for row in merged:
        pid = row["id"]
        if pid not in prompts:
            continue
        meta = prompts[pid]
        raw = meta["raw_prompt"]
        entry = meta["entry_point"]
        if "error" in row:
            n_total += 1
            n_error += 1
            scored_rows.append({"id": pid, "passed": False, "result": f"GEN_ERROR: {row['error']}"})
            continue
        ans = row.get("answer_text", "") or ""
        body = ans.split("```python\n", 1)[-1].split("```")[0]
        # Same re-indent as run_carve_dream.py — Dream prompt opens code fence with the function header.
        lines = body.split("\n")
        for _idx, _line in enumerate(lines):
            if _line.strip():
                if not _line[:1].isspace():
                    lines[_idx] = "    " + _line
                break
        body_reindented = "\n".join(lines)
        candidate = raw + "\n" + body_reindented
        completion = sanitize(candidate, entry)
        extracted_rows.append({"id": pid, "completion": completion})
        problem = {
            "task_id": meta["task_id"],
            "prompt": "",
            "test": meta["test"],
            "entry_point": entry,
        }
        result = check_correctness(problem, completion, timeout=TIMEOUT_SECONDS, completion_id=0)
        passed = bool(result.get("passed", False))
        if passed:
            n_pass += 1
        n_total += 1
        scored_rows.append({"id": pid, "task_id": meta["task_id"], "passed": passed, "result": result.get("result", "")})

    (out_dir / "extracted.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in extracted_rows) + "\n", encoding="utf-8")
    (out_dir / "scored.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scored_rows) + "\n", encoding="utf-8")
    pass_at_1 = (n_pass / n_total) if n_total else 0.0
    summary = {
        "config": out_dir.name,
        "n_total": n_total,
        "n_passed": n_pass,
        "n_error": n_error,
        "pass_at_1": pass_at_1,
    }
    summary["perf"] = aggregate_perf_metrics(merged, n_scored=n_total)
    attach_summary_metadata(summary, out_dir)
    (out_dir / "summary.md").write_text(
        f"# {out_dir.name} — DAEDAL Dream HumanEval Pass@1\n\n"
        f"- problems: **{n_total}**\n"
        f"- passed: **{n_pass}**\n"
        f"- gen errors: {n_error}\n"
        f"- **Pass@1: {pass_at_1*100:.2f}%**\n",
        encoding="utf-8",
    )
    _run_cfg = read_run_config(out_dir)
    with (out_dir / "summary.md").open("a", encoding="utf-8") as _f_md:
        _f_md.write("\n" + render_config_md(_run_cfg))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()

    if args.merge:
        merge_and_score(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
