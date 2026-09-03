"""MATH500 driver for CARVE v2 (insertion-branch gate, commit-from-chosen-branch).

Mirrors `benchmark/gsm8k/run_carve_dream.py` for MATH-500 (4-shot Minerva, max_len=512).

Differences vs gsm8k v2 runner:
  - prompt builder + scoring are MATH500's (extract_boxed + grade_math)
  - num_shots=4, default max_len/steps=512
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

from scripts.carve_dream import run_carve_dream  # noqa: E402
from scripts.dream_backend import load_model_and_tokenizer  # noqa: E402
from benchmark._shared.math_grade import extract_boxed, grade_math  # noqa: E402
from benchmark.math500.run_vanilla_dream import (  # noqa: E402
    _gold_for,
    _question_for,
    build_input_ids,
)
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
    JS_THRESHOLD,
    INSERT_K,
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "carve", "dream", "math500"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"


def run_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts)
    all_prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    shard = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] device={args.device} n={len(shard)}", flush=True)

    print(f"[shard {args.shard_id}] loading model on {args.device}", flush=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, args.device)
    print(
        f"[shard {args.shard_id}] L0={CFG.l0} insertion_mode={"mid_uncert"} "
        f"frontier_conf_thr={0.5} k={INSERT_K} interval={CFG.carve_interval} "
        f"crop={"stop"} js_threshold={JS_THRESHOLD}",
        flush=True,
    )

    write_run_config(out_dir, args, task="math500", method="carve_dream", num_shots=4, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(shard):
            pid = row["id"]
            q = _question_for(row)
            try:
                prompt_ids = build_input_ids(tokenizer, q)
                t0 = time.time()
                rec = run_carve_dream(
                    model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
                    seed=CFG.seed, L0=CFG.l0, max_len=CFG.lmax, steps=CFG.steps,
                    js_threshold=JS_THRESHOLD, insert_k=INSERT_K,
                    carve_interval=CFG.carve_interval,
                    mid_window=CFG.mid_window,
                    temperature=CFG.temperature, top_p=0.9,
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
                    "method": "carve_dream",
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

    n_pass = 0
    n_total = 0
    n_error = 0
    n_parseable = 0
    extracted_rows: List[Dict] = []
    scored_rows: List[Dict] = []
    n_expand_attempts: List[int] = []
    n_expand_accepted: List[int] = []
    inserts_used_vals: List[int] = []
    for row in merged:
        pid = row["id"]
        if pid not in prompts:
            continue
        meta = prompts[pid]
        gold = _gold_for(meta)
        if "error" in row:
            n_total += 1
            n_error += 1
            extracted_rows.append({"id": pid, "predicted": None})
            scored_rows.append({"id": pid, "passed": False, "predicted": None, "gold": gold,
                                "result": f"GEN_ERROR: {row['error']}",
                                "subject": meta.get("subject"), "level": meta.get("level")})
            continue
        if "n_expand_attempts" in row:
            n_expand_attempts.append(int(row["n_expand_attempts"]))
            n_expand_accepted.append(int(row["n_expand_accepted"]))
            inserts_used_vals.append(int(row.get("inserts_used", 0)))
        ans = row.get("answer_text", "") or ""
        boxed = extract_boxed(ans)
        if boxed is not None:
            n_parseable += 1
        passed, reason = grade_math(boxed, gold)
        if passed:
            n_pass += 1
        n_total += 1
        extracted_rows.append({"id": pid, "predicted": boxed})
        scored_rows.append({
            "id": pid, "passed": passed, "predicted": boxed, "gold": gold,
            "result": reason, "subject": meta.get("subject"), "level": meta.get("level"),
        })

    (out_dir / "extracted.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in extracted_rows) + "\n", encoding="utf-8")
    (out_dir / "scored.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scored_rows) + "\n", encoding="utf-8")
    pass_at_1 = (n_pass / n_total) if n_total else 0.0
    parseable_pct = (n_parseable / n_total) if n_total else 0.0
    summary = {
        "config": out_dir.name,
        "n_total": n_total,
        "n_passed": n_pass,
        "n_error": n_error,
        "n_parseable": n_parseable,
        "parseable_pct": parseable_pct,
        "pass_at_1": pass_at_1,
    }
    if n_expand_attempts:
        summary["mean_expand_attempts"] = sum(n_expand_attempts) / len(n_expand_attempts)
        summary["mean_expand_accepted"] = sum(n_expand_accepted) / len(n_expand_attempts)
        summary["mean_accept_rate"] = (
            sum(n_expand_accepted) / max(1, sum(n_expand_attempts))
        )
        summary["mean_inserts_used"] = sum(inserts_used_vals) / len(inserts_used_vals)
    summary["perf"] = aggregate_perf_metrics(merged, n_scored=n_total)
    attach_summary_metadata(summary, out_dir)
    (out_dir / "summary.md").write_text(
        f"# {out_dir.name} — CARVE v2 MATH500 accuracy\n\n"
        f"- problems: **{n_total}**\n"
        f"- passed: **{n_pass}**\n"
        f"- gen errors: {n_error}\n"
        f"- parseable predictions (had \\boxed{{}}): {n_parseable}/{n_total} ({parseable_pct*100:.1f}%)\n"
        f"- **Accuracy: {pass_at_1*100:.2f}%**\n"
        + (
            f"- mean expand attempts/sample: {summary.get('mean_expand_attempts'):.1f}\n"
            f"- mean expansions accepted/sample: {summary.get('mean_expand_accepted'):.1f}\n"
            f"- mean accept rate: {summary.get('mean_accept_rate', 0)*100:.1f}%\n"
            f"- mean inserts used/sample: {summary.get('mean_inserts_used', 0):.1f}\n"
            if "mean_expand_attempts" in summary else ""
        ),
        encoding="utf-8",
    )
    _run_cfg_for_md = read_run_config(out_dir)
    with (out_dir / "summary.md").open("a", encoding="utf-8") as _f_md:
        _f_md.write("\n" + render_config_md(_run_cfg_for_md))
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

    if not args.merge and CFG.l0 is None:
        p.error("--L0 is required for CARVE v2 (no prefill probe)")

    if args.merge:
        merge_and_score(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
