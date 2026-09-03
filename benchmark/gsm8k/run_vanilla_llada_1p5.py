"""GSM8K driver — vanilla LLaDA-1.5.

DAEDAL prompt (math expert + <reasoning>/<answer>/\\boxed{} format) wrapped in
LLaDA-1.5 plain chat template (NO <reasoning> prefix). 0-shot.

Defaults per LLaDA EVAL.md (LLaDA 1.5 section) for GSM8K:
    gen_length=CFG.lmax, block_length=16, confidence_eos_eot_inf=True,
    logits_eos_inf=False.
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

from scripts.llada_1p5_backend import load_model_and_tokenizer, run_vanilla_llada  # noqa: E402
from benchmark._shared.llada_1p5_prompts import build_llada_1p5_prompt_ids, gsm8k_doc_to_text  # noqa: E402
from benchmark._shared.math_grade import grade_gsm8k_boxed  # noqa: E402
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
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "vanilla", "llada-1.5", "gsm8k"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "GSAI-ML/LLaDA-1.5"


def _question_for(row: Dict) -> str:
    return row.get("question") or row.get("raw_prompt") or row.get("prompt", "")


def _gold_for(row: Dict):
    if "gold_number" in row:
        return row["gold_number"]
    return row.get("gold_answer")


def run_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts)
    all_prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    shard = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] device={args.device} n={len(shard)}", flush=True)

    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, args.device)
    print(
        f"[shard {args.shard_id}] gen_len={CFG.lmax} steps={CFG.steps} "
        f"block_len={CFG.block_length} conf_eos_eot_inf={True}",
        flush=True,
    )

    write_run_config(out_dir, args, task="gsm8k", method="vanilla_llada_1p5", num_shots=0, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(shard):
            pid = row["id"]
            question = _question_for(row)
            try:
                user_text = gsm8k_doc_to_text({"question": question})
                prompt_ids = build_llada_1p5_prompt_ids(tokenizer, user_text)
                t0 = time.time()
                rec = run_vanilla_llada(
                    model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
                    seed=CFG.seed,
                    gen_length=CFG.lmax,
                    steps=CFG.steps,
                    block_length=CFG.block_length,
                    cfg_scale=0.0,
                    temperature=CFG.temperature,
                    remasking="low_confidence",
                    logits_eos_inf=CFG.logits_eos_inf,
                    confidence_eos_eot_inf=CFG.confidence_eos_eot_inf,
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
                    "method": "vanilla_llada_1p5",
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

    n_pass = n_total = n_error = n_parseable = 0
    extracted_rows: List[Dict] = []
    scored_rows: List[Dict] = []
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
            scored_rows.append({"id": pid, "passed": False, "predicted": None, "gold": gold, "result": f"GEN_ERROR: {row['error']}"})
            continue
        ans = row.get("answer_text", "") or ""
        passed, reason, pred = grade_gsm8k_boxed(ans, gold)
        if pred is not None:
            n_parseable += 1
        if passed:
            n_pass += 1
        n_total += 1
        extracted_rows.append({"id": pid, "predicted": pred})
        scored_rows.append({"id": pid, "passed": passed, "predicted": pred, "gold": gold, "result": reason})

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
    summary["perf"] = aggregate_perf_metrics(merged, n_scored=n_total)
    attach_summary_metadata(summary, out_dir)
    (out_dir / "summary.md").write_text(
        f"# {out_dir.name} — Vanilla LLaDA-1.5 GSM8K Pass@1\n\n"
        f"- problems: **{n_total}**\n"
        f"- passed: **{n_pass}**\n"
        f"- gen errors: {n_error}\n"
        f"- parseable predictions: {n_parseable}/{n_total} ({parseable_pct*100:.1f}%)\n"
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
