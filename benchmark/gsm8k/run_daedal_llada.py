"""GSM8K driver — DAEDAL on LLaDA-8B-Instruct (DAEDAL prompt + DAEDAL algorithm)."""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path
from typing import Dict, List


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.daedal_llada import run_daedal_llada  # noqa: E402
from scripts.llada_backend import load_model_and_tokenizer  # noqa: E402
from benchmark._shared.llada_prompts import build_llada_prompt_ids, gsm8k_doc_to_text  # noqa: E402
from benchmark._shared.math_grade import grade_gsm8k_boxed  # noqa: E402
from benchmark._shared.results_io import (  # noqa: E402
    shard_output_path, merged_output_path, write_run_config,
    attach_summary_metadata, aggregate_perf_metrics, render_config_md, read_run_config,
)
from benchmark._shared.paper_config import (  # noqa: E402
    resolve,
    DAEDAL,
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "daedal", "llada-8b", "gsm8k"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"


def _question_for(row: Dict) -> str:
    return row.get("question") or row.get("raw_prompt") or row.get("prompt", "")


def _gold_for(row: Dict):
    return row.get("gold_number", row.get("gold_answer"))


def run_shard(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    shard = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] device={args.device} n={len(shard)}", flush=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, args.device)
    print(f"[shard {args.shard_id}] L0={CFG.l0} max_len={CFG.lmax} block={CFG.block_length} ef={8}", flush=True)
    write_run_config(out_dir, args, task="gsm8k", method="daedal_llada", num_shots=0, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(shard):
            pid = row["id"]
            try:
                user_text = gsm8k_doc_to_text({"question": _question_for(row)})
                prompt_ids = build_llada_prompt_ids(tokenizer, user_text)
                t0 = time.time()
                rec = run_daedal_llada(
                    model=model, tokenizer=tokenizer, prompt_ids=prompt_ids, seed=1337,
                    initial_gen_length=CFG.l0, max_gen_length=CFG.lmax,
                    block_length=CFG.block_length, expansion_factor=8, low_conf_threshold=0.1,
                    eos_confidence_threshold=0.5,
                    expand_eos_confidence_threshold=0.9,
                    eos_check_tokens=32, cfg_scale=0.0,
                    temperature=CFG.temperature, max_iterations=CFG.steps,
                )
                rec["id"] = pid
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(f"[shard {args.shard_id}] [{idx+1}/{len(shard)}] {pid} done in {time.time()-t0:.1f}s", flush=True)
            except Exception as exc:
                rec = {"id": pid, "method": "daedal_llada",
                       "error": f"{type(exc).__name__}: {exc}",
                       "trace": traceback.format_exc(limit=5)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(f"[shard {args.shard_id}] [{idx+1}/{len(shard)}] {pid} FAILED: {exc}", flush=True)
    print(f"[shard {args.shard_id}] DONE -> {out_path}", flush=True)


def merge_and_score(args):
    out_dir = Path(args.output_dir)
    prompts = {r["id"]: r for r in (json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip())}
    merged: List[Dict] = []
    for sid in range(args.num_shards):
        p = shard_output_path(out_dir, sid, args.num_shards)
        if not p.exists(): print(f"WARN missing {p}", flush=True); continue
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            try: merged.append(json.loads(line))
            except json.JSONDecodeError as e: print(f"WARN corrupt {p}: {e}", flush=True)
    merged.sort(key=lambda r: r["id"])
    merged_output_path(out_dir).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n", encoding="utf-8")
    print(f"merged {len(merged)} rows", flush=True)
    n_pass = n_total = n_error = n_parseable = 0
    extracted_rows, scored_rows = [], []
    for row in merged:
        pid = row["id"]
        if pid not in prompts: continue
        gold = _gold_for(prompts[pid])
        if "error" in row:
            n_total += 1; n_error += 1
            extracted_rows.append({"id": pid, "predicted": None})
            scored_rows.append({"id": pid, "passed": False, "predicted": None, "gold": gold,
                                "result": f"GEN_ERROR: {row['error']}"}); continue
        ans = row.get("answer_text", "") or ""
        passed, reason, pred = grade_gsm8k_boxed(ans, gold)
        if pred is not None: n_parseable += 1
        if passed: n_pass += 1
        n_total += 1
        extracted_rows.append({"id": pid, "predicted": pred})
        scored_rows.append({"id": pid, "passed": passed, "predicted": pred, "gold": gold, "result": reason})
    (out_dir / "extracted.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in extracted_rows) + "\n", encoding="utf-8")
    (out_dir / "scored.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scored_rows) + "\n", encoding="utf-8")
    pass_at_1 = (n_pass / n_total) if n_total else 0.0
    summary = {"config": out_dir.name, "n_total": n_total, "n_passed": n_pass, "n_error": n_error,
               "n_parseable": n_parseable, "parseable_pct": n_parseable / max(1, n_total),
               "pass_at_1": pass_at_1}
    summary["perf"] = aggregate_perf_metrics(merged, n_scored=n_total)
    attach_summary_metadata(summary, out_dir)
    (out_dir / "summary.md").write_text(
        f"# {out_dir.name} — DAEDAL LLaDA GSM8K Pass@1\n\n"
        f"- problems: **{n_total}**\n- passed: **{n_pass}**\n- gen errors: {n_error}\n"
        f"- **Pass@1: {pass_at_1*100:.2f}%**\n", encoding="utf-8")
    with (out_dir / "summary.md").open("a", encoding="utf-8") as f:
        f.write("\n" + render_config_md(read_run_config(out_dir)))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    (merge_and_score if args.merge else run_shard)(args)


if __name__ == "__main__":
    main()
