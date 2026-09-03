"""MBPP driver — vanilla LLaDA-1.5 (DAEDAL prompt, 0-shot).

Defaults per LLaDA EVAL.md (LLaDA 1.5 section) for MBPP:
    gen_length=CFG.lmax, block_length=32, confidence_eos_eot_inf=True,
    logits_eos_inf=False.
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path
from typing import Dict, List


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.llada_1p5_backend import load_model_and_tokenizer, run_vanilla_llada  # noqa: E402
from benchmark._shared.llada_1p5_prompts import build_llada_1p5_prompt_ids, mbpp_doc_to_text  # noqa: E402
from benchmark._shared.sandbox import run_program  # noqa: E402
from benchmark.humaneval.run_vanilla_dream import sanitize  # noqa: E402
from benchmark.mbpp.score import build_program  # noqa: E402
from benchmark._shared.results_io import (  # noqa: E402
    shard_output_path, merged_output_path, write_run_config,
    attach_summary_metadata, aggregate_perf_metrics, render_config_md, read_run_config,
)
from benchmark._shared.paper_config import (  # noqa: E402
    resolve,
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "vanilla", "llada-1.5", "mbpp"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "GSAI-ML/LLaDA-1.5"
TIMEOUT_SECONDS = 5.0


def run_shard(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    shard = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] device={args.device} n={len(shard)}", flush=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, args.device)
    print(f"[shard {args.shard_id}] gen_len={CFG.lmax} steps={CFG.steps} block={CFG.block_length}", flush=True)
    write_run_config(out_dir, args, task="mbpp", method="vanilla_llada_1p5", num_shots=0, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(shard):
            pid = row["id"]
            try:
                user_text = mbpp_doc_to_text({"text": row.get("text", row.get("raw_prompt", "")),
                                              "test_list": row["test_list"]})
                prompt_ids = build_llada_1p5_prompt_ids(tokenizer, user_text)
                t0 = time.time()
                rec = run_vanilla_llada(
                    model=model, tokenizer=tokenizer, prompt_ids=prompt_ids, seed=1337,
                    gen_length=CFG.lmax, steps=CFG.steps, block_length=CFG.block_length,
                    cfg_scale=0.0, temperature=CFG.temperature, remasking="low_confidence",
                    logits_eos_inf=CFG.logits_eos_inf,
                    confidence_eos_eot_inf=CFG.confidence_eos_eot_inf,
                )
                rec["id"] = pid
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(f"[shard {args.shard_id}] [{idx+1}/{len(shard)}] {pid} done in {time.time()-t0:.1f}s", flush=True)
            except Exception as exc:
                rec = {"id": pid, "method": "vanilla_llada_1p5",
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
    merged.sort(key=lambda r: prompts.get(r["id"], {}).get("task_id", 0))
    merged_output_path(out_dir).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n", encoding="utf-8")
    print(f"merged {len(merged)} rows", flush=True)
    n_pass = n_total = n_error = 0
    extracted_rows, scored_rows = [], []
    for row in merged:
        pid = row["id"]
        if pid not in prompts: continue
        meta = prompts[pid]
        entry = meta.get("entry_point") or None
        if "error" in row:
            n_total += 1; n_error += 1
            scored_rows.append({"id": pid, "task_id": meta["task_id"], "passed": False,
                                "result": f"GEN_ERROR: {row['error']}", "extraction_method": "error"}); continue
        ans = row.get("answer_text", "") or ""
        if "[DONE]" in ans:
            ans = ans.split("[DONE]", 1)[0]
        body = ans.split("```python\n", 1)[-1].split("```")[0]
        completion = sanitize(body, entry)
        extracted_rows.append({"id": pid, "task_id": meta["task_id"],
                               "completion": completion, "extraction_method": "ast_sanitize"})
        program = build_program(meta.get("test_imports", []) or [], completion, meta["test_list"])
        result = run_program(program, timeout=TIMEOUT_SECONDS)
        passed = bool(result["passed"])
        if passed: n_pass += 1
        n_total += 1
        scored_rows.append({"id": pid, "task_id": meta["task_id"], "passed": passed,
                            "result": str(result["result"]), "extraction_method": "ast_sanitize"})
    (out_dir / "extracted.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in extracted_rows) + "\n", encoding="utf-8")
    (out_dir / "scored.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scored_rows) + "\n", encoding="utf-8")
    pass_at_1 = (n_pass / n_total) if n_total else 0.0
    summary = {"config": out_dir.name, "n_total": n_total, "n_passed": n_pass, "n_error": n_error,
               "pass_at_1": pass_at_1}
    summary["perf"] = aggregate_perf_metrics(merged, n_scored=n_total)
    attach_summary_metadata(summary, out_dir)
    (out_dir / "summary.md").write_text(
        f"# {out_dir.name} — Vanilla LLaDA-1.5 MBPP-sanitized Pass@1\n\n"
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
