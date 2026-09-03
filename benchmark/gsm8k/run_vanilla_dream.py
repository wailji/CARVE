"""GSM8K driver — faithful reproduction of the Dream paper recipe.

Mirror of `benchmark/humaneval/run_vanilla_dream.py`, adapted for GSM8K
(`HKUNLP/Dream:eval_instruct/eval.sh` + `lm_eval/tasks/gsm8k/gsm8k_cot.yaml`,
num_fewshot=0 → bare question via chat template; max_new_tokens=256,
diffusion_steps=256, temperature=0.1, top_p=0.9, alg=entropy).

There is NO gen_prefix for GSM8K (unlike HumanEval — the model just opens
the assistant turn and writes free-form CoT).

Scoring: extract the LAST number (`[-+]?\\d*\\.?\\d+`) from answer_text
and compare to `gold_number` with absolute tolerance 1e-4.

Usage (per shard):
  python benchmark/gsm8k/run_vanilla_dream.py \\
    --prompts outputs/benchmarks/gsm8k/prompts_full.jsonl \\
    --output-dir outputs/benchmarks/gsm8k/paper_repro/vanilla_L256_t0.1/ \\
    --shard-id 0 --num-shards 8 --device cuda:0

Merge step:
  python benchmark/gsm8k/run_vanilla_dream.py --merge \\
    --output-dir outputs/benchmarks/gsm8k/paper_repro/vanilla_L256_t0.1/ \\
    --prompts outputs/benchmarks/gsm8k/prompts_full.jsonl \\
    --num-shards 8
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.dream_backend import load_model_and_tokenizer, set_seed, _flop_ctx, _read_flops  # noqa: E402
from benchmark._shared.results_io import (  # noqa: E402
    shard_output_path,
    merged_output_path,
    write_run_config,
    attach_summary_metadata,
    aggregate_perf_metrics,
    render_config_md,
    read_run_config,
)
from benchmark._shared.math_grade import grade_gsm8k  # noqa: E402
from benchmark._shared.paper_config import (  # noqa: E402
    resolve,
    DREAM_ALG,
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "vanilla", "dream", "gsm8k"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"


def build_input_ids(tokenizer, prompt_text: str) -> torch.LongTensor:
    """GSM8K: 8-shot CoT prompt wrapped by chat template, no gen_prefix.

    Expects the FULL pre-formatted prompt (8 shots + test query ending with
    `Q: ... \\n A:`). Built by build_dataset.py per lm-eval gsm8k_cot.
    """
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def _question_for(row: Dict) -> str:
    """Return the full pre-formatted prompt (8-shot + test query)."""
    return row.get("prompt") or row.get("raw_prompt") or row.get("question", "")


def _gold_for(row: Dict):
    if "gold_number" in row:
        return row["gold_number"]
    return row.get("gold_answer")


# ---------------------------------------------------------------------------
# Per-shard generation
# ---------------------------------------------------------------------------
def run_shard(args: argparse.Namespace) -> None:
    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts)
    all_prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    shard = [p for i, p in enumerate(all_prompts) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] device={device} n={len(shard)}", flush=True)

    print(f"[shard {args.shard_id}] loading model on {device}", flush=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, device)
    set_seed(1337)

    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.mask_token_id = tokenizer.mask_token_id
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = tokenizer.eos_token_id
    gen_cfg.alg = "entropy"
    gen_cfg.alg_temp = None
    gen_cfg.temperature = CFG.temperature
    gen_cfg.top_p = 0.9
    gen_cfg.steps = CFG.steps
    gen_cfg.max_new_tokens = CFG.lmax

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    bsz = max(1, int(1))

    write_run_config(out_dir, args, task="gsm8k", method="vanilla", num_shots=8, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for batch_start in range(0, len(shard), bsz):
            batch = shard[batch_start : batch_start + bsz]
            pids = [r["id"] for r in batch]
            qs = [_question_for(r) for r in batch]
            try:
                ids_list = [build_input_ids(tokenizer, q) for q in qs]
                lens = [t.shape[1] for t in ids_list]
                Lmax = max(lens)
                # LEFT-pad with pad_token (= eos). attn=0 on pads, 1 on real prompt.
                B = len(ids_list)
                input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
                attn_mask = torch.zeros((B, Lmax), dtype=torch.long)
                for b, t in enumerate(ids_list):
                    L = t.shape[1]
                    input_ids[b, Lmax - L : Lmax] = t[0]
                    attn_mask[b, Lmax - L : Lmax] = 1
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                t0 = time.time()
                fc = _flop_ctx()
                with fc, torch.inference_mode():
                    output_ids = model.diffusion_generate(
                        inputs=input_ids,
                        attention_mask=attn_mask,
                        generation_config=gen_cfg,
                    )
                elapsed = time.time() - t0
                total_flops_batch = _read_flops(fc)
                per_sample_flops = int(total_flops_batch / max(1, B))
                # Canvas starts at position Lmax for every sample.
                for b, pid in enumerate(pids):
                    new_ids = output_ids[b, Lmax:].tolist()
                    if eos_id in new_ids:
                        new_ids = new_ids[: new_ids.index(eos_id)]
                    answer_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    rec = {
                        "id": pid,
                        "mode": "vanilla_paper_repro",
                        "seed": 1337,
                        "steps": CFG.steps,
                        "max_new_tokens": CFG.lmax,
                        "temperature": CFG.temperature,
                        "top_p": 0.9,
                        "alg": "entropy",
                        "answer_text": answer_text,
                        "answer_token_length": len(new_ids),
                        "prompt_token_length": lens[b],
                        "batch_size": B,
                        "elapsed_seconds": elapsed / B,
                        "elapsed_seconds_batch": elapsed,
                        "total_flops": per_sample_flops,
                        "total_flops_batch": total_flops_batch,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[shard {args.shard_id}] batch [{batch_start}-{batch_start+B-1}] done in {elapsed:.1f}s ({elapsed/B:.1f}s/prompt)", flush=True)
                continue
            except Exception as exc:
                err_trace = traceback.format_exc(limit=5)
                for pid in pids:
                    rec = {
                        "id": pid,
                        "mode": "vanilla_paper_repro",
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": err_trace,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[shard {args.shard_id}] batch [{batch_start}-{batch_start+len(pids)-1}] FAILED: {exc}", flush=True)
    print(f"[shard {args.shard_id}] DONE -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Merge + score
# ---------------------------------------------------------------------------
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
            if line.strip():
                merged.append(json.loads(line))
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
        passed, reason, pred = grade_gsm8k(ans, gold)
        if pred is not None:
            n_parseable += 1
        if passed:
            n_pass += 1
        n_total += 1
        extracted_rows.append({"id": pid, "predicted": pred})
        scored_rows.append({
            "id": pid,
            "passed": passed,
            "predicted": pred,
            "gold": gold,
            "result": reason,
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
    summary["perf"] = aggregate_perf_metrics(merged, n_scored=n_total)
    attach_summary_metadata(summary, out_dir)
    (out_dir / "summary.md").write_text(
        f"# {out_dir.name} — Paper-repro GSM8K Pass@1\n\n"
        f"- problems: **{n_total}**\n"
        f"- passed: **{n_pass}**\n"
        f"- gen errors: {n_error}\n"
        f"- parseable predictions: {n_parseable}/{n_total} ({parseable_pct*100:.1f}%)\n"
        f"- **Pass@1: {pass_at_1*100:.2f}%**\n",
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

    if args.merge:
        merge_and_score(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
