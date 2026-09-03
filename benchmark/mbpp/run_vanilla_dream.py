"""MBPP-sanitized driver — faithful reproduction of the Dream paper recipe.

Mirrors `benchmark/humaneval/run_vanilla_dream.py` but with the MBPP recipe
from HKUNLP/Dream:eval_instruct/lm_eval/tasks/mbpp/mbpp_instruct.yaml:

  - dataset: mbpp (we use config 'sanitized', the existing repo convention; the
    Dream paper uses 'full'. See benchmark/mbpp/build_dataset.py for rationale.)
  - doc_to_text:
        "You are an expert Python programmer, and here is your task: {text}
         Your code should pass these tests:\n\n{test_list[0..2]}"
  - gen_prefix:
        "Here is the completed function:\n```python\n"
    (The gen_prefix is APPENDED to the assistant turn — i.e. the model
     continues from inside an open ```python fence with NO function header
     pre-supplied. This differs from HumanEval, where the entry-point
     signature/docstring are prepended; MBPP has no signature in the prompt.)
  - generation_kwargs from eval.sh:
        max_new_tokens=CFG.lmax, diffusion_steps=1024, dtype=bfloat16,
        temperature=CFG.temperature, top_p=0.9, alg=entropy

Scoring uses our own AST-based sanitize (re-imported from
`benchmark.humaneval.run_vanilla_dream.sanitize`, which is verbatim from
HKUNLP/Dream:eval_instruct/.../sanitize_utils.py). The candidate program is
`extracted_code + "\n" + "\n".join(test_imports + test_list)` and is exec'd
in a sandboxed subprocess with a 5s timeout (mirrors
benchmark/_shared/sandbox.run_program). This matches the existing
outputs/benchmarks/mbpp/carve/_baseline_vanilla/scored.jsonl row format
({id, task_id, passed, result, extraction_method}).

Usage (per shard):
  python benchmark/mbpp/run_vanilla_dream.py \\
    --prompts outputs/benchmarks/mbpp/prompts_full.jsonl \\
    --output-dir outputs/benchmarks/mbpp/paper_repro/vanilla_L1024_t0.1/ \\
    --shard-id 0 --num-shards 8 --device cuda:0

Merge step:
  python benchmark/mbpp/run_vanilla_dream.py --merge \\
    --output-dir outputs/benchmarks/mbpp/paper_repro/vanilla_L1024_t0.1/ \\
    --prompts outputs/benchmarks/mbpp/prompts_full.jsonl \\
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
from benchmark._shared.sandbox import run_program  # noqa: E402
from benchmark.humaneval.run_vanilla_dream import sanitize  # noqa: E402
from benchmark.mbpp.score import build_program  # noqa: E402
from benchmark._shared.paper_config import (  # noqa: E402
    resolve,
    DREAM_ALG,
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "vanilla", "dream", "mbpp"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"
TIMEOUT_SECONDS = 5.0

# With the new build_dataset.py, `raw_prompt` already contains the full
# 4-shot prefix + test query (DOC_TO_TEXT applied per shot). So we feed it
# directly to the chat template without re-wrapping.
GEN_PREFIX = "Here is the completed function:\n```python\n"


# ---------------------------------------------------------------------------
# Prompt building (matches mbpp_instruct.yaml: chat-template + gen_prefix)
# ---------------------------------------------------------------------------
def build_input_ids(tokenizer, raw_prompt: str, test_list: List[str]) -> torch.LongTensor:
    # raw_prompt is already a fully-formatted 4-shot+test-query block.
    messages = [{"role": "user", "content": raw_prompt}]
    base_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prefix_ids = tokenizer(GEN_PREFIX, return_tensors="pt", add_special_tokens=False).input_ids
    return torch.cat([base_ids, prefix_ids], dim=1)


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

    write_run_config(out_dir, args, task="mbpp", method="vanilla", num_shots=4, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for batch_start in range(0, len(shard), bsz):
            batch = shard[batch_start : batch_start + bsz]
            pids = [r["id"] for r in batch]
            raws = [r["raw_prompt"] for r in batch]
            tlists = [r["test_list"] for r in batch]
            try:
                ids_list = [build_input_ids(tokenizer, raws[k], tlists[k]) for k in range(len(batch))]
                lens = [t.shape[1] for t in ids_list]
                Lmax = max(lens)
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
                print(
                    f"[shard {args.shard_id}] batch [{batch_start}-{batch_start+B-1}] "
                    f"done in {elapsed:.1f}s ({elapsed/B:.1f}s/prompt)",
                    flush=True,
                )
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
                print(
                    f"[shard {args.shard_id}] batch [{batch_start}-{batch_start+len(pids)-1}] "
                    f"FAILED: {exc}",
                    flush=True,
                )
    print(f"[shard {args.shard_id}] DONE -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Merge + score
# ---------------------------------------------------------------------------
def merge_and_score(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    prompts_path = Path(args.prompts)
    prompts = {
        r["id"]: r
        for r in (json.loads(l) for l in prompts_path.read_text().splitlines() if l.strip())
    }

    merged: List[Dict] = []
    for sid in range(args.num_shards):
        shard_path = shard_output_path(out_dir, sid, args.num_shards)
        if not shard_path.exists():
            print(f"WARN missing {shard_path}", flush=True)
            continue
        for line in shard_path.read_text().splitlines():
            if line.strip():
                merged.append(json.loads(line))
    # Stable sort by task_id (numeric) when present, else by id.
    merged.sort(key=lambda r: prompts.get(r["id"], {}).get("task_id", 0))
    merged_path = merged_output_path(out_dir)
    merged_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
        encoding="utf-8",
    )
    print(f"merged {len(merged)} rows -> {merged_path}", flush=True)

    n_pass = 0
    n_total = 0
    n_error = 0
    scored_rows: List[Dict] = []
    extracted_rows: List[Dict] = []
    for row in merged:
        pid = row["id"]
        if pid not in prompts:
            continue
        meta = prompts[pid]
        entry = meta.get("entry_point") or None
        if "error" in row:
            n_total += 1
            n_error += 1
            scored_rows.append({
                "id": pid,
                "task_id": meta["task_id"],
                "passed": False,
                "result": f"GEN_ERROR: {row['error']}",
                "extraction_method": "error",
            })
            continue
        ans = row.get("answer_text", "") or ""
        # Strip Dream-Instruct stop marker [DONE] BEFORE the python-block split,
        # so we don't capture another problem the model continued into.
        if "[DONE]" in ans:
            ans = ans.split("[DONE]", 1)[0]
        # paper convention: r.split('```python\n', 1)[-1].split('```')[0]
        body = ans.split("```python\n", 1)[-1].split("```")[0]
        # MBPP has no signature prefix to prepend (unlike HumanEval). The body
        # is the full function definition; sanitize handles the AST extraction.
        completion = sanitize(body, entry)
        extracted_rows.append({
            "id": pid,
            "task_id": meta["task_id"],
            "completion": completion,
            "extraction_method": "ast_sanitize",
        })
        program = build_program(
            meta.get("test_imports", []) or [],
            completion,
            meta["test_list"],
        )
        result = run_program(program, timeout=TIMEOUT_SECONDS)
        passed = bool(result["passed"])
        if passed:
            n_pass += 1
        n_total += 1
        scored_rows.append({
            "id": pid,
            "task_id": meta["task_id"],
            "passed": passed,
            "result": str(result["result"]),
            "extraction_method": "ast_sanitize",
        })

    (out_dir / "extracted.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in extracted_rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "scored.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scored_rows) + "\n",
        encoding="utf-8",
    )
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
        f"# {out_dir.name} — MBPP-sanitized paper-repro Pass@1\n\n"
        f"- problems: **{n_total}**\n"
        f"- passed: **{n_pass}**\n"
        f"- gen errors: {n_error}\n"
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
