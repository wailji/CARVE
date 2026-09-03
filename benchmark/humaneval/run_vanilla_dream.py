"""HumanEval driver — faithful reproduction of the Dream paper recipe.

Why this exists: our `vanilla_L128_entropy` Pass@1 (31.71%) is far below the
~55% reported for Dream-v0-Instruct-7B in the paper. Inspecting the official
eval (HKUNLP/Dream:eval_instruct/eval.sh + lm_eval/tasks/humaneval/*.yaml)
revealed three deltas that together explain the gap:

  1. max_new_tokens=768, diffusion_steps=768 (we used 128/128).
  2. temperature=0.1, top_p=0.9 (we used greedy temperature=0.0).
  3. Prompt format uses doc_to_text + gen_prefix:
        user:   "Write a solution to the following problem and make sure that
                 it passes the tests:\n```{prompt}"
        asst*:  "Here is the completed function:\n```python\n{prompt}\n"
     The asterisked prefix is *prepended* to the assistant turn (i.e. the
     model continues from inside an open code fence with the function header
     already visible).
  4. Extraction uses an AST-based `sanitize` (longest-valid-AST + reachable
     defs from entrypoint) instead of a simple regex fence match.

This script reproduces (1)-(4) exactly, sharded 8-way across GPUs.

Usage (per shard):
  python benchmark/humaneval/run_vanilla_dream.py \\
    --prompts outputs/benchmarks/humaneval/prompts_full.jsonl \\
    --output-dir outputs/benchmarks/humaneval/paper_repro/vanilla_L768_t0.1/ \\
    --shard-id 0 --num-shards 8 --device cuda:0

Merge step:
  python benchmark/humaneval/run_vanilla_dream.py --merge \\
    --output-dir outputs/benchmarks/humaneval/paper_repro/vanilla_L768_t0.1/ \\
    --prompts outputs/benchmarks/humaneval/prompts_full.jsonl \\
    --num-shards 8
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
from human_eval.execution import check_correctness  # noqa: E402
from benchmark._shared.paper_config import (  # noqa: E402
    resolve,
    DREAM_ALG,
)

# This runner is one cell of Table 1; its recipe comes from paper_config.
METHOD, MODEL, TASK = "vanilla", "dream", "humaneval"
CFG = resolve(METHOD, MODEL, TASK)

MODEL_PATH = "Dream-org/Dream-v0-Instruct-7B"
TIMEOUT_SECONDS = 5.0

DOC_TO_TEXT = (
    "Write a solution to the following problem and make sure that it passes "
    "the tests:\n```{prompt}"
)
GEN_PREFIX = "Here is the completed function:\n```python\n{prompt}\n"


# ---------------------------------------------------------------------------
# AST-based extraction (verbatim from HKUNLP/Dream eval_instruct/.../sanitize_utils.py)
# ---------------------------------------------------------------------------
def _refine_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def _syntax_check(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False


def _extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 100:
        lines = lines[:100]
    max_valid_lines = 0
    max_valid_snippet = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            snippet = "\n".join(lines[i:j + 1])
            if _syntax_check(snippet):
                valid_line_count = sum(1 for line in lines[i:j + 1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = snippet
    return max_valid_snippet


def _get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps: Dict[str, Set[str]] = {}
    for name, node in nodes:
        deps: Set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    deps.add(child.attr)
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps


def _get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited: Set[str] = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def _get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def _has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))


def sanitize(text: str, entrypoint: Optional[str] = None) -> str:
    text = _refine_text(text)
    code = _extract_longest_valid_code(text)
    if not code:
        return ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    definitions: Dict[str, Tuple[str, ast.AST]] = {}
    imports: List[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = ("class", node)
        elif isinstance(node, ast.FunctionDef):
            if _has_return_statement(node):
                definitions[node.name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = _get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)
    if entrypoint:
        name2deps = _get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = _get_function_dependency(entrypoint, name2deps)
    else:
        reachable = set(definitions.keys())
    out: List[str] = [ast.unparse(n) for n in imports]
    for name, (_, node) in definitions.items():
        if not entrypoint or name in reachable:
            out.append(ast.unparse(node))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Prompt building (matches doc_to_text + gen_prefix from humaneval_instruct.yaml)
# ---------------------------------------------------------------------------
def build_input_ids(tokenizer, raw_prompt: str) -> torch.LongTensor:
    user_text = DOC_TO_TEXT.format(prompt=raw_prompt)
    messages = [{"role": "user", "content": user_text}]
    base_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prefix_text = GEN_PREFIX.format(prompt=raw_prompt)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids
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

    write_run_config(out_dir, args, task="humaneval", method="vanilla", num_shots=0, recipe=CFG)
    out_path = shard_output_path(out_dir, args.shard_id, args.num_shards)
    with out_path.open("w", encoding="utf-8") as f:
        for batch_start in range(0, len(shard), bsz):
            batch = shard[batch_start : batch_start + bsz]
            pids = [r["id"] for r in batch]
            raws = [r["raw_prompt"] for r in batch]
            try:
                ids_list = [build_input_ids(tokenizer, r) for r in raws]
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

    # Score using paper's sanitize.
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
        raw = meta["raw_prompt"]
        entry = meta["entry_point"]
        if "error" in row:
            n_total += 1
            n_error += 1
            scored_rows.append({"id": pid, "passed": False, "result": f"GEN_ERROR: {row['error']}"})
            continue
        ans = row.get("answer_text", "") or ""
        # paper: r.split('```python\n', 1)[-1].split('```')[0]
        body = ans.split("```python\n", 1)[-1].split("```")[0]
        # The full candidate for sanitize is doc["prompt"] + "\n" + body
        candidate_src = raw + "\n" + body
        completion = sanitize(candidate_src, entry)
        extracted_rows.append({"id": pid, "completion": completion})
        problem = {
            "task_id": meta["task_id"],
            "prompt": "",          # full standalone module is in `completion`
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
        f"# {out_dir.name} — Paper-repro Pass@1\n\n"
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
