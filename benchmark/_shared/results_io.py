"""Shared output-layout helpers for benchmark runners.

Layout convention (post 2026-05 reorganization):

    {out_dir}/
        outputs.jsonl              ← merged per-prompt outputs (vanilla or CARVE)
        extracted.jsonl            ← extracted completions (post-merge)
        scored.jsonl               ← per-prompt pass/fail (post-merge)
        summary.json               ← aggregate metrics + full run_config + timestamps
        summary.md                 ← human-readable summary, including config block
        run_config.json            ← STATIC-SCHEMA config + start timestamp (every shard
                                     writes the same content modulo shard_id; last wins)
        shards/
            outputs.shard{i}of{N}.jsonl   ← per-shard outputs
            shard_{i}.log                 ← per-shard logs (written by sbatch launcher)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# ─── canonical run-config schema ────────────────────────────────────────────
# Every run_config.json has exactly these fields, in this order. Fields that
# don't apply to a particular runner are explicitly null (e.g. CARVE knobs in
# a vanilla run, or `max_new_tokens` in a CARVE run that uses `max_len`).
# This makes summaries trivially comparable across runners and across time.
CANONICAL_FIELDS = [
    # identity
    "task", "method", "num_shots",
    # paths
    "prompts", "output_dir",
    # sharding
    "num_shards", "shard_id_writer", "device",
    # sampling
    "seed", "temperature", "top_p", "alg", "alg_temp", "batch_size",
    # generation budget (vanilla uses max_new_tokens, CARVE uses max_len)
    "steps", "max_new_tokens", "max_len",
    # CARVE — prefill probe
    "L0", "l0_floor", "eos_thr", "disable_prefill",
    # CARVE — JS gate / expansion
    "js_threshold", "insert_k", "carve_interval", "disable_expand",
    # CARVE — schedule + EOS handling
    "legacy_schedule", "eos_crop_mode",
    # CARVE v2 — insertion-branch gate
    "insertion_mode", "frontier_conf_thr",
    # CARVE LLaDA
    "reveal_strategy", "high_conf_threshold", "initial_gen_length", "max_gen_length", "block_length",
    # DAEDAL
    "expansion_factor", "low_conf_threshold", "eos_confidence_threshold",
    "expand_eos_confidence_threshold", "eos_check_tokens", "max_iterations",
    # Vanilla LLaDA
    "gen_length", "cfg_scale", "remasking", "logits_eos_inf", "confidence_eos_eot_inf",
    # Model selection
    "model_path",
    # misc
    "limit", "num_prompts",
]


def shards_dir(out_dir) -> Path:
    """Return out_dir/shards (creates it if missing)."""
    d = Path(out_dir) / "shards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def shard_output_path(out_dir, shard_id: int, num_shards: int) -> Path:
    return shards_dir(out_dir) / f"outputs.shard{shard_id}of{num_shards}.jsonl"


def merged_output_path(out_dir) -> Path:
    return Path(out_dir) / "outputs.jsonl"


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    return v


def build_canonical_config(
    args: argparse.Namespace,
    *,
    task: str,
    method: str,
    num_shots: int,
) -> Dict:
    """Build the static-schema config dict for one run.

    Every field listed in CANONICAL_FIELDS is present, with the value from
    `args` if argparse has it, else None. Identity fields (task / method /
    num_shots) come from the caller — they're not argparse flags.
    """
    cfg: Dict = {k: None for k in CANONICAL_FIELDS}
    cfg["task"] = task
    cfg["method"] = method
    cfg["num_shots"] = num_shots
    src = vars(args)
    for k in CANONICAL_FIELDS:
        if k in src:
            cfg[k] = _jsonable(src[k])
    cfg["shard_id_writer"] = getattr(args, "shard_id", None)
    return cfg


def write_run_config(
    out_dir,
    args: argparse.Namespace,
    *,
    task: str,
    method: str,
    num_shots: int,
    recipe=None,
) -> None:
    """Each shard writes this at startup. Content is identical across shards
    (modulo shard_id_writer), so race order doesn't matter.

    `recipe` is the resolved paper_config.Config for this cell. It is recorded
    verbatim, so every run carries the exact decoding settings it used even
    though those are no longer command-line flags."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "config": build_canonical_config(args, task=task, method=method, num_shots=num_shots),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "raw_args": {k: _jsonable(v) for k, v in vars(args).items()},
    }
    if recipe is not None:
        cfg["recipe"] = {k: _jsonable(v) for k, v in asdict(recipe).items()}
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def read_run_config(out_dir) -> Dict:
    p = Path(out_dir) / "run_config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def render_config_md(cfg: Dict) -> str:
    """Render the (canonical) run config as a markdown block."""
    static = cfg.get("config", {}) or {}
    started = cfg.get("started_at", "?")
    lines = ["## Run configuration", "", f"- `started_at`: `{started}`"]
    for k in CANONICAL_FIELDS:
        v = static.get(k)
        v_str = "`null`" if v is None else f"`{v}`"
        lines.append(f"- `{k}`: {v_str}")
    return "\n".join(lines) + "\n"


def attach_summary_metadata(summary: Dict, out_dir) -> Dict:
    """Adds run_config (from disk) + merged_at timestamp into the summary dict."""
    summary["merged_at"] = datetime.now().isoformat(timespec="seconds")
    summary["run_config"] = read_run_config(out_dir)
    return summary


def aggregate_perf_metrics(merged_rows, *, n_scored: int) -> Dict:
    """Aggregate FLOPs / forward-count / wall-time from per-row JSONL.

    Conventions (held by every runner today):
      - `total_flops` is **per-sample** (vanilla divides batch FLOPs by B; CARVE is B=1)
      - `elapsed_seconds` is **per-sample** (same B-division for vanilla)
      - `fwd_count` is the number of model forwards used to produce that one sample

    Rows with an `error` key are skipped (they have no perf data).

    Returned keys land verbatim under `summary["perf"]`:
      - total_flops             : sum across all successful rows
      - mean_flops_per_sample   : sum / n_scored (uses n_scored, not n_success, so a run with errors is penalised honestly)
      - total_fwds              : sum
      - mean_fwds_per_sample    : sum / n_scored
      - total_compute_seconds   : sum of per-sample elapsed (= what the run would have taken if serialised, NOT true wall-clock since shards run in parallel)
      - mean_seconds_per_sample : sum / n_scored
      - n_with_perf             : how many rows actually carried perf data
    """
    n_scored = max(1, int(n_scored))
    flops, fwds, wall = [], [], []
    for r in merged_rows:
        if "error" in r:
            continue
        flops.append(int(r.get("total_flops") or 0))
        fwds.append(int(r.get("fwd_count") or 0))
        wall.append(float(r.get("elapsed_seconds") or 0.0))
    s_flops, s_fwds, s_wall = sum(flops), sum(fwds), sum(wall)
    n_with_perf = sum(1 for x in flops if x > 0)
    return {
        "total_flops": s_flops,
        "mean_flops_per_sample": s_flops / n_scored,
        "total_fwds": s_fwds,
        "mean_fwds_per_sample": s_fwds / n_scored,
        "total_compute_seconds": round(s_wall, 3),
        "mean_seconds_per_sample": round(s_wall / n_scored, 3),
        "n_with_perf": n_with_perf,
    }
