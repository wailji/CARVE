#!/usr/bin/env python
"""Run the cells of Table 1 — as SLURM jobs, or directly on this machine.

Each cell is one (method, model, task) run. Its prompts are split into
nodes x 8 shards, one shard per GPU, and merged once they finish.

    # submit all 36 cells to SLURM (the default)
    python scripts/run.py --all

    # one cell
    python scripts/run.py --method carve --model dream --task humaneval

    # show the plan without submitting
    python scripts/run.py --all --dry-run

    # run here instead, on this machine's GPUs, no scheduler
    python scripts/run.py --all --local --gpus 8

Nodes per cell default to 4 for HumanEval (164 prompts) and 16 for MBPP,
MATH-500 and GSM8K; override with --nodes.

Cluster settings come from the environment, so nothing site-specific is baked
into the repo. Anything left empty is omitted from the job script and the
cluster default applies:

    CARVE_PARTITION  CARVE_QOS  CARVE_ACCOUNT  CARVE_EXCLUDE  CARVE_NODELIST
    CARVE_GRES       (default gpu:8)
    CARVE_CPUS       (default 64)
    CARVE_MEM        (default 512G)
    CARVE_TIME       (default 24:00:00)
    CARVE_CONDA_ENV  conda env to activate inside the job
    CARVE_PREAMBLE   extra shell line(s), e.g. 'module load rocm'

Results land in outputs/benchmarks/<task>/table1/<method>__<model>/summary.json.
Print the finished table with:  python scripts/table1.py

The decoding recipe is never passed on the command line: a cell is fully
determined by (method, model, task), and each runner reads its own settings
from benchmark/_shared/paper_config.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBDIR = REPO / "jobs"          # generated sbatch scripts (gitignored)
LOGDIR = JOBDIR / "logs"

# Seconds between launching consecutive shards on a node, so eight processes
# don't hit the HuggingFace cache and host RAM at the same instant.
STAGGER_SECONDS = 3

METHODS = ["vanilla", "daedal", "carve"]
MODELS = ["dream", "llada-8b", "llada-1.5"]
TASKS = ["humaneval", "mbpp", "math500", "gsm8k"]

# ── which runner implements each (method, model) ────────────────────────────
RUNNER = {
    ("vanilla", "dream"):     "run_vanilla_dream",
    ("vanilla", "llada-8b"):  "run_vanilla_llada",
    ("vanilla", "llada-1.5"): "run_vanilla_llada_1p5",
    ("daedal",  "dream"):     "run_daedal_dream",
    ("daedal",  "llada-8b"):  "run_daedal_llada",
    ("daedal",  "llada-1.5"): "run_daedal_llada_1p5",
    ("carve",   "dream"):     "run_carve_dream",
    ("carve",   "llada-8b"):  "run_carve_llada",
    ("carve",   "llada-1.5"): "run_carve_llada_1p5",
}

# Nodes per cell. HumanEval is 164 prompts; MBPP and MATH-500 are 500 each and
# GSM8K is 1319, so those get four times the width.
NODES = {"humaneval": 4, "mbpp": 16, "math500": 16, "gsm8k": 16}

GPUS_PER_NODE = 8

# ── cluster settings, all overridable from the environment ─────────────────
ENV = {
    "partition": os.environ.get("CARVE_PARTITION", ""),
    "qos":       os.environ.get("CARVE_QOS", ""),
    "account":   os.environ.get("CARVE_ACCOUNT", ""),
    "exclude":   os.environ.get("CARVE_EXCLUDE", ""),
    "nodelist":  os.environ.get("CARVE_NODELIST", ""),
    "gres":      os.environ.get("CARVE_GRES", f"gpu:{GPUS_PER_NODE}"),
    "cpus":      os.environ.get("CARVE_CPUS", "64"),
    "mem":       os.environ.get("CARVE_MEM", "512G"),
    "time":      os.environ.get("CARVE_TIME", "24:00:00"),
    "conda_env": os.environ.get("CARVE_CONDA_ENV", ""),
    "preamble":  os.environ.get("CARVE_PREAMBLE", ""),
}


def sbatch_lines(*, gres="", cpus="", mem="", time_="") -> str:
    """Optional #SBATCH directives. Empty settings are omitted entirely."""
    out = []
    for key in ("partition", "qos", "account", "exclude", "nodelist"):
        if ENV[key]:
            out.append(f"#SBATCH --{key}={ENV[key]}")
    if gres:
        out.append(f"#SBATCH --gres={gres}")
    if cpus:
        out.append(f"#SBATCH --cpus-per-task={cpus}")
    if mem:
        out.append(f"#SBATCH --mem={mem}")
    if time_:
        out.append(f"#SBATCH --time={time_}")
    return "\n".join(out)


def job_setup() -> str:
    """Environment activation inside the job. Empty -> whatever is on PATH."""
    parts = []
    if ENV["preamble"]:
        parts.append(ENV["preamble"])
    if ENV["conda_env"]:
        parts.append(
            'if [ -n "${CONDA_EXE:-}" ]; then\n'
            '  source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"\n'
            'elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then\n'
            '  source "$HOME/miniconda3/etc/profile.d/conda.sh"\n'
            'elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then\n'
            '  source "$HOME/anaconda3/etc/profile.d/conda.sh"\n'
            'fi\n'
            f'conda activate {ENV["conda_env"]}'
        )
    return "\n".join(parts)


GEN = """#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-{maxarray}
#SBATCH --chdir={repo}
#SBATCH --output={logdir}/{tag}_%A_%a.log
{sbatch}
set -uo pipefail
{setup}
export PYTHONNOUSERSITE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
NSHARDS={nshards}
NGPUS={gpus}
OUT={outdir}
mkdir -p "$OUT/shards"
NODE=$SLURM_ARRAY_TASK_ID
echo "=== {tag} node $NODE on $(hostname) $(date '+%F %T') nshards=$NSHARDS ==="
for i in $(seq 0 $((NGPUS - 1))); do
  SID=$((NODE * NGPUS + i))
  echo "NODE_HOST=$(hostname)" > "$OUT/shards/shard_$SID.log"
  export TMPDIR=${{TMPDIR:-/tmp}}/$USER/$SLURM_JOB_ID/$SID; mkdir -p "$TMPDIR"
  export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
  sleep $((i * {stagger}))
  CUDA_VISIBLE_DEVICES=$i HIP_VISIBLE_DEVICES=$i {python} {runner} \\
    --prompts {prompts} --output-dir "$OUT" \\
    --num-shards $NSHARDS --shard-id $SID --device cuda:0 \\
    >> "$OUT/shards/shard_$SID.log" 2>&1 &
done
wait
echo "=== node $NODE done $(date '+%F %T') ==="
"""

MERGE = """#!/bin/bash
#SBATCH --job-name={tag}_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --chdir={repo}
#SBATCH --output={logdir}/{tag}_merge_%j.log
{sbatch}
set -uo pipefail
{setup}
export PYTHONNOUSERSITE=1
{python} {runner} --prompts {prompts} --output-dir {outdir} \\
  --num-shards {nshards} --merge
"""


def paths(method, model, task):
    runner = f"benchmark/{task}/{RUNNER[(method, model)]}.py"
    prompts = f"outputs/benchmarks/{task}/prompts_full.jsonl"
    outdir = f"outputs/benchmarks/{task}/table1/{method}__{model}"
    return runner, prompts, outdir


# ── SLURM ───────────────────────────────────────────────────────────────────
def submit_cell(method, model, task, *, nodes, gpus, python, dry) -> bool:
    runner, prompts, outdir = paths(method, model, task)
    nshards = nodes * gpus
    tag = f"t1_{task}_{method}_{model}".replace(".", "p")
    fields = dict(tag=tag, repo=REPO, logdir=LOGDIR, outdir=outdir, runner=runner,
                  prompts=prompts, nshards=nshards, gpus=gpus, python=python,
                  maxarray=nodes - 1, stagger=STAGGER_SECONDS, setup=job_setup())
    gen = GEN.format(sbatch=sbatch_lines(gres=ENV["gres"], cpus=ENV["cpus"],
                                         mem=ENV["mem"], time_=ENV["time"]), **fields)
    # merge is CPU-only: no gres, small allocation
    merge = MERGE.format(sbatch=sbatch_lines(cpus="8", mem="32G", time_="2:00:00"),
                         **fields)

    if dry:
        print(f"  {method}/{model}/{task:<10} {nodes:>2} nodes x {gpus} "
              f"= {nshards:>3} shards  ->  {outdir}")
        return True

    JOBDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    gpath, mpath = JOBDIR / f"{tag}.sh", JOBDIR / f"{tag}_merge.sh"
    gpath.write_text(gen)
    mpath.write_text(merge)

    j = subprocess.run(["sbatch", "--parsable", str(gpath)],
                       capture_output=True, text=True, cwd=REPO)
    if j.returncode != 0:
        print(f"  !! sbatch failed for {tag}: {j.stderr.strip()}", file=sys.stderr)
        return False
    arr = j.stdout.strip()
    m = subprocess.run(["sbatch", "--parsable", f"--dependency=afterany:{arr}", str(mpath)],
                       capture_output=True, text=True, cwd=REPO)
    if m.returncode != 0:
        print(f"  !! merge sbatch failed for {tag}: {m.stderr.strip()}", file=sys.stderr)
        return False
    print(f"  {method}/{model}/{task:<10} gen={arr} ({nodes}x{gpus})  "
          f"merge={m.stdout.strip()}")
    return True


# ── local ───────────────────────────────────────────────────────────────────
def run_cell_local(method, model, task, *, gpus, python, dry) -> bool:
    runner, prompts, outdir = paths(method, model, task)
    base = [python, runner, "--prompts", prompts, "--output-dir", outdir,
            "--num-shards", str(gpus)]
    tag = f"{method}/{model}/{task}"
    if dry:
        print(f"  {tag}: {gpus} shards locally  ->  {outdir}")
        return True

    (REPO / outdir / "shards").mkdir(parents=True, exist_ok=True)
    print(f"\n=== {tag}  ({gpus} shards) ===", flush=True)
    procs = []
    for i in range(gpus):
        tmp = Path(tempfile.gettempdir()) / f"carve-{os.getpid()}" / str(i)
        (tmp / "torchinductor").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), HIP_VISIBLE_DEVICES=str(i),
                   PYTHONNOUSERSITE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
                   TMPDIR=str(tmp), TORCHINDUCTOR_CACHE_DIR=str(tmp / "torchinductor"))
        cmd = base + ["--shard-id", str(i), "--device", "cuda:0"]
        log = (REPO / outdir / "shards" / f"shard_{i}.log").open("w")
        procs.append((i, subprocess.Popen(cmd, cwd=REPO, env=env,
                                          stdout=log, stderr=subprocess.STDOUT), log))
        if i + 1 < gpus:
            time.sleep(STAGGER_SECONDS)
    failed = []
    for i, proc, log in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            failed.append(i)
    if failed:
        print(f"  !! shards {failed} failed — see {outdir}/shards/shard_<i>.log", flush=True)
        return False
    print("  shards done, merging...", flush=True)
    if subprocess.run(base + ["--merge"], cwd=REPO).returncode != 0:
        print(f"  !! merge failed for {tag}", flush=True)
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="all 36 cells")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--local", action="store_true",
                   help="run on this machine instead of submitting to SLURM")
    p.add_argument("--nodes", type=int, default=0,
                   help="nodes per cell (default: 4 for HumanEval, 16 otherwise)")
    p.add_argument("--gpus", type=int, default=GPUS_PER_NODE, help="GPUs per node")
    p.add_argument("--python", default="python")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if a.all:
        cells = [(me, mo, ta) for me in METHODS for mo in MODELS for ta in TASKS]
    else:
        if not (a.method and a.model and a.task):
            p.error("give --all, or all three of --method --model --task")
        cells = [(a.method, a.model, a.task)]

    if a.gpus < 1:
        p.error("--gpus must be >= 1")
    if not a.local and not a.dry_run and shutil.which("sbatch") is None:
        p.error("sbatch not found on PATH.\n"
                "Submit from a login node, or use --local to run here instead.")

    missing = sorted({c[2] for c in cells
                      if not (REPO / f"outputs/benchmarks/{c[2]}/prompts_full.jsonl").exists()})
    if missing and not a.dry_run:
        p.error("prompt files missing for " + ", ".join(missing)
                + "\nBuild them first:  python scripts/build_datasets.py")

    print(f"# {len(cells)} cell(s) — "
          + (f"running here on {a.gpus} GPU(s)" if a.local else "submitting to SLURM"),
          flush=True)

    ok, bad = 0, []
    for me, mo, ta in cells:
        nodes = a.nodes or NODES.get(ta, 16)
        done = (run_cell_local(me, mo, ta, gpus=a.gpus,
                               python=a.python, dry=a.dry_run)
                if a.local else
                submit_cell(me, mo, ta, nodes=nodes, gpus=a.gpus,
                            python=a.python, dry=a.dry_run))
        ok += bool(done)
        if not done:
            bad.append(f"{me}/{mo}/{ta}")
            # A submission failure is almost always systemic (bad partition,
            # QoS, account, gres). Stop rather than repeat it 35 more times.
            if not a.local:
                print("\nStopped after the first submission failure — the cause is\n"
                      "usually a cluster setting. Check CARVE_PARTITION / CARVE_QOS /\n"
                      "CARVE_ACCOUNT / CARVE_GRES against `sinfo`, then re-run.",
                      file=sys.stderr)
                break

    if a.dry_run:
        total = sum((a.nodes or NODES.get(ta, 16)) for _, _, ta in cells)
        if not a.local:
            print(f"\n# {total} nodes total across {len(cells)} array job(s), "
                  f"each with a dependent CPU-only merge job")
        return

    if a.local:
        print(f"\n=== {ok}/{len(cells)} cells completed ===")
    else:
        print(f"\n=== {ok}/{len(cells)} cells submitted ===")
    if bad:
        print("failed: " + ", ".join(bad))
    if ok:
        if not a.local:
            print("Watch with:                squeue -u $USER")
        print("When everything finishes:  python scripts/table1.py")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
