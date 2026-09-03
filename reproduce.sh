#!/usr/bin/env bash
#
# Reproduce Table 1 of the CARVE paper.
#
#   conda activate carve
#   bash reproduce.sh
#
# Run from a login node. Downloads the four benchmarks, then submits all 36
# cells to SLURM: one job array per cell, each array task a node with 8 GPUs,
# plus a dependent merge job. 4 nodes for HumanEval, 16 for the others.
#
# Works with no configuration -- the job requests a generic gpu:8 and inherits
# your activated environment. Optional CARVE_* variables (PARTITION, QOS,
# ACCOUNT, GRES, CPUS, MEM, TIME, CONDA_ENV, PREAMBLE, EXCLUDE, NODELIST)
# override the defaults; see scripts/run.py.
#
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
ARGS=()
[ -n "${NODES:-}" ]   && ARGS+=(--nodes "$NODES")
[ -n "${DRY_RUN:-}" ] && ARGS+=(--dry-run)
[ -n "${LOCAL:-}" ]   && ARGS+=(--local)

echo "=============================================================="
echo " CARVE — Table 1"
echo " host   : $(hostname)"
echo " started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

echo
echo ">>> [1/2] Building datasets"
"$PY" scripts/build_datasets.py || { echo "dataset build FAILED"; exit 1; }

echo
echo ">>> [2/2] Launching the 36 cells"
"$PY" scripts/run.py --all "${ARGS[@]}"
RC=$?

echo
echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
exit $RC
