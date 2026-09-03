"""The complete Table-1 recipe: one entry per (method, model, task).

Every value a decoder needs is derived here. Nothing is a command-line flag,
because nothing varies at run time -- a cell is fully determined by which of the
36 combinations you are running.

Sources: paper Appendix Table 5 (canvas and the per-model CARVE knobs) and
Section 4.3 (L0 = Lmax/2, steps = Lmax).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

METHODS = ("vanilla", "daedal", "carve")
MODELS = ("dream", "llada-8b", "llada-1.5")
TASKS = ("humaneval", "mbpp", "math500", "gsm8k")

SEED = 1337

# ── Appendix Table 5: maximum canvas Lmax per (model, task) ─────────────────
CANVAS: Dict[str, Dict[str, int]] = {
    "dream":     {"humaneval": 128, "mbpp": 256, "math500": 512, "gsm8k": 256},
    "llada-1.5": {"humaneval": 512, "mbpp": 512, "math500": 512, "gsm8k": 512},
    "llada-8b":  {"humaneval": 512, "mbpp": 256, "math500": 512, "gsm8k": 512},
}

# ── Appendix Table 5: per-model knobs ──────────────────────────────────────
# block=None means full-canvas decoding (Dream). All three methods of a model
# decode at the same temperature, so the rows differ only in the algorithm.
TEMPERATURE: Dict[str, float] = {"dream": 0.04, "llada-8b": 0.03, "llada-1.5": 0.05}
BLOCK: Dict[str, Optional[int]] = {"dream": None, "llada-8b": 64, "llada-1.5": 32}
MID_WINDOW: Dict[str, int] = {"dream": 12, "llada-8b": 8, "llada-1.5": 4}
CARVE_INTERVAL: Dict[str, int] = {"dream": 16, "llada-8b": 1, "llada-1.5": 1}

# CARVE, identical for every model.
JS_THRESHOLD = 0.05    # accept an expansion iff mean JS divergence is below this
INSERT_K = 16          # masks added per accepted expansion (Table 5, k)

# Dream's sampler settings for the fixed-length baseline.
DREAM_TOP_P = 0.9
DREAM_ALG = "entropy"

# DAEDAL chooses its own length, so it uses its published budget rather than the
# Table-5 canvas -- identical for every model and task (Li et al., 2025).
# On Dream it uses DAEDAL's standard block of 32: it is semi-autoregressive and
# degenerates at full-canvas block size.
DAEDAL = dict(
    initial_gen_length=64,
    max_gen_length=2048,
    max_iterations=4096,
    expansion_factor=8,
    high_conf_threshold=0.9,
    low_conf_threshold=0.1,
    eos_confidence_threshold=0.5,
    expand_eos_confidence_threshold=0.9,
    eos_check_tokens=32,
    cfg_scale=0.0,
)
DAEDAL_BLOCK: Dict[str, int] = {"dream": 32, "llada-8b": 64, "llada-1.5": 32}


@dataclass(frozen=True)
class Config:
    """Everything one cell needs. Unused fields for a method are None."""
    method: str
    model: str
    task: str
    seed: int = SEED
    # canvas
    l0: Optional[int] = None
    lmax: Optional[int] = None
    steps: Optional[int] = None
    block_length: Optional[int] = None
    # sampling
    temperature: float = 0.0
    top_p: Optional[float] = None
    # CARVE
    carve_interval: Optional[int] = None
    mid_window: Optional[int] = None
    # LLaDA EOS handling
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False


def resolve(method: str, model: str, task: str) -> Config:
    """The decoding configuration for one cell of Table 1."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")

    lmax = CANVAS[model][task]
    temp = TEMPERATURE[model]
    dream = model == "dream"

    # LLaDA-8B on HumanEval is the only cell that suppresses EOS in the logits
    # instead of only in the confidence that drives reveal ordering.
    he_8b = model == "llada-8b" and task == "humaneval"
    eos = {} if dream else dict(logits_eos_inf=he_8b,
                                confidence_eos_eot_inf=not he_8b)

    if method == "vanilla":
        return Config(method, model, task,
                      lmax=lmax, steps=lmax,
                      block_length=None if dream else BLOCK[model],
                      temperature=temp,
                      top_p=DREAM_TOP_P if dream else None,
                      **eos)

    if method == "daedal":
        return Config(method, model, task,
                      l0=DAEDAL["initial_gen_length"],
                      lmax=DAEDAL["max_gen_length"],
                      steps=DAEDAL["max_iterations"],
                      block_length=DAEDAL_BLOCK[model],
                      temperature=temp)

    # CARVE -- start at Lmax/2 and grow up to Lmax (Section 4.3).
    return Config(method, model, task,
                  l0=lmax // 2, lmax=lmax, steps=lmax,
                  block_length=None if dream else BLOCK[model],
                  temperature=temp,
                  top_p=DREAM_TOP_P if dream else None,
                  carve_interval=CARVE_INTERVAL[model],
                  mid_window=MID_WINDOW[model],
                  **eos)
