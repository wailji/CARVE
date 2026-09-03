"""carve_dream.py — CARVE on Dream (full-canvas masked diffusion).

Implements Algorithm 1 of the paper. At each denoising step:

  1. F_base = forward(current canvas).
  2. Every `carve_interval` steps, while L < max_len, propose inserting
     `insert_k` masks at the most under-specified gap (highest windowed
     entropy over `mid_window` positions), and run a second forward F_ins on
     that expanded canvas.
  3. Accept the expansion iff the mean Jensen-Shannon divergence between
     F_base and F_ins, over aligned still-masked positions, is below
     `js_threshold`. Otherwise discard it and continue on the original canvas.
  4. Reveal using the CHOSEN branch's logits on the chosen branch's canvas —
     tokens are never committed from a geometry the model did not see.
  5. On a committed EOS before the canvas end, crop there, fill any remaining
     masks by argmax with one final forward, and return.

Reveal uses the adaptive schedule n_s = ceil(|M_s| / (T - s)) (paper Eq. 29),
which guarantees progress at every step and keeps pace with a growing canvas.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ─── reuse infra from dream_backend ────────────────────────────────────────
# Identity helpers + flop-counter shim + forward kernel are unchanged from v1
# so we just import them rather than copying.
from scripts.dream_backend import (  # noqa: E402
    load_model_and_tokenizer,
    set_seed,
    forward_logits,
    _flop_ctx,
    _read_flops,
    _first_eos_in_response,
    _crop_to_eos,
    _top_p_filter,
    _top_k_filter,
    entropy_reveal_step,
)


# ───────────────────────────── insertion primitives ─────────────────────────────



def _insert_at_gap(
    x: torch.Tensor,
    att1d: torch.Tensor,
    prompt_len: int,
    active_len: int,
    gap: int,            # 0..active_len; insert k masks BEFORE response position `gap`
    k: int,
    mask_token_id: int,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Insert k masks at response position `gap` (i.e. between old positions
    gap-1 and gap; gap=active_len means tail-append; gap=0 means head-prepend).
    Returns (x_new, att_new, new_active_len, k_inserted).
    """
    k = min(k, max_len - active_len)
    if k <= 0:
        return x, att1d, active_len, 0
    abs_g = prompt_len + gap
    new_chunk = torch.full((x.shape[0], k), mask_token_id, device=x.device, dtype=x.dtype)
    new_att = torch.ones((x.shape[0], k), device=att1d.device, dtype=att1d.dtype)
    x_new = torch.cat([x[:, :abs_g], new_chunk, x[:, abs_g:]], dim=1)
    att_new = torch.cat([att1d[:, :abs_g], new_att, att1d[:, abs_g:]], dim=1)
    return x_new, att_new, active_len + k, k


def propose_insertion_gap(
    F_base_logits: torch.Tensor,   # (1, prompt_len + active_len, V)
    prompt_len: int,
    active_len: int,
    *,
    mid_window: int = 12,
) -> int:
    """Return the gap position g in [0, active_len] where k masks should go.

    Mid-insert (paper Section 5.3): pick the gap whose surrounding `mid_window`
    response positions carry the highest summed entropy, i.e. the most
    under-specified region of the current canvas. `mid_window` must be a
    positive even integer; the window covers [g - W/2, g + W/2).

    Falls back to appending at the tail when the canvas is too small to fit the
    requested window.
    """
    if active_len < 2:
        return active_len            # too small for an interior gap
    if mid_window < 2 or mid_window % 2 != 0:
        raise ValueError(f"mid_window must be a positive EVEN integer (got {mid_window})")
    if active_len < mid_window:
        return active_len

    # Response-only logits, shape (active_len, V).
    abs_pos = torch.arange(prompt_len, prompt_len + active_len, device=F_base_logits.device)
    probs = F.softmax(F_base_logits[0, abs_pos].float(), dim=-1)
    H = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)  # (L,)
    half = mid_window // 2
    # Windowed sum via cumsum: score(g) = sum H[g-half : g+half].
    cum = F.pad(H.cumsum(0), (1, 0))
    gs = torch.arange(half, active_len - half + 1, device=H.device)
    scores = cum[gs + half] - cum[gs - half]
    return int(gs[int(scores.argmax().item())].item())


def js_damage_aligned(
    F_base: torch.Tensor,         # (1, prompt_len + active_len, V)
    F_ins: torch.Tensor,          # (1, prompt_len + active_len + k, V)
    x_base: torch.Tensor,
    mask_token_id: int,
    prompt_len: int,
    active_len: int,
    gap: int,
    k: int,
    eps: float = 1e-12,
) -> float:
    """Mean JS divergence between base and inserted-branch logits, over base
    response positions that are MASK. Aligned positions: j_base = j (if j<gap)
    else j+k. Newly inserted positions [gap, gap+k) on the inserted branch are
    skipped (they don't exist in base). Committed (non-MASK) base positions
    are skipped (those are frozen tokens we don't care about for damage).
    """
    if active_len == 0:
        return 0.0

    abs_resp = torch.arange(prompt_len, prompt_len + active_len, device=x_base.device)
    response_tokens = x_base[0, abs_resp]
    is_mask = (response_tokens == mask_token_id)
    if not bool(is_mask.any()):
        return 0.0

    mask_pos_base = is_mask.nonzero(as_tuple=False).squeeze(-1)         # (N,) response indices
    shift = (mask_pos_base >= gap).long() * k                            # (N,)
    mask_pos_ins = mask_pos_base + shift                                 # aligned response indices

    p = F.softmax(F_base[0, prompt_len + mask_pos_base].float(), dim=-1).clamp_min(eps)
    q = F.softmax(F_ins[0, prompt_len + mask_pos_ins].float(), dim=-1).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)
    return float(js.mean().item())


# ───────────────────────────── CARVE v2 main loop ─────────────────────────────

def run_carve_dream(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    seed: int,
    max_len: int,
    steps: int,
    js_threshold: float,             # damage threshold; accept if damage < threshold
    insert_k: int,
    L0: int,                          # always required in v2 — no prefill probe
    carve_interval: int = 1,
    mid_window: int = 12,             # even, >= 2
    temperature: float = 0.0,
    top_p: Optional[float] = None,
) -> Dict:
    """CARVE v2: insertion-branch gate, commit-from-chosen-branch.

    Per diffusion step:
      F_base = forward(canvas)
      F_chosen = F_base
      if expandable AND step % carve_interval == 0:
          gap = propose_insertion_gap(F_base, mid_window=w)
          x_ins, att_ins, _ = _insert_at_gap(x, att, gap, k)
          F_ins = forward(x_ins, att_ins)
          damage = js_damage_aligned(F_base, F_ins, gap, k)
          if damage < js_threshold:
              x, att, active_len = x_ins, att_ins, active_len + k
              F_chosen = F_ins
      x = entropy_reveal_step(x, F_chosen, ...)   # ← reveal from chosen branch

    No prefill probe: L0 is required and used as the fixed initial canvas.
    No legacy scheduler: always adaptive `ceil(M / remaining_steps)`.
    """
    set_seed(seed)
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started_at = time.time()
    fwd_count = 0
    inserts_used = 0
    n_expand_attempts = 0
    n_expand_accepted = 0

    flop_ctx = _flop_ctx()
    flop_ctx.__enter__()

    # ── No prefill probe: L0 fixed ──
    active_len = max(1, min(max_len, int(L0)))
    response = torch.full((1, active_len), mask_token_id, device=device, dtype=torch.long)
    x = torch.cat([prompt_ids, response], dim=1)
    att1d = torch.ones((1, prompt_len + active_len), device=device, dtype=torch.bool)

    branch_interval = max(1, int(carve_interval))
    js_history: List[Dict] = []
    canvas_growth: List[Tuple[int, int]] = [(0, active_len)]
    early_exit_step: Optional[int] = None
    eos_crop_locked = False
    eos_crop_steps: List[Tuple[int, int]] = []

    for step in range(steps):
        # Early exit when canvas is fully revealed AND no further expansion possible.
        abs_resp = torch.arange(prompt_len, prompt_len + active_len, device=device)
        if not (x[0, abs_resp] == mask_token_id).any():
            if eos_crop_locked or active_len >= max_len:
                early_exit_step = step
                break

        F_base = forward_logits(model, x, att1d)
        fwd_count += 1
        F_chosen = F_base
        chosen_canvas: Tuple[torch.Tensor, torch.Tensor, int] = (x, att1d, active_len)

        do_expand = False
        gap_chosen: Optional[int] = None
        damage: Optional[float] = None
        if (not eos_crop_locked) and (step % branch_interval == 0) and active_len < max_len:
            gap_chosen = propose_insertion_gap(
                F_base, prompt_len, active_len,
                mid_window=mid_window,
            )
            x_ins, att_ins, new_len_ins, k_inserted = _insert_at_gap(
                x, att1d, prompt_len, active_len, gap_chosen, insert_k, mask_token_id, max_len,
            )
            if k_inserted > 0:
                F_ins = forward_logits(model, x_ins, att_ins)
                fwd_count += 1
                damage = js_damage_aligned(
                    F_base, F_ins, x, mask_token_id,
                    prompt_len, active_len, gap_chosen, k_inserted,
                )
                n_expand_attempts += 1
                if damage < js_threshold:
                    do_expand = True
                    n_expand_accepted += 1
                    chosen_canvas = (x_ins, att_ins, new_len_ins)
                    F_chosen = F_ins
                    inserts_used += k_inserted

        # Commit using the chosen branch's logits ON the chosen branch's canvas.
        x, att1d, active_len = chosen_canvas
        x = entropy_reveal_step(
            x, F_chosen, mask_token_id, step, steps,
            temperature=temperature, top_p=top_p,
        )

        # ── EOS crop: stop as soon as an EOS is committed before the canvas end ──
        eos_pos = _first_eos_in_response(x, prompt_len, active_len, eos_token_id)
        if eos_pos is not None and eos_pos + 1 < active_len:
            x, att1d, active_len = _crop_to_eos(x, att1d, prompt_len, eos_pos)
            eos_crop_locked = True
            eos_crop_steps.append((step + 1, active_len))
            # Argmax-fill any remaining MASK positions in the cropped canvas
            # using one fresh F_base forward (logits depend on the new shorter canvas).
            abs_resp = torch.arange(prompt_len, prompt_len + active_len, device=device)
            if (x[0, abs_resp] == mask_token_id).any():
                F_final = forward_logits(model, x, att1d)
                fwd_count += 1
                mask_index = (x == mask_token_id)
                if mask_index.any():
                    mask_logits = F_final[mask_index]
                    argmax_tokens = mask_logits.argmax(dim=-1)
                    x_new = x.clone()
                    x_new[mask_index] = argmax_tokens
                    x = x_new
            early_exit_step = step + 1
            canvas_growth.append((step + 1, active_len))
            break

        canvas_growth.append((step + 1, active_len))
        if damage is not None:
            js_history.append({
                "step": step + 1,
                "active_len": active_len,
                "gap": gap_chosen,
                "damage": damage,
                "action": "expand" if do_expand else "skip",
            })

    elapsed = time.time() - started_at
    peak_mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0.0

    response_ids = x[0, prompt_len : prompt_len + active_len].tolist()
    answer_ids: List[int] = []
    for tid in response_ids:
        if tid == eos_token_id:
            break
        answer_ids.append(tid)
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    full_response_text = tokenizer.decode(response_ids, skip_special_tokens=False)

    flop_ctx.__exit__(None, None, None)
    total_flops = _read_flops(flop_ctx)

    return {
        "method": "carve_dream",
        "seed": seed,
        "L0": int(L0),
        "carve_interval": branch_interval,
        "mid_window": int(mid_window),
        "eos_crop_steps": eos_crop_steps,
        "max_len": max_len,
        "steps": steps,
        "js_threshold": js_threshold,
        "insert_k": insert_k,
        "temperature": temperature,
        "top_p": top_p,
        "answer_text": answer_text,
        "answer_token_length": len(answer_ids),
        "response_ids_all": response_ids,
        "full_response_text": full_response_text,
        "final_active_slots": active_len,
        "fwd_count": fwd_count,
        "total_flops": total_flops,
        "inserts_used": inserts_used,
        "n_expand_attempts": n_expand_attempts,
        "n_expand_accepted": n_expand_accepted,
        "early_exit_step": early_exit_step,
        "elapsed_seconds": round(elapsed, 3),
        "peak_mem_gb": peak_mem_gb,
        "js_history": js_history,
        "canvas_growth": canvas_growth,
    }
