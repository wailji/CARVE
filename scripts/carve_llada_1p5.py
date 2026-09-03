"""carve_llada_1p5.py — CARVE on LLaDA-1.5 (blockwise semi-autoregressive).

Identical algorithm to carve_llada.py, bound to the LLaDA-1.5 backend. Kept as
its own module so each model of Table 1 is self-contained.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from scripts.llada_1p5_backend import (
    LLADA_MASK_ID,
    LLADA_EOS_ID,
    add_gumbel_noise,
    daedal_confthr_reveal_step,
    forward_logits,
    insert_masks_at,
    llada_lowconf_reveal_step,
    set_seed,
    _flop_ctx,
    _read_flops,
    _first_eos_in_response,
    _crop_to_eos,
)




def _propose_insertion_pos_in_block(
    logits: torch.Tensor,         # (1, L, V) — base logits
    block_start: int,
    block_end: int,
    *,
    mid_window: int = 8,
) -> int:
    """Absolute position to insert k masks INSIDE [block_start, block_end).

    Mid-insert (paper Section 5.3): pick the gap whose surrounding `mid_window`
    positions carry the highest summed entropy — the most under-specified part
    of the current block. Falls back to the block end when the block is too
    small for the requested window.
    """
    block_len = block_end - block_start
    if block_len < 2:
        return block_end
    if mid_window < 2 or mid_window % 2 != 0:
        raise ValueError(f"mid_window must be a positive EVEN integer (got {mid_window})")
    if block_len < mid_window:
        return block_end

    probs = F.softmax(logits[0, block_start:block_end].float(), dim=-1)
    H = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    half = mid_window // 2
    cum = F.pad(H.cumsum(0), (1, 0))
    gs = torch.arange(half, block_len - half + 1, device=H.device)
    scores = cum[gs + half] - cum[gs - half]
    return block_start + int(gs[int(scores.argmax().item())].item())


def _js_damage_full_response(
    F_base: torch.Tensor,
    F_ins: torch.Tensor,
    x_base: torch.Tensor,
    mask_token_id: int,
    prompt_len: int,
    active_len: int,
    insert_pos: int,
    k: int,
    eps: float = 1e-12,
) -> float:
    """Mean JS divergence over ALL response MASK positions."""
    if active_len == 0:
        return 0.0
    abs_resp = torch.arange(prompt_len, prompt_len + active_len, device=x_base.device)
    is_mask = x_base[0, abs_resp] == mask_token_id
    if not bool(is_mask.any()):
        return 0.0
    mask_pos_base = abs_resp[is_mask]
    shift = (mask_pos_base >= insert_pos).long() * k
    mask_pos_ins = mask_pos_base + shift

    p = F.softmax(F_base[0, mask_pos_base].float(), dim=-1).clamp_min(eps)
    q = F.softmax(F_ins[0, mask_pos_ins].float(), dim=-1).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)
    return float(js.mean().item())


def run_carve_llada(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,    # (1, P)
    *,
    seed: int,
    initial_gen_length: int,
    max_gen_length: int,
    max_steps: int,
    block_length: int = 32,
    js_threshold: float = 0.05,
    insert_k: int = 4,
    carve_interval: int = 1,
    mid_window: int = 8,              # even, >= 2
    temperature: float = 0.0,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    mask_token_id: int = LLADA_MASK_ID,
    eos_token_id: int = LLADA_EOS_ID,
    eot_token_id: int = 126348,  # LLADA_EOT_ID
) -> Dict:
    """CARVE v2 on LLaDA-1.5: semi-AR + insertion-branch JS gate."""
    set_seed(seed)
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started_at = time.time()
    fwd_count = 0
    inserts_used = 0
    n_expand_attempts = 0
    n_expand_accepted = 0
    js_history: List[Dict] = []
    canvas_growth: List[Tuple[int, int]] = []
    early_exit_step: Optional[int] = None
    eos_crop_step: Optional[int] = None
    eos_crop_active_len: Optional[int] = None

    flop_ctx = _flop_ctx()
    flop_ctx.__enter__()

    active_len = max(1, min(max_gen_length, int(initial_gen_length)))
    response = torch.full((1, active_len), mask_token_id, device=device, dtype=torch.long)
    x = torch.cat([prompt_ids, response], dim=1)
    attn = torch.ones((1, prompt_len + active_len), device=device, dtype=torch.long)
    canvas_growth.append((0, active_len))

    current_pos = prompt_len
    steps_in_block_so_far = 0
    max_steps_per_block = max(1, block_length)
    current_block_growth = 0
    branch_interval = max(1, int(carve_interval))

    step = 0
    while step < max_steps:
        total_len = prompt_len + active_len
        if current_pos >= total_len:
            break

        block_end = min(current_pos + block_length, total_len)
        block_masks = (x[0, current_pos:block_end] == mask_token_id)
        if not bool(block_masks.any()):
            current_pos = block_end
            steps_in_block_so_far = 0
            continue

        F_base = forward_logits(model, x, attn)
        fwd_count += 1
        F_chosen = F_base
        chosen_canvas: Tuple[torch.Tensor, torch.Tensor, int] = (x, attn, active_len)

        do_expand = False
        damage: Optional[float] = None
        insert_pos: Optional[int] = None
        block_end_chosen = block_end
        room_left = active_len < max_gen_length
        gate_step = (step % branch_interval == 0)

        if room_left and gate_step:
            insert_pos = _propose_insertion_pos_in_block(
                F_base, current_pos, block_end,
                mid_window=mid_window,
            )
            x_ins, attn_ins, k_inserted = insert_masks_at(
                x, attn, insert_pos, insert_k, mask_token_id, prompt_len + max_gen_length,
            )
            if k_inserted > 0:
                F_ins = forward_logits(model, x_ins, attn_ins)
                fwd_count += 1
                damage = _js_damage_full_response(
                    F_base, F_ins, x, mask_token_id,
                    prompt_len, active_len, insert_pos, k_inserted,
                )
                n_expand_attempts += 1
                if damage < js_threshold:
                    do_expand = True
                    n_expand_accepted += 1
                    chosen_canvas = (x_ins, attn_ins, active_len + k_inserted)
                    F_chosen = F_ins
                    inserts_used += k_inserted
                    block_end_chosen = block_end + k_inserted
                    if insert_pos < block_end:
                        current_block_growth += k_inserted

        x, attn, active_len = chosen_canvas

        remaining_in_block = max(1, max_steps_per_block + current_block_growth - steps_in_block_so_far)
        x = llada_lowconf_reveal_step(
            x, F_chosen, mask_token_id,
            block_start=current_pos, block_end=block_end_chosen,
            remaining_steps_in_block=remaining_in_block,
            temperature=temperature,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
            eos_token_id=eos_token_id,
            eot_token_id=eot_token_id,
        )

        steps_in_block_so_far += 1
        canvas_growth.append((step + 1, active_len))
        if damage is not None:
            js_history.append({
                "step": step + 1,
                "active_len": active_len,
                "current_pos": current_pos,
                "block_end": block_end_chosen,
                "insert_pos": insert_pos,
                "damage": damage,
                "action": "expand" if do_expand else "skip",
            })

        # ── EOS crop: stop once an EOS is committed before the canvas end ──
        eos_pos = _first_eos_in_response(x, prompt_len, active_len, eos_token_id)
        if eos_pos is not None and eos_pos + 1 < active_len:
            x, attn, active_len = _crop_to_eos(x, attn, prompt_len, eos_pos)
            abs_resp = torch.arange(prompt_len, prompt_len + active_len, device=device)
            if (x[0, abs_resp] == mask_token_id).any():
                F_final = forward_logits(model, x, attn)
                fwd_count += 1
                mask_index = (x == mask_token_id)
                if mask_index.any():
                    mask_logits = F_final[mask_index]
                    argmax_tokens = mask_logits.argmax(dim=-1)
                    x_new = x.clone()
                    x_new[mask_index] = argmax_tokens
                    x = x_new
            eos_crop_step = step + 1
            eos_crop_active_len = active_len
            early_exit_step = step + 1
            canvas_growth.append((step + 1, active_len))
            step += 1
            break

        effective_block_end = min(
            current_pos + block_length + current_block_growth,
            prompt_len + active_len,
        )
        if not bool((x[0, current_pos:effective_block_end] == mask_token_id).any()):
            current_pos = effective_block_end
            steps_in_block_so_far = 0
            current_block_growth = 0

        step += 1

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
        "method": "carve_llada_1p5",
        "model_family": "llada_1p5",
        "seed": seed,
        "initial_gen_length": int(initial_gen_length),
        "max_gen_length": int(max_gen_length),
        "block_length": int(block_length),
        "max_steps": int(max_steps),
        "actual_steps": step,
        "mid_window": int(mid_window),
        "eos_crop_step": eos_crop_step,
        "eos_crop_active_len": eos_crop_active_len,
        "early_exit_step": early_exit_step,
        "js_threshold": js_threshold,
        "insert_k": insert_k,
        "carve_interval": int(branch_interval),
        "temperature": temperature,
        "logits_eos_inf": bool(logits_eos_inf),
        "confidence_eos_eot_inf": bool(confidence_eos_eot_inf),
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
        "elapsed_seconds": round(elapsed, 3),
        "peak_mem_gb": peak_mem_gb,
        "js_history": js_history,
        "canvas_growth": canvas_growth,
    }
