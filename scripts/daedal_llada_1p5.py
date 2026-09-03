"""daedal_llada_1p5.py — DAEDAL on LLaDA-1.5.

Verbatim copy of daedal_llada.py with the backend import redirected to
scripts.llada_1p5_backend. All DAEDAL hyperparameters are at paper defaults.
block_length=32 (our internal convention for DAEDAL+CARVE).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from scripts.llada_1p5_backend import (
    LLADA_EOS_ID,
    LLADA_MASK_ID,
    add_gumbel_noise,
    forward_logits,
    set_seed,
    _flop_ctx,
    _read_flops,
)


def _calculate_eos_confidence(
    logits: torch.Tensor,
    total_length: int,
    prompt_length: int,
    eos_check_tokens: int,
    eos_token_id: int,
) -> float:
    if eos_token_id is None:
        return 0.0
    confidences = F.softmax(logits.to(torch.float32), dim=-1)
    predicted_tokens = torch.argmax(logits, dim=-1)
    eos_confs: List[float] = []
    start_scan_pos = total_length - 1
    end_scan_pos = prompt_length - 1
    for pos in range(start_scan_pos, end_scan_pos, -1):
        if len(eos_confs) >= eos_check_tokens:
            break
        if int(predicted_tokens[0, pos].item()) == eos_token_id:
            eos_confs.append(float(confidences[0, pos, eos_token_id].item()))
    return sum(eos_confs) / max(1, eos_check_tokens)


def run_daedal_llada(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,    # (1, P)
    *,
    seed: int,
    initial_gen_length: int = 64,
    max_gen_length: int = 2048,
    block_length: int = 32,
    expansion_factor: int = 8,
    high_conf_threshold: float = 0.9,
    low_conf_threshold: float = 0.1,
    eos_confidence_threshold: float = 0.5,
    expand_eos_confidence_threshold: float = 0.9,
    eos_check_tokens: int = 32,
    cfg_scale: float = 0.0,
    temperature: float = 0.0,
    max_iterations: int = 4096,
    mask_token_id: int = LLADA_MASK_ID,
    eos_token_id: int = LLADA_EOS_ID,
) -> Dict:
    """DAEDAL on LLaDA-1.5. Single-sequence (batch=1)."""
    set_seed(seed)
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    prompt_length = prompt_ids.shape[1]

    started_at = time.time()
    fwd_count = 0
    inserts_used = 0
    n_stage1_expansions = 0
    n_stage2_expansion_events = 0
    canvas_growth: List[Tuple[int, int]] = []

    flop_ctx = _flop_ctx()
    flop_ctx.__enter__()

    gen_length = max(1, min(max_gen_length, int(initial_gen_length)))
    response = torch.full((1, gen_length), mask_token_id, device=device, dtype=torch.long)
    x = torch.cat([prompt_ids, response], dim=1)
    attn = torch.ones((1, prompt_length + gen_length), device=device, dtype=torch.long)
    canvas_growth.append((0, gen_length))

    # ─── Stage 1 ───
    while True:
        total_length = prompt_length + gen_length
        logits = forward_logits(model, x, attn)
        fwd_count += 1
        eos_conf = _calculate_eos_confidence(
            logits, total_length, prompt_length, eos_check_tokens, eos_token_id,
        )
        if eos_conf >= eos_confidence_threshold or gen_length >= max_gen_length:
            break
        new_gen = min(gen_length + expansion_factor, max_gen_length)
        if new_gen <= gen_length:
            break
        new_chunk = torch.full((1, new_gen - gen_length), mask_token_id, device=device, dtype=torch.long)
        new_attn = torch.ones((1, new_gen - gen_length), device=device, dtype=torch.long)
        x = torch.cat([x, new_chunk], dim=1)
        attn = torch.cat([attn, new_attn], dim=1)
        gen_length = new_gen
        n_stage1_expansions += 1
        canvas_growth.append((-n_stage1_expansions, gen_length))

    # ─── Intermediate buffer ───
    extra = min(max_gen_length - gen_length, eos_check_tokens // 2)
    if extra > 0:
        new_chunk = torch.full((1, extra), eos_token_id, device=device, dtype=torch.long)
        new_attn = torch.ones((1, extra), device=device, dtype=torch.long)
        x = torch.cat([x, new_chunk], dim=1)
        attn = torch.cat([attn, new_attn], dim=1)
        gen_length += extra

    # ─── Stage 2 ───
    current_pos = prompt_length
    denoise_only_mode = False
    last_x: Optional[torch.Tensor] = None

    iteration = 0
    while iteration < max_iterations:
        total_length = prompt_length + gen_length
        if current_pos >= total_length:
            break
        if last_x is not None and last_x.shape == x.shape and torch.equal(x, last_x):
            break
        last_x = x.clone()

        if gen_length >= max_gen_length and not denoise_only_mode:
            denoise_only_mode = True

        logits = forward_logits(model, x, attn)
        fwd_count += 1
        pred = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
        confs_full = F.softmax(logits.to(torch.float32), dim=-1)
        pred_conf = torch.gather(confs_full, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)
        eos_conf = _calculate_eos_confidence(
            logits, total_length, prompt_length, eos_check_tokens, eos_token_id,
        )

        block_end = min(current_pos + block_length, total_length)
        in_block = torch.zeros((1, x.shape[1]), dtype=torch.bool, device=device)
        in_block[:, current_pos:block_end] = True
        currently_masked = x == mask_token_id

        high_conf = (pred_conf > high_conf_threshold) & in_block & currently_masked & (pred != mask_token_id)

        if not bool(high_conf[0, current_pos:block_end].any()):
            valid = in_block[0] & currently_masked[0]
            candidates = torch.where(valid)[0]
            if candidates.numel() > 0:
                cand_conf = pred_conf[0, candidates]
                cand_tok = pred[0, candidates]
                sorted_conf, sort_idx = torch.sort(cand_conf, descending=True)
                best = -1
                for li in sort_idx.tolist():
                    if int(cand_tok[li].item()) != mask_token_id:
                        best = int(candidates[li].item())
                        break
                if best != -1:
                    high_conf[0, best] = True
                else:
                    stuck_logits = logits[0, candidates].clone().to(torch.float32)
                    stuck_logits[:, mask_token_id] = float("-inf")
                    new_p = F.softmax(stuck_logits, dim=-1)
                    new_c, new_t = torch.max(new_p, dim=-1)
                    best_local = int(torch.argmax(new_c).item())
                    best_pos = int(candidates[best_local].item())
                    pred[0, best_pos] = new_t[best_local]
                    high_conf[0, best_pos] = True

        expand_indices = torch.zeros((1, x.shape[1]), dtype=torch.bool, device=device)
        will_expand = (
            (not denoise_only_mode)
            and eos_conf < expand_eos_confidence_threshold
            and gen_length < max_gen_length
            and current_pos < total_length
        )
        if will_expand:
            potential = (pred_conf < low_conf_threshold) & in_block & currently_masked & (~high_conf)
            cand = torch.where(potential[0])[0]
            if cand.numel() > 0:
                cc = pred_conf[0, cand]
                _, lowest_local = torch.topk(cc, 1, largest=False)
                expand_indices[0, cand[lowest_local]] = True

        x_new = x.clone()
        x_new[high_conf] = pred[high_conf]

        if not bool(expand_indices.any()):
            x = x_new
        else:
            n_expand_positions = int(expand_indices.sum().item())
            new_gen_length = min(
                gen_length + n_expand_positions * (expansion_factor - 1),
                max_gen_length,
            )
            new_total_len = prompt_length + new_gen_length
            buf = torch.full((1, new_total_len), eos_token_id, dtype=torch.long, device=device)
            new_attn = torch.zeros((1, new_total_len), dtype=attn.dtype, device=device)

            buf[0, :prompt_length] = x_new[0, :prompt_length]
            new_attn[0, :prompt_length] = 1
            write_ptr = prompt_length
            for j in range(prompt_length, prompt_length + gen_length):
                if write_ptr >= new_total_len:
                    break
                if expand_indices[0, j]:
                    end_w = min(write_ptr + expansion_factor, new_total_len)
                    buf[0, write_ptr:end_w] = mask_token_id
                    new_attn[0, write_ptr:end_w] = 1
                    inserts_used += end_w - write_ptr
                    write_ptr = end_w
                else:
                    buf[0, write_ptr] = x_new[0, j]
                    new_attn[0, write_ptr] = 1
                    write_ptr += 1
            actual_new_gen = write_ptr - prompt_length
            x = buf[:, : prompt_length + actual_new_gen].contiguous()
            attn = new_attn[:, : prompt_length + actual_new_gen].contiguous()
            gen_length = actual_new_gen
            n_stage2_expansion_events += 1
            canvas_growth.append((iteration + 1, gen_length))

        total_length = prompt_length + gen_length
        block_end_after = min(current_pos + block_length, total_length)
        if not bool((x[0, current_pos:block_end_after] == mask_token_id).any()):
            current_pos = block_end_after

        iteration += 1

    elapsed = time.time() - started_at

    response_ids = x[0, prompt_length : prompt_length + gen_length].tolist()
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
        "method": "daedal_llada_1p5",
        "model_family": "llada_1p5",
        "seed": seed,
        "initial_gen_length": int(initial_gen_length),
        "max_gen_length": int(max_gen_length),
        "block_length": int(block_length),
        "expansion_factor": int(expansion_factor),
        "high_conf_threshold": high_conf_threshold,
        "low_conf_threshold": low_conf_threshold,
        "eos_confidence_threshold": eos_confidence_threshold,
        "expand_eos_confidence_threshold": expand_eos_confidence_threshold,
        "eos_check_tokens": int(eos_check_tokens),
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "answer_text": answer_text,
        "answer_token_length": len(answer_ids),
        "response_ids_all": response_ids,
        "full_response_text": full_response_text,
        "final_active_slots": gen_length,
        "fwd_count": fwd_count,
        "total_flops": total_flops,
        "inserts_used": inserts_used,
        "n_stage1_expansions": n_stage1_expansions,
        "n_stage2_expansion_events": n_stage2_expansion_events,
        "iterations": iteration,
        "elapsed_seconds": round(elapsed, 3),
        "canvas_growth": canvas_growth,
    }
