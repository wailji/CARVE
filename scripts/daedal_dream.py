"""daedal_dream.py — DAEDAL training-free variable-length denoising on Dream.

Direct port of `DAEDAL/models/LLaDA_DAEDAL.py::generate()` (LLaDA's algorithm,
LLaDA's hyperparameters), adapted to Dream:

  - swap LLaDA's `model(x, attention_mask=...).logits` for Dream's
    `forward_logits(model, x, att1d)` from scripts.dream_backend (handles the
    "full" sentinel for unpadded canvases AND Dream's logits-shift convention)
  - swap LLaDA's mask_id (126336) → Dream's mask_id (defaults to tokenizer.mask_token_id)
  - swap LLaDA's eos_token_id (126081) → Dream's eos_token_id
    (defaults to tokenizer.eos_token_id)
  - keep ALL other DAEDAL hyperparameters at paper defaults

Single-sequence (batch=1) for parity with our existing CARVE runners.

DAEDAL hyperparameter defaults (Li et al. 2025 + DAEDAL/scripts/eval_LLaDA_DAEDAL.sh):
  initial_gen_length = 64
  max_gen_length     = 2048
  block_length       = 32
  expansion_factor   = 8
  high_conf_threshold       = 0.9
  low_conf_threshold        = 0.1
  eos_confidence_threshold  = 0.5    # initial-length probe
  expand_eos_confidence_threshold = 0.9   # stop expanding when EOS conf clears this
  eos_check_tokens   = 32

FLOPs counter wraps the entire run via `_flop_ctx()`.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from scripts.dream_backend import (
    forward_logits,
    set_seed,
    _flop_ctx,
    _read_flops,
)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _calculate_eos_confidence(
    logits: torch.Tensor,         # (1, L, V)
    total_length: int,            # absolute position one-past-end of valid canvas
    prompt_length: int,
    eos_check_tokens: int,
    eos_token_id: int,
) -> float:
    """Average predicted-EOS confidence over the LAST `eos_check_tokens` positions
    in the response that the model PREDICTS as EOS. Mirrors DAEDAL's _calculate_eos_confidence."""
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


def run_daedal_dream(
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
    max_iterations: int = 4096,    # safety bound on Stage-2 loop
    mask_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Dict:
    """DAEDAL on Dream. Single-sequence (batch=1).

    Stage 1: Initial Length Adjustment
      Repeat: forward(canvas) → AVG predicted-EOS conf over last `eos_check_tokens`
              if AVG < eos_confidence_threshold AND gen_length < max_gen_length:
                expand by `expansion_factor` masks at the tail
              else: stop

    Intermediate: pad with EOS for `eos_check_tokens // 2` positions then re-mask
    the original gen region (mirrors DAEDAL's intermediate_x_tensor block).

    Stage 2: Iterative Denoising and Mask Insertion
      Per iter: forward(canvas) → confidences
        Reveal positions in current block where conf > 0.9 (fallback: best-conf in block)
        At positions in current block where conf < 0.1: mark for expansion
            (cap at 1 expansion per iter per DAEDAL's `num_to_expand = min(1, ...)` line)
        Insert `expansion_factor` masks at each marked position
        Advance current_pos when block is fully filled (no MASK left)
      Stop when current_pos passes total_length, or stagnant.
    """
    if mask_token_id is None:
        mask_token_id = int(tokenizer.mask_token_id)
    if eos_token_id is None:
        eos_token_id = int(tokenizer.eos_token_id)

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

    # ─── Stage 1: Initial Length Adjustment ───
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

    # ─── Intermediate buffer (DAEDAL adds eos_check_tokens // 2 EOS at tail) ───
    extra = min(max_gen_length - gen_length, eos_check_tokens // 2)
    if extra > 0:
        new_chunk = torch.full((1, extra), eos_token_id, device=device, dtype=torch.long)
        new_attn = torch.ones((1, extra), device=device, dtype=torch.long)
        x = torch.cat([x, new_chunk], dim=1)
        attn = torch.cat([attn, new_attn], dim=1)
        gen_length += extra

    # ─── Stage 2: Iterative Denoising and Mask Insertion ───
    current_pos = prompt_length
    denoise_only_mode = False
    last_x: Optional[torch.Tensor] = None

    iteration = 0
    while iteration < max_iterations:
        total_length = prompt_length + gen_length
        if current_pos >= total_length:
            break
        if last_x is not None and last_x.shape == x.shape and torch.equal(x, last_x):
            break  # stagnant
        last_x = x.clone()

        # Enter denoise-only mode if max length reached.
        if gen_length >= max_gen_length and not denoise_only_mode:
            denoise_only_mode = True

        logits = forward_logits(model, x, attn)
        fwd_count += 1
        pred = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)            # (1, L)
        confs_full = F.softmax(logits.to(torch.float32), dim=-1)
        pred_conf = torch.gather(confs_full, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)  # (1, L)
        eos_conf = _calculate_eos_confidence(
            logits, total_length, prompt_length, eos_check_tokens, eos_token_id,
        )

        block_end = min(current_pos + block_length, total_length)
        in_block = torch.zeros((1, x.shape[1]), dtype=torch.bool, device=device)
        in_block[:, current_pos:block_end] = True
        currently_masked = x == mask_token_id

        high_conf = (pred_conf > high_conf_threshold) & in_block & currently_masked & (pred != mask_token_id)

        # Fallback: if no position in the block clears 0.9, take the best non-mask candidate
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

        # Expansion candidates: low-conf masked positions in the block
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
                # Per DAEDAL: pick the SINGLE lowest-conf position to expand at this iter
                _, lowest_local = torch.topk(cc, 1, largest=False)
                expand_indices[0, cand[lowest_local]] = True

        # Apply reveal first.
        x_new = x.clone()
        x_new[high_conf] = pred[high_conf]

        if not bool(expand_indices.any()):
            x = x_new
        else:
            # Apply expansion: rebuild canvas inserting expansion_factor masks at each expand position
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

        # Advance current_pos if current block is fully filled
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
        "method": "daedal_dream",
        "model_family": "dream",
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
