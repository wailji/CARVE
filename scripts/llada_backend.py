"""llada_backend.py — LLaDA-8B-Instruct backend for our CSG framework.

Mirrors the role of `scripts/dream_backend.py` for Dream:

  - load_model_and_tokenizer: AutoModel + AutoTokenizer (left-padding, bfloat16).
  - prepare_prompt: chat-template with optional `<reasoning> ` assistant prefix
    (DAEDAL convention — used for vanilla, CARVE, and DAEDAL on LLaDA).
  - forward_logits: single forward pass returning (1, L, V) logits, wrapped
    by an outer FlopCounter context (caller manages the context).
  - reveal step kernels:
      * llada_lowconf_reveal_step  — LLaDA's `low_confidence + num_transfer_tokens`
        schedule, restricted to a block range. Matches LLaDA's vanilla generate().
      * daedal_confthr_reveal_step — DAEDAL's confidence-threshold reveal
        (conf > 0.9 within block, single-position fallback otherwise).
  - shared helpers (re-exported from dream_backend where model-agnostic):
        set_seed, _flop_ctx, _read_flops, _first_eos_in_response, _crop_to_eos.

Constants (from LLaDA's tokenizer):
  - LLADA_MASK_ID = 126336
  - LLADA_EOT_ID  = 126348  (|EOT|)
  - LLADA_EOS_ID  = 126081  (|EOS|)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Re-export shared utilities so callers can import everything from one place.
from scripts.dream_backend import (  # noqa: F401
    set_seed,
    _flop_ctx,
    _read_flops,
    _first_eos_in_response,
    _crop_to_eos,
)


LLADA_MASK_ID: int = 126336
LLADA_EOT_ID: int = 126348
LLADA_EOS_ID: int = 126081


def load_model_and_tokenizer(model_path: str, device: str):
    """Load LLaDA model + tokenizer. Forces left padding (per LLaDA convention)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == LLADA_MASK_ID:
        raise RuntimeError("LLaDA tokenizer pad_token_id collides with mask_id; aborting.")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer




@torch.inference_mode()
def forward_logits(
    model,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LLaDA forward pass. Returns (B, L, V) logits in bfloat16.

    Unlike Dream, LLaDA's HF wrapper takes `(input_ids, attention_mask)` directly
    and returns `.logits` aligned 1:1 with input positions (no shift).
    """
    return model(x, attention_mask=attention_mask).logits


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Per LLaDA's `generate.py`: low-precision Gumbel reduces sample quality;
    use float32 (DAEDAL also uses float32). When temperature == 0, returns
    logits unchanged (downstream argmax is invariant to the exp-normalisation)."""
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(num_masks: int, remaining_steps: int) -> int:
    """Adaptive per-step reveal count for LLaDA's `low_confidence` schedule.

    Matches the spirit of LLaDA's `get_num_transfer_tokens` but computed
    per-step adaptively (ceil(num_masks / remaining_steps)) so it works with
    expanding canvases too. When remaining_steps is 0/1, drains everything.
    """
    if num_masks <= 0:
        return 0
    if remaining_steps <= 1:
        return num_masks
    return -(-num_masks // remaining_steps)  # ceil division


@torch.inference_mode()
def llada_lowconf_reveal_step(
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    block_start: int,
    block_end: int,
    remaining_steps_in_block: int,
    *,
    temperature: float = 0.0,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: int = LLADA_EOS_ID,
    eot_token_id: int = LLADA_EOT_ID,
) -> torch.Tensor:
    """LLaDA-native reveal restricted to [block_start, block_end).

    Procedure (per LLaDA generate.py:74-118):
      1. predicted_tokens = argmax(gumbel_noise(logits))
      2. score = softmax(logits)[predicted_token]                   # confidence
      3. mask out positions outside the block (-inf)                # block-restrict
      4. num_to_reveal = ceil(masks_in_block / remaining_steps)
      5. reveal top-`num_to_reveal` masked positions by score.

    Optional flags mirror vanilla run_vanilla_llada (LLaDA appendix B.4):
      - logits_eos_inf: set logits[:,:,EOS] = -inf BEFORE gumbel/argmax
      - confidence_eos_eot_inf: use a separate logits clone with EOS+EOT zeroed
        for the per-position confidence score that drives reveal order
        (prevents the model from committing EOS/EOT-confident positions early,
        which is critical for math/long-reasoning tasks).

    Returns the updated x. Operates on batch=1.
    """
    device = x.device
    L = x.shape[1]
    is_mask = x == mask_token_id

    # forward_logits returns an inference-mode tensor; clone before any in-place
    # modification (mirrors the same pattern in run_vanilla_llada).
    logits = logits.clone()
    if logits_eos_inf:
        logits[:, :, eos_token_id] = float("-inf")

    logits_noisy = add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_noisy, dim=-1)  # (B, L)

    if confidence_eos_eot_inf:
        logits_for_conf = logits.clone()
        logits_for_conf[:, :, eos_token_id] = float("-inf")
        logits_for_conf[:, :, eot_token_id] = float("-inf")
        probs = F.softmax(logits_for_conf.to(torch.float32), dim=-1)
    else:
        probs = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # Restrict to current block AND currently-masked positions
    out_of_block = torch.ones((1, L), dtype=torch.bool, device=device)
    out_of_block[:, block_start:block_end] = False
    x0_p = x0_p.masked_fill(out_of_block, float("-inf"))
    x0_p = x0_p.masked_fill(~is_mask, float("-inf"))

    masks_in_block = int((is_mask[0, block_start:block_end]).sum().item())
    if masks_in_block == 0:
        return x

    num_to_reveal = get_num_transfer_tokens(masks_in_block, remaining_steps_in_block)
    if num_to_reveal <= 0:
        return x

    _, sel_idx = torch.topk(x0_p[0], k=num_to_reveal, dim=-1)
    x_new = x.clone()
    x_new[0, sel_idx] = x0[0, sel_idx]
    return x_new


@torch.inference_mode()
def daedal_confthr_reveal_step(
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    block_start: int,
    block_end: int,
    *,
    high_conf_threshold: float = 0.9,
    temperature: float = 0.0,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    eos_token_id: int = LLADA_EOS_ID,
    eot_token_id: int = LLADA_EOT_ID,
) -> torch.Tensor:
    """DAEDAL's confidence-threshold reveal restricted to [block_start, block_end).

    Mirrors `LLaDA_DAEDAL.generate()` lines 167-205:
      1. predicted_tokens = argmax(gumbel_noise(logits))
      2. confidences = softmax(logits)[predicted_token]
      3. high_conf = (confidences > 0.9) & in_block & is_mask & (pred != mask_id)
      4. If no high_conf in the block: take the single best non-mask candidate;
         if all candidates predict mask_id, re-pick after zeroing mask logit.

    Optional flags mirror run_vanilla_llada (see llada_lowconf_reveal_step for
    details). Off by default to keep DAEDAL's published behaviour intact.

    Operates on batch=1.
    """
    device = x.device
    L = x.shape[1]
    is_mask = x == mask_token_id

    logits = logits.clone()
    if logits_eos_inf:
        logits[:, :, eos_token_id] = float("-inf")

    logits_noisy = add_gumbel_noise(logits, temperature)
    pred = torch.argmax(logits_noisy, dim=-1)  # (B, L)

    if confidence_eos_eot_inf:
        logits_for_conf = logits.clone()
        logits_for_conf[:, :, eos_token_id] = float("-inf")
        logits_for_conf[:, :, eot_token_id] = float("-inf")
        probs = F.softmax(logits_for_conf.to(torch.float32), dim=-1)
    else:
        probs = F.softmax(logits.to(torch.float32), dim=-1)
    conf = torch.gather(probs, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)  # (B, L)

    in_block = torch.zeros((1, L), dtype=torch.bool, device=device)
    in_block[:, block_start:block_end] = True

    high_conf = (conf > high_conf_threshold) & in_block & is_mask & (pred != mask_token_id)

    if not bool(high_conf[0, block_start:block_end].any()):
        # Fallback: pick the single best masked position in the block.
        valid = in_block[0] & is_mask[0]
        candidates = torch.where(valid)[0]
        if candidates.numel() == 0:
            return x
        cand_conf = conf[0, candidates]
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
            new_probs = F.softmax(stuck_logits, dim=-1)
            new_conf, new_tok = torch.max(new_probs, dim=-1)
            best_local = int(torch.argmax(new_conf).item())
            best_pos = int(candidates[best_local].item())
            pred[0, best_pos] = new_tok[best_local]
            high_conf[0, best_pos] = True

    x_new = x.clone()
    x_new[high_conf] = pred[high_conf]
    return x_new


def run_vanilla_llada(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,    # (1, P)
    *,
    seed: int,
    gen_length: int = 512,
    steps: Optional[int] = None,
    block_length: Optional[int] = None,
    cfg_scale: float = 0.0,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    mask_token_id: int = LLADA_MASK_ID,
    eos_token_id: int = LLADA_EOS_ID,
    eot_token_id: int = LLADA_EOT_ID,
) -> dict:
    """Vanilla LLaDA generation per LLaDA's `generate.py`.

    Implements the exact algorithm from `LLaDA/generate.py::generate()` for batch=1:
      - x = [prompt | MASK * gen_length]
      - block_length defaults to gen_length (no semi-AR — LLaDA Tab 1 setting)
      - steps defaults to gen_length
      - per-block num_transfer_tokens schedule from get_num_transfer_tokens
      - low_confidence remasking via softmax(logits)[predicted_token]
      - logits_eos_inf / confidence_eos_eot_inf appendix-B.4 flags supported

    FLOPs counter wraps every forward; returned dict matches the run_carve* shape.
    """
    set_seed(seed)
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    prompt_length = prompt_ids.shape[1]

    if steps is None:
        steps = gen_length
    if block_length is None:
        block_length = gen_length
    if gen_length % block_length != 0:
        raise ValueError(f"gen_length ({gen_length}) must be divisible by block_length ({block_length})")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(f"steps ({steps}) must be divisible by num_blocks ({num_blocks})")
    steps_per_block = steps // num_blocks

    import time as _time
    t0 = _time.time()
    flop_ctx = _flop_ctx()
    flop_ctx.__enter__()

    x = torch.full((1, prompt_length + gen_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt_ids
    attn = torch.ones((1, prompt_length + gen_length), device=device, dtype=torch.long)
    prompt_index = (x != mask_token_id)
    fwd_count = 0

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_token_id)
        # Precompute per-step transfer counts for this block (LLaDA's static schedule).
        masks_in_block = int(block_mask_index.sum().item())
        base = masks_in_block // steps_per_block
        rem = masks_in_block % steps_per_block
        num_transfer = [base + (1 if i < rem else 0) for i in range(steps_per_block)]

        for i in range(steps_per_block):
            if num_transfer[i] <= 0:
                continue
            mask_index = (x == mask_token_id)
            if not bool(mask_index.any()):
                break
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_token_id
                x_ = torch.cat([x, un_x], dim=0)
                un_attn = attn.clone()
                attn_ = torch.cat([attn, un_attn], dim=0)
                logits = model(x_, attention_mask=attn_).logits
                fwd_count += 1
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = forward_logits(model, x, attn)
                fwd_count += 1

            # forward_logits is wrapped in inference_mode, so its return tensor is
            # an inference-tensor and cannot be mutated in place. Clone before any
            # in-place operation (logits_eos_inf / confidence_eos_eot_inf flags).
            logits = logits.clone()
            if logits_eos_inf:
                logits[:, :, eos_token_id] = float("-inf")

            logits_noisy = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            if confidence_eos_eot_inf:
                logits_for_conf = logits.clone()
                logits_for_conf[:, :, eos_token_id] = float("-inf")
                logits_for_conf[:, :, eot_token_id] = float("-inf")
                p = F.softmax(logits_for_conf.to(torch.float32), dim=-1)
            else:
                p = F.softmax(logits.to(torch.float32), dim=-1)

            if remasking == "low_confidence":
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
            else:
                raise NotImplementedError(remasking)

            # Mask out positions outside the current block (no future-block reveal)
            x0_p[:, block_end:] = float("-inf")
            # Score = confidence on currently-masked positions only; non-mask = -inf
            x0_p = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

            k = int(num_transfer[i])
            if k <= 0:
                continue
            _, sel = torch.topk(x0_p[0], k=k)
            x[0, sel] = x0[0, sel]

    flop_ctx.__exit__(None, None, None)
    total_flops = _read_flops(flop_ctx)
    elapsed = _time.time() - t0

    response_ids = x[0, prompt_length:].tolist()
    answer_ids: List[int] = []
    for tid in response_ids:
        if tid == eos_token_id:
            break
        answer_ids.append(tid)
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    full_response_text = tokenizer.decode(response_ids, skip_special_tokens=False)

    return {
        "method": "vanilla_llada",
        "model_family": "llada",
        "seed": seed,
        "gen_length": int(gen_length),
        "steps": int(steps),
        "block_length": int(block_length),
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "remasking": remasking,
        "logits_eos_inf": bool(logits_eos_inf),
        "confidence_eos_eot_inf": bool(confidence_eos_eot_inf),
        "answer_text": answer_text,
        "answer_token_length": len(answer_ids),
        "response_ids_all": response_ids,
        "full_response_text": full_response_text,
        "final_active_slots": int(gen_length),
        "fwd_count": fwd_count,
        "total_flops": total_flops,
        "elapsed_seconds": round(elapsed, 3),
    }


def insert_masks_at(
    x: torch.Tensor,
    attn: torch.Tensor,
    insert_pos: int,
    k: int,
    mask_token_id: int,
    max_total_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Insert k masks at absolute position `insert_pos`. Returns (x_new, attn_new, k_inserted).

    `attn` must be (B, L) with 1s for real tokens. New mask positions also get attn=1.
    """
    cur_len = x.shape[1]
    k = min(k, max_total_len - cur_len)
    if k <= 0:
        return x, attn, 0
    new_chunk = torch.full((x.shape[0], k), mask_token_id, device=x.device, dtype=x.dtype)
    new_attn = torch.ones((x.shape[0], k), device=attn.device, dtype=attn.dtype)
    x_new = torch.cat([x[:, :insert_pos], new_chunk, x[:, insert_pos:]], dim=1)
    attn_new = torch.cat([attn[:, :insert_pos], new_attn, attn[:, insert_pos:]], dim=1)
    return x_new, attn_new, k
