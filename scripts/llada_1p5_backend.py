"""llada_1p5_backend.py — LLaDA-1.5 backend for our CSG framework.

Drop-in replacement for llada_backend.py, targeting GSAI-ML/LLaDA-1.5.

Key differences from llada_backend.py:
  - MODEL_PATH = "GSAI-ML/LLaDA-1.5"
  - prepare_prompt() uses DAEDAL's `<reasoning> ` assistant prefix by default
    (matches DAEDAL/scripts/eval_LLaDA_1p5_*.sh).
  - All other kernels (forward_logits, reveal steps, insert_masks_at, etc.)
    are identical — same architecture, same tokenizer family, same special
    token IDs (verified: mask_id=126336, eos_id=126081, eot_id=126348).

Per-task vanilla defaults from LLaDA EVAL.md Table (LLaDA 1.5 section):
  GSM8K:    gen_length=256,  block_length=16,  confidence_eos_eot_inf=True
  MATH-500: gen_length=1024, block_length=128, confidence_eos_eot_inf=True
  HumanEval: gen_length=512, block_length=32,  confidence_eos_eot_inf=True
  MBPP:     gen_length=512,  block_length=32,  confidence_eos_eot_inf=True
  (logits_eos_inf=False for all tasks)

DAEDAL/CARVE-v2 runners use block_length=32 (our internal convention).
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


MODEL_PATH: str = "GSAI-ML/LLaDA-1.5"

# Same tokenizer family as LLaDA-8B-Instruct — identical special token IDs.
LLADA_MASK_ID: int = 126336
LLADA_EOT_ID: int = 126348
LLADA_EOS_ID: int = 126081


def load_model_and_tokenizer(model_path: str, device: str):
    """Load LLaDA-1.5 model + tokenizer. Forces left padding (per LLaDA convention)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == LLADA_MASK_ID:
        raise RuntimeError("LLaDA-1.5 tokenizer pad_token_id collides with mask_id; aborting.")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def prepare_prompt(
    tokenizer,
    prompt_text: str,
    *,
    assistant_prefix: Optional[str] = "<reasoning> ",
) -> torch.LongTensor:
    """Apply LLaDA-1.5 chat template with DAEDAL's `<reasoning> ` prefix by default.

    Matches DAEDAL/scripts/eval_LLaDA_1p5_*.sh's `assistant_prefix=<reasoning> `.
    Pass `assistant_prefix=None` to skip (reproduces LLaDA/chat.py raw behavior).

    Returns a (1, L) LongTensor.
    """
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if assistant_prefix:
        chat = chat + assistant_prefix
    enc = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"]


@torch.inference_mode()
def forward_logits(
    model,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LLaDA forward pass. Returns (B, L, V) logits in bfloat16."""
    return model(x, attention_mask=attention_mask).logits


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Per LLaDA's generate.py: float32 Gumbel noise. temp=0 → unchanged logits."""
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(num_masks: int, remaining_steps: int) -> int:
    """Adaptive per-step reveal count (ceil division). Drains everything when <=1 step."""
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
    """LLaDA-native reveal restricted to [block_start, block_end). Batch=1."""
    device = x.device
    L = x.shape[1]
    is_mask = x == mask_token_id

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
    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

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
    """DAEDAL's confidence-threshold reveal restricted to [block_start, block_end). Batch=1."""
    device = x.device
    L = x.shape[1]
    is_mask = x == mask_token_id

    logits = logits.clone()
    if logits_eos_inf:
        logits[:, :, eos_token_id] = float("-inf")

    logits_noisy = add_gumbel_noise(logits, temperature)
    pred = torch.argmax(logits_noisy, dim=-1)

    if confidence_eos_eot_inf:
        logits_for_conf = logits.clone()
        logits_for_conf[:, :, eos_token_id] = float("-inf")
        logits_for_conf[:, :, eot_token_id] = float("-inf")
        probs = F.softmax(logits_for_conf.to(torch.float32), dim=-1)
    else:
        probs = F.softmax(logits.to(torch.float32), dim=-1)
    conf = torch.gather(probs, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)

    in_block = torch.zeros((1, L), dtype=torch.bool, device=device)
    in_block[:, block_start:block_end] = True

    high_conf = (conf > high_conf_threshold) & in_block & is_mask & (pred != mask_token_id)

    if not bool(high_conf[0, block_start:block_end].any()):
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
    """Vanilla LLaDA-1.5 generation (identical algorithm to llada_backend.run_vanilla_llada)."""
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

            x0_p[:, block_end:] = float("-inf")
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
        "method": "vanilla_llada_1p5",
        "model_family": "llada_1p5",
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
    """Insert k masks at absolute position `insert_pos`. Returns (x_new, attn_new, k_inserted)."""
    cur_len = x.shape[1]
    k = min(k, max_total_len - cur_len)
    if k <= 0:
        return x, attn, 0
    new_chunk = torch.full((x.shape[0], k), mask_token_id, device=x.device, dtype=x.dtype)
    new_attn = torch.ones((x.shape[0], k), device=attn.device, dtype=attn.dtype)
    x_new = torch.cat([x[:, :insert_pos], new_chunk, x[:, insert_pos:]], dim=1)
    attn_new = torch.cat([attn[:, :insert_pos], new_attn, attn[:, insert_pos:]], dim=1)
    return x_new, attn_new, k
