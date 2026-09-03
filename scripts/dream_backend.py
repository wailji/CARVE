"""dream_backend.py — Dream-v0-Instruct-7B backend.

Model loading, the forward pass, and the reveal kernel shared by all three
Dream decoders (vanilla / DAEDAL / CARVE). The LLaDA counterpart is
scripts/llada_backend.py.

  load_model_and_tokenizer  AutoModel + AutoTokenizer, bfloat16
  forward_logits            one forward pass, matching Dream's own _sample()
  entropy_reveal_step       entropy-ordered reveal with the adaptive schedule
                            n_s = ceil(|M_s| / (T - s))  (paper Eq. 29)
  insert_masks              canvas expansion primitive used by CARVE
  _first_eos_in_response / _crop_to_eos   EOS-crop primitives
  _flop_ctx / _read_flops   FLOP counting around each forward
"""
from __future__ import annotations

import contextlib
import math
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Optional FLOP counting via torch's built-in dispatch-mode counter. Each decode
# is wrapped in this context so the returned record carries `total_flops`, summed
# over every forward pass. Falls back to a nullcontext (and total_flops=0) on
# torch versions without FlopCounterMode.
try:
    from torch.utils.flop_counter import FlopCounterMode  # type: ignore[attr-defined]
    _HAS_FLOP_COUNTER = True
except ImportError:
    FlopCounterMode = None  # type: ignore[assignment]
    _HAS_FLOP_COUNTER = False


def _flop_ctx():
    """Return a FlopCounterMode if available, else nullcontext."""
    if _HAS_FLOP_COUNTER:
        return FlopCounterMode(display=False)
    return contextlib.nullcontext()


def _read_flops(ctx) -> int:
    """Pull total FLOPs out of a (possibly null) FlopCounterMode."""
    if not _HAS_FLOP_COUNTER or isinstance(ctx, contextlib.nullcontext):
        return 0
    try:
        return int(ctx.get_total_flops())
    except Exception:
        return 0
from transformers import AutoModel, AutoTokenizer


# ───────────────────────────── infra ─────────────────────────────

def load_model_and_tokenizer(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def prepare_prompt(tokenizer, prompt: str) -> torch.LongTensor:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_attention(att1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tok_idx = att1d.long().cumsum(-1) - 1
    tok_idx.masked_fill_(att1d == 0, 1)
    att4d = torch.logical_and(
        att1d.unsqueeze(1).unsqueeze(-2),
        att1d.unsqueeze(1).unsqueeze(-1),
    )
    return att4d, tok_idx


@torch.inference_mode()
def forward_logits(model, x: torch.Tensor, att1d: torch.Tensor) -> torch.Tensor:
    """Forward pass matching Dream's `_sample` exactly.

    - When att1d has any zeros (real padding), build the 4D attention mask
      and explicit position ids and pass them to the model.
    - When att1d is all ones (no padding — our usual case), pass the string
      sentinel "full" and `position_ids=None`. This routes SDPA through the
      attn_mask=None kernel (different bf16 numerics than the all-True mask
      kernel), and lets the model fill in default position ids — exactly
      what `model.diffusion_generate` does in the no-padding branch.
    - Logits are kept in their native dtype (bfloat16 for Dream-7B) to match
      vanilla's numerics; previously we cast to float32 here, which made
      every downstream softmax/entropy/sample slightly different from Dream.
    """
    if bool((att1d == 0).any()):
        att4d, tok_idx = build_attention(att1d)
        logits = model(x, att4d, tok_idx).logits
    else:
        logits = model(x, "full", None).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


# ───────────────────────────── CARVE primitives ─────────────────────────────







def _first_eos_in_response(
    x: torch.Tensor,
    prompt_len: int,
    active_len: int,
    eos_token_id: int,
) -> Optional[int]:
    """Return 0-indexed position of the first EOS in the response slice, or None."""
    if active_len <= 0:
        return None
    response = x[0, prompt_len : prompt_len + active_len]
    eos_hits = (response == eos_token_id).nonzero(as_tuple=False)
    if eos_hits.numel() == 0:
        return None
    return int(eos_hits[0].item())


def _crop_to_eos(
    x: torch.Tensor,
    att1d: torch.Tensor,
    prompt_len: int,
    eos_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Truncate canvas to `prompt + first (eos_pos+1) response positions`.
    Drops every token (mask or committed) to the right of the EOS."""
    new_active_len = eos_pos + 1
    new_total = prompt_len + new_active_len
    return x[:, :new_total].contiguous(), att1d[:, :new_total].contiguous(), new_active_len




def _top_p_filter(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    drop = cum > top_p
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask = mask.scatter_(-1, sorted_idx, drop)
    return logits.masked_fill(mask, float("-inf"))


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Match Dream's top_k_logits: keep top-k by value, mask the rest with -inf."""
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def entropy_reveal_step(
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    step: int,
    total_steps: int,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Entropy-ordered reveal with the adaptive commit schedule.

    Commits ceil(M / remaining_steps) tokens per step, matched to the CURRENT
    mask count and the remaining budget (paper Eq. 29). This guarantees
    progress at every step and keeps pace with a canvas that grows mid-decode,
    which a static schedule keyed to the initial mask count cannot do.

    Token sampling and confidence ordering follow Dream:
      temperature > 0  -> multinomial sample from tempered + top_p logits;
                          confidence = neg-entropy of the SAME distribution.
      temperature == 0 -> argmax; confidence = neg-entropy of raw softmax.
    """
    # top_k hardcoded to 50 to match Dream's vanilla pipeline (which inherits
    # the HuggingFace GenerationConfig default top_k=50 and applies it inside
    # sample_tokens). We never override it in our gen_cfg, so vanilla actually
    # samples from top_p(0.9) ∩ top_k(50). Mirroring that here.
    TOP_K_DEFAULT = 50

    mask_index = x == mask_token_id
    if not mask_index.any():
        return x
    mask_logits = logits[mask_index]
    if temperature and temperature > 0:
        eff_logits = mask_logits / max(temperature, 1e-8)
        eff_logits = _top_p_filter(eff_logits, top_p)
        eff_logits = _top_k_filter(eff_logits, TOP_K_DEFAULT)
        probs = F.softmax(eff_logits, dim=-1)
        # Match Dream's sample_tokens: Categorical(probs).sample() with argmax
        # fallback when bf16 underflow makes probs not sum to 1 (a real edge
        # case after top_p+top_k filtering — Dream wraps this in try/except too).
        try:
            sampled_tokens = torch.distributions.Categorical(probs=probs).sample()
        except Exception:
            sampled_tokens = probs.argmax(dim=-1)
    else:
        # In greedy mode Dream still applies top_p / top_k to the logits before
        # argmax (because the filters mask logits to -inf, which can change which
        # token is the max). Mirror that.
        if top_p is not None and top_p < 1.0:
            mask_logits = _top_p_filter(mask_logits, top_p)
        mask_logits = _top_k_filter(mask_logits, TOP_K_DEFAULT)
        probs = F.softmax(mask_logits, dim=-1)
        sampled_tokens = probs.argmax(dim=-1)
    neg_entropy = (probs * probs.clamp_min(1e-10).log()).sum(dim=-1)

    num_mask = int(mask_index.sum().item())
    remaining_steps = max(1, total_steps - step)
    n_transfer = min(max(1, math.ceil(num_mask / remaining_steps)), num_mask)

    full_conf = torch.full(x.shape, float("-inf"), device=x.device, dtype=neg_entropy.dtype)
    full_conf[mask_index] = neg_entropy
    _, transfer_idx = torch.topk(full_conf, n_transfer, dim=-1)

    x_new = x.clone()
    sampled_full = torch.full_like(x, mask_token_id)
    sampled_full[mask_index] = sampled_tokens
    rows = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_idx)
    x_new[rows, transfer_idx] = sampled_full[rows, transfer_idx]
    return x_new


# ───────────────────────────── CARVE main loop ─────────────────────────────






# ───────────────────────────── CLI ─────────────────────────────



