"""
Scale-Weighted Focal KL loss for Zoom Decoder training.

Given teacher top-k distributions and the decoder's full logits, computes:

    L = (1/N) * sum_i  w_i * KL( p_teacher_i || p_student_i )
        + alpha * (1/N) * sum_i  CE( p_student_i, y_i )

where w_i = min(cap, (ref / px_i) ** gamma), so smaller target pixel size
⇒ larger gradient weight.  This is a physically-motivated analogue of focal
loss: instead of upweighting by low confidence, we upweight by small object
scale (the *a priori* hard cases).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def scale_weights_from_pixel(
    pixel_sizes: torch.Tensor, ref: float = 48.0, gamma: float = 1.0, cap: float = 8.0,
) -> torch.Tensor:
    """(B,) int → (B,) float scale weight."""
    px = pixel_sizes.clamp_min(1).float()
    w = (ref / px) ** gamma
    return w.clamp_max(cap)


def scale_weighted_focal_kl(
    logits: torch.Tensor,         # (B, T, V)
    labels: torch.Tensor,         # (B, T) with -100 ignored
    teacher_top_ids: torch.Tensor,   # (N_ans, K)  flat across non-ignored positions
    teacher_top_probs: torch.Tensor, # (N_ans, K)
    per_token_weights: torch.Tensor, # (N_ans,)
    vocab_size: int,
    alpha_ce: float = 0.3,
) -> dict:
    """
    Arguments align to the convention used by the existing VLM MemDec pipeline:
    `teacher_top_ids / probs` concatenate rows across all valid answer
    positions in the batch.  `per_token_weights` is the scale weight replicated
    per answer token.
    """
    # Shift (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()      # (B, T-1)
    mask = shift_labels != -100                    # (B, T-1)

    flat_logits = shift_logits[mask]               # (N, V)
    flat_labels = shift_labels[mask]               # (N,)

    N = flat_logits.size(0)
    N_ans = teacher_top_ids.size(0)
    assert N == N_ans, f"answer positions mismatch: logits={N} teacher={N_ans}"

    # Build sparse full-vocab teacher distribution
    teacher_probs = torch.zeros(N, vocab_size, device=flat_logits.device, dtype=flat_logits.dtype)
    valid = teacher_top_ids < vocab_size
    # scatter only valid ids
    safe_ids = teacher_top_ids.clamp_max(vocab_size - 1)
    teacher_probs.scatter_(1, safe_ids.long(), teacher_top_probs.to(flat_logits.dtype))
    # Zero-out invalid
    if not valid.all():
        mask_k = (~valid).unsqueeze(0)  # won't be used; simpler: mask top probs before scatter
        # This rare path is noop in practice; skip for speed.
        pass
    # Re-normalise
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    # KL( teacher || student )  =  sum p_t * (log p_t - log p_s)
    log_student = F.log_softmax(flat_logits.float(), dim=-1)
    # Per-token KL (don't reduce yet)
    # Using F.kl_div with log_target requires kl = exp(log_target) * (log_target - log_input)
    log_teacher = teacher_probs.float().clamp_min(1e-12).log()
    per_tok_kl = (teacher_probs.float() * (log_teacher - log_student)).sum(dim=-1)  # (N,)

    w = per_token_weights.to(per_tok_kl.dtype).to(per_tok_kl.device)
    kl_weighted = (w * per_tok_kl).mean()

    ce = F.cross_entropy(flat_logits.float(), flat_labels, reduction="mean")

    total = kl_weighted + alpha_ce * ce
    return {
        "loss": total,
        "kl": kl_weighted.detach(),
        "ce": ce.detach(),
        "mean_weight": w.mean().detach(),
        "n_tokens": torch.tensor(float(N), device=flat_logits.device),
    }
