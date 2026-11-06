"""Masked objectives for the reference byte model.

Decoder distillation compares probabilities of the same observed patch event,
never logits from different vocabularies. The complete reference training recipe
is experimental and is not a reproduction of the paper's architecture.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean() if values.numel() else values.sum()
    if mask.shape != values.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} must match {tuple(values.shape)}.")
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


class BoundaryLoss(nn.Module):
    """Binary cross entropy for boundaries after observed byte positions."""

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError("Boundary logits and targets must have the same shape.")
        if torch.any((targets != 0) & (targets != 1)):
            raise ValueError("Boundary targets must be binary.")
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
        return _masked_mean(loss, mask)


class EmbeddingAlignmentLoss(nn.Module):
    """Masked MSE over matching teacher/student representations (the n=0 baseline).

    Passing states after matching frozen global layers also supports n>0
    alignment; extracting and aligning those states is the caller's job.
    """

    def forward(self, student: Tensor, teacher: Tensor, mask: Tensor | None = None) -> Tensor:
        if student.shape != teacher.shape or student.ndim < 2:
            raise ValueError(
                "Teacher and student representations must have matching [..., hidden] shapes."
            )
        values = (student.float() - teacher.detach().float()).square().mean(dim=-1)
        return _masked_mean(values, mask)


class ByteCrossEntropyLoss(nn.Module):
    """Next-byte/control-symbol CE; labels are already shifted, with -100 ignored."""

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.ndim < 2 or logits.shape[:-1] != labels.shape:
            raise ValueError("Byte logits must have shape [..., vocabulary] matching labels [...].")
        if labels.dtype != torch.long:
            raise ValueError("Byte labels must have torch.long dtype.")
        valid = labels != -100
        if not valid.any():
            return logits.float().sum() * 0
        return F.cross_entropy(logits.float()[valid], labels[valid])


def _log1mexp(log_probability: Tensor) -> Tensor:
    # Both branches stay in domain because callers clamp log_probability below 0.
    return torch.where(
        log_probability < -math.log(2),
        torch.log1p(-torch.exp(log_probability)),
        torch.log(-torch.expm1(log_probability)),
    )


class DecoderDistillationLoss(nn.Module):
    """Temperature-modulated Bernoulli CE on aligned observed token/patch events.

    Inputs are scalar log probabilities per patch, including the boundary event
    on the student side, with shape [batch, patches]. They are not vocabulary
    distributions. Teacher targets are detached. This implements the paper's
    comparison function, independently of the reference architecture around it.
    """

    def __init__(self, temperature: float = 5.0) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Distillation temperature must be finite and positive.")
        self.temperature = temperature

    def forward(
        self,
        student_patch_log_probs: Tensor,
        teacher_token_log_probs: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if (
            student_patch_log_probs.ndim != 2
            or student_patch_log_probs.shape != teacher_token_log_probs.shape
        ):
            raise ValueError(
                "Distillation requires matching [batch, patches] scalar log-probability tensors."
            )
        for value in (student_patch_log_probs, teacher_token_log_probs):
            if not torch.isfinite(value).all() or torch.any(value > 0):
                raise ValueError(
                    "Distillation inputs must be finite non-positive log probabilities."
                )
        log_student = student_patch_log_probs.float() / self.temperature
        log_teacher = teacher_token_log_probs.detach().float() / self.temperature
        # A log probability of exactly zero otherwise produces log(0) and 0*inf.
        log_student = log_student.clamp_max(-torch.finfo(torch.float32).eps)
        target = log_teacher.exp()
        values = -(target * log_student + (-torch.expm1(log_teacher)) * _log1mexp(log_student))
        return _masked_mean(values, mask)
