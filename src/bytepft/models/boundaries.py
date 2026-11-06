"""Causal end-of-patch decisions for the experimental reference baseline.

These boundaries are decided after observing a byte. They intentionally do
not implement a lookahead/fused-symbol boundary formulation.
"""

from __future__ import annotations

import torch
from torch import nn


def validate_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, length]")
    if attention_mask.dtype != torch.bool and not torch.all(
        (attention_mask == 0) | (attention_mask == 1)
    ):
        raise ValueError("attention_mask may contain only 0 and 1")
    mask = attention_mask.bool()
    if mask.shape[1] > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("Only right-padded attention masks are supported")
    return mask


def constrain_boundaries(
    proposed: torch.Tensor,
    attention_mask: torch.Tensor,
    min_patch_size: int = 1,
    max_patch_size: int = 16,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply length constraints in byte order without closing an open suffix.

    BOS/EOS form a boundary even when a minimum patch length has not elapsed.
    No decision depends on the length of a later suffix in the batch.
    """
    mask = validate_attention_mask(attention_mask)
    if proposed.shape != mask.shape:
        raise ValueError("Boundary proposals and attention mask must have the same shape")
    if proposed.dtype != torch.bool and not torch.all((proposed == 0) | (proposed == 1)):
        raise ValueError("Boundary proposals may contain only 0 and 1")
    for value in (min_patch_size, max_patch_size):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Patch size constraints must be integers")
    if not 1 <= min_patch_size <= max_patch_size:
        raise ValueError("Require 1 <= min_patch_size <= max_patch_size")
    if input_ids is not None and input_ids.shape != mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    result = torch.zeros_like(mask)
    lengths = torch.zeros(mask.shape[0], dtype=torch.long, device=mask.device)
    for position in range(mask.shape[1]):
        valid = mask[:, position]
        lengths = lengths + valid.long()
        control = (
            (input_ids[:, position] == 257) | (input_ids[:, position] == 258)
            if input_ids is not None
            else torch.zeros_like(valid)
        )
        chosen = valid & (
            control
            | (lengths >= max_patch_size)
            | (proposed[:, position].bool() & (lengths >= min_patch_size))
        )
        result[:, position] = chosen
        lengths = torch.where(chosen, 0, lengths)
    return result


def fixed_boundaries(
    attention_mask: torch.Tensor,
    patch_size: int = 4,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    return constrain_boundaries(
        torch.zeros_like(attention_mask, dtype=torch.bool),
        attention_mask,
        min_patch_size=1,
        max_patch_size=patch_size,
        input_ids=input_ids,
    )


class BoundaryPredictor(nn.Module):
    """Learned causal MLP plus explicit fixed and supervised oracle modes."""

    def __init__(
        self,
        hidden_size: int,
        mode: str = "learned",
        patch_size: int = 4,
        min_patch_size: int = 1,
        max_patch_size: int = 16,
        threshold: float = 0.5,
    ):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        for name, value in (
            ("hidden_size", hidden_size),
            ("patch_size", patch_size),
            ("min_patch_size", min_patch_size),
            ("max_patch_size", max_patch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if mode not in ("fixed", "oracle", "learned"):
            raise ValueError("mode must be 'fixed', 'oracle', or 'learned'")
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between zero and one")
        if not 1 <= min_patch_size <= max_patch_size:
            raise ValueError("Require 1 <= min_patch_size <= max_patch_size")
        if not min_patch_size <= patch_size <= max_patch_size:
            raise ValueError("patch_size must fall within the min/max patch constraints")
        self.mode = mode
        self.patch_size = patch_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.threshold = threshold
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1)
        )

    def scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, length, hidden]")
        return self.mlp(hidden_states).squeeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        oracle_boundaries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.scores(hidden_states)
        mask = validate_attention_mask(attention_mask)
        if logits.shape != mask.shape:
            raise ValueError("Hidden sequence and attention mask shapes do not agree")
        if self.mode == "fixed":
            boundaries = fixed_boundaries(mask, self.patch_size, input_ids)
        elif self.mode == "oracle":
            if oracle_boundaries is None:
                raise ValueError("Oracle patching requires source-token boundary supervision")
            boundaries = constrain_boundaries(
                oracle_boundaries,
                mask,
                self.min_patch_size,
                self.max_patch_size,
                input_ids,
            )
        else:
            boundaries = constrain_boundaries(
                logits.sigmoid() >= self.threshold,
                mask,
                self.min_patch_size,
                self.max_patch_size,
                input_ids,
            )
        return logits.masked_fill(~mask, 0), boundaries
