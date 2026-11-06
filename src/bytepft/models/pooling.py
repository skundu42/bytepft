"""Differentiable pooling and explicit byte-to-patch alignment metadata."""

from dataclasses import dataclass

import torch
from torch import nn

from .boundaries import validate_attention_mask


@dataclass
class PatchMetadata:
    ends: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    patch_mask: torch.Tensor
    complete: torch.Tensor
    byte_to_patch: torch.Tensor
    attention_mask: torch.Tensor
    boundaries: torch.Tensor

    @property
    def patch_count(self) -> int:
        return int(self.patch_mask.sum().item())

    @property
    def average_bytes_per_patch(self) -> float:
        count = self.patch_count
        return float(self.attention_mask.sum().item()) / count if count else 0.0


class PatchPool(nn.Module):
    """Pool every valid position exactly once, retaining an incomplete suffix.

    The suffix is useful for coverage and training alignment but its global
    state must not be exposed by depooling until a real end boundary occurs.
    """

    def __init__(self, mode: str = "last"):
        super().__init__()
        if mode not in ("last", "mean"):
            raise ValueError("Pooling mode must be 'last' or 'mean'")
        self.mode = mode

    def forward(
        self,
        byte_hidden_states: torch.Tensor,
        boundaries: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, PatchMetadata]:
        mask = validate_attention_mask(attention_mask)
        if byte_hidden_states.ndim != 3 or byte_hidden_states.shape[:2] != mask.shape:
            raise ValueError("Byte hidden states must have shape [batch, length, hidden]")
        if boundaries.shape != mask.shape:
            raise ValueError("Boundaries and attention mask must have the same shape")
        if boundaries.dtype != torch.bool and not torch.all((boundaries == 0) | (boundaries == 1)):
            raise ValueError("Boundaries may contain only 0 and 1")
        boundary_mask = boundaries.bool() & mask
        batch, length, hidden = byte_hidden_states.shape
        valid_lengths = mask.sum(-1)
        all_ends: list[list[int]] = []
        for row in range(batch):
            row_ends = boundary_mask[row].nonzero(as_tuple=False).flatten().tolist()
            final = int(valid_lengths[row].item()) - 1
            if final >= 0 and (not row_ends or row_ends[-1] != final):
                row_ends.append(final)
            all_ends.append(row_ends)
        patches = max((len(row) for row in all_ends), default=0)
        device = byte_hidden_states.device
        starts = torch.zeros((batch, patches), dtype=torch.long, device=device)
        ends = torch.zeros_like(starts)
        lengths = torch.zeros_like(starts)
        patch_mask = torch.zeros((batch, patches), dtype=torch.bool, device=device)
        complete = torch.zeros_like(patch_mask)
        byte_to_patch = torch.full((batch, length), -1, dtype=torch.long, device=device)
        # Add a differentiable zero so even an empty tensor remains connected to
        # its inputs for callers testing gradients on a fully padded batch.
        pooled = byte_hidden_states.new_zeros((batch, patches, hidden))
        pooled = pooled + byte_hidden_states.sum() * 0
        for row, row_ends in enumerate(all_ends):
            start = 0
            for patch, end in enumerate(row_ends):
                starts[row, patch] = start
                ends[row, patch] = end
                lengths[row, patch] = end - start + 1
                patch_mask[row, patch] = True
                complete[row, patch] = boundary_mask[row, end]
                byte_to_patch[row, start : end + 1] = patch
                pooled[row, patch] = (
                    byte_hidden_states[row, end]
                    if self.mode == "last"
                    else byte_hidden_states[row, start : end + 1].mean(0)
                )
                start = end + 1
        metadata = PatchMetadata(
            ends=ends,
            starts=starts,
            lengths=lengths,
            patch_mask=patch_mask,
            complete=complete,
            byte_to_patch=byte_to_patch,
            attention_mask=mask,
            boundaries=boundary_mask,
        )
        return pooled, metadata


PatchPooling = PatchPool
