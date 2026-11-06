"""Expose only completed-patch context to the byte decoder."""

import torch
from torch import nn

from .pooling import PatchMetadata


class PatchDepool(nn.Module):
    """At observed byte t, return the latest patch whose end is <= t.

    Logits at position t predict byte t+1. A patch ending at t therefore may
    inform those logits, while an incomplete current patch may not.
    """

    def forward(self, patch_hidden_states: torch.Tensor, metadata: PatchMetadata) -> torch.Tensor:
        if patch_hidden_states.ndim != 3:
            raise ValueError("Patch hidden states must have shape [batch, patches, hidden]")
        batch, patches, hidden = patch_hidden_states.shape
        if patch_hidden_states.shape[:2] != metadata.patch_mask.shape:
            raise ValueError("Patch states and alignment metadata have incompatible shapes")
        length = metadata.attention_mask.shape[1]
        if patches == 0:
            return patch_hidden_states.new_zeros((batch, length, hidden)) + (
                patch_hidden_states.sum() * 0
            )
        # In each row all true boundaries precede the optional incomplete suffix.
        # Their cumulative count identifies the most recent completed patch.
        indices = metadata.boundaries.long().cumsum(-1) - 1
        gathered = patch_hidden_states.gather(
            1, indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden)
        )
        valid = (indices >= 0) & metadata.attention_mask
        return gathered * valid.to(gathered.dtype).unsqueeze(-1)


PatchDepooling = PatchDepool
