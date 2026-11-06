"""Small contracts for pretrained global transformer adapters."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class GlobalOutput:
    """Patch-aligned final transformer states and an optional HF KV cache."""

    hidden_states: torch.Tensor
    past_key_values: Any = None


@dataclass
class TeacherOutput:
    """Teacher outputs in the source tokenizer's representation space."""

    logits: torch.Tensor
    hidden_states: torch.Tensor


class GlobalAdapter(nn.Module):
    """Common parameter-freezing behavior for global transformer adapters."""

    hidden_size: int
    source_model_id: str | None
    source_revision: str | None
    kind: str
    original_parameter_count: int

    def freeze(self) -> None:
        self.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        self.requires_grad_(True)
