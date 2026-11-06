"""Independent byte/control output head for the reference architecture."""

import torch
from torch import nn


class BytePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 259) -> None:
        super().__init__()
        if hidden_size < 1 or vocab_size < 256:
            raise ValueError("invalid byte head dimensions")
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)
