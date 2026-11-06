"""Causal local decoder with exact recurrent incremental state."""

import torch
from torch import nn


class LocalByteDecoder(nn.Module):
    def __init__(self, hidden_size: int, global_hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        if min(hidden_size, global_hidden_size, num_layers) < 1:
            raise ValueError("decoder dimensions and depth must be positive")
        self.local_projection = nn.Linear(hidden_size, hidden_size)
        self.context_projection = nn.Linear(global_hidden_size, hidden_size, bias=False)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, local_features: torch.Tensor, context: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if local_features.shape[:2] != context.shape[:2]:
            raise ValueError("local and global context sequence dimensions must agree")
        if attention_mask.shape != local_features.shape[:2]:
            raise ValueError("decoder attention_mask shape must match local features")
        hidden, _ = self.step(local_features, context)
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)

    def step(
        self,
        local_features: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = self.local_projection(local_features) + self.context_projection(context)
        output, next_state = self.rnn(combined, state)
        return self.norm(output + combined), next_state
