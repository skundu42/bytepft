"""CPU-friendly causal recurrent reference encoder (not the research mLSTM)."""

import torch
from torch import nn


class LocalByteEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[:2] != attention_mask.shape:
            raise ValueError("encoder expects embeddings [B,L,H] and attention_mask [B,L]")
        if embeddings.shape[1] == 0:
            return embeddings
        hidden, _ = self.step(embeddings)
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)

    def step(
        self, embeddings: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_state = self.rnn(embeddings, state)
        return self.norm(output + embeddings), next_state
