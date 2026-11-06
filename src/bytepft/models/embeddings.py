"""Trainable byte and control embeddings for the reference architecture."""

import torch
from torch import nn

from .boundaries import validate_attention_mask


class ByteEmbedding(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 259, padding_idx: int = 256):
        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (hidden_size, vocab_size, padding_idx)
        ):
            raise ValueError("Embedding dimensions and padding index must be integers")
        if hidden_size < 1 or vocab_size <= padding_idx or padding_idx < 0:
            raise ValueError("Invalid embedding dimension, vocabulary size, or padding index")
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be an int64 [batch, length] tensor")
        hidden = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = input_ids != self.padding_idx
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        mask = validate_attention_mask(attention_mask)
        return hidden * mask.to(hidden.dtype).unsqueeze(-1)


# Plural spelling retained as a convenient component name.
ByteEmbeddings = ByteEmbedding
