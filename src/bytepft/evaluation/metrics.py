"""Likelihood normalized by raw UTF-8 bytes, excluding document controls."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bytepft.training.datasets import ByteDataset, collate_documents

if TYPE_CHECKING:
    from bytepft.models.byte_model import ByteModel


@torch.no_grad()
def evaluate_model(
    model: ByteModel,
    documents: Sequence[str],
    *,
    max_bytes: int | None = 256,
    batch_size: int = 2,
) -> dict[str, float | int]:
    """Evaluate each document independently; preserve Unicode on truncation.

    Byte probabilities are taken from the complete 259-way head. BOS, EOS and
    padding are excluded from the denominator and from loss summation, so this
    metric should not be confused with cross-tokenizer token perplexity.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if model.config.boundary_mode == "oracle":
        raise ValueError(
            "Evaluation without a source tokenizer requires fixed or learned boundaries."
        )
    dataset = ByteDataset(documents, max_bytes=max_bytes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_documents)
    total_nll = 0.0
    byte_count = 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            ids = batch["input_ids"].to(model.device)
            mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            logits = model(ids, mask).logits.float()
            raw = (labels >= 0) & (labels <= 255)
            count = int(raw.sum().item())
            if count:
                total_nll += float(
                    F.cross_entropy(logits[raw], labels[raw], reduction="sum").item()
                )
                byte_count += count
    finally:
        model.train(was_training)
    if byte_count == 0:
        raise ValueError("Evaluation requires at least one raw UTF-8 byte after truncation.")
    nll = total_nll / byte_count
    if not math.isfinite(nll):
        raise ValueError("Evaluation produced nonfinite byte likelihoods.")
    return {
        "nll": nll,
        "bits_per_byte": nll / math.log(2),
        "byte_count": byte_count,
        "document_count": len(dataset),
    }
