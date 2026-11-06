"""Document-preserving local/Hugging Face loading and deterministic collation."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258


def load_texts(
    source: str | Path | Sequence[str] | None = None,
    *,
    text_field: str = "text",
    hf_dataset: str | None = None,
    split: str = "train",
) -> list[str]:
    """Read .txt documents, JSONL records, directories, or an explicit HF dataset.

    Each UTF-8 text file is one document; each JSONL row is one document. Input
    documents are never concatenated. HF loading requires the ``training`` extra.
    A nonexistent local path is never silently interpreted as a dataset name.
    """
    if hf_dataset is not None:
        if source is not None:
            raise ValueError("Provide either a local source or hf_dataset, not both.")
        try:
            load_dataset = import_module("datasets").load_dataset
        except ImportError as exc:
            raise ImportError(
                "Hugging Face datasets require: pip install 'bytepft[training]'."
            ) from exc
        dataset = load_dataset(hf_dataset, split=split)
        try:
            values = list(dataset[text_field])
        except KeyError as exc:
            raise ValueError(f"HF dataset has no text field {text_field!r}.") from exc
        return _validate_texts(values)
    if source is None:
        raise ValueError("A local source, sequence of documents, or hf_dataset is required.")
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return _validate_texts(list(source))
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".jsonl"}
        )
        if not files:
            raise ValueError(f"No .txt or .jsonl documents found in {path}.")
        return [text for file in files for text in load_texts(file, text_field=text_field)]
    if path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8", newline="") as handle:
            return [handle.read()]
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Unsupported dataset format {path.suffix!r}; expected .txt or .jsonl.")
    documents: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc.msg}") from exc
            if (
                not isinstance(item, dict)
                or text_field not in item
                or not isinstance(item[text_field], str)
            ):
                raise ValueError(
                    f"Expected string field {text_field!r} in {path}, line {line_number}."
                )
            documents.append(item[text_field])
    return documents


def _validate_texts(values: list[Any]) -> list[str]:
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Every dataset document must be a string.")
    return values


def deterministic_split(
    documents: Sequence[str], validation_fraction: float = 0.1, *, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split at document boundaries with a private seeded random generator."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0, 1).")
    _validate_texts(list(documents))
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    count = 0
    if validation_fraction > 0 and len(indices) > 1:
        count = max(1, min(len(indices) - 1, round(len(indices) * validation_fraction)))
    validation = [documents[idx] for idx in indices[:count]]
    train = [documents[idx] for idx in indices[count:]]
    return train, validation


class ByteDataset(Dataset[dict[str, Any]]):
    """One BOS + UTF-8 document + EOS per item, with already-shifted labels.

    A byte limit keeps the longest complete UTF-8 prefix within that budget.
    ``text`` remains exactly aligned with the bytes for source-token supervision.
    """

    def __init__(self, documents: Sequence[str], max_bytes: int | None = None) -> None:
        _validate_texts(list(documents))
        if max_bytes is not None and (
            isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer or None.")
        self.texts = [
            text
            if max_bytes is None
            else text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
            for text in documents
        ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text = self.texts[index]
        byte_ids = list(text.encode("utf-8"))
        ids = torch.tensor([BOS_ID, *byte_ids, EOS_ID], dtype=torch.long)
        labels = torch.tensor([*byte_ids, EOS_ID, -100], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "labels": labels,
            "text": text,
            "byte_count": len(byte_ids),
        }


def collate_documents(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad independent documents; retain text/counts for teacher alignment."""
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    max_length = max(item["input_ids"].numel() for item in items)
    ids = torch.full((len(items), max_length), PAD_ID, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, item in enumerate(items):
        length = item["input_ids"].numel()
        ids[index, :length] = item["input_ids"]
        labels[index, :length] = item["labels"]
        mask[index, :length] = item["attention_mask"]
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": labels,
        "texts": [item["text"] for item in items],
        "byte_counts": [item["byte_count"] for item in items],
    }


def source_boundary_targets(
    input_ids: Tensor, token_ends: Sequence[Sequence[int]]
) -> tuple[Tensor, Tensor]:
    """Binary boundary supervision and mask at BOS-offset raw-byte positions."""
    if input_ids.ndim != 2 or len(token_ends) != input_ids.shape[0]:
        raise ValueError("Expected matching [batch, length] IDs and source alignments.")
    targets = torch.zeros_like(input_ids, dtype=torch.float32)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, ends in enumerate(token_ends):
        previous = -1
        for end in ends:
            if (
                isinstance(end, bool)
                or not isinstance(end, int)
                or end <= previous
                or end + 1 >= input_ids.shape[1]
            ):
                raise ValueError("Token ends must increase and fall within the input after BOS.")
            targets[index, end + 1] = 1
            previous = end
        if ends:
            raw_positions = input_ids[index, 1 : ends[-1] + 2]
            if torch.any((raw_positions < 0) | (raw_positions > 255)):
                raise ValueError("Boundary supervision must align only to raw input bytes.")
            mask[index, 1 : ends[-1] + 2] = True
    return targets, mask
