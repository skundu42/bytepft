"""Exact raw-byte source alignment and observed patch-event probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SourceAlignment:
    """Source token IDs and their inclusive end offsets in UTF-8 bytes."""

    token_ids: list[int]
    token_ends: list[int]


def _byte_level_decoder() -> dict[str, int]:
    # Inverse of the reversible GPT-2/ByteLevel byte-to-Unicode alphabet.
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    alphabet = dict(zip(visible, visible, strict=True))
    extra = 0
    for value in range(256):
        if value not in alphabet:
            alphabet[value] = 256 + extra
            extra += 1
    return {chr(character): value for value, character in alphabet.items()}


def extract_source_alignment(text: str, source_tokenizer: Any) -> SourceAlignment:
    """Extract verified ByteLevel BPE boundaries, including split UTF-8 characters.

    Qwen3 uses this reversible byte alphabet. Character offset mappings and
    decoding individual source tokens are unsafe: either can lose split bytes.
    Any unsupported mapping or special token fails explicitly.
    """
    if not isinstance(text, str):
        raise TypeError("Source alignment requires text.")
    token_ids = list(source_tokenizer.encode(text, add_special_tokens=False))
    decoder = getattr(source_tokenizer, "byte_decoder", None) or _byte_level_decoder()
    special_ids = set(getattr(source_tokenizer, "all_special_ids", []))
    pieces: list[bytes] = []
    token_ends: list[int] = []
    length = 0
    for token_id in token_ids:
        if token_id in special_ids:
            raise ValueError(
                "Source alignment does not support special tokens inside ordinary text."
            )
        spelling = source_tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(spelling, str) or not spelling:
            raise ValueError(f"Source token {token_id} has no nonempty byte spelling.")
        try:
            piece = bytes(decoder[character] for character in spelling)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                "Source tokenizer is not compatible with verified ByteLevel byte alignment."
            ) from exc
        pieces.append(piece)
        length += len(piece)
        token_ends.append(length - 1)
    if b"".join(pieces) != text.encode("utf-8"):
        raise ValueError("Source token bytes do not reconstruct the original UTF-8 text exactly.")
    return SourceAlignment(token_ids=token_ids, token_ends=token_ends)


def collect_patch_log_probs(
    logits: Tensor,
    boundary_logits: Tensor,
    input_ids: Tensor,
    token_ends: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    """Sum the reference model's byte and Bernoulli boundary event log probs.

    ``input_ids`` is [B,L] with BOS at position 0 and raw bytes starting at 1;
    ``logits[:,t]`` predicts ``input_ids[:,t+1]``. ``boundary_logits[:,t]``
    predicts the boundary *after the observed input at t*. Every token's final
    byte has boundary target 1; other bytes have boundary target 0. EOS/control
    symbols are excluded. Returns [B,max_tokens] log probabilities and bool mask.

    This is a causal, factorized reference joint event, not the paper's 512-class
    fused head. The teacher and student still describe the same complete patch.
    """
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Expected byte logits [B,L,V] and matching input IDs [B,L].")
    if boundary_logits.shape != input_ids.shape or len(token_ends) != input_ids.shape[0]:
        raise ValueError("Boundary logits and source-alignment batch must match input IDs.")
    if input_ids.dtype != torch.long:
        raise ValueError("Input IDs must have torch.long dtype.")
    max_tokens = max((len(ends) for ends in token_ends), default=0)
    rows: list[Tensor] = []
    masks: list[Tensor] = []
    log_probs = F.log_softmax(logits.float(), dim=-1)
    for batch_idx, ends in enumerate(token_ends):
        previous = -1
        patches: list[Tensor] = []
        for end in ends:
            if (
                isinstance(end, bool)
                or not isinstance(end, int)
                or end <= previous
                or end + 1 >= input_ids.shape[1]
            ):
                raise ValueError(
                    "Token end offsets must increase and lie within raw input bytes after BOS."
                )
            observed_positions = torch.arange(previous + 2, end + 2, device=input_ids.device)
            byte_ids = input_ids[batch_idx, observed_positions]
            if torch.any((byte_ids < 0) | (byte_ids > 255)):
                raise ValueError(
                    "Source-aligned positions must contain raw bytes, not control symbols."
                )
            byte_log_probs = log_probs[batch_idx, observed_positions - 1, byte_ids]
            event_logits = boundary_logits[batch_idx, observed_positions].float()
            flags = torch.zeros_like(event_logits)
            flags[-1] = 1
            boundary_log_probs = -F.binary_cross_entropy_with_logits(
                event_logits, flags, reduction="none"
            )
            patches.append((byte_log_probs + boundary_log_probs).sum())
            previous = end
        row = torch.stack(patches) if patches else logits[batch_idx].sum().reshape(1)[:0]
        rows.append(F.pad(row, (0, max_tokens - len(patches))))
        masks.append(torch.arange(max_tokens, device=input_ids.device) < len(patches))
    if not rows:
        return logits.new_zeros((0, max_tokens)), torch.zeros(
            (0, max_tokens), dtype=torch.bool, device=logits.device
        )
    return torch.stack(rows), torch.stack(masks)
