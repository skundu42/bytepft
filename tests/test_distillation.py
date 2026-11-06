import math

import pytest
import torch

from bytepft.training.distillation import (
    _byte_level_decoder,
    collect_patch_log_probs,
    extract_source_alignment,
)


class ByteBPETokenizer:
    all_special_ids: list[int] = []

    def __init__(self, pieces: list[bytes]):
        inverse = {value: key for key, value in _byte_level_decoder().items()}
        self.pieces = ["".join(inverse[value] for value in piece) for piece in pieces]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(len(self.pieces)))

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[token_id]


def test_source_alignment_retains_utf8_characters_split_between_tokens():
    text = "A🌍é"
    raw = text.encode("utf-8")
    tokenizer = ByteBPETokenizer([raw[:2], raw[2:4], raw[4:]])
    alignment = extract_source_alignment(text, tokenizer)
    assert alignment.token_ids == [0, 1, 2]
    assert alignment.token_ends == [1, 3, 6]


def test_source_alignment_handles_empty_and_rejects_mismatches_or_specials():
    assert extract_source_alignment("", ByteBPETokenizer([])).token_ends == []
    with pytest.raises(ValueError, match="reconstruct"):
        extract_source_alignment("a", ByteBPETokenizer([b"b"]))
    tokenizer = ByteBPETokenizer([b"a"])
    tokenizer.all_special_ids = [0]
    with pytest.raises(ValueError, match="special"):
        extract_source_alignment("a", tokenizer)


def test_patch_probability_sums_same_byte_and_boundary_events_with_gradients():
    logits = torch.zeros(2, 5, 259, requires_grad=True)
    boundary_logits = torch.zeros(2, 5, requires_grad=True)
    ids = torch.tensor([[257, 65, 66, 67, 258], [257, 68, 258, 256, 256]])
    values, mask = collect_patch_log_probs(logits, boundary_logits, ids, [[1, 2], [0]])
    assert values.shape == mask.shape == (2, 2)
    per_event = -math.log(259) - math.log(2)
    torch.testing.assert_close(values, torch.tensor([[2 * per_event, per_event], [per_event, 0]]))
    assert mask.tolist() == [[True, True], [True, False]]
    loss = -(values * mask).sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert boundary_logits.grad.abs().sum() > 0
    assert logits.grad[0, 3:].abs().sum() == 0
    assert boundary_logits.grad[0, 0] == 0


def test_patch_probability_rejects_control_symbols_and_bad_offsets():
    logits = torch.zeros(1, 3, 259)
    boundaries = torch.zeros(1, 3)
    ids = torch.tensor([[257, 65, 258]])
    with pytest.raises(ValueError, match="increase"):
        collect_patch_log_probs(logits, boundaries, ids, [[0, 0]])
    with pytest.raises(ValueError, match="control"):
        collect_patch_log_probs(logits, boundaries, ids, [[1]])
