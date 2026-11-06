import pytest
import torch

from bytepft.models.depooling import PatchDepool
from bytepft.models.embeddings import ByteEmbedding
from bytepft.models.pooling import PatchPool


@pytest.mark.parametrize("mode", ["last", "mean"])
def test_pooling_coverage_alignment_and_gradients(mode):
    hidden = torch.arange(24.0).reshape(2, 6, 2).requires_grad_()
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    boundaries = torch.tensor([[0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0]])
    pooled, metadata = PatchPool(mode)(hidden, boundaries, mask)
    assert pooled.shape == (2, 3, 2)
    assert metadata.starts.tolist() == [[0, 2, 5], [0, 1, 0]]
    assert metadata.ends.tolist() == [[1, 4, 5], [0, 2, 0]]
    assert metadata.lengths.tolist() == [[2, 3, 1], [1, 2, 0]]
    assert metadata.complete.tolist() == [[True, True, False], [True, False, False]]
    assert metadata.byte_to_patch.tolist() == [[0, 0, 1, 1, 1, 2], [0, 1, 1, -1, -1, -1]]
    assert metadata.patch_count == 5
    assert metadata.average_bytes_per_patch == 9 / 5
    for batch in range(2):
        covered = []
        for patch in metadata.patch_mask[batch].nonzero().flatten().tolist():
            start, end = metadata.starts[batch, patch], metadata.ends[batch, patch]
            covered.extend(range(int(start), int(end) + 1))
            expected = (
                hidden[batch, end] if mode == "last" else hidden[batch, start : end + 1].mean(0)
            )
            assert torch.equal(pooled[batch, patch], expected)
        assert covered == list(range(int(mask[batch].sum())))
    pooled.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert hidden.grad[1, 3:].abs().sum() == 0


def test_depool_only_exposes_latest_completed_patch():
    hidden = torch.randn(1, 7, 3)
    boundaries = torch.tensor([[0, 1, 0, 0, 1, 0, 0]])
    mask = torch.ones(1, 7, dtype=torch.bool)
    _, metadata = PatchPool()(hidden, boundaries, mask)
    patches = torch.tensor([[[10.0], [20.0], [999.0]]], requires_grad=True)
    decoded = PatchDepool()(patches, metadata)
    assert decoded.squeeze(-1).tolist() == [[0.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0]]
    decoded.sum().backward()
    assert patches.grad.squeeze(-1).tolist() == [[3.0, 3.0, 0.0]]


def test_depool_padding_and_all_padding():
    hidden = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    pooled, metadata = PatchPool()(hidden, torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]), mask)
    decoded = PatchDepool()(pooled, metadata)
    assert torch.equal(decoded[1], torch.zeros_like(decoded[1]))
    assert decoded[0, 2:].abs().sum() == 0
    pooled, metadata = PatchPool()(hidden, torch.zeros_like(mask), torch.zeros_like(mask))
    assert pooled.shape == (2, 0, 3)
    assert metadata.patch_count == 0
    assert metadata.average_bytes_per_patch == 0
    decoded = PatchDepool()(pooled, metadata)
    decoded.sum().backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))


def test_pooling_empty_sequences():
    pooled, metadata = PatchPool()(torch.empty(2, 0, 3), torch.empty(2, 0), torch.empty(2, 0))
    assert pooled.shape == (2, 0, 3)
    assert PatchDepool()(pooled, metadata).shape == (2, 0, 3)


def test_pooling_rejects_nonbinary_boundaries():
    with pytest.raises(ValueError, match="only 0 and 1"):
        PatchPool()(torch.zeros(1, 2, 4), torch.tensor([[1.0, float("nan")]]), torch.ones(1, 2))


def test_embedding_padding_mask_gradients_and_seed():
    torch.manual_seed(7)
    embedding = ByteEmbedding(8)
    original = embedding.weight.detach().clone()
    torch.manual_seed(7)
    assert torch.equal(original, ByteEmbedding(8).weight)
    ids = torch.tensor([[257, 42, 256]])
    output = embedding(ids)
    assert output.shape == (1, 3, 8)
    assert output[0, -1].abs().sum() == 0
    output.sum().backward()
    assert embedding.weight.grad[42].abs().sum() > 0
    assert embedding.weight.grad[256].abs().sum() == 0
    with pytest.raises(ValueError, match="int64"):
        embedding(ids.float())
