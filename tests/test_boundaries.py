import pytest
import torch

from bytepft.models.boundaries import BoundaryPredictor, constrain_boundaries, fixed_boundaries


def test_fixed_patching_leaves_incomplete_suffix_open():
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    boundaries = fixed_boundaries(mask, patch_size=2)
    assert boundaries.tolist() == [
        [False, True, False, True, False],
        [False, True, False, False, False],
    ]
    assert torch.equal(fixed_boundaries(mask[:, :3], 2), boundaries[:, :3])


def test_control_symbols_close_patches():
    ids = torch.tensor([[257, 97, 98, 99, 258, 256]])
    mask = ids != 256
    result = fixed_boundaries(mask, 2, ids)
    assert result.tolist() == [[True, False, True, False, True, False]]


def test_boundary_length_constraints():
    mask = torch.ones(1, 9, dtype=torch.bool)
    assert constrain_boundaries(mask, mask, 2, 4).tolist() == [
        [False, True, False, True, False, True, False, True, False]
    ]
    assert constrain_boundaries(~mask, mask, 2, 4).tolist() == [
        [False, False, False, True, False, False, False, True, False]
    ]


def test_learned_boundary_scores_have_gradients_and_masking():
    torch.manual_seed(11)
    predictor = BoundaryPredictor(8)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    mask = torch.tensor([[1] * 7, [1] * 3 + [0] * 4])
    logits, boundaries = predictor(hidden, mask)
    assert logits.shape == boundaries.shape == mask.shape
    assert boundaries.dtype == torch.bool
    assert not boundaries[1, 3:].any()
    logits.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert hidden.grad[1, 3:].abs().sum() == 0
    assert all(parameter.grad is not None for parameter in predictor.parameters())
    assert sum(parameter.grad.abs().sum() for parameter in predictor.parameters()) > 0


def test_oracle_requires_boundaries_and_preserves_order():
    predictor = BoundaryPredictor(8, mode="oracle")
    hidden = torch.randn(1, 6, 8)
    mask = torch.ones(1, 6, dtype=torch.bool)
    oracle = torch.tensor([[0, 0, 1, 0, 1, 0]])
    with pytest.raises(ValueError, match="requires"):
        predictor(hidden, mask)
    _, actual = predictor(hidden, mask, oracle_boundaries=oracle)
    assert torch.equal(actual, oracle.bool())


@pytest.mark.parametrize("mode", ["fixed", "learned"])
def test_boundary_decisions_are_prefix_consistent(mode):
    torch.manual_seed(3)
    predictor = BoundaryPredictor(4, mode=mode, min_patch_size=2, max_patch_size=6)
    hidden = torch.randn(1, 18, 4)
    mask = torch.ones(1, 18, dtype=torch.bool)
    _, full = predictor(hidden, mask)
    for prefix in range(1, 18):
        _, partial = predictor(hidden[:, :prefix], mask[:, :prefix])
        assert torch.equal(partial, full[:, :prefix])


def test_invalid_masks_and_shapes():
    with pytest.raises(ValueError, match="right-padded"):
        fixed_boundaries(torch.tensor([[1, 0, 1]]), 2)
    with pytest.raises(ValueError, match="only 0 and 1"):
        fixed_boundaries(torch.tensor([[1, 2]]), 2)
    with pytest.raises(ValueError, match="positive integer"):
        fixed_boundaries(torch.ones(1, 2), 0)
    with pytest.raises(ValueError, match="same shape"):
        constrain_boundaries(torch.ones(1, 2), torch.ones(1, 3))
    with pytest.raises(ValueError, match="min_patch_size"):
        BoundaryPredictor(4, min_patch_size=5, max_patch_size=2)
    with pytest.raises(ValueError, match="integers"):
        constrain_boundaries(torch.ones(1, 2), torch.ones(1, 2), max_patch_size=2.5)
    with pytest.raises(ValueError, match="only 0 and 1"):
        constrain_boundaries(torch.tensor([[0, 2]]), torch.ones(1, 2))
    with pytest.raises(ValueError, match="integer"):
        BoundaryPredictor(4, max_patch_size=True)


def test_empty_boundary_inputs():
    predictor = BoundaryPredictor(4)
    logits, boundaries = predictor(torch.empty(2, 0, 4), torch.empty(2, 0, dtype=torch.bool))
    assert logits.shape == boundaries.shape == (2, 0)
