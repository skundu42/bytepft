import math

import pytest
import torch
from torch.nn import functional as F

from bytepft.training.losses import (
    BoundaryLoss,
    ByteCrossEntropyLoss,
    DecoderDistillationLoss,
    EmbeddingAlignmentLoss,
)


def test_boundary_masked_gradients():
    logits = torch.zeros(1, 3, requires_grad=True)
    targets = torch.tensor([[0, 1, 1]])
    mask = torch.tensor([[True, True, False]])
    loss = BoundaryLoss()(logits, targets, mask)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.25, -0.25, 0.0]]))


def test_embedding_alignment_detaches_teacher_and_masks_padding():
    student = torch.zeros(1, 2, 3, requires_grad=True)
    teacher = torch.ones_like(student, requires_grad=True)
    loss = EmbeddingAlignmentLoss()(student, teacher, torch.tensor([[True, False]]))
    assert loss.item() == 1
    loss.backward()
    assert teacher.grad is None
    assert student.grad[0, 0].abs().sum() > 0
    assert student.grad[0, 1].abs().sum() == 0


def test_byte_ce_shift_is_caller_responsibility_and_ignored_targets_are_safe():
    logits = torch.randn(2, 3, 259, requires_grad=True)
    labels = torch.tensor([[65, 258, -100], [66, 67, -100]])
    actual = ByteCrossEntropyLoss()(logits, labels)
    expected = F.cross_entropy(logits.reshape(-1, 259), labels.reshape(-1))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 2].abs().sum() == 0
    empty = ByteCrossEntropyLoss()(logits, torch.full((2, 3), -100))
    assert empty.item() == 0
    empty.backward()


def test_distillation_matches_scalar_bernoulli_comparison():
    student = torch.tensor([[-3.0, -7.0]], requires_grad=True)
    teacher = torch.tensor([[-2.0, -4.0]], requires_grad=True)
    temperature = 5
    result = DecoderDistillationLoss(temperature)(student, teacher)
    p = torch.exp(student / temperature)
    q = torch.exp(teacher.detach() / temperature)
    expected = -(q * p.log() + (1 - q) * (1 - p).log()).mean()
    torch.testing.assert_close(result, expected)
    result.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_distillation_stable_for_extreme_probabilities_and_padding():
    student = torch.tensor([[-100000.0, 0.0, -3.0]], requires_grad=True)
    teacher = torch.tensor([[-90000.0, 0.0, -4.0]])
    loss = DecoderDistillationLoss()(student, teacher, torch.tensor([[True, True, False]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad[0, 2] == 0


def test_losses_reject_incompatible_shapes_and_probabilities():
    with pytest.raises(ValueError, match="same shape"):
        BoundaryLoss()(torch.zeros(1, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="binary"):
        BoundaryLoss()(torch.zeros(1, 2), torch.tensor([[0.0, 0.5]]))
    with pytest.raises(ValueError, match="matching"):
        EmbeddingAlignmentLoss()(torch.zeros(1, 2, 3), torch.zeros(1, 3, 3))
    with pytest.raises(ValueError, match="matching"):
        DecoderDistillationLoss()(torch.zeros(1, 2, 259), torch.zeros(1, 2, 1000))
    with pytest.raises(ValueError, match="non-positive"):
        DecoderDistillationLoss()(torch.ones(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="positive"):
        DecoderDistillationLoss(0)
