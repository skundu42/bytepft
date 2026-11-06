import pytest
import torch

from bytepft.adapters.qwen import TinyGlobalAdapter
from bytepft.config import BytePFTConfig
from bytepft.models.byte_model import ByteModel
from bytepft.models.decoder import LocalByteDecoder
from bytepft.models.encoder import LocalByteEncoder
from bytepft.models.heads import BytePredictionHead


def tiny_model(*, mode="fixed", pooling="last"):
    torch.manual_seed(17)
    adapter = TinyGlobalAdapter(hidden_size=16, intermediate_size=32, num_hidden_layers=1)
    config = BytePFTConfig(
        hidden_size=16,
        encoder_layers=1,
        decoder_layers=1,
        bytes_per_patch=2,
        max_patch_size=8,
        boundary_mode=mode,
        pooling=pooling,
    )
    return ByteModel(adapter, config)


def test_model_component_shapes_and_finite_outputs():
    model = tiny_model()
    ids = torch.tensor([[257, 65, 66, 67, 68, 258], [257, 70, 258, 256, 256, 256]])
    labels = torch.tensor([[65, 66, 67, 68, 258, -100], [70, 258, -100, -100, -100, -100]])
    result = model(ids, labels=labels)
    assert result.logits.shape == (2, 6, 259)
    assert result.encoder_hidden_states.shape == (2, 6, 16)
    assert result.patch_embeddings.shape == result.global_hidden_states.shape == (2, 4, 16)
    assert result.boundaries.shape == result.boundary_logits.shape == (2, 6)
    assert result.loss.ndim == 0 and torch.isfinite(result.loss)
    assert torch.isfinite(result.logits).all()
    result.loss.backward()


@pytest.mark.parametrize("mode", ["fixed", "learned"])
@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_future_bytes_cannot_change_earlier_predictions(mode, pooling):
    model = tiny_model(mode=mode, pooling=pooling).eval()
    ids = torch.tensor([[257, 65, 66, 67, 68, 69, 70, 71, 72, 258]])
    original = model(ids)
    for cut in (1, 3, 6, 8):
        changed = ids.clone()
        changed[:, cut + 1 : -1] = 120
        actual = model(changed)
        torch.testing.assert_close(actual.logits[:, : cut + 1], original.logits[:, : cut + 1])
        assert torch.equal(actual.boundaries[:, : cut + 1], original.boundaries[:, : cut + 1])
        prefix = model(ids[:, : cut + 1])
        torch.testing.assert_close(prefix.logits, original.logits[:, : cut + 1])


@pytest.mark.parametrize("mode", ["fixed", "learned"])
def test_right_padding_preserves_unpadded_outputs(mode):
    model = tiny_model(mode=mode).eval()
    short = torch.tensor([[257, 65, 66, 67, 258]])
    batched = torch.tensor(
        [[257, 65, 66, 67, 258, 256, 256, 256], [257, 70, 71, 72, 73, 74, 75, 258]]
    )
    expected = model(short)
    actual = model(batched)
    torch.testing.assert_close(actual.logits[:1, :5], expected.logits, atol=1e-6, rtol=1e-5)
    assert not actual.boundaries[0, 5:].any()


@pytest.mark.parametrize("layers", [1, 2])
def test_local_encoder_incremental_matches_full(layers):
    torch.manual_seed(2)
    encoder = LocalByteEncoder(8, layers).eval()
    embeddings = torch.randn(2, 9, 8)
    full = encoder(embeddings, torch.ones(2, 9))
    state = None
    incremental = []
    for position in range(9):
        value, state = encoder.step(embeddings[:, position : position + 1], state)
        incremental.append(value)
    torch.testing.assert_close(torch.cat(incremental, dim=1), full, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("layers", [1, 2])
def test_local_decoder_incremental_matches_full(layers):
    torch.manual_seed(3)
    decoder = LocalByteDecoder(8, 16, layers).eval()
    features = torch.randn(2, 9, 8)
    context = torch.randn(2, 9, 16)
    full = decoder(features, context, torch.ones(2, 9))
    state = None
    incremental = []
    for position in range(9):
        value, state = decoder.step(
            features[:, position : position + 1], context[:, position : position + 1], state
        )
        incremental.append(value)
    torch.testing.assert_close(torch.cat(incremental, dim=1), full, atol=1e-6, rtol=1e-5)


def test_output_head_is_normalized_and_differentiable():
    head = BytePredictionHead(8)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    logits = head(hidden)
    assert logits.shape == (2, 5, 259)
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(2, 5))
    logits[..., 42].sum().backward()
    assert hidden.grad.abs().sum() > 0
    assert head.projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "ids,mask",
    [
        (torch.empty(1, 0, dtype=torch.long), None),
        (torch.tensor([257, 65]), None),
        (torch.tensor([[257.0, 65.0]]), None),
        (torch.tensor([[256, 256]]), None),
        (torch.tensor([[257, 65]]), torch.tensor([[1, 0, 0]])),
        (torch.tensor([[257, 65, 258]]), torch.tensor([[1, 0, 1]])),
        (torch.tensor([[257, 65]]), torch.tensor([[1, 2]])),
        (torch.tensor([[257, 65]]), torch.tensor([[1.0, float("nan")]])),
    ],
)
def test_model_rejects_invalid_input(ids, mask):
    model = tiny_model()
    with pytest.raises(ValueError):
        model(ids, attention_mask=mask)


def test_oracle_mode_requires_supervision():
    model = tiny_model(mode="oracle")
    with pytest.raises(ValueError, match="Oracle"):
        model(torch.tensor([[257, 65, 66]]))


def test_teacher_patch_state_alignment_is_checked():
    model = tiny_model()
    ids = torch.tensor([[257, 65, 66]])
    with pytest.raises(ValueError, match="align"):
        model(ids, teacher_patch_states=torch.zeros(1, 99, 16))


def test_untrained_stats_do_not_invent_evaluation_results():
    model = tiny_model()
    stats = model.stats()
    assert stats["training"]["trained"] is False
    assert stats["training"]["quality_validated"] is False
    assert stats["metrics"] == {}
    assert stats["patch_count"] is None
    assert stats["average_bytes_per_patch"] is None
    assert stats["new_parameters"] > 0
    assert stats["parameters"] == stats["new_parameters"] + stats["original_parameters"]
