import torch

from bytepft.adapters.qwen import TinyGlobalAdapter
from bytepft.config import BytePFTConfig
from bytepft.models.byte_model import ByteModel
from bytepft.training.losses import BoundaryLoss


def model_and_batch():
    torch.manual_seed(101)
    model = ByteModel(
        TinyGlobalAdapter(hidden_size=16, num_hidden_layers=1, intermediate_size=32),
        BytePFTConfig(
            hidden_size=16,
            encoder_layers=1,
            decoder_layers=1,
            bytes_per_patch=2,
            max_patch_size=8,
        ),
    )
    ids = torch.tensor([[257, 65, 66, 67, 68, 258]])
    labels = torch.tensor([[65, 66, 67, 68, 258, -100]])
    return model, ids, labels


def test_stage1_frozen_backbone_preserves_gradient_to_local_interface():
    model, ids, labels = model_and_batch()
    model.global_adapter.freeze()
    model.train()
    assert not model.global_adapter.model.training
    original = {
        name: value.detach().clone() for name, value in model.global_adapter.named_parameters()
    }
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    output = model(ids, labels=labels)
    boundary_targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]])
    loss = output.loss + BoundaryLoss()(output.boundary_logits, boundary_targets)
    loss.backward()
    for module in (
        model.embeddings,
        model.encoder,
        model.boundary,
        model.to_global,
        model.decoder,
        model.head,
    ):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    assert all(parameter.grad is None for parameter in model.global_adapter.parameters())
    optimizer.step()
    for name, parameter in model.global_adapter.named_parameters():
        assert torch.equal(parameter, original[name])


def test_stage2_unfreezing_updates_global_transformer():
    model, ids, labels = model_and_batch()
    model.global_adapter.freeze()
    model.global_adapter.unfreeze()
    model.train()
    assert model.global_adapter.model.training
    parameter = model.global_adapter.backbone.layers[0].self_attn.q_proj.weight
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    output = model(ids, labels=labels)
    output.loss.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(parameter, before)


def test_external_teacher_states_are_detached():
    model, ids, labels = model_and_batch()
    preliminary = model(ids)
    teacher_states = torch.randn_like(preliminary.patch_embeddings, requires_grad=True)
    output = model(ids, labels=labels, teacher_patch_states=teacher_states)
    output.loss.backward()
    assert teacher_states.grad is None
    assert model.encoder.rnn.weight_ih_l0.grad.abs().sum() > 0
