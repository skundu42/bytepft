import math
from dataclasses import replace

import pytest
import torch

from bytepft.adapters.qwen import TinyGlobalAdapter
from bytepft.config import BytePFTConfig, BytePFTTrainingConfig, Stage1Config, Stage2Config
from bytepft.exceptions import CompatibilityError
from bytepft.models.byte_model import ByteModel
from bytepft.serialization import load_pretrained
from bytepft.training.trainer import BytePFTTrainer


def cpu_model(seed=5, precision="bf16", *, loaded_dtype=torch.bfloat16):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ByteModel(
            TinyGlobalAdapter(hidden_size=16, num_hidden_layers=1, intermediate_size=32),
            BytePFTConfig(
                hidden_size=16,
                encoder_layers=1,
                decoder_layers=1,
                bytes_per_patch=2,
                max_patch_size=8,
                precision=precision,
            ),
        )
    return model.to(device="cpu", dtype=loaded_dtype)


def training_config(path, *, stage1=3, stage2=0, accumulation=2):
    return BytePFTTrainingConfig(
        stage1=Stage1Config(steps=stage1),
        stage2=Stage2Config(steps=stage2),
        batch_size=1,
        gradient_accumulation_steps=accumulation,
        learning_rate=0.005,
        validation_fraction=0,
        checkpoint_interval=1,
        max_bytes=16,
        seed=53,
        output_dir=str(path),
    )


def test_cpu_bf16_amp_keeps_master_parameters_and_adam_moments_in_fp32(tmp_path):
    model = cpu_model()
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    original_global = {
        name: parameter.detach().float().clone()
        for name, parameter in model.global_adapter.named_parameters()
    }
    original_head = model.head.projection.weight.detach().float().clone()
    trainer = BytePFTTrainer(model, training_config(tmp_path))
    history = trainer.fit(["abcabc", "bcbcbc"])
    assert len(history) == 3
    assert all(
        math.isfinite(record["loss"]) and math.isfinite(record["grad_norm"]) for record in history
    )
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(
        g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients
    )
    assert not torch.equal(model.head.projection.weight, original_head)
    for name, parameter in model.global_adapter.named_parameters():
        assert torch.equal(parameter, original_global[name])

    state = torch.load(tmp_path / "checkpoint-final" / "trainer_state.pt", weights_only=True)
    assert "scaler" in state
    # bf16 on CPU needs no loss scaling; its empty scaler state is still saved
    # so the checkpoint format also supports CUDA fp16 training.
    assert state["scaler"] == {}
    assert "device_rng" not in state
    moments = [
        tensor
        for values in state["optimizer"]["state"].values()
        for key, tensor in values.items()
        if key in {"exp_avg", "exp_avg_sq"}
    ]
    assert moments and all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in moments)
    restored = load_pretrained(tmp_path / "checkpoint-final", device="cpu")
    assert restored.config.precision == "bf16"
    assert all(parameter.dtype == torch.float32 for parameter in restored.parameters())


@pytest.mark.parametrize("checkpoint", ["stage1-step-1", "stage2-step-1"])
def test_cpu_bf16_amp_resumption_matches_uninterrupted_training(tmp_path, checkpoint):
    documents = ["ababa", "bcabc", "cabca"]
    continuous = cpu_model(seed=13)
    config = training_config(tmp_path / "continuous", stage1=2, stage2=2)
    continuous_trainer = BytePFTTrainer(continuous, config)
    expected = continuous_trainer.fit(documents)

    resumed = cpu_model(seed=99)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    resumed_trainer = BytePFTTrainer(resumed, resumed_config)
    actual = resumed_trainer.fit(documents, resume_from=tmp_path / "continuous" / checkpoint)
    assert actual == expected
    assert resumed.training_status == continuous.training_status
    assert resumed.patch_statistics == continuous.patch_statistics
    assert resumed.metrics == continuous.metrics
    for name, parameter in continuous.state_dict().items():
        assert torch.equal(parameter, resumed.state_dict()[name]), name

    continuous_state = torch.load(
        tmp_path / "continuous" / "checkpoint-final" / "trainer_state.pt", weights_only=True
    )
    resumed_state = torch.load(
        tmp_path / "resumed" / "checkpoint-final" / "trainer_state.pt", weights_only=True
    )
    assert continuous_state["scaler"] == resumed_state["scaler"] == {}
    assert continuous_state["scheduler"] == resumed_state["scheduler"]
    for parameter_id, values in continuous_state["optimizer"]["state"].items():
        for name, tensor in values.items():
            assert torch.equal(tensor, resumed_state["optimizer"]["state"][parameter_id][name])


def test_cpu_fp16_training_is_rejected_before_any_update(tmp_path):
    model = cpu_model(precision="fp16", loaded_dtype=torch.float16)
    with pytest.raises(CompatibilityError, match="CUDA AMP"):
        BytePFTTrainer(model, training_config(tmp_path, stage1=1))
    assert model.training_status["stage1_steps"] == 0


def test_gradient_accumulation_matches_equivalent_batch_on_repeated_document(tmp_path):
    accumulated = cpu_model(precision="fp32", loaded_dtype=torch.float32)
    batched = cpu_model(precision="fp32", loaded_dtype=torch.float32)
    accumulated_config = training_config(tmp_path / "accumulated", stage1=2, accumulation=2)
    batched_config = replace(
        accumulated_config,
        batch_size=2,
        gradient_accumulation_steps=1,
        output_dir=str(tmp_path / "batched"),
    )
    BytePFTTrainer(accumulated, accumulated_config).fit(["abcabc"])
    BytePFTTrainer(batched, batched_config).fit(["abcabc"])
    for name, parameter in accumulated.state_dict().items():
        torch.testing.assert_close(parameter, batched.state_dict()[name], atol=2e-6, rtol=2e-5)
    assert accumulated.metrics["training_counts"]["documents_seen"] == 4
    assert batched.metrics["training_counts"]["documents_seen"] == 4
