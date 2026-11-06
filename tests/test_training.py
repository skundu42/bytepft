from dataclasses import replace

import pytest
import torch

from bytepft.adapters.qwen import TinyGlobalAdapter
from bytepft.config import (
    BytePFTConfig,
    BytePFTTrainingConfig,
    Stage1Config,
    Stage2Config,
)
from bytepft.models.byte_model import ByteModel
from bytepft.training.trainer import BytePFTTrainer


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(seed=7, *, boundary_mode="fixed"):
    torch.manual_seed(seed)
    adapter = TinyGlobalAdapter(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        vocab_size=32,
    )
    adapter.model.config.bos_token_id = 1
    adapter.model.config.eos_token_id = 2
    return ByteModel(
        adapter,
        BytePFTConfig(
            hidden_size=16,
            encoder_layers=1,
            decoder_layers=1,
            bytes_per_patch=2,
            max_patch_size=4,
            boundary_mode=boundary_mode,
        ),
    )


def training_config(path, *, stage1=2, stage2=0, mode="diagnostic", **kwargs):
    return BytePFTTrainingConfig(
        stage1=Stage1Config(steps=stage1, mode=mode),
        stage2=Stage2Config(steps=stage2),
        batch_size=2,
        learning_rate=0.01,
        max_bytes=32,
        validation_fraction=0,
        checkpoint_interval=1,
        validation_interval=2,
        output_dir=str(path),
        seed=17,
        **kwargs,
    )


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def test_stage1_only_updates_local_parameters_with_finite_gradients(tmp_path):
    model = tiny_model()
    before = snapshot(model)
    config = training_config(tmp_path, stage1=3)
    history = BytePFTTrainer(model, config).fit(["ababa", "babab"])
    assert len(history) == 3
    assert {record["stage"] for record in history} == {"stage1"}
    assert model.training_status["stage1"] == "completed"
    assert model.training_status["stage2"] == "not_started"
    assert model.training_status["stage2_steps"] == 0
    changed_local = []
    for name, parameter in model.named_parameters():
        if name.startswith("global_adapter."):
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            assert not parameter.requires_grad
            assert parameter.grad is None
        elif not torch.equal(parameter, before[name]):
            changed_local.append(name)
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    assert any(name.startswith("encoder.") for name in changed_local)
    assert any(name.startswith("decoder.") for name in changed_local)
    assert any(name.startswith("head.") for name in changed_local)
    assert all(torch.isfinite(torch.tensor(record["grad_norm"])) for record in history)


def test_stage2_unfreezing_updates_global_parameters(tmp_path):
    model = tiny_model()
    before = snapshot(model.global_adapter)
    config = training_config(tmp_path, stage1=0, stage2=2)
    history = BytePFTTrainer(model, config).fit(["abcabc", "bcabca"])
    assert {record["stage"] for record in history} == {"stage2"}
    assert model.training_status["stage2"] == "completed"
    changed = [
        name
        for name, parameter in model.global_adapter.named_parameters()
        if not torch.equal(parameter, before[name])
    ]
    assert changed
    assert all(parameter.requires_grad for parameter in model.global_adapter.parameters())


def test_stage2_can_explicitly_keep_global_frozen(tmp_path):
    model = tiny_model()
    before = snapshot(model.global_adapter)
    config = training_config(tmp_path, stage1=0, stage2=1)
    config.stage2.unfreeze_global = False
    BytePFTTrainer(model, config).fit(["abcabc"])
    for name, parameter in model.global_adapter.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_tiny_model_actually_overfits_repeated_corpus(tmp_path):
    model = tiny_model()
    config = training_config(tmp_path, stage1=60)
    config.learning_rate = 0.025
    config.checkpoint_interval = 60
    history = BytePFTTrainer(model, config).fit(["abcabcabc"] * 4)
    initial = sum(record["byte_ce"] for record in history[:5]) / 5
    final = sum(record["byte_ce"] for record in history[-5:]) / 5
    assert final < initial * 0.35, (initial, final)
    assert final < 1.2
    assert model.training_status["trained"]
    # This is a structural overfit result, not a claim of general language quality.
    assert not model.training_status["quality_validated"]
    continuation = model.generate("abc", max_new_bytes=6, temperature=0, return_bytes=True)
    assert continuation.startswith(b"abc")
    assert len(continuation) > 3


class TinyByteLevelTokenizer:
    all_special_ids = [1, 2]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) - ord("a") + 3 for character in text]

    def convert_ids_to_tokens(self, token_id):
        return chr(token_id - 3 + ord("a"))


def test_aligned_stage1_runs_real_compatible_losses_and_keeps_teacher_frozen(tmp_path):
    model = tiny_model(boundary_mode="learned")
    before = snapshot(model.global_adapter)
    local_before = snapshot(model.head)
    config = training_config(tmp_path, stage1=3, mode="aligned")
    history = BytePFTTrainer(model, config, source_tokenizer=TinyByteLevelTokenizer()).fit(
        ["ababa", "baba", ""]
    )
    for record in history:
        assert {"boundary", "embedding", "distillation", "byte_ce"} <= record.keys()
        assert all(torch.isfinite(torch.tensor(record[key])) for key in ("loss", "distillation"))
    for name, parameter in model.global_adapter.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        assert parameter.grad is None
    assert any(
        not torch.equal(parameter, local_before[name])
        for name, parameter in model.head.named_parameters()
    )
    assert model.metrics["training_counts"]["source_tokens_seen"] > 0


@pytest.mark.parametrize("checkpoint", ["stage1-step-2", "stage1-final", "stage2-step-2"])
def test_resumption_matches_continuous_parameters_history_and_counts(tmp_path, checkpoint):
    documents = ["abcabc", "bcabca", "cabcab", "ababab"]
    continuous = tiny_model(seed=23)
    config = training_config(
        tmp_path / "continuous",
        stage1=4,
        stage2=4,
        gradient_accumulation_steps=2,
        warmup_steps=1,
    )
    expected_history = BytePFTTrainer(continuous, config).fit(documents)
    resumed = tiny_model(seed=999)
    resumed_config = replace(config, output_dir=str(tmp_path / checkpoint))
    history = BytePFTTrainer(resumed, resumed_config).fit(
        documents, resume_from=tmp_path / "continuous" / checkpoint
    )
    assert history == expected_history
    assert resumed.training_status == continuous.training_status
    assert resumed.patch_statistics == continuous.patch_statistics
    assert resumed.metrics["training_counts"] == continuous.metrics["training_counts"]
    for name, parameter in continuous.named_parameters():
        torch.testing.assert_close(
            parameter, dict(resumed.named_parameters())[name], rtol=0, atol=0
        )


def test_resume_rejects_changed_data_and_training_configuration(tmp_path):
    model = tiny_model()
    config = training_config(tmp_path / "original", stage1=2)
    BytePFTTrainer(model, config).fit(["ababa", "babab"])
    checkpoint = tmp_path / "original" / "stage1-step-1"
    altered = replace(config, output_dir=str(tmp_path / "changed"), learning_rate=0.1)
    with pytest.raises(ValueError, match="original training configuration"):
        BytePFTTrainer(tiny_model(), altered).fit(["ababa", "babab"], resume_from=checkpoint)
    same_config = replace(config, output_dir=str(tmp_path / "changed-data"))
    with pytest.raises(ValueError, match="documents differ"):
        BytePFTTrainer(tiny_model(), same_config).fit(["different"], resume_from=checkpoint)


def test_aligned_stage1_requires_source_tokenizer_and_fit_requires_steps(tmp_path):
    config = training_config(tmp_path, stage1=1, mode="aligned")
    with pytest.raises(ValueError, match="original source tokenizer"):
        BytePFTTrainer(tiny_model(), config).fit(["abc"])
    config = training_config(tmp_path, stage1=0, stage2=0)
    with pytest.raises(ValueError, match="at least one"):
        BytePFTTrainer(tiny_model(), config).fit(["abc"])
