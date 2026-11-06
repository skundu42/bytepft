"""Opt-in GPU tests: RUN_MPS_TESTS=1 uv run pytest tests/test_mps.py."""

import os
from dataclasses import replace

import pytest
import torch

from bytepft import (
    BytePFT,
    BytePFTConfig,
    BytePFTTrainer,
    BytePFTTrainingConfig,
    Stage1Config,
    Stage2Config,
)
from bytepft.evaluation import evaluate_model
from bytepft.inference.generation import create_cache, incremental_logits

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.mps,
    pytest.mark.skipif(
        os.environ.get("RUN_MPS_TESTS") != "1" or not torch.backends.mps.is_available(),
        reason="Requires explicit RUN_MPS_TESTS=1 and available Apple GPU",
    ),
]


def converter(mode="fixed", pooling="last", device="mps"):
    return BytePFT.from_tiny(
        config=BytePFTConfig(
            hidden_size=16,
            encoder_layers=1,
            decoder_layers=1,
            bytes_per_patch=2,
            boundary_mode=mode,
            pooling=pooling,
        ),
        global_hidden_size=16,
        seed=31,
        device=device,
    )


@pytest.mark.parametrize("mode", ["fixed", "learned"])
@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_mps_forward_causality_and_incremental_match_cpu(mode, pooling):
    cpu = converter(mode, pooling, "cpu").model.eval()
    mps = converter(mode, pooling).model.eval()
    ids = torch.tensor([[257, *"hé🙂abc".encode()]], dtype=torch.long)
    cache = create_cache(mps)
    with torch.no_grad():
        expected = cpu(ids).logits
        actual = mps(ids.to("mps")).logits.cpu()
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
        for length, byte in enumerate(ids[0].tolist(), 1):
            cached = incremental_logits(mps, byte, cache).cpu()
            torch.testing.assert_close(cached, actual[0, length - 1], atol=3e-5, rtol=3e-4)
        mutated = ids.clone()
        mutated[:, 5:] = 97
        changed = mps(mutated.to("mps")).logits.cpu()
        torch.testing.assert_close(changed[:, :5], actual[:, :5], atol=3e-5, rtol=3e-4)


def test_mps_both_stages_resume_and_persistence(tmp_path):
    instance = converter()
    model = instance.model
    initial_global = {
        n: p.detach().cpu().clone() for n, p in model.global_adapter.named_parameters()
    }
    config = BytePFTTrainingConfig(
        stage1=Stage1Config(steps=2),
        stage2=Stage2Config(steps=1),
        batch_size=1,
        learning_rate=0.01,
        validation_fraction=0,
        max_bytes=16,
        checkpoint_interval=1,
        output_dir=str(tmp_path / "continuous"),
    )
    history = instance.retrofit(["abcabc"], training_config=config)
    frozen = BytePFT.load(tmp_path / "continuous" / "stage1-final")
    for name, parameter in frozen.adapter.named_parameters():
        assert torch.equal(parameter, initial_global[name])
    assert any(
        not torch.equal(p.detach().cpu(), initial_global[n])
        for n, p in model.global_adapter.named_parameters()
    )
    assert all(torch.isfinite(torch.tensor(row["grad_norm"])) for row in history)
    resumed = converter()
    actual_history = resumed.retrofit(
        ["abcabc"],
        training_config=replace(config, output_dir=str(tmp_path / "resumed")),
        resume_from=tmp_path / "continuous" / "stage1-step-1",
    )
    assert len(actual_history) == len(history)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(
            parameter, resumed.model.state_dict()[name], atol=1e-6, rtol=1e-5
        )
    loaded = BytePFT.load(tmp_path / "continuous", device="mps")
    expected = instance.generate("abc", max_new_bytes=12, temperature=0, return_bytes=True)
    assert loaded.generate("abc", max_new_bytes=12, temperature=0, return_bytes=True) == expected
    assert (
        loaded.generate("abc", max_new_bytes=12, temperature=0, use_cache=False, return_bytes=True)
        == expected
    )


def test_mps_real_tiny_training_reduces_byte_nll(tmp_path, request):
    instance = converter()
    docs = ["abcabcabc"] * 2
    initial = evaluate_model(instance.model, docs)["nll"]
    instance.retrofit(
        docs,
        stage1_steps=60,
        learning_rate=0.025,
        batch_size=2,
        validation_fraction=0,
        output_dir=str(tmp_path),
    )
    final = evaluate_model(instance.model, docs)["nll"]
    request.node.user_properties.extend([("initial_nll", initial), ("final_nll", final)])
    assert final < initial * 0.3
    assert instance.generate(
        "abc", max_new_bytes=12, temperature=0, return_bytes=True
    ) == instance.generate(
        "abc", max_new_bytes=12, temperature=0, use_cache=False, return_bytes=True
    )


def test_mps_right_padded_batch_matches_cpu():
    cpu = converter(device="cpu").model.eval()
    mps = converter().model.eval()
    ids = torch.tensor([[257, 97, 98, 258, 256, 256], [257, 98, 97, 98, 97, 258]])
    with torch.no_grad():
        expected = cpu(ids).logits
        actual = mps(ids.to("mps")).logits.cpu()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


def test_mps_aligned_stage1_teacher_and_losses(tmp_path):
    class ByteLevelFixture:
        all_special_ids = [1, 2]

        def encode(self, text, add_special_tokens=False):
            return [ord(char) - ord("a") + 3 for char in text]

        def convert_ids_to_tokens(self, token_id):
            return chr(token_id - 3 + ord("a"))

    model = converter(mode="learned").model
    model.global_adapter.model.config.bos_token_id = 1
    model.global_adapter.model.config.eos_token_id = 2
    initial = {n: p.detach().cpu().clone() for n, p in model.global_adapter.named_parameters()}
    config = BytePFTTrainingConfig(
        stage1=Stage1Config(steps=1, mode="aligned"),
        batch_size=2,
        validation_fraction=0,
        output_dir=str(tmp_path),
        max_bytes=16,
    )
    history = BytePFTTrainer(model, config, source_tokenizer=ByteLevelFixture()).fit(
        ["ababa", "ba"]
    )
    for key in ("boundary", "embedding", "distillation", "byte_ce", "grad_norm"):
        assert torch.isfinite(torch.tensor(history[0][key]))
    for name, parameter in model.global_adapter.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter.detach().cpu(), initial[name])
