"""Mechanics tests use explicitly marked synthetic, randomly initialized models."""

import importlib
import math

import pytest
import torch

from bytepft.adapters import TinyGlobalAdapter
from bytepft.config import BytePFTConfig
from bytepft.evaluation import benchmark_generation, evaluate_model
from bytepft.exceptions import NotAdaptedError
from bytepft.inference.generation import GenerationResult, create_cache, incremental_logits
from bytepft.models.byte_model import ByteModel


def tiny_model(**kwargs):
    torch.manual_seed(7)
    model = ByteModel(
        TinyGlobalAdapter(hidden_size=16, num_attention_heads=2, num_key_value_heads=1),
        BytePFTConfig(hidden_size=16, encoder_layers=1, decoder_layers=1, **kwargs),
    )
    # This enables generation mechanics tests; it is not a claim of adaptation.
    model.training_status["trained"] = True
    return model


@pytest.mark.parametrize("mode", ["fixed", "learned"])
@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_cached_logits_and_greedy_match_reference(mode, pooling):
    model = tiny_model(
        boundary_mode=mode,
        pooling=pooling,
        min_patch_size=2,
        max_patch_size=5,
        bytes_per_patch=3,
    ).eval()
    observed = [257, *"hé🙂!?".encode()]
    cache = create_cache(model)
    with torch.no_grad():
        for end, input_id in enumerate(observed, start=1):
            cached = incremental_logits(model, input_id, cache)
            reference = model(torch.tensor([observed[:end]])).logits[0, -1]
            torch.testing.assert_close(cached, reference, atol=2e-5, rtol=2e-5)
    cached_bytes = model.generate(
        "hé🙂", max_new_bytes=15, temperature=0, use_cache=True, return_bytes=True
    )
    reference_bytes = model.generate(
        "hé🙂", max_new_bytes=15, temperature=0, use_cache=False, return_bytes=True
    )
    assert cached_bytes == reference_bytes
    assert cache.patches_processed < len(observed)


def test_controls_suppressed_and_generation_limit_uses_raw_bytes():
    model = tiny_model()
    with torch.no_grad():
        model.head.projection.weight.zero_()
        model.head.projection.bias.fill_(-20)
        model.head.projection.bias[256:258] = 100
        model.head.projection.bias[ord("a")] = 50
    result = model.generate("π", max_new_bytes=9, temperature=0, return_details=True)
    assert isinstance(result, GenerationResult)
    assert result.text == "π" + "a" * 9
    assert result.data == "π".encode() + b"a" * 9
    assert result.generated_ids == [97] * 9
    assert result.generated_bytes == 9 and result.prompt_bytes == 2
    assert not result.stopped_on_eos
    assert model.training
    assert result.prefill_seconds > 0
    assert result.time_to_first_byte_seconds >= result.prefill_seconds
    assert result.bytes_per_second > 0
    assert result.decode_global_calls < result.generated_bytes
    assert result.to_dict()["data_hex"] == result.data.hex()


def test_eos_stopping_and_zero_byte_budget():
    model = tiny_model()
    with torch.no_grad():
        model.head.projection.weight.zero_()
        model.head.projection.bias.fill_(-100)
        model.head.projection.bias[258] = 100
    result = model.generate("hello", max_new_bytes=10, temperature=0, return_details=True)
    assert result.stopped_on_eos
    assert result.data == b"hello"
    assert result.generated_bytes == 0
    assert result.time_to_first_byte_seconds is None
    assert result.global_calls_per_generated_byte is None
    empty = model.generate("", max_new_bytes=0, return_details=True)
    assert empty.data == b""
    assert empty.global_calls == 0


def test_multibyte_output_buffers_partial_suffix(monkeypatch):
    module = importlib.import_module("bytepft.inference.generation")
    model = tiny_model()
    outputs = iter([0xE2, 0x82, 0xAC, 0xE2, 0x82])
    monkeypatch.setattr(module, "_sample", lambda *args: next(outputs))
    result = model.generate("ह", max_new_bytes=5, return_details=True)
    assert result.text == "ह€"
    assert result.incomplete_utf8_suffix == b"\xe2\x82"
    assert result.data == "ह€".encode() + b"\xe2\x82"


def test_sampling_has_private_rng_and_respects_top_k():
    model = tiny_model().eval()
    with torch.no_grad():
        model.head.projection.bias[258] = -100
    state = torch.random.get_rng_state().clone()
    first = model.generate(
        "Hi",
        max_new_bytes=20,
        seed=3,
        temperature=0.8,
        top_k=4,
        top_p=0.7,
        return_bytes=True,
    )
    second = model.generate(
        "Hi",
        max_new_bytes=20,
        seed=3,
        temperature=0.8,
        top_k=4,
        top_p=0.7,
        return_bytes=True,
    )
    assert first == second
    assert torch.equal(state, torch.random.get_rng_state())
    greedy = model.generate("Hi", max_new_bytes=20, temperature=0, return_bytes=True)
    top_one = model.generate("Hi", max_new_bytes=20, temperature=10, top_k=1, return_bytes=True)
    assert greedy == top_one


def test_pretraining_oracle_and_argument_guards():
    model = tiny_model()
    model.training_status["trained"] = False
    with pytest.raises(NotAdaptedError, match="Adaptation"):
        model.generate("x")
    oracle = tiny_model(boundary_mode="oracle")
    with pytest.raises(ValueError, match="training-only"):
        oracle.generate("x")
    model.training_status["trained"] = True
    for kwargs in [
        {"max_new_bytes": -1},
        {"max_new_bytes": True},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"top_p": 0},
        {"top_p": 2},
        {"top_k": -2},
        {"top_k": 1.5},
        {"return_bytes": True, "return_details": True},
    ]:
        with pytest.raises(ValueError):
            model.generate("x", **kwargs)
    with pytest.raises(TypeError):
        model.generate(b"raw")
    assert model.training


def test_evaluation_normalizes_only_raw_utf8_bytes_and_restores_mode():
    model = tiny_model()
    with torch.no_grad():
        model.head.projection.weight.zero_()
        model.head.projection.bias.zero_()
    metrics = evaluate_model(model, ["é", "🙂", "a", ""], batch_size=2)
    assert metrics["byte_count"] == 7
    assert metrics["document_count"] == 4
    assert metrics["nll"] == pytest.approx(math.log(259))
    assert metrics["bits_per_byte"] == pytest.approx(math.log2(259))
    assert model.training
    with pytest.raises(ValueError, match="at least one"):
        evaluate_model(model, [""])
    assert model.training


def test_benchmark_reports_observed_cpu_metrics():
    model = tiny_model()
    with torch.no_grad():
        model.head.projection.bias[258] = -100
    report = benchmark_generation(model, "Hi", max_new_bytes=5, repeats=2)
    assert report["device"] == "cpu"
    assert report["peak_gpu_memory_bytes"] is None
    assert report["model_tensor_memory_bytes"] > 0
    assert report["aggregate_bytes_per_second"] > 0
    assert len(report["measurements"]) == 2
    assert report["measurements"][0]["generated_bytes"] == 5
