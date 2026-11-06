import json

import pytest
import torch
from safetensors import safe_open

from bytepft.adapters.qwen import TinyGlobalAdapter
from bytepft.config import BytePFTConfig, BytePFTTrainingConfig, Stage1Config
from bytepft.exceptions import CheckpointError, NotAdaptedError
from bytepft.models.byte_model import ByteModel
from bytepft.serialization import load_pretrained, save_pretrained
from bytepft.training.trainer import BytePFTTrainer


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    torch.manual_seed(42)
    return ByteModel(
        TinyGlobalAdapter(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=32,
            vocab_size=32,
        ),
        BytePFTConfig(hidden_size=16, encoder_layers=1, decoder_layers=1, bytes_per_patch=2),
    )


def test_trained_checkpoint_roundtrip_matches_logits_greedy_bytes_and_metadata(tmp_path):
    model = tiny_model()
    config = BytePFTTrainingConfig(
        stage1=Stage1Config(steps=3),
        batch_size=2,
        learning_rate=0.01,
        max_bytes=16,
        validation_fraction=0,
        output_dir=str(tmp_path / "training"),
    )
    BytePFTTrainer(model, config).fit(["abcabc", "bcabca"])
    model.eval()
    inputs = torch.tensor([[257, 97, 98, 99]])
    with torch.no_grad():
        expected_logits = model(inputs).logits
    expected_bytes = model.generate("abc", max_new_bytes=8, temperature=0, return_bytes=True)
    save_pretrained(model, tmp_path / "saved")
    restored = load_pretrained(tmp_path / "saved", device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs).logits, expected_logits, rtol=0, atol=0)
    assert (
        restored.generate("abc", max_new_bytes=8, temperature=0, return_bytes=True)
        == expected_bytes
    )
    assert restored.training_status == model.training_status
    assert restored.metrics == model.metrics
    assert restored.patch_statistics == model.patch_statistics
    assert restored.checkpoint_metadata["format_version"] == 1
    assert not any(parameter.requires_grad for parameter in restored.global_adapter.parameters())


def test_untrained_roundtrip_preserves_generation_guard(tmp_path):
    model = tiny_model()
    save_pretrained(model, tmp_path)
    restored = load_pretrained(tmp_path)
    assert not restored.training_status["trained"]
    with pytest.raises(NotAdaptedError, match="Adaptation"):
        restored.generate("hello")


def test_safetensors_roundtrip_preserves_shared_qwen_embedding_weights(tmp_path):
    model = tiny_model()
    assert (
        model.global_adapter.get_input_embeddings().weight.data_ptr()
        == model.global_adapter.get_output_embeddings().weight.data_ptr()
    )
    save_pretrained(model, tmp_path)
    with safe_open(tmp_path / "model.safetensors", framework="pt") as handle:
        keys = list(handle.keys())
        # safetensors.save_model records aliases in metadata instead of duplicate tensors.
        alias_keys = {
            "global_adapter.model.model.embed_tokens.weight",
            "global_adapter.model.lm_head.weight",
        }
        assert len(alias_keys & set(keys)) == 1
        assert any(key in (handle.metadata() or {}) for key in alias_keys)
    restored = load_pretrained(tmp_path)
    assert (
        restored.global_adapter.get_input_embeddings().weight.data_ptr()
        == restored.global_adapter.get_output_embeddings().weight.data_ptr()
    )


@pytest.mark.parametrize(
    "corruption", ["unsupported_version", "invalid_json", "missing_weights", "bad_weights"]
)
def test_corrupt_or_incompatible_checkpoints_fail_clearly(tmp_path, corruption):
    save_pretrained(tiny_model(), tmp_path)
    path = tmp_path / "bytepft_config.json"
    if corruption == "unsupported_version":
        config = json.loads(path.read_text())
        config["format_version"] = 999
        path.write_text(json.dumps(config))
    elif corruption == "invalid_json":
        path.write_text("{not-json}")
    elif corruption == "missing_weights":
        (tmp_path / "model.safetensors").unlink()
    else:
        (tmp_path / "model.safetensors").write_bytes(b"corrupt checkpoint")
    with pytest.raises(CheckpointError):
        load_pretrained(tmp_path)
