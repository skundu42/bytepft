import json
import math

import pytest
import torch
import yaml

from bytepft import BytePFT, BytePFTConfig
from bytepft.cli import main
from bytepft.exceptions import CompatibilityError, NotAdaptedError


def architecture():
    return BytePFTConfig(
        hidden_size=16,
        encoder_layers=1,
        decoder_layers=1,
        bytes_per_patch=2,
        max_patch_size=8,
    )


def test_tiny_api_initialization_is_reproducible_and_preserves_caller_rng():
    torch.manual_seed(199)
    before = torch.get_rng_state().clone()
    first = BytePFT.from_tiny(config=architecture(), seed=7, device="cpu", global_hidden_size=16)
    assert torch.equal(torch.get_rng_state(), before)
    second = BytePFT.from_tiny(config=architecture(), seed=7, device="cpu", global_hidden_size=16)
    for name, parameter in first.model.state_dict().items():
        assert torch.equal(parameter, second.model.state_dict()[name])
    assert str(first.model.device) == "cpu"


def test_api_retrofit_save_load_and_byte_generation(tmp_path, monkeypatch):
    converter = BytePFT.from_tiny(config=architecture(), global_hidden_size=16)
    original = {
        name: value.detach().clone() for name, value in converter.adapter.state_dict().items()
    }
    assert converter.configure(hidden_size=24, encoder_layers=2) is converter
    for name, value in converter.adapter.state_dict().items():
        assert torch.equal(value, original[name])
    history = converter.retrofit(
        ["abc", "abd"],
        stage1_steps=2,
        stage2_steps=0,
        batch_size=1,
        learning_rate=0.01,
        output_dir=str(tmp_path / "training"),
        validation_fraction=0,
        max_bytes=16,
    )
    assert len(history) == 2 and all(math.isfinite(item["loss"]) for item in history)
    assert converter.stats()["training"]["stage1_steps"] == 2
    assert converter.stats()["training"]["stage2_steps"] == 0
    assert converter.stats()["training"]["trained"] is True
    assert converter.stats()["training"]["quality_validated"] is False

    def reject_source_generation(*args, **kwargs):
        pytest.fail("Byte inference must not call source-model subword generation")

    monkeypatch.setattr(converter.adapter.model, "generate", reject_source_generation)
    expected = converter.generate("ab", max_new_bytes=5, temperature=0, return_bytes=True)
    converter.save_pretrained(tmp_path / "saved")
    restored = BytePFT.load(tmp_path / "saved", device="cpu")
    actual = restored.generate("ab", max_new_bytes=5, temperature=0, return_bytes=True)
    assert actual == expected
    assert isinstance(actual, bytes) and actual.startswith(b"ab")
    assert restored.stats()["training"] == converter.stats()["training"]
    with pytest.raises(ValueError, match="trained"):
        converter.configure(hidden_size=16)


def test_api_rejects_generation_before_training_and_incompatible_precision():
    converter = BytePFT.from_tiny(config=architecture())
    with pytest.raises(NotAdaptedError):
        converter.generate("hello", max_new_bytes=1)
    with pytest.raises(CompatibilityError, match="fp16"):
        converter.configure(precision="fp16")
    with pytest.raises(ValueError, match="source model"):
        converter.retrofit(["hello"], stage1_steps=1, stage1_mode="aligned")


def test_tiny_api_honors_requested_cpu_precision():
    config = architecture()
    config.precision = "bf16"
    converter = BytePFT.from_tiny(config=config, device="cpu")
    assert converter.model.embeddings.weight.dtype == torch.bfloat16
    output = converter.model(
        torch.tensor([[257, 65, 66, 258]]), labels=torch.tensor([[65, 66, 258, -100]])
    )
    assert output.logits.dtype == torch.bfloat16
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in converter.model.parameters() if p.grad is not None
    )


def test_cli_help_describes_installed_commands(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    for command in ("train-tiny", "retrofit", "inspect", "generate", "evaluate", "benchmark"):
        assert command in output


def test_cli_tiny_training_inspection_evaluation_generation_and_benchmark(tmp_path, capsys):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "architecture": architecture().to_dict(),
                "training": {
                    "stage1": {"steps": 2, "mode": "diagnostic"},
                    "batch_size": 1,
                    "validation_fraction": 0,
                    "max_bytes": 16,
                },
            }
        )
    )
    output_dir = tmp_path / "output"
    assert main(["train-tiny", "--config", str(config_path), "--output", str(output_dir)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["stats"]["training"]["stage1_steps"] == 2
    assert result["stats"]["training"]["stage2_steps"] == 0

    assert main(["inspect", str(output_dir)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["adapter_kind"] == "tiny"
    assert result["training_status"]["trained"]

    data_path = tmp_path / "evaluation.txt"
    data_path.write_text("abc", encoding="utf-8")
    assert main(["evaluate", "--model", str(output_dir), "--dataset", str(data_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["byte_count"] == 3 and math.isfinite(result["bits_per_byte"])

    assert (
        main(["generate", "--model", str(output_dir), "--prompt", "hello", "--max-new-bytes", "0"])
        == 0
    )
    assert capsys.readouterr().out.rstrip("\n") == "hello"

    assert main(["benchmark", "--model", str(output_dir), "--max-new-bytes", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["device"] == "cpu"
    assert result["requested_max_new_bytes"] == 2
    assert result["peak_gpu_memory_bytes"] is None


def test_cli_reports_bad_checkpoints_and_yaml_without_tracebacks(tmp_path, capsys):
    assert main(["generate", "--model", str(tmp_path / "missing")]) == 1
    assert "Cannot load" in capsys.readouterr().err
    config = tmp_path / "bad.yaml"
    config.write_text("unknown_section: true\n")
    assert main(["train-tiny", "--config", str(config), "--output", str(tmp_path / "out")]) == 1
    assert "Unknown config sections" in capsys.readouterr().err
    config.write_text("architecture: [invalid]\n")
    assert main(["train-tiny", "--config", str(config), "--output", str(tmp_path / "out")]) == 1
    assert "bytepft:" in capsys.readouterr().err
