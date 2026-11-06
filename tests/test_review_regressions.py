import json

import pytest
import torch

from bytepft.adapters.qwen import TinyGlobalAdapter
from bytepft.config import BytePFTConfig, BytePFTTrainingConfig, Stage1Config
from bytepft.exceptions import CheckpointError
from bytepft.models.byte_model import ByteModel
from bytepft.serialization import load_pretrained, save_pretrained
from bytepft.training.trainer import BytePFTTrainer


def small_model():
    return ByteModel(
        TinyGlobalAdapter(hidden_size=16, num_hidden_layers=1, intermediate_size=32),
        BytePFTConfig(
            hidden_size=16,
            encoder_layers=1,
            decoder_layers=1,
            bytes_per_patch=2,
            max_patch_size=8,
        ),
    )


def test_resume_completed_run_is_a_noop_with_unchanged_weights(tmp_path):
    config = BytePFTTrainingConfig(
        stage1=Stage1Config(steps=1),
        batch_size=1,
        max_bytes=8,
        validation_fraction=0,
        output_dir=str(tmp_path / "first"),
    )
    trainer = BytePFTTrainer(small_model(), config)
    history = trainer.fit(["abc"])
    expected = {
        name: tensor.detach().clone() for name, tensor in trainer.model.state_dict().items()
    }
    resumed = BytePFTTrainer(small_model(), config)
    actual_history = resumed.fit(["abc"], resume_from=tmp_path / "first" / "checkpoint-final")
    assert actual_history == history
    assert resumed.model.training_status["stage1_steps"] == 1
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_checkpoint_load_validates_saved_byte_mapping(tmp_path):
    model = small_model()
    save_pretrained(model, tmp_path)
    path = tmp_path / "byte_tokenizer.json"
    config = json.loads(path.read_text())
    config["pad_token_id"] = 0
    path.write_text(json.dumps(config))
    with pytest.raises((ValueError, CheckpointError)):
        load_pretrained(tmp_path)


@pytest.mark.parametrize("documents,max_bytes", [(["", ""], 8), (["💡"], 1)])
def test_training_rejects_zero_raw_bytes_before_updating_weights(tmp_path, documents, max_bytes):
    model = small_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainer = BytePFTTrainer(
        model,
        BytePFTTrainingConfig(
            stage1=Stage1Config(steps=1),
            batch_size=1,
            max_bytes=max_bytes,
            validation_fraction=0,
            output_dir=str(tmp_path),
        ),
    )
    with pytest.raises(ValueError, match="byte"):
        trainer.fit(documents)
    assert model.training_status["stage1_steps"] == 0
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
