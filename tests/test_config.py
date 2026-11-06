import math

import pytest

from bytepft.config import BytePFTConfig, BytePFTTrainingConfig, Stage1Config, Stage2Config


def test_config_roundtrip_preserves_nested_training_settings():
    config = BytePFTTrainingConfig(
        stage1=Stage1Config(steps=3, mode="aligned", temperature=2),
        stage2=Stage2Config(steps=4, global_learning_rate=1e-5),
        batch_size=1,
        seed=7,
    )
    restored = BytePFTTrainingConfig.from_dict(config.to_dict())
    assert restored == config
    assert isinstance(restored.stage1, Stage1Config)
    assert isinstance(restored.stage2, Stage2Config)


@pytest.mark.parametrize(
    "values",
    [
        {"hidden_size": 0},
        {"hidden_size": 1.5},
        {"encoder_layers": True},
        {"decoder_layers": 0},
        {"max_patch_size": 2.5},
        {"min_patch_size": 8, "max_patch_size": 4},
        {"bytes_per_patch": 0.5},
        {"bytes_per_patch": math.nan},
        {"bytes_per_patch": math.inf},
        {"boundary_threshold": math.nan},
        {"boundary_threshold": 1},
        {"boundary_mode": "lookahead"},
        {"pooling": "attention"},
        {"precision": "int8"},
    ],
)
def test_model_config_rejects_invalid_or_unsupported_settings(values):
    with pytest.raises((ValueError, TypeError)):
        BytePFTConfig(**values)


def test_unimplemented_mlstm_architecture_is_explicit():
    with pytest.raises(NotImplementedError, match="reference"):
        BytePFTConfig(architecture="mlstm")


@pytest.mark.parametrize(
    "name",
    ["boundary_weight", "embedding_weight", "distillation_weight", "byte_ce_weight", "temperature"],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -1])
def test_stage1_rejects_nonfinite_or_negative_loss_parameters(name, value):
    with pytest.raises((ValueError, TypeError)):
        Stage1Config(**{name: value})


@pytest.mark.parametrize("config", [Stage1Config, Stage2Config])
@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_stage_steps_are_nonnegative_integers(config, steps):
    with pytest.raises((ValueError, TypeError)):
        config(steps=steps)


@pytest.mark.parametrize("value", [math.nan, math.inf, 0, -1])
def test_stage2_learning_rate_is_finite_positive(value):
    with pytest.raises((ValueError, TypeError)):
        Stage2Config(global_learning_rate=value)


@pytest.mark.parametrize("name", ["learning_rate", "weight_decay", "max_grad_norm"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -1])
def test_training_numeric_values_are_finite_and_valid(name, value):
    with pytest.raises((ValueError, TypeError)):
        BytePFTTrainingConfig(**{name: value})


@pytest.mark.parametrize(
    "name",
    [
        "batch_size",
        "gradient_accumulation_steps",
        "max_bytes",
        "validation_interval",
        "checkpoint_interval",
        "warmup_steps",
        "seed",
    ],
)
@pytest.mark.parametrize("value", [1.5, True])
def test_training_count_values_are_integers(name, value):
    with pytest.raises((ValueError, TypeError)):
        BytePFTTrainingConfig(**{name: value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1])
def test_validation_split_bounds(value):
    with pytest.raises((ValueError, TypeError)):
        BytePFTTrainingConfig(validation_fraction=value)
