"""Small validated configuration objects; no implicit research presets."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _integer(name: str, value: Any, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(name: str, value: Any, minimum: float = 0.0, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if positive else 'at least'} {minimum}")


@dataclass
class BytePFTConfig:
    hidden_size: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 2
    bytes_per_patch: float = 4.0
    boundary_mode: str = "fixed"
    min_patch_size: int = 1
    max_patch_size: int = 16
    boundary_threshold: float = 0.5
    pooling: str = "last"
    architecture: str = "reference"
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.architecture != "reference":
            raise NotImplementedError(
                "Only architecture='reference' is implemented; mLSTM support is deferred"
            )
        for name in ("hidden_size", "encoder_layers", "decoder_layers", "min_patch_size"):
            _integer(name, getattr(self, name))
        _integer("max_patch_size", self.max_patch_size)
        _finite("bytes_per_patch", self.bytes_per_patch, positive=True)
        _finite("boundary_threshold", self.boundary_threshold, positive=True)
        if self.max_patch_size < self.min_patch_size:
            raise ValueError("max_patch_size must be >= min_patch_size")
        if not self.min_patch_size <= self.bytes_per_patch <= self.max_patch_size:
            raise ValueError("bytes_per_patch must be between min_patch_size and max_patch_size")
        if self.boundary_mode not in {"fixed", "learned", "oracle"}:
            raise ValueError("boundary_mode must be fixed, learned, or oracle")
        if self.pooling not in {"last", "mean"}:
            raise ValueError("pooling must be last or mean")
        if not 0 < self.boundary_threshold < 1:
            raise ValueError("boundary_threshold must be between 0 and 1")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")

    @property
    def patch_size(self) -> int:
        return round(self.bytes_per_patch)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1Config:
    steps: int = 0
    mode: str = "diagnostic"
    boundary_weight: float = 1.0
    embedding_weight: float = 1.0
    distillation_weight: float = 1.0
    byte_ce_weight: float = 1.0
    temperature: float = 5.0

    def __post_init__(self) -> None:
        _integer("stage1 steps", self.steps, 0)
        if self.mode not in {"diagnostic", "aligned"}:
            raise ValueError("stage1 mode must be diagnostic or aligned")
        for name in ("boundary_weight", "embedding_weight", "distillation_weight"):
            _finite(name, getattr(self, name))
        _finite("byte_ce_weight", self.byte_ce_weight, positive=True)
        _finite("temperature", self.temperature, positive=True)


@dataclass
class Stage2Config:
    steps: int = 0
    unfreeze_global: bool = True
    global_learning_rate: float | None = None

    def __post_init__(self) -> None:
        _integer("stage2 steps", self.steps, 0)
        if not isinstance(self.unfreeze_global, bool):
            raise ValueError("unfreeze_global must be boolean")
        if self.global_learning_rate is not None:
            _finite("global_learning_rate", self.global_learning_rate, positive=True)


@dataclass
class BytePFTTrainingConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_bytes: int = 256
    validation_fraction: float = 0.1
    validation_interval: int = 20
    checkpoint_interval: int = 50
    early_stopping_patience: int | None = None
    warmup_steps: int = 0
    seed: int = 42
    output_dir: str = "outputs/bytepft"
    text_field: str = "text"
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_bytes",
            "validation_interval",
            "checkpoint_interval",
        ):
            _integer(name, getattr(self, name))
        _finite("learning_rate", self.learning_rate, positive=True)
        _finite("max_grad_norm", self.max_grad_norm, positive=True)
        _finite("weight_decay", self.weight_decay)
        _finite("validation_fraction", self.validation_fraction)
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        _integer("warmup_steps", self.warmup_steps, 0)
        _integer("seed", self.seed, 0)
        if self.early_stopping_patience is not None:
            _integer("early_stopping_patience", self.early_stopping_patience)
        if not isinstance(self.stage1, Stage1Config) or not isinstance(self.stage2, Stage2Config):
            raise ValueError("stage1 and stage2 must use the corresponding configuration objects")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BytePFTTrainingConfig":
        values = dict(data)
        values["stage1"] = Stage1Config(**values.get("stage1", {}))
        values["stage2"] = Stage2Config(**values.get("stage2", {}))
        return cls(**values)
