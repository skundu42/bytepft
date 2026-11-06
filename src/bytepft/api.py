"""Public conversion API. Models never fall back to subword generation."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from bytepft.adapters.qwen import QwenGlobalAdapter, TinyGlobalAdapter
from bytepft.config import BytePFTConfig, BytePFTTrainingConfig, Stage1Config, Stage2Config
from bytepft.exceptions import CompatibilityError
from bytepft.models.byte_model import ByteModel
from bytepft.serialization import load_pretrained
from bytepft.training.trainer import BytePFTTrainer


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise CompatibilityError("CUDA was requested but is unavailable")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise CompatibilityError("Apple MPS was requested but is unavailable")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise CompatibilityError("Supported devices are cpu, cuda, mps and auto")
    return resolved


class BytePFT:
    """Configure, adapt and persist a byte interface around a Qwen3 backbone.

    Loading pretrained weights alone does not produce an adapted byte model.
    Defaults use CPU; selecting ``auto`` explicitly permits an available accelerator.
    """

    def __init__(self, adapter: QwenGlobalAdapter, config: BytePFTConfig | None = None) -> None:
        self.adapter = adapter
        self.model = ByteModel(adapter, config)
        self.trainer: BytePFTTrainer | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen3-0.6B-Base",
        *,
        device: str = "cpu",
        precision: str = "fp32",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "BytePFT":
        adapter = QwenGlobalAdapter.from_pretrained(
            model_id,
            device=resolve_device(device),
            precision=precision,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(adapter, BytePFTConfig(precision=precision))

    @classmethod
    def from_tiny(
        cls,
        *,
        config: BytePFTConfig | None = None,
        seed: int = 42,
        device: str = "cpu",
        global_hidden_size: int = 32,
    ) -> "BytePFT":
        # Private RNG scope makes initialization reproducible without resetting caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            adapter = TinyGlobalAdapter(hidden_size=global_hidden_size)
            instance = cls(adapter, config)
        resolved = resolve_device(device)
        precision = instance.model.config.precision
        if precision == "fp16" and resolved.type == "cpu":
            raise CompatibilityError("fp16 on CPU is unsupported; use fp32 or bf16")
        if precision == "bf16" and resolved.type == "mps":
            raise CompatibilityError("The reference GRU path uses fp32 or fp16 on MPS")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        instance.model.to(device=resolved, dtype=dtype)
        return instance

    def configure(self, **settings: Any) -> "BytePFT":
        if self.model.training_status["stage1_steps"] or self.model.training_status["stage2_steps"]:
            raise ValueError(
                "Cannot reinitialize a trained byte interface; create another converter"
            )
        config = replace(self.model.config, **settings)
        device = self.model.device
        if config.precision == "fp16" and device.type == "cpu":
            raise CompatibilityError("fp16 on CPU is unsupported; use fp32 or bf16")
        if config.precision == "bf16" and device.type == "mps":
            raise CompatibilityError("The reference GRU path uses fp32 or fp16 on MPS")
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            config.precision
        ]
        self.model = ByteModel(self.adapter, config).to(device=device, dtype=dtype)
        return self

    def retrofit(
        self,
        dataset: Any,
        *,
        stage1_steps: int = 1000,
        stage2_steps: int = 0,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        output_dir: str = "outputs/bytepft",
        stage1_mode: str | None = None,
        training_config: BytePFTTrainingConfig | None = None,
        validation_dataset: Any = None,
        resume_from: str | Path | None = None,
        **training_settings: Any,
    ) -> list[dict[str, Any]]:
        mode = stage1_mode or ("diagnostic" if self.adapter.kind == "tiny" else "aligned")
        config = training_config or BytePFTTrainingConfig(
            stage1=Stage1Config(steps=stage1_steps, mode=mode),
            stage2=Stage2Config(steps=stage2_steps),
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            **training_settings,
        )
        if training_config is not None and training_settings:
            raise ValueError("Put training settings in training_config when one is supplied")
        source_tokenizer = None
        if config.stage1.steps and config.stage1.mode == "aligned":
            if not self.adapter.source_model_id:
                raise ValueError("Aligned Stage 1 requires a pretrained source model identity")
            from transformers import AutoTokenizer

            source_tokenizer = AutoTokenizer.from_pretrained(
                self.adapter.source_model_id,
                revision=self.adapter.source_revision,
                trust_remote_code=False,
            )
        self.trainer = BytePFTTrainer(self.model, config, source_tokenizer=source_tokenizer)
        return self.trainer.fit(
            dataset, validation_dataset=validation_dataset, resume_from=resume_from
        )

    def generate(self, prompt: str = "", **settings: Any) -> Any:
        return self.model.generate(prompt, **settings)

    def save_pretrained(self, directory: str | Path) -> None:
        self.model.save_pretrained(str(directory))

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> "BytePFT":
        model = load_pretrained(directory, device=str(resolve_device(device)))
        instance = cls.__new__(cls)
        instance.model = model
        instance.adapter = model.global_adapter
        instance.trainer = None
        return instance

    def stats(self) -> dict[str, Any]:
        return self.model.stats()
