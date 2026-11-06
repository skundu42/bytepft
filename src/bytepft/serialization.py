"""Portable, offline byte checkpoints with safetensors model weights."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_model, save_model

from bytepft.adapters.qwen import QwenGlobalAdapter
from bytepft.config import BytePFTConfig
from bytepft.exceptions import CheckpointError
from bytepft.models.byte_model import ByteModel
from bytepft.tokenization.byte_tokenizer import ByteTokenizer

FORMAT_VERSION = 1


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file, rejecting nonfinite metrics."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_pretrained(model: ByteModel, directory: str | Path) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    weights_path = path / "model.safetensors"
    temporary = path / "model.safetensors.tmp"
    save_model(model, str(temporary))
    os.replace(temporary, weights_path)
    metadata = {
        "format_version": FORMAT_VERSION,
        "bytepft_version": "0.1.0",
        "saved_at": datetime.now(UTC).isoformat(),
        "source_model": model.global_adapter.source_model_id,
        "source_revision": model.global_adapter.source_revision,
    }
    model.checkpoint_metadata = metadata
    write_json(
        path / "bytepft_config.json",
        {
            "format_version": FORMAT_VERSION,
            "config": model.config.to_dict(),
            "global_config": model.global_adapter.hf_config,
            "adapter_kind": model.global_adapter.kind,
            "source_model_id": model.global_adapter.source_model_id,
            "source_revision": model.global_adapter.source_revision,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "training_status": model.training_status,
            "metrics": model.metrics,
            "patch_statistics": model.patch_statistics,
            "checkpoint_metadata": metadata,
            "global_frozen": not any(p.requires_grad for p in model.global_adapter.parameters()),
        },
    )
    model.tokenizer.save_pretrained(path)


def load_pretrained(directory: str | Path, *, device: str = "cpu") -> ByteModel:
    path = Path(directory)
    try:
        data = json.loads((path / "bytepft_config.json").read_text(encoding="utf-8"))
        if data["format_version"] != FORMAT_VERSION:
            raise CheckpointError(f"Unsupported checkpoint version: {data['format_version']}")
        config = BytePFTConfig(**data["config"])
        adapter = QwenGlobalAdapter.from_config(
            data["global_config"],
            kind=data["adapter_kind"],
            source_model_id=data["source_model_id"],
            source_revision=data.get("source_revision"),
        )
        model = ByteModel(adapter, config)
        model.tokenizer = ByteTokenizer.from_pretrained(path)
        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        if data["dtype"] not in dtypes:
            raise CheckpointError("Unsupported checkpoint tensor dtype")
        model.to(device=device, dtype=dtypes[data["dtype"]])
        load_model(model, str(path / "model.safetensors"), strict=True, device=device)
        model.training_status = data["training_status"]
        model.metrics = data["metrics"]
        model.patch_statistics = data["patch_statistics"]
        model.checkpoint_metadata = data["checkpoint_metadata"]
        if data.get("global_frozen"):
            adapter.freeze()
        model.eval()
        return model
    except (
        OSError,
        KeyError,
        TypeError,
        RuntimeError,
        SafetensorError,
        json.JSONDecodeError,
    ) as exc:
        raise CheckpointError(f"Cannot load BytePFT checkpoint at {path}: {exc}") from exc
