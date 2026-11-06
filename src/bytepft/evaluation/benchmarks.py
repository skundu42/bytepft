"""Measured generation timing for an explicit model/device/workload."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

from bytepft.inference.generation import GenerationResult, generate

if TYPE_CHECKING:
    from bytepft.models.byte_model import ByteModel


def benchmark_generation(
    model: ByteModel,
    prompt: str = "Hello",
    *,
    max_new_bytes: int = 128,
    use_cache: bool = True,
    repeats: int = 1,
) -> dict[str, Any]:
    """Report observed throughput, with EOS-dependent output lengths visible.

    No source-model speed comparison is inferred. CPU and MPS peak accelerator
    memory is unavailable here and reported as null rather than estimated.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    measurements: list[dict[str, Any]] = []
    for _ in range(repeats):
        result = generate(
            model,
            prompt,
            max_new_bytes=max_new_bytes,
            temperature=0,
            use_cache=use_cache,
            return_details=True,
        )
        assert isinstance(result, GenerationResult)
        entry = asdict(result)
        del entry["data"], entry["incomplete_utf8_suffix"], entry["generated_ids"], entry["text"]
        measurements.append(entry)
    memory_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model.parameters())
    memory_bytes += sum(tensor.numel() * tensor.element_size() for tensor in model.buffers())
    total_seconds = sum(item["generation_seconds"] for item in measurements)
    total_bytes = sum(item["generated_bytes"] for item in measurements)
    return {
        "device": str(model.device),
        "dtype": str(model.embeddings.weight.dtype),
        "cached": use_cache,
        "prompt": prompt,
        "requested_max_new_bytes": max_new_bytes,
        "repeats": repeats,
        "model_tensor_memory_bytes": memory_bytes,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None
        ),
        "aggregate_bytes_per_second": total_bytes / total_seconds if total_seconds > 0 else 0.0,
        "measurements": measurements,
        "conditions": "single stream, greedy, no warmup; includes Python/CPU sampling overhead",
    }
