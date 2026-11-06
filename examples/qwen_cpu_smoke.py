"""Download Qwen3-0.6B-Base, run one CPU gradient step, and verify persistence.

This is an integration check, not a completed adaptation experiment. It downloads
approximately 1.2 GB and writes approximately 2.4 GB of fp32 checkpoint weights.
No GPU or MPS operation is performed. Run with ``uv run python
examples/qwen_cpu_smoke.py --output outputs/qwen-cpu-smoke``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from bytepft.adapters import QwenGlobalAdapter
from bytepft.config import BytePFTConfig
from bytepft.models.byte_model import ByteModel
from bytepft.serialization import load_pretrained


def run_smoke(
    output: str | Path = "outputs/qwen-cpu-smoke",
    *,
    source_model: str = "Qwen/Qwen3-0.6B-Base",
    local_files_only: bool = False,
    threads: int = 2,
) -> dict[str, Any]:
    """Execute a real, small integration check with all tensors on CPU."""
    if threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(threads)
    torch.manual_seed(23)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "source_model": source_model,
        "device": "cpu",
        "precision": "fp32",
        "gpu_used": False,
        "platform": platform.platform(),
        "cpu_architecture": platform.machine(),
        "threads": threads,
        "torch_version": torch.__version__,
        "purpose": "one real frozen-global gradient step and checkpoint round trip",
        "full_adaptation_completed": False,
        "quality_evaluated": False,
    }
    started = time.perf_counter()
    adapter = QwenGlobalAdapter.from_pretrained(
        source_model, device="cpu", precision="fp32", local_files_only=local_files_only
    )
    report["load_seconds"] = time.perf_counter() - started
    report["source_revision"] = adapter.source_revision
    report["original_parameters"] = adapter.original_parameter_count
    adapter.freeze()
    model = ByteModel(
        adapter,
        BytePFTConfig(hidden_size=16, encoder_layers=1, decoder_layers=1, boundary_mode="fixed"),
    )
    del adapter
    model.train()
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    versions = {
        name: parameter._version for name, parameter in model.global_adapter.named_parameters()
    }
    local_before = model.to_global.weight.detach().clone()
    local_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    report["trainable_parameters"] = sum(parameter.numel() for parameter in local_parameters)
    optimizer = torch.optim.AdamW(local_parameters, lr=1e-3)
    ids = torch.tensor([[257, 72, 105, 258]], dtype=torch.long)
    labels = torch.tensor([[72, 105, 258, -100]], dtype=torch.long)
    started = time.perf_counter()
    result = model(ids, labels=labels)
    assert result.loss is not None
    if not torch.isfinite(result.loss):
        raise RuntimeError("The real Qwen integration produced a nonfinite loss.")
    result.loss.backward()
    gradients = [parameter.grad for parameter in local_parameters if parameter.grad is not None]
    finite = bool(gradients) and all(
        torch.isfinite(gradient).all().item() for gradient in gradients
    )
    if not finite:
        raise RuntimeError(
            "The real Qwen integration produced missing or nonfinite local gradients."
        )
    if any(parameter.grad is not None for parameter in model.global_adapter.parameters()):
        raise RuntimeError("Frozen Qwen parameters unexpectedly received gradients.")
    report["loss"] = float(result.loss.detach())
    report["logits_shape"] = list(result.logits.shape)
    report["finite_local_gradients"] = finite
    optimizer.step()
    frozen_unchanged = all(
        parameter._version == versions[name]
        for name, parameter in model.global_adapter.named_parameters()
    )
    local_updated = not torch.equal(local_before, model.to_global.weight)
    if not frozen_unchanged or not local_updated:
        raise RuntimeError(
            "The expected frozen/global versus trainable/local update invariant failed."
        )
    report["global_parameters_unchanged"] = frozen_unchanged
    report["local_projection_updated"] = local_updated
    report["gradient_step_seconds"] = time.perf_counter() - started
    # A single manual smoke step is intentionally insufficient for generation.
    model.training_status.update(stage1_steps=1, stage1="smoke_step_only", trained=False)
    model.eval()
    with torch.no_grad():
        expected_logits = model(ids).logits.detach().clone()
    started = time.perf_counter()
    model.save_pretrained(str(output))
    report["save_seconds"] = time.perf_counter() - started
    report["checkpoint_bytes"] = (output / "model.safetensors").stat().st_size
    # Release the source model and all optimizer/graph references before loading.
    del result, optimizer, gradients, local_parameters, local_before, model
    gc.collect()
    started = time.perf_counter()
    loaded = load_pretrained(output, device="cpu")
    report["reload_seconds"] = time.perf_counter() - started
    with torch.no_grad():
        actual_logits = loaded(ids).logits
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-6)
    report["round_trip_max_abs_logit_difference"] = float(
        (actual_logits - expected_logits).abs().max()
    )
    report["round_trip_matches"] = True
    report["training_status"] = dict(loaded.training_status)
    del loaded
    gc.collect()
    (output / "smoke_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/qwen-cpu-smoke")
    parser.add_argument("--source-model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_smoke(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
