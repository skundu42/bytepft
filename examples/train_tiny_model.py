"""Overfit a small synthetic corpus using CPU and a randomly initialized backbone.

Run: uv run python examples/train_tiny_model.py
The saved model generates with its own byte head. This demonstration does not
establish pretrained-Qwen adaptation or general language-model quality.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from bytepft import BytePFT, BytePFTConfig, BytePFTTrainingConfig
from bytepft.evaluation import evaluate_model
from bytepft.inference import GenerationResult


def run_example(
    *,
    output: str | Path = "outputs/tiny-example",
    config_path: str | Path | None = None,
    steps: int | None = None,
    stage2_steps: int | None = None,
    boundary_mode: str | None = None,
    threads: int = 2,
) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(threads)
    path = Path(config_path or Path(__file__).resolve().parents[1] / "configs/tiny_smoke.yaml")
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    architecture = BytePFTConfig(**settings["architecture"])
    if boundary_mode is not None:
        architecture.boundary_mode = boundary_mode
        architecture.__post_init__()
    training = BytePFTTrainingConfig.from_dict(settings["training"])
    training.output_dir = str(output)
    if steps is not None:
        training.stage1.steps = steps
    if stage2_steps is not None:
        training.stage2.steps = stage2_steps
        training.stage2.unfreeze_global = True
    training.stage1.__post_init__()
    training.stage2.__post_init__()
    training.__post_init__()
    documents = ["hello byte world\n"] * 4
    converter = BytePFT.from_tiny(config=architecture, seed=training.seed, device="cpu")
    initial = evaluate_model(converter.model, documents)
    started = time.perf_counter()
    history = converter.retrofit(documents, training_config=training)
    training_seconds = time.perf_counter() - started
    final = evaluate_model(converter.model, documents)
    generated = converter.generate("hello", max_new_bytes=48, temperature=0, return_details=True)
    assert isinstance(generated, GenerationResult)
    report: dict[str, Any] = {
        "experiment": "synthetic overfit with randomly initialized global transformer",
        "device": "cpu",
        "gpu_used": False,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "threads": threads,
        "seed": training.seed,
        "architecture": architecture.to_dict(),
        "training": training.to_dict(),
        "training_seconds": training_seconds,
        "initial_byte_nll": initial["nll"],
        "final_byte_nll": final["nll"],
        "initial_bits_per_byte": initial["bits_per_byte"],
        "final_bits_per_byte": final["bits_per_byte"],
        "first_training_loss": history[0]["loss"],
        "last_training_loss": history[-1]["loss"],
        "relative_nll_reduction": 1 - final["nll"] / initial["nll"],
        "generation": generated.to_dict(),
        "stats": converter.stats(),
        "general_language_quality_evaluated": False,
    }
    destination = Path(output) / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--output", default="outputs/tiny-example")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stage2-steps", type=int)
    parser.add_argument("--boundary-mode", choices=("fixed", "learned"))
    parser.add_argument("--threads", type=int, default=2)
    print(json.dumps(run_example(**vars(parser.parse_args())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
