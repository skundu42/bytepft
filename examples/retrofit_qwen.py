"""Run explicit Stage 1/Stage 2 Qwen adaptation on a local dataset.

The default configuration is an experimental GPU training preset. Full Qwen
adaptation has not been validated by the included CPU smoke check. Example:

uv run python examples/retrofit_qwen.py --dataset data/train.jsonl --device cuda

For a deliberately small CPU experiment, also specify --precision fp32 and
--stage1-steps 1. No accelerator is selected implicitly.
"""

import argparse
import json
from pathlib import Path

import yaml

from bytepft import BytePFT, BytePFTConfig, BytePFTTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / "configs/qwen3_06b.yaml")
    )
    parser.add_argument("--output", default="outputs/qwen-byte")
    parser.add_argument("--stage1-steps", type=int)
    parser.add_argument("--stage2-steps", type=int)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    architecture = BytePFTConfig(**settings["architecture"])
    if args.precision is not None:
        architecture.precision = args.precision
        architecture.__post_init__()
    if args.device in {"cpu", "mps"} and architecture.precision != "fp32":
        parser.error("This preset uses accelerator precision; pass --precision fp32 for CPU/MPS.")
    training = BytePFTTrainingConfig.from_dict(settings["training"])
    training.output_dir = args.output
    if args.stage1_steps is not None:
        training.stage1.steps = args.stage1_steps
    if args.stage2_steps is not None:
        training.stage2.steps = args.stage2_steps
    training.stage1.__post_init__()
    training.stage2.__post_init__()
    training.__post_init__()
    converter = BytePFT.from_pretrained(
        args.model, device=args.device, precision=architecture.precision
    )
    converter.configure(**architecture.to_dict())
    converter.retrofit(args.dataset, training_config=training, resume_from=args.resume_from)
    print(json.dumps(converter.stats(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
