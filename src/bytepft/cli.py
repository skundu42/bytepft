"""Command-line entry points with explicit CPU defaults."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from bytepft import BytePFT, BytePFTConfig, BytePFTTrainingConfig, Stage1Config, Stage2Config
from bytepft.evaluation.benchmarks import benchmark_generation
from bytepft.evaluation.metrics import evaluate_model
from bytepft.training.datasets import load_texts


def _output(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bytepft", description="Adapt Qwen3 transformers to UTF-8 bytes"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Inspect architecture metadata without loading weights"
    )
    inspect.add_argument("model")
    for name, help_text in (
        ("retrofit", "Run explicit adaptation stages"),
        ("train-tiny", "Train an offline tiny reference model"),
    ):
        sub = commands.add_parser(name, help=help_text)
        if name == "retrofit":
            sub.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
            sub.add_argument("--dataset", required=True)
        else:
            sub.add_argument("--dataset", help="Defaults to a tiny synthetic corpus")
        sub.add_argument("--config", help="YAML with architecture and training mappings")
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", default="cpu", help="cpu (default), cuda, mps, auto")
        sub.add_argument("--stage1-steps", type=int)
        sub.add_argument("--stage2-steps", type=int)
        sub.add_argument("--resume-from")
    generate = commands.add_parser("generate", help="Generate from a trained byte checkpoint")
    generate.add_argument("--model", required=True)
    generate.add_argument("--prompt", default="")
    generate.add_argument("--max-new-bytes", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument(
        "--reference", action="store_true", help="Use the uncached correctness path"
    )
    generate.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate", help="Measure NLL and bits per UTF-8 byte")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--text-field", default="text")
    evaluate.add_argument("--max-bytes", type=int, default=256)
    evaluate.add_argument("--device", default="cpu")
    benchmark = commands.add_parser(
        "benchmark", help="Measure real byte generation latency and throughput"
    )
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--prompt", default="Hello")
    benchmark.add_argument("--max-new-bytes", type=int, default=128)
    benchmark.add_argument("--repeats", type=int, default=1)
    benchmark.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            path = Path(args.model)
            if (path / "bytepft_config.json").exists():
                _output(json.loads((path / "bytepft_config.json").read_text()))
            else:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
                _output(
                    {
                        "model_type": config.model_type,
                        "supported": config.model_type == "qwen3",
                        "config": config.to_dict(),
                    }
                )
        elif args.command in {"retrofit", "train-tiny"}:
            data = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
            if not isinstance(data, dict):
                raise ValueError("Configuration YAML must be a mapping")
            for section in ("architecture", "training"):
                if section in data and not isinstance(data[section], dict):
                    raise ValueError(f"Configuration section {section!r} must be a mapping")
            unknown = set(data) - {"architecture", "training"}
            if unknown:
                raise ValueError(f"Unknown config sections: {sorted(unknown)}")
            tiny = args.command == "train-tiny"
            architecture = BytePFTConfig(**data.get("architecture", {}))
            training = BytePFTTrainingConfig.from_dict(data.get("training", {}))
            training.output_dir = args.output
            if not data.get("training"):
                training.stage1 = Stage1Config(
                    steps=60 if tiny else 1000, mode="diagnostic" if tiny else "aligned"
                )
                training.stage2 = Stage2Config(steps=0)
            if args.stage1_steps is not None:
                training.stage1.steps = args.stage1_steps
            if args.stage2_steps is not None:
                training.stage2.steps = args.stage2_steps
            training.stage1.__post_init__()
            training.stage2.__post_init__()
            training.__post_init__()
            converter = (
                BytePFT.from_tiny(config=architecture, device=args.device)
                if tiny
                else BytePFT.from_pretrained(
                    args.model, device=args.device, precision=architecture.precision
                )
            )
            if not tiny:
                converter.configure(**architecture.to_dict())
            documents = (
                load_texts(args.dataset, text_field=training.text_field)
                if args.dataset
                else ["hello byte world\n"] * 4
            )
            history = converter.retrofit(
                documents, training_config=training, resume_from=args.resume_from
            )
            _output({"last_step": history[-1] if history else None, "stats": converter.stats()})
        else:
            converter = BytePFT.load(args.model, device=args.device)
            if args.command == "generate":
                print(
                    converter.generate(
                        args.prompt,
                        max_new_bytes=args.max_new_bytes,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        use_cache=not args.reference,
                    )
                )
            elif args.command == "evaluate":
                _output(
                    evaluate_model(
                        converter.model,
                        load_texts(args.dataset, text_field=args.text_field),
                        max_bytes=args.max_bytes,
                    )
                )
            else:
                _output(
                    benchmark_generation(
                        converter.model,
                        args.prompt,
                        max_new_bytes=args.max_new_bytes,
                        repeats=args.repeats,
                    )
                )
        return 0
    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        NotImplementedError,
        yaml.YAMLError,
    ) as exc:
        print(f"bytepft: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
