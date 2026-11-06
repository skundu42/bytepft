"""Measure cached versus reference execution of the same trained BytePFT model.

This is not a source-subword-model performance comparison. A matched-workload
source baseline remains deferred; fewer patches alone do not establish speedup.
"""

import argparse
import json
from pathlib import Path

from bytepft import BytePFT
from bytepft.evaluation import benchmark_generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-new-bytes", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    converter = BytePFT.load(args.model, device=args.device)
    cached = converter.generate(
        args.prompt,
        max_new_bytes=args.max_new_bytes,
        temperature=0,
        use_cache=True,
        return_bytes=True,
    )
    reference = converter.generate(
        args.prompt,
        max_new_bytes=args.max_new_bytes,
        temperature=0,
        use_cache=False,
        return_bytes=True,
    )
    if cached != reference:
        raise RuntimeError(
            "Cached and reference greedy byte outputs differ; investigate before timing."
        )
    report = {
        "comparison": "same BytePFT weights: cached versus uncached reference execution",
        "greedy_outputs_match": True,
        "source_model_comparison_measured": False,
        "cached": benchmark_generation(
            converter.model,
            args.prompt,
            max_new_bytes=args.max_new_bytes,
            repeats=args.repeats,
            use_cache=True,
        ),
        "reference": benchmark_generation(
            converter.model,
            args.prompt,
            max_new_bytes=args.max_new_bytes,
            repeats=args.repeats,
            use_cache=False,
        ),
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
