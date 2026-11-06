# BytePFT

[![CI](https://github.com/skundu42/bytepft/actions/workflows/ci.yml/badge.svg)](https://github.com/skundu42/bytepft/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Retrofitting language models, one byte at a time.**

BytePFT is a Python library for adapting pretrained Qwen3 language models to
UTF-8 bytes. It trains a byte-level encoder and decoder around the existing
transformer, with patch-based processing and cached byte generation.

BytePFT is experimental. The reference implementation is tested on CPU and Apple
MPS in fp32; full pretrained adaptation and retained language quality have not yet
been evaluated. See [project status](https://github.com/skundu42/bytepft/blob/main/STATUS.md) for results and limitations.

## Installation

Requires Python 3.11+.

```sh
pip install bytepft
```

Before the first PyPI release, install from source:

```sh
pip install git+https://github.com/skundu42/bytepft.git
```

## Quickstart

Train a tiny model locally, without downloading pretrained weights:

```python
from bytepft import BytePFT, BytePFTConfig

model = BytePFT.from_tiny(
    config=BytePFTConfig(hidden_size=32, encoder_layers=1, decoder_layers=1),
    device="cpu",
)
model.retrofit(
    ["hello byte world\n"] * 4,
    stage1_steps=100,
    learning_rate=0.01,
    validation_fraction=0,
    output_dir="outputs/tiny",
)

model = BytePFT.load("outputs/tiny")
print(model.generate("hello", max_new_bytes=32, temperature=0))
```

This example uses a randomly initialized tiny transformer. To adapt pretrained
weights, use `BytePFT.from_pretrained("Qwen/Qwen3-0.6B-Base")` and follow the
[training guide](https://github.com/skundu42/bytepft/blob/main/docs/training.md). Adaptation must complete before generation.

## Features

- UTF-8 byte encoding with separate padding, beginning and end symbols.
- Fixed, oracle and learned patch boundaries.
- Frozen-backbone Stage 1 and optional end-to-end Stage 2 training.
- Resumable checkpoints, validation and bits-per-byte evaluation.
- Greedy and sampled generation with local state and global KV caching.

Defaults use CPU. Select `device="mps"` explicitly for Apple GPU execution;
[MPS validation](https://github.com/skundu42/bytepft/blob/main/docs/mps.md) currently covers fp32. CUDA remains unvalidated.

## Command line

```sh
bytepft train-tiny --output outputs/tiny --device cpu
bytepft generate --model outputs/tiny --prompt "hello" --max-new-bytes 32
bytepft inspect Qwen/Qwen3-0.6B-Base
```

Run `bytepft --help` for all commands.

## Documentation

- [Training and checkpoints](https://github.com/skundu42/bytepft/blob/main/docs/training.md)
- [Architecture](https://github.com/skundu42/bytepft/blob/main/docs/architecture.md)
- [Research notes](https://github.com/skundu42/bytepft/blob/main/docs/research.md)
- [Examples](https://github.com/skundu42/bytepft/tree/main/examples)
- [Contributing](https://github.com/skundu42/bytepft/blob/main/CONTRIBUTING.md) · [Releases](https://github.com/skundu42/bytepft/blob/main/docs/releases.md) · [Changelog](https://github.com/skundu42/bytepft/blob/main/CHANGELOG.md)

Inspired by [*Retrofitting language models to operate over bytes*, Nature (2026)](https://www.nature.com/articles/s41586-026-11111-4).
BytePFT uses a simplified reference architecture; the research notes document its deviations.

## License

[Apache-2.0](https://github.com/skundu42/bytepft/blob/main/LICENSE). See [NOTICE](https://github.com/skundu42/bytepft/blob/main/NOTICE) for attribution.
