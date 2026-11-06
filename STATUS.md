# BytePFT 0.1.0 status

Last updated: 2026-11-06. Local execution used CPU only; no GPU training or PyPI
publication was performed. This is an experimental reference library, with a
working end-to-end tiny workflow and a real pretrained-backbone integration test.
It is not a validated reproduction of the full research architecture.

Local verification: **197 tests passed, 1 opt-in pretrained test skipped**; the
pretrained smoke script was executed separately. Ruff and mypy (31 source files)
pass. Wheel and source distributions build and pass `twine check`.
The built wheel also installs and exposes its CLI in an isolated environment.
GitHub CI passes on Python 3.11, 3.12 and 3.13;
the [recorded run](docs/results/ci_validation.json) identifies the verified commit.

## Implemented and tested

- Installable Python package, typed public API, validated configurations and CLI.
- Deterministic UTF-8 byte encoding, separate control IDs, raw bytes, batching,
  right padding, incremental text decoding and tokenizer persistence.
- Trainable embeddings, causal GRU local encoder/decoder, fixed/oracle/learned
  boundary prediction, last/mean pooling and completed-patch-only depooling.
- Qwen3 compatibility checks, original pretrained transformer weights,
  positional/mask handling, freezing/unfreezing and native global KV caches.
- Byte-only greedy and temperature/top-k/top-p generation. Cached and full
  reference paths agree; no source tokenizer is needed at inference.
- Document-preserving text/JSONL/directory loading, deterministic splitting,
  byte truncation at Unicode boundaries and optional Hugging Face dataset loading.
- Diagnostic Stage 1 and aligned Stage 1 with exact source-token byte extraction,
  boundary BCE, n=0 embedding alignment and compatible token-event distillation.
- Explicit Stage 2, separate learning rates, gradient accumulation/clipping,
  scheduling, validation, early stopping and optional activation checkpointing.
- CPU fp32 and bf16 AMP training, fp32 master weights/optimizer moments, resumable
  checkpoints across both stages, dataset fingerprints and saved random states.
- Safetensors persistence, offline loading, corrupt checkpoint validation,
  untrained-generation guards and structured statistics.
- Measured byte NLL, bits per byte, generation timing, bytes/second, patch calls
  and model tensor memory. No source-model speedup is inferred.
- Unit, gradient, causality, training, generation, API, CLI and persistence tests;
  wheel/source builds; CI and an explicit maintainer release workflow.

## Observed CPU experiments

The [tiny experiment](docs/results/tiny_cpu_example.json) trained a 54,500-parameter
randomly initialized reference model for 100 diagnostic Stage 1 steps while keeping
its global transformer frozen. Training byte NLL decreased from **5.689678 to
0.002905**. Greedy generation from `hello` produced `hello byte world\n` and EOS.
This measures synthetic training-corpus memorization, not general language quality.

The [pretrained integration](docs/results/qwen_cpu_smoke.json) loaded the actual
**Qwen/Qwen3-0.6B-Base** checkpoint on CPU: 596,049,920 source parameters and
45,204 new parameters in its deliberately small local interface. One gradient
step had finite loss **6.297652**; local weights updated, and all global parameter
versions stayed unchanged with no global gradients. The checkpoint saved and
reloaded offline with maximum absolute logit difference **0.0**. Its status remains
`smoke_step_only`, `trained=false`; generation is correctly blocked.

Actual source-tokenizer alignment was also checked on multilingual text, split
UTF-8 code points, emoji, whitespace and NUL. Neither this nor a single gradient
step establishes successful pretrained-model adaptation.

## Implemented but unvalidated here

- CUDA bf16/fp16 AMP (including GradScaler), CUDA memory measurements and MPS
  fp32 execution. CPU tests do not establish accelerator compatibility or speed.
- Optional remote Hugging Face dataset ingestion and activation checkpointing
  in long pretrained runs. CPU-scale activation checkpointing is tested.
- PyPI Trusted Publishing requires maintainer configuration and an explicit
  publishing action. It has not been configured or executed here.

## Not run or not supported

- Full Qwen adaptation, retained-capability evaluation, code-completion or
  character-reasoning quality suites, and matched source-model benchmarks.
- The research mLSTM implementation, noncausal prefill boundaries, fused
  byte/boundary output, suffix embedding augmentation and n=4 alignment.
- Non-Qwen3 adapters, optimized kernels, batched generation, distributed
  training, serving integrations and quantization-aware adaptation.

## Known limits

The reference architecture uses GRUs, causal boundaries and separate output
heads. Aligned Stage 1 uses teacher global context and n=0 input alignment;
it is a weaker research baseline. Oracle training boundaries preserve source
token alignment, while inference enforces configured length constraints.
Stage 2 supervises compression with fixed-size boundary targets. A requested
learned bytes-per-patch value is a target surrogate, not a guaranteed measured rate.

Incremental generation handles a single stream and prefills sequentially.
Invalid generated UTF-8 is replaced in text views; incomplete suffixes remain
buffered and are available in raw-byte results. Full checkpoints retain source
embeddings/head for alignment even though byte inference does not use them.
AMP retains fp32 master weights, so GPU memory needs exceed half-weight loading.
Resumption requires the same configuration and documents; exact numerical
reproducibility is only tested on the same CPU backend.

Large pretrained adaptation should run on explicitly chosen training hardware.
No such run has been started on this device. See [research mapping](docs/research.md),
[architecture](docs/architecture.md) and [training guide](docs/training.md).
