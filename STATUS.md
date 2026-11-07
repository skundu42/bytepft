# BytePFT 0.1.0 status

Last updated: 2026-11-07. The experimental reference library has a working tiny
workflow and bounded pretrained Qwen runs that completed both adaptation
stages in fp32 on CPU and Apple MPS.

Latest CPU verification: **209 tests passed, 9 explicit GPU/pretrained cases skipped**.
The combined CPU/MPS suite passed **217 tests**, with the separate opt-in
pretrained case skipped. The real pretrained experiments ran separately.
Ruff and mypy pass.
Wheel and source distributions build and pass `twine check`.
The built wheel also installs and exposes its CLI in an isolated environment.
The previous GitHub matrix passed on Python 3.11, 3.12 and 3.13;
the [recorded run](docs/results/ci_validation.json) identifies that commit.
Regular CI now uses one Python 3.13 CPU job for tests and Ruff. Release builds
and publishing checks run separately when a GitHub release is published.

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

The CPU adaptation pilot completed eight aligned Stage 1 steps and three
unfrozen Stage 2 steps on Qwen3-0.6B-Base.
A sampled attention weight was identical to the source after Stage 1 and changed
after Stage 2. Offline reload reproduced the logits exactly. Cached and reference
generation agreed on three prompts, and the CLI reproduced the held-out score
and a greedy sample. On four disjoint hand-written test documents (175 bytes),
bits per byte went from **8.285** at initialization to **8.076** after Stage 1
and **7.506** after Stage 2. Original Qwen scored **1.441** along its canonical
tokenization, normalized over the same bytes. Pilot completions were repetitive
or contained invalid UTF-8. [CPU pilot report](docs/results/qwen_cpu_adaptation_pilot.json)

## Observed MPS experiments

An **MPS fp32** validation run passed eight dedicated GPU cases and
the combined 205-test suite with CPU fallback disabled. Tiny training byte NLL
fell from **5.762887 to 0.001340**. Real Qwen3-0.6B-Base also passed a frozen-global
gradient step and exact save/load on MPS. [MPS guide](docs/mps.md) ·
[Test report](docs/results/mps_validation.json) ·
[Pretrained smoke report](docs/results/qwen_mps_smoke.json)

A subsequent MPS fp32 run completed **500 aligned Stage 1 steps and 150
unfrozen Stage 2 steps** on Qwen3-0.6B-Base with operator fallback disabled.
Training and checkpoint writes took 247.3 seconds on an M3 Pro. On 128 held-out
WikiText-2 prefixes (16,126 bytes), BytePFT improved from **8.277** bits per byte
at initialization to **3.683** after Stage 1 and **3.398** after Stage 2.
Original Qwen scored **1.445** along its canonical tokenization on the same bytes.
CPU reload reproduced logits exactly. MPS reload matched the CPU likelihood
within 0.000001 bits per byte, and cached and reference greedy generation agreed
on all four prompts. Greedy completions repeated common words and seeded samples
remained incoherent. [MPS adaptation report](docs/results/qwen_mps_adaptation.json)

An expanded-data continuation warm-started that checkpoint and added **1,200
unfrozen Stage 2 updates** with a fresh optimizer. The run drew 2,400 samples
from 46,444 WikiText-2 training chunks and processed 544,107 UTF-8 bytes.
On the same 128 test excerpts, it improved from **3.398 to 2.706** bits per
byte. Original Qwen remained ahead at **1.445**. Offline CPU and MPS evaluation
agreed within 0.000001 bits per byte. Greedy generation remained repetitive,
and all 250 complete raw-byte patches on the validation split reached the
16-byte limit. [Expanded-data run report](docs/results/qwen_mps_expanded.json)

A time-matched Stage 2 comparison warm-started both arms from the expanded-data
checkpoint. One arm used periodic labels and the other used Qwen token ends
as supervision for the same causal boundary predictor. The periodic and Qwen
arms completed 301 and 267 updates, respectively, in about 240 seconds each.
On the same 128 test excerpts they scored **2.711** and **2.731** bits per byte.
The Qwen-label arm's paired difference was **+0.020** bits per byte, with a
95% document bootstrap interval of **+0.009 to +0.031** for this one seeded pair.

Qwen supervision reduced the mean complete test patch from 15.99 to 4.87 bytes
and the share hitting the 16-byte cap from 99.9% to 1.6%. Both checkpoints
still generated repetitive text on all four prompts. Cached and uncached
generation matched after offline reload. [Boundary comparison report](docs/results/qwen_mps_boundary_ab.json)

## Implemented but unvalidated here

- CUDA bf16/fp16 AMP (including GradScaler), CUDA memory measurements and
  MPS mixed precision. MPS validation currently covers fp32 only.
- Optional remote Hugging Face dataset ingestion and activation checkpointing
  in long pretrained runs. CPU-scale activation checkpointing is tested.
- PyPI publishing uses Trusted Publishing: publishing a GitHub release publishes
  the version from its tag. A build-only manual dispatch validates the release
  without uploading to PyPI.

## Not run or not supported

- Sustained Qwen adaptation, retained-capability evaluation, code-completion or
  character-reasoning quality suites, and matched source-model speed benchmarks.
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

[Research mapping](docs/research.md) · [Architecture](docs/architecture.md) ·
[Training guide](docs/training.md)
