# Apple MPS validation

Bounded fp32 validation passed on an Apple M3 Pro with 18 GB unified memory,
macOS 26.6.2 and PyTorch 2.15.0. It includes a two-stage pretrained Qwen run
with held-out likelihood and generation checks.

## Reproduce the checks

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 RUN_MPS_TESTS=1 uv run pytest tests/test_mps.py -q
```

The eight opt-in cases cover CPU/MPS forward agreement, causality, fixed/learned
boundaries, last/mean pooling, padding, incremental/reference agreement, frozen
and aligned Stage 1, unfrozen tiny Stage 2, resumption, persistence and tiny-corpus
overfitting. The combined suite passed 205 tests with one separate pretrained
integration test skipped. See the [measured report](results/mps_validation.json).
GPU tests stay disabled in ordinary CI unless explicitly enabled.

For the real pretrained backbone, using already cached source weights:

```sh
uv run python examples/qwen_cpu_smoke.py --device mps --local-files-only \
  --output outputs/qwen-mps-smoke
```

The existing smoke script retains its CPU default and accepts an explicit MPS
device. It loads Qwen3-0.6B-Base, runs one frozen-global gradient step with a small
16-dimensional local interface, saves a checkpoint, releases the model and reloads
it on MPS. Omit `--local-files-only` only when you intend to download weights.

The [real MPS result](results/qwen_mps_smoke.json) records finite loss 6.297651,
finite local gradients, updated local weights, unchanged global parameter versions
and no global gradients. Save/load reproduced logits with absolute difference 0.0.
After the step, MPS tensor allocation was 2,400,221,696 bytes and driver allocation
was 2,830,876,672 bytes. These are snapshots, not peak memory measurements. The
short step took 1.71 seconds including first-use overhead; it is not a throughput
estimate for a normal training sequence or larger local interface.

## Validated scope

Use `device="mps"` and `precision="fp32"`. The standard Qwen YAML defaults to bf16
for CUDA experiments; change it to fp32 for MPS. The pretrained retrofit example
accepts `--device mps --precision fp32`. MPS fp16/bf16 adaptation is not supported
by this reference training policy.

## Pretrained adaptation on WikiText-2

An MPS fp32 run completed 500 aligned Stage 1 updates and 150 unfrozen Stage 2
updates on Qwen3-0.6B-Base. It used batch size 1, at most 128 UTF-8 bytes per
document, and activation checkpointing. Stage 2 updated a sampled Qwen
attention weight that stayed identical to the source through Stage 1. Training
and checkpoint writes took 247.3 seconds. MPS reported 8.24 GB of driver
allocation after training, a snapshot taken before moving the model to CPU.

The offline MPS checkpoint reproduced the held-out CPU likelihood within
0.000001 bits per byte. Cached and uncached greedy generation agreed on four
prompts across both devices. All four greedy completions repeated common words.
Operator fallback was disabled. Byte generation uses CPU sampling and control
work as part of the reference implementation.
[Experiment settings and results](results/qwen_mps_adaptation.json)

The expanded-data run reused the trained weights and reset the optimizer for
1,200 more unfrozen Stage 2 updates. It kept the original validation and test
excerpts and used two training chunks of up to 256 bytes per update. Training
and checkpoint writes took 1,016.6 seconds. The final checkpoint had the lowest
validation loss among the saved candidates and scored 2.706 bits per byte on
the test set, compared with 3.398 for the previous checkpoint and 1.445 for
original Qwen. [Expanded-data run report](results/qwen_mps_expanded.json)
