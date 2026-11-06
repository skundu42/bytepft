# Apple MPS validation

Bounded fp32 validation passed on an Apple M3 Pro with 18 GB unified memory,
macOS 26.6.2 and PyTorch 2.15.0 on 2026-11-06. This explicitly used the GPU.
Full pretrained adaptation and language-quality evaluation were not performed.

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

Tiny Stage 2 works on MPS. Unfreezing all 596 million pretrained parameters has
not been tested here and needs a separate memory-budgeted experiment. No speedup
over CPU, sustained-run stability, full adaptation or retained-language-quality
claim follows from these checks. Byte generation still performs deliberate CPU
sampling/control work; unsupported-operator fallback was disabled in the tests.
