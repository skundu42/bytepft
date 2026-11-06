# Changelog

## 0.1.0 — Unreleased

Initial experimental release, available from the source checkout. Not published
to PyPI.

### Added

- Fixed UTF-8 byte vocabulary with separate controls, raw-byte support, batching,
  tokenizer persistence and incremental text decoding.
- Causal GRU byte encoder/decoder, fixed/learned/oracle patching, last/mean
  pooling, causal depooling and a trainable byte/control prediction head.
- Dense Qwen3 adapter using Hugging Face transformer layers, explicit
  configuration checks, frozen/global training control and native KV caches.
- Offline tiny reference workflow and a real 0.6B base-checkpoint CPU smoke test.
- Diagnostic and source-aligned Stage 1 objectives; explicit optional Stage 2;
  validation, gradient accumulation/clipping, learning-rate scheduling, early
  stopping and resumable stage checkpoints.
- Greedy and sampled byte generation with cached and uncached reference paths.
- Safetensors model persistence, offline loading and training-state checkpoints.
- Byte-normalized evaluation, measured generation benchmarking, CLI commands,
  examples, CPU tests and release tooling.

### Validation scope

CPU tests demonstrate tiny-corpus loss reduction, frozen/unfrozen parameter
behavior, aligned scalar event losses, deterministic resumption and persistence.
The real pretrained base checkpoint passes one local gradient update and an exact
save/load logit comparison. That smoke experiment is **not full adaptation** and
does not establish retained source-model quality.

### Known limitations

- The reference GRU, causal boundary and separate-head architecture differs from
  the cited Nature method; exact mLSTM/fused-boundary support is not implemented.
- Full pretrained adaptation, downstream task quality and matched source-model
  performance comparisons have not been executed.
- MPS fp32 passed bounded training/inference and real pretrained integration
  checks on an M3 Pro. CUDA and MPS mixed precision remain unvalidated. CPU is the default.
- Inference supports a single generation stream; batched generation and
  distributed training are not implemented.
- Source embedding/output matrices are retained in checkpoints for Stage 1;
  inference does not use the source tokenizer.
