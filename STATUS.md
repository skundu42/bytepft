# Implementation status

Work in progress. All local validation is explicitly CPU-only. No GPU run is authorized.
No pretrained model has yet completed adaptation or functional evaluation in BytePFT.

## Milestones

1. Package, byte tokenizer, constrained fixed patching, CLI and tests.
2. Causal reference neural model, tiny overfit and byte generation.
3. Qwen3 adapter, compatibility checks and CPU-scale integration.
4. Oracle boundary alignment, independently tested losses, Stage 1 and checkpoints.
5. Cached generation compared with a full-forward reference.
6. Stage 2, evaluation and measured benchmarks.
7. Documentation, CI, release configuration and build artifacts.

## Scope decisions

- Version 0.1 implements an explicitly experimental reference architecture: GRU local
  encoder/decoder, causal boundary MLP, last/mean pooling, latest completed patch context,
  and independent byte and boundary heads. It is not the research mLSTM/fused-output model.
- Qwen3 is the only supported pretrained family. Prefer Qwen/Qwen3-0.6B-Base;
  Qwen/Qwen3-0.6B is the separately published post-trained checkpoint.
- The first alignment baseline uses n=0 embedding alignment. No paper preset is claimed.
- Full pretrained adaptation and meaningful model-quality evaluation require a dedicated
  training run. They must remain explicitly unverified until executed.
