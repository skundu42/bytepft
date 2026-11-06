# Training and checkpoint resumption

Training is an explicit research experiment. Loading a source checkpoint creates
randomly initialized byte-interface parameters; it does not complete adaptation.
The tiny workflow validates mechanics and can overfit a small corpus on CPU.
The full pretrained workflow requires a separate quality evaluation before any
claim about preserving Qwen capabilities.

## Data

`load_texts` accepts a local path or sequence of Unicode documents:

```python
from bytepft.training.datasets import load_texts

documents = load_texts("data/train.jsonl", text_field="text")
```

One `.txt` file is one document, including its original line endings. Each JSONL
record supplies one document through the configured string field. A directory
loads `.txt` and `.jsonl` files recursively in sorted order. Separate documents
are never concatenated.

The optional Hugging Face loader is explicit:

```python
documents = load_texts(hf_dataset="your-dataset-id", split="train", text_field="text")
```

Install the `training` extra first. A nonexistent local path is not automatically
treated as a remote dataset identifier.

The trainer adds BOS/EOS per document and right-pads batches. Labels are shifted
once by the collator; padding and the final unused prediction use `-100`.
`max_bytes` keeps the longest complete UTF-8 prefix within the limit, so teacher
text and bytes remain aligned. This is prefix truncation, not sliding windows.
At least one training document must contain raw bytes after truncation.

Without an explicit `validation_dataset`, the trainer splits at document
boundaries with `validation_fraction` and the recorded seed. Supply independent
validation data for useful quality estimates; a tiny training-corpus evaluation
only demonstrates overfitting.

## Stage 1

Stage 1 freezes the entire source adapter. New byte modules remain trainable.
Two modes are available:

| Mode | Inputs and objectives | Intended use |
| --- | --- | --- |
| `diagnostic` | Byte CE plus fixed-target boundary BCE | Offline tiny models and debugging |
| `aligned` | Source-token oracle patches, teacher context, boundary BCE, `n=0` embedding MSE, observed patch-event distillation and byte CE | Experimental pretrained adaptation |

`BytePFT.retrofit` defaults to diagnostic mode for a tiny adapter and aligned
mode for a pretrained source. Aligned mode loads the original source tokenizer
only for training. The tokenizer's raw token bytes must exactly reconstruct each
UTF-8 document; unhandled special tokens and mismatches raise an error.

The teacher event probability is the observed next source token's probability.
The student event probability is the product of that patch's byte predictions
and its boundary/non-boundary probabilities. The scalar events, rather than
incompatible vocabulary vectors, enter a temperature-modulated Bernoulli CE.
Teacher targets and teacher global context are detached. The embedding MSE uses
pooled student representations and source embeddings (`n=0`), not the paper's
preferred four-layer alignment. See [research details](research.md).

The current architecture is a reference baseline: causal boundary MLP, GRU local
modules, independent byte/boundary event factors and configurable local/global
widths. It does not implement the paper's preferred lookahead/fused architecture.
There is no paper-faithful named preset.

## Stage 2 and typed configuration

Stage 2 runs only if its step target is positive. Set `unfreeze_global=False` to
keep the global transformer frozen, or use the default `True` for end-to-end
updates. The reference Stage 2 objective combines byte CE and fixed-target
boundary BCE. `bytes_per_patch` sets the rounded fixed patch size/supervision
target; it is not a guarantee of the learned mode's measured average.

```python
from bytepft import (
    BytePFT,
    BytePFTTrainingConfig,
    Stage1Config,
    Stage2Config,
)

converter = BytePFT.from_pretrained("Qwen/Qwen3-0.6B-Base", device="cpu")
converter.configure(boundary_mode="learned", bytes_per_patch=4)
training = BytePFTTrainingConfig(
    stage1=Stage1Config(steps=1000, mode="aligned"),
    stage2=Stage2Config(
        steps=1000,
        unfreeze_global=True,
        global_learning_rate=1e-5,
    ),
    batch_size=2,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    max_bytes=256,
    validation_fraction=0.1,
    validation_interval=20,
    checkpoint_interval=50,
    early_stopping_patience=5,
    warmup_steps=20,
    seed=42,
    output_dir="outputs/qwen-byte",
)
history = converter.retrofit("data/train.jsonl", training_config=training)
```

This shows configuration, not a completed or quality-validated pretrained run.
When `training_config` is supplied, it defines stage targets and optimizer
settings; put additional training settings in that object rather than passing
conflicting convenience arguments.

Stage 1 loss weights are individually configurable through `Stage1Config`.
AdamW uses a local learning rate and optional distinct global learning rate.
The scheduler applies warmup followed by linear decay within each stage.
Gradients are accumulated, clipped and checked for finiteness. Validation occurs
at the configured interval and early stopping tracks validation NLL. Activation
checkpointing can be requested through `activation_checkpointing=True`.

`precision="fp32"` uses ordinary fp32 training. `precision="bf16"` enables
training-loss autocast on CPU/CUDA, while `precision="fp16"` adaptation requires
CUDA and uses `GradScaler`. Master parameters and Adam moments remain fp32:
starting mixed-precision training upcasts any lower-precision loaded parameters,
so its memory requirements exceed merely loading half-precision weights.
Autocast wraps training loss computation; validation and saved master parameters
are fp32. The source-loading precision also controls the initial inference dtype.
The MPS reference training path is fp32 only. CUDA/MPS execution and numerical
stability have not been validated.

## Checkpoint contents

Each stage writes periodic `stage1-step-N` or `stage2-step-N` directories and a
`stage1-final`/`stage2-final` directory. The trainer also saves `checkpoint-final`
and the final model at `output_dir`.

| File | Contents |
| --- | --- |
| `model.safetensors` | Byte modules and source transformer weights, with shared tensors deduplicated |
| `bytepft_config.json` | Architecture, global config, source identity/revision, dtype, training status, metrics and patch statistics |
| `byte_tokenizer.json` | Fixed byte/control mapping |
| `training_config.json` | Exact training configuration for resumption |
| `trainer_state.pt` | Optimizer, scheduler, AMP scaler, CPU/batch and active-device RNG, stage, history, counters and data fingerprint |
| `metrics.jsonl` in the output directory | Per-step observed losses and gradient norm |

A model-only `save_pretrained` checkpoint supports offline loading and inference
after completed adaptation. Resuming optimizer training requires a trainer
checkpoint containing the additional training-state files. Source tokenizer
files are unnecessary for inference; restarting aligned training still requires
the source tokenizer.

## Exact resumption

Use the **same complete training configuration**, including final stage targets,
learning rates, schedule and split seed. Only `output_dir` may differ. Source
documents and validation split must match their recorded fingerprint. Increasing
the target step count is not exact resumption and is rejected.

```python
from dataclasses import replace

resumed = BytePFT.load("outputs/qwen-byte/stage1-step-500", device="cpu")
resume_config = replace(training, output_dir="outputs/qwen-resumed")
resumed.retrofit(
    "data/train.jsonl",
    training_config=resume_config,
    resume_from="outputs/qwen-byte/stage1-step-500",
)
```

The model architecture is checked before restoring parameters. Training state
restores the optimizer/scheduler when continuing that stage and creates the
appropriate optimizer when crossing the Stage 1/Stage 2 boundary. CPU tests
compare final fp32 parameters and history exactly against a continuous run.
Stages that reached their target or stopped early remain finished when resumed;
resumption does not take an extra optimizer step. Terminal checkpoints are
finalized with evaluation metrics and can be written to the new output directory.
Cross-hardware/dtype reproducibility and accelerator RNG resumption have not been
validated; see `STATUS.md` for tested dtype combinations.

Generation is guarded until the trainer completes an adaptation stage. A
successful one-step integration smoke test intentionally leaves `trained=False`.
Even after a completed tiny or short research run, `quality_validated=False`
remains the honest status until a separate quality experiment establishes more.

## Device choice and evaluation

CPU is the default for every public workflow. `device="auto"` explicitly allows
an available CUDA/MPS accelerator; choose `device="cuda"` or `"mps"` only when you
intend to use one. No GPU was used for the checked-in validation results. Full
pretrained adaptation was not executed on this device.

Evaluate held-out text with `evaluate_model` or `bytepft evaluate`. NLL is summed
only for raw-byte targets and divided by their count; bits per byte is NLL divided
by `log(2)`. EOS, BOS and padding do not enter that denominator. Training byte CE
also supervises EOS, so it is a different logged quantity. Do not compare token
perplexity across tokenizers as though it measured the same units.

The [CPU smoke report](results/qwen_cpu_smoke.json) records one finite frozen-global
gradient update on the actual base checkpoint and an exact save/load logit
comparison. It contains no language-quality evaluation or completed adaptation.
