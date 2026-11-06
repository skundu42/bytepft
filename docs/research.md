# Research relationship and implementation status

BytePFT 0.1.0 is an independently implemented research library inspired by
[Minixhofer et al., *Retrofitting language models to operate over bytes*, Nature,
2026](https://www.nature.com/articles/s41586-026-11111-4).
It implements a **reference baseline**, not an exact reproduction of the paper.
The architecture and training Methods sections were read before implementation.
The authors' [arXiv version](https://arxiv.org/html/2512.15586v2) is a second
accessible account of the architecture and objectives.

The authors' official implementation was inspected at commit
`2d8178b16ce73cd40ee31a8973d450defa1cdfef`.
This pin records the source reviewed; it is not a dependency or a checkpoint
compatibility claim. BytePFT loads its own checkpoint format.

## Component mapping

| Paper Methods component | BytePFT implementation | Relationship and deviations |
| --- | --- | --- |
| Tokenization and embedding | `tokenization/byte_tokenizer.py`, `models/embeddings.py` | Fixed ordinary byte IDs, separate control symbols, trainable byte embeddings. The longest matching source-vocabulary suffix augmentation is omitted. |
| Local encoder | `models/encoder.py` | Causal PyTorch GRU reference implementation. The paper uses mLSTM and feedforward blocks. |
| Boundary predictor | `models/boundaries.py` | Fixed, externally supplied oracle, or causal learned sigmoid boundaries. The paper's preferred prefill predictor uses one-byte lookahead and projected cosine distance; it is not implemented. |
| Pooling | `models/pooling.py` | Last-position pooling matches the paper's basic aggregation. Mean pooling is an additional diagnostic option. The baseline projects from local to global dimension, unlike the paper's equal-width architecture. |
| Global model | `adapters/qwen.py` | Reuses Hugging Face Qwen3 transformer weights and native attention/cache machinery. A tiny randomly initialized Qwen3 adapter is available for CPU tests. It is not a pretrained checkpoint. |
| Depooling | `models/depooling.py` | Uses only the latest completed patch state. Current unfinished patches never provide future-byte information. |
| Local decoder | `models/decoder.py` | Causal PyTorch GRU reference implementation with projected global context. The paper uses an mLSTM stack. |
| Language-modelling head | `models/heads.py` | 259 byte/control classes plus a separate causal boundary predictor. The paper's joint 512-way byte/boundary head is not implemented. |
| Boundary objective | `training/losses.py:BoundaryLoss` | Masked binary CE on boundaries after observed bytes. Causal boundary features differ from the paper's lookahead predictor. |
| Encoder objective | `training/losses.py:EmbeddingAlignmentLoss` | Masked MSE on source embeddings and projected oracle patches: the simpler `n=0` baseline. The paper aligns after four shared frozen global layers. |
| Decoder distillation | `training/losses.py:DecoderDistillationLoss`, `training/distillation.py` | Temperature-modulated Bernoulli CE compares scalar observed patch-event probabilities. This comparison function follows the paper. The reference architecture uses a factorized byte/Bernoulli event instead of the fused head. |
| Optional data CE | `training/losses.py:ByteCrossEntropyLoss` | Teacher-forced next-byte/control CE with ignored padding. Labels are shifted by the collator, not inside the loss. |
| Training stages | `training/trainer.py` | Stage 1 freezes the global adapter and trains local modules; Stage 2 explicitly chooses whether to unfreeze it. Reference training settings do not constitute a reported-paper preset. |
| Autoregressive inference | `inference/generation.py` | Generates from the byte head; source BPE tokenization is not used. Incremental encoder/decoder state and native global KV caches are compared against a full reference path. |

Paths in the table are relative to `src/bytepft`.

## Official source examined

The following modules were reviewed at the commit above, in the source released
with the cited Nature paper:

- Local-model modules: recurrent encoder/decoder blocks and boundary prediction.
- Hugging Face implementation: boundary prediction, pooling, depooling,
  incremental state and the fused head.
- `src/olmo_core/nn/transformer/model.py`: `teacher_forward`,
  `_compute_local_encoder_loss`, `_compute_alm_style_loss`, and Stage 1
  orchestration.
- Probability utilities: stable `log1mexp` and Bernoulli objectives.
- Stage 1 launch scripts: reported loss weights, distillation temperature and
  four-layer encoder-state alignment settings.
- `QWEN3_06B_RESULTS.md`: the authors' small-Qwen results, **not BytePFT results**.

## Alignment and loss semantics

Source tokens and byte logits have different vocabulary sizes, so their vectors
cannot be passed directly into a KL-divergence objective. BytePFT instead compares
probabilities of the same observed byte-string-and-boundary event.

For a source token whose bytes occupy positions `a..b`, the reference student
event log probability is:

```text
student_log_p = sum_t=a..b [
    log P(next_byte_t | bytes before t)
    + log P(boundary_t = (t == b) | bytes through t)
]
```

The teacher event probability is the source model's probability of that observed
next source token at the preceding source-token position. Both are scalar log
probabilities per aligned token. Given temperature `tau`, the loss is binary
cross entropy between `exp(teacher_log_p / tau)` and
`exp(student_log_p / tau)`. Calculations stay in log space for stability.
The implemented comparison defaults to `tau=5`. Padding events are masked,
and teacher targets are detached.

The paper uses a fused byte/boundary distribution, teacher global representations
for the decoder's Stage 1 input, and an encoder objective after frozen global
layers. Its reported loss weights are boundary 4, encoder 1, decoder distillation
1 and data CE 1. BytePFT does not expose those weights as a paper-faithful preset.
The causal boundary features, threshold/length-constrained patch decisions,
separate Bernoulli event factor, `n=0` alignment and GRU blocks are material
deviations. Mathematical compatibility of the scalar event comparison does not
establish equivalence of the complete models. The Nature Methods also states
that including the optional data CE gives up exact teacher-matching optimality.

Qwen source token boundaries are recovered from their reversible ByteLevel byte
spellings. Individual token decoding and Unicode character offsets are unsafe
because tokens may split a multibyte UTF-8 character. Each alignment is checked
by concatenating all source token bytes and requiring exact equality to the
document's UTF-8 encoding. Unhandled special tokens and reconstruction mismatches
raise errors.

## Verified checkpoint identity

The [Qwen3-0.6B-Base model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/README.md)
identifies it as a pretrained base model. The
[Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/README.md)
identifies the checkpoint without `-Base` as post-trained. Examples therefore
prefer the base checkpoint for initial adaptation experiments, while the adapter
can validate either checkpoint's architecture.

Their model configurations share hidden size 1024, 28 transformer layers, 16
query heads, 8 key/value heads, explicit head dimension 128, vocabulary size
151936, and tied input/output embeddings. Explicit `head_dim` matters: it must
not be inferred as `hidden_size // num_attention_heads`. Their EOS IDs and
context configurations differ, so the adapter reads actual model configuration
rather than hardcoding a Qwen-wide assumption. Compare the
[base config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)
and [post-trained config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json).

## Validation and limits

CPU tests cover event-probability arithmetic, extreme-probability stability,
gradient propagation, frozen targets, masks, invalid shapes, document isolation,
and byte boundary alignment across split Unicode code points.

The actual `Qwen/Qwen3-0.6B-Base` tokenizer was downloaded without model weights
and verified on empty text, English, Devanagari, emoji with modifiers and a zero
width joiner, accented Latin, Chinese, whitespace and a NUL byte. Reconstruction
was exact for every tested document. This validates source-token byte extraction;
it does not demonstrate model adaptation or language quality.

See the repository's `STATUS.md` for the current executable validation results,
including any real pretrained-checkpoint test. Random-weight CPU adapter tests,
tiny-corpus overfitting and successful serialization do not demonstrate retained
Qwen capabilities. Full-model adaptation, model-quality comparisons and GPU
performance remain separate experiments. No speedup or quality-parity claim is
made.

## Attribution and licenses

The reviewed official source license and the
[Qwen3-0.6B-Base model license](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/LICENSE)
are Apache-2.0. BytePFT original code is Apache-2.0. The implementation uses
PyTorch and Hugging Face; no code was copied from the reviewed training framework.
The mathematical methods and architectural inspiration are attributed above;
pretrained model licenses continue to apply to any redistributed weights.
