# Reference architecture and tensor alignment

BytePFT 0.1.0 is a research baseline, not an exact reproduction of
[*Retrofitting language models to operate over bytes*](https://www.nature.com/articles/s41586-026-11111-4).
The paper's architecture/Methods and official source were inspected before this
design. [research.md](research.md) records component-level deviations.

## Tensors and modules

Let `B` be batch size, `L` the padded byte/control sequence length, `P` the maximum
patch count in the batch, `D` the local hidden width and `H` the source
transformer's hidden width.

| Transformation | Shape | Implementation |
| --- | --- | --- |
| UTF-8 text plus document controls | `[B,L]`, integer IDs | `ByteTokenizer`, `ByteDataset` |
| Byte/control embeddings | `[B,L,D]` | `ByteEmbedding` |
| Causal local encoder | `[B,L,D]` | `LocalByteEncoder` (GRU) |
| Boundary logits and decisions | `[B,L]` each | `BoundaryPredictor` |
| Last/mean pooling | `[B,P,D]` | `PatchPool` |
| Local-to-global projection | `[B,P,H]` | `ByteModel.to_global` |
| Causal patch transformer | `[B,P,H]` | `QwenGlobalAdapter` |
| Latest completed patch depooling | `[B,L,H]` | `PatchDepool` |
| Context projection and local decoding | `[B,L,D]` | `LocalByteDecoder` (GRU) |
| Next-byte/control logits | `[B,L,259]` | `BytePredictionHead` |

Ordinary byte IDs are fixed at `0..255`, with PAD=256, BOS=257 and EOS=258. The
embedding layer zeros masked positions; masks must be binary and right padding
must be contiguous. The global adapter supplies native Qwen3 rotary positions,
causal attention and attention-mask handling. Qwen3's explicit `head_dim` is read
from its configuration rather than inferred from `hidden_size / heads`.

The model retains source input/output matrices for aligned training and
checkpoint compatibility. Byte inference enters the source backbone through
`inputs_embeds`; it does not use the source embedding lookup, tokenizer or LM
head for generation.

## Boundaries and metadata

Fixed patching closes a patch at a configurable rounded byte count. Learned
patching thresholds a causal MLP and obeys minimum/maximum patch lengths. Oracle
patching accepts externally supplied boundaries for forward diagnostics and
source-aligned training. BOS/EOS close their control patches. Oracle decisions
must preserve the source-token axis rather than inserting arbitrary boundaries
for length constraints.

`PatchMetadata` records byte-to-patch assignments, patch starts/ends, masks,
completion state and patch lengths. Every valid input position belongs to
one contiguous patch. A final unfinished patch is represented for full coverage
but is not available as global context until its closing boundary occurs.

`bytes_per_patch` is a fixed-patching/supervision setting. Learned patch lengths
also depend on weights, threshold and constraints; report their measured average
rather than treating that setting as an observed compression result.

## Autoregressive alignment

At input position `t`, the model has observed byte `x[t]`; its logits predict
`x[t+1]`. A boundary at `t` closes the patch ending at `t`, so its contextualized
global representation may be used to predict `x[t+1]`. An unfinished patch ending
later than `t` cannot contribute to that prediction.

For example, with a BOS patch and two-byte patches:

```text
observed:           BOS   a   b   c   d
completed patch:     0        1       2
latest context:      0   0   1   1   2
prediction target:   a   b   c   d   next
```

Using the current patch's final representation for all bytes inside that patch
would leak future bytes. Depooling uses only the latest completed patch, while
the local GRUs remain causal. Padded positions do not add model context.

The data collator performs the prediction shift once: inputs are
`BOS + bytes + EOS`, labels are `bytes + EOS + ignored`. CE receives those already
shifted labels. Source-aligned event collection likewise reads the target byte
at position `t` from logits at `t-1`; source token end offsets are inclusive raw
byte offsets and acquire a `+1` position offset for BOS.

## Stage-specific computation

Diagnostic Stage 1 runs student patch representations through a frozen global
transformer and trains with byte CE plus boundary BCE. Aligned Stage 1 instead
uses oracle source boundaries, compares projected encoder patches against source
embeddings (`n=0`), and supplies detached teacher global states to the byte decoder.
Scalar observed token/byte-patch probabilities support the distillation objective.
Frozen parameters have no parameter gradients; differentiable student paths
still train the new interface.

Stage 2 uses the actual student global states and optionally unfreezes global
parameters. Its current boundary objective uses a fixed-target surrogate. This
differs from the paper's complete architecture and training recipe.

## Incremental generation

The cached generation path maintains:

- Encoder and decoder recurrent states.
- Current patch length and optional mean-pooling sum.
- Latest completed patch's global context.
- Hugging Face Qwen3 KV cache and processed patch positions.

Prefill consumes BOS and prompt bytes sequentially. Each new observed byte
advances local state. The global transformer advances only when a patch closes.
The uncached reference path recomputes the full observed prefix and is intended
for correctness comparisons. CPU tests compare both paths. Current generation
supports one stream, not padded batched decoding.

Greedy decoding uses temperature zero; sampling supports temperature, top-k and
top-p. PAD/BOS outputs are suppressed, EOS stops decoding and the byte limit
counts ordinary emitted bytes. Incremental UTF-8 decoding buffers an incomplete
suffix; malformed complete sequences use replacement characters in the text
view. Raw-byte and detailed outputs retain every sampled byte.

Checkpoints preserve architecture, source identity, weights and honest training
status. Trainer checkpoints additionally preserve optimizer/scheduler and random
state. See [training.md](training.md) for the resume contract and
[STATUS.md](../STATUS.md) for actual validation results and remaining work.
