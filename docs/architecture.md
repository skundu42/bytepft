# Reference architecture and milestone plan

BytePFT 0.1 is a research baseline, not an exact reproduction of the research. The Nature Methods
and official research implementation were inspected before this design.

Input IDs `[B,L]` use ordinary bytes 0–255, PAD=256, BOS=257 and EOS=258. A learned byte
embedding and causal GRU encoder produce `[B,L,D_local]`. A causal MLP predicts whether
the observed byte closes a patch. Fixed and oracle boundaries are diagnostic alternatives.
BOS and EOS close patches. An unfinished final patch is recorded for complete coverage
but contributes no global context until it closes.

Pooling selects the last encoder state (or an experimental mean) and projects it to
`[B,P,D_global]`. A Hugging Face Qwen3 backbone contextualizes patches causally. At byte
position t, depooling uses only the latest completed patch ending at or before t. Combining
this with local features, a causal GRU decoder and a 259-way head predicts the next byte
or EOS. PAD and BOS are never generation outputs. Raw-byte output is separate from text;
the text path buffers incomplete UTF-8 suffixes.

The fast generation path caches local recurrent states, patch-pooling state, latest global
context and the Qwen KV cache. The global model runs only when a patch completes. A slower
full-forward path exists as a correctness reference. No source tokenizer is needed for
either inference path.

Stage 1 freezes pretrained parameters and trains the byte interface. Diagnostic mode uses
byte CE; aligned mode uses source token byte boundaries, n=0 embedding alignment and
token-event probability distillation. Stage 2 explicitly opts into global weight updates.
Checkpoints persist architecture, source identity, training status, optimizer, schedule,
random state and data fingerprint where appropriate.

Implementation follows the seven milestones in STATUS.md. CPU unit tests and a tiny Qwen
configuration validate mechanics; they do not establish pretrained-model quality.
