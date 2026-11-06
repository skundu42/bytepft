# Contributing to BytePFT

BytePFT prioritizes a small, understandable and verifiable byte-model workflow.
Contributions should preserve causal alignment, document boundaries and the
distinction between reference baselines and research reproductions.

## Development setup

Use Python 3.11+ and `uv`:

```sh
uv sync
uv run pytest
uv run ruff check .
uv run mypy
uv build
```

The ordinary suite is CPU compatible and must not download pretrained weights.
Use `uv sync --extra training` when working on the optional Hugging Face dataset
integration. Tests use tiny randomly initialized Qwen3 configurations to exercise
the real Hugging Face transformer implementation.

Defaults use CPU. Do not introduce automatic GPU execution into tests or
examples. CUDA/MPS tests and pretrained-weight downloads must be explicit opt-ins
and describe their hardware and download requirements. A contributor must choose
to run those experiments; CI should remain usable without them.

## Making a change

Keep changes focused, match the surrounding style and add tests for observable
behavior. Test the failure path when adding validation or serialization code.
For neural components, cover shapes, masks, finite gradients and relevant causal
invariants. Changing future bytes must not change predictions for earlier bytes.
Patch construction must account for every valid byte once, in order.

For training changes, check that Stage 1 leaves global parameters unchanged and
that optional Stage 2 unfreezing updates them. Resume tests should compare the
same training configuration and data against a continuous run, including optimizer,
scheduler, random sampling state and history. For inference changes, compare
cached logits or greedy bytes against the uncached reference implementation.

Run the tests appropriate to the change, then the ordinary suite and tooling
before proposing a release. Report the commands actually run and any unavailable
validation. Avoid claiming that a successful forward pass, random-weight test or
single gradient step proves model quality.

## Research and API changes

Read the relevant methods and upstream implementation before adding a derived
architecture or objective. Record the paper section, source revision, deviations
and validation in [docs/research.md](docs/research.md). Preserve applicable licenses
when incorporating third-party code. Original additions use Apache-2.0.

Compare teacher and student distributions only when their events and axes are
mathematically aligned. Source-token logits and byte logits cannot be directly
KL-diverged. Document byte offsets, BOS/EOS treatment and the prediction shift
when adding an alignment method.

Do not expose an unfinished adapter or command as supported. Add model-family
compatibility checks rather than relying on coincidentally similar module paths.
The current exact-paper mLSTM and fused-boundary architecture remain unsupported.

## Reporting an issue

Include a minimal example, Python/PyTorch/Transformers versions, device and dtype,
configuration, traceback and whether the model is random, source pretrained or a
trained BytePFT checkpoint. For resume issues, include the checkpoint stage/step
and whether the config or documents changed. Never include credentials or private
training data in a public report.

## Release checklist

1. Update [CHANGELOG.md](CHANGELOG.md) and choose a release tag such as `v0.1.0`.
2. Run the ordinary CPU suite, Ruff, mypy and `uv build`.
3. Inspect wheel/sdist contents and install the wheel into a clean environment.
4. Review [STATUS.md](STATUS.md), examples and measured results for accurate claims.
5. Complete the one-time [PyPI setup](docs/releases.md), then publish a GitHub
   release from the tested commit. The release tag sets the package version and
   triggers publication automatically. A manual workflow dispatch builds and
   tests a candidate without publishing. Ordinary pushes and pull requests do
   not publish packages.

Do not add fabricated authorship or automatic co-author trailers to commits.
