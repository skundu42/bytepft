# Releasing BytePFT

Publishing a GitHub release automatically builds and publishes that version to
PyPI. The [publish workflow](../.github/workflows/publish.yml) runs tests, builds
the wheel and source distribution, checks their metadata, and uploads through
PyPI Trusted Publishing. No stored API token is needed.

## One-time PyPI setup

Sign in to the PyPI account that should own the package, with two-factor
authentication enabled. For a project's first publication, add a
[pending publisher](https://pypi.org/manage/account/publishing/) with these values:

| Field | Value |
| --- | --- |
| PyPI project name | `bytepft` |
| GitHub owner | `skundu42` |
| GitHub repository | `bytepft` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

For an existing PyPI project, its owner adds the same publisher under the
project's Publishing settings. The GitHub workflow uses the `pypi` environment.
There is no manual approval gate in this setup: publishing the GitHub release
is the maintainer's publishing decision. Adding required environment reviewers
later will pause publishing until they approve.

Do not share your PyPI password, recovery codes or API token. Account-side
publisher configuration must be completed before the first published GitHub
release. See the [official setup guide](https://docs.pypi.org/trusted-publishers/).

## Publish a version

1. Merge the intended changes into `main` and check that CI passes.
2. Update the changelog and open [New release](https://github.com/skundu42/bytepft/releases/new).
3. Create or select the tag for the tested commit, for example `v0.1.0`.
4. Publish the release. The workflow automatically publishes `bytepft==0.1.0`.

Draft releases and ordinary tag pushes do not publish. Published prereleases
also publish to PyPI: use a PEP 440 tag such as `v0.2.0rc1` and mark the GitHub
release as a prerelease. Local version suffixes such as `+local` are rejected.

The release tag is authoritative. The build updates `pyproject.toml` and the root
package version in `uv.lock` only in the runner's checkout; dependency pins stay
unchanged. It does not commit a version bump back to `main`. Runtime
`bytepft.__version__` and checkpoint metadata read the installed package version.

Use a new version for every release. PyPI does not permit replacing previously
uploaded distribution files. If publication fails, inspect the workflow logs and
rerun the failed job after fixing publisher configuration; do not move a published
release tag to different source code.

## Check a release without publishing

Open **Actions → Publish to PyPI → Run workflow** and enter the proposed tag,
for example `v0.1.1`. This builds the selected workflow ref with that package
version and uploads distribution artifacts to the run. It does not create a tag,
create a release or publish to PyPI.

Regular CI contains only one CPU job for tests and Ruff. Building distributions
and validating PyPI metadata belong to the release workflow. Mypy and local
build checks remain available to contributors.
