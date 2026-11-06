"""Apply a GitHub release tag to the build checkout without changing dependency pins."""

import os
import subprocess
import tomllib
from pathlib import Path

from packaging.version import InvalidVersion, Version


def release_version(tag: str) -> str:
    """Accept a public PEP 440 version, optionally prefixed by a lowercase v."""
    if not tag or tag != tag.strip():
        raise ValueError("A release tag such as v0.1.0 is required")
    try:
        parsed = Version(tag.removeprefix("v"))
    except InvalidVersion as exc:
        raise ValueError(f"Release tag {tag!r} is not a valid Python version") from exc
    if parsed.local is not None:
        raise ValueError("Local version suffixes (+...) cannot be published to PyPI")
    return str(parsed)


def main() -> None:
    value = release_version(os.environ.get("RELEASE_TAG", ""))
    lock_path = Path("uv.lock")
    before = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    subprocess.run(["uv", "version", "--no-sync", value], check=True)
    after = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    if [p for p in before["package"] if p["name"] != "bytepft"] != [
        p for p in after["package"] if p["name"] != "bytepft"
    ]:
        raise RuntimeError("Version preparation changed dependency pins; review the lock file")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"version={value}\n")


if __name__ == "__main__":
    main()
