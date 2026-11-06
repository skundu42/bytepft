"""Release tags must produce matching package and runtime versions."""

import os
import runpy
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

from bytepft import __version__
from bytepft.adapters import TinyGlobalAdapter
from bytepft.config import BytePFTConfig
from bytepft.models.byte_model import ByteModel
from bytepft.serialization import save_pretrained

ROOT = Path(__file__).resolve().parents[1]
release_version = runpy.run_path(str(ROOT / "scripts/prepare_release.py"))["release_version"]


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("v0.1.0", "0.1.0"),
        ("1.2.3", "1.2.3"),
        ("v0.2.0rc1", "0.2.0rc1"),
        ("v0.2.0-beta.1", "0.2.0b1"),
    ],
)
def test_public_release_tags(tag, expected):
    assert release_version(tag) == expected


@pytest.mark.parametrize("tag", ["", "main", " v1.0.0", "v1.0.0\n", "v1.0.0+local", "$(id)"])
def test_invalid_release_tags_are_rejected(tag):
    with pytest.raises(ValueError):
        release_version(tag)


def test_preparing_release_changes_only_project_version_and_preserves_dependency_pins(tmp_path):
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "NOTICE"):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    before = tomllib.loads((tmp_path / "uv.lock").read_text())
    output = tmp_path / "github-output"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_release.py")],
        cwd=tmp_path,
        env={**os.environ, "RELEASE_TAG": "v0.2.0rc1", "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        tomllib.loads((tmp_path / "pyproject.toml").read_text())["project"]["version"] == "0.2.0rc1"
    )
    after = tomllib.loads((tmp_path / "uv.lock").read_text())
    assert next(p["version"] for p in after["package"] if p["name"] == "bytepft") == "0.2.0rc1"
    assert [p for p in before["package"] if p["name"] != "bytepft"] == [
        p for p in after["package"] if p["name"] != "bytepft"
    ]
    assert output.read_text() == "version=0.2.0rc1\n"


def test_checkpoint_records_installed_distribution_version(tmp_path):
    import json

    assert __version__ == version("bytepft")
    model = ByteModel(TinyGlobalAdapter(hidden_size=16), BytePFTConfig(hidden_size=16))
    save_pretrained(model, tmp_path)
    config = json.loads((tmp_path / "bytepft_config.json").read_text())
    assert config["checkpoint_metadata"]["bytepft_version"] == __version__
