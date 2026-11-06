"""Opt-in test: downloads the actual pretrained 0.6B checkpoint; CPU only."""

import importlib.util
import os
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN_INTEGRATION") != "1",
    reason="Set RUN_QWEN_INTEGRATION=1 to download and validate the real Qwen checkpoint on CPU.",
)
def test_real_pretrained_qwen_cpu_gradient_and_checkpoint(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "qwen_cpu_smoke.py"
    spec = importlib.util.spec_from_file_location("qwen_cpu_smoke", example)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_smoke(tmp_path)
    assert report["device"] == "cpu" and not report["gpu_used"]
    assert report["original_parameters"] > 500_000_000
    assert report["finite_local_gradients"]
    assert report["global_parameters_unchanged"]
    assert report["local_projection_updated"]
    assert report["round_trip_matches"]
    assert not report["training_status"]["trained"]
