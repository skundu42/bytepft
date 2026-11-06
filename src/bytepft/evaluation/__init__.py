"""Observed model-quality and generation metrics."""

from bytepft.evaluation.benchmarks import benchmark_generation
from bytepft.evaluation.metrics import evaluate_model

__all__ = ["benchmark_generation", "evaluate_model"]
