"""Supported global transformer adapters."""

from bytepft.adapters.base import GlobalAdapter, GlobalOutput, TeacherOutput
from bytepft.adapters.qwen import QwenGlobalAdapter, TinyGlobalAdapter

__all__ = [
    "GlobalAdapter",
    "GlobalOutput",
    "TeacherOutput",
    "QwenGlobalAdapter",
    "TinyGlobalAdapter",
]
