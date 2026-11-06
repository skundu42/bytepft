"""Incremental state for one byte-generation stream."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GenerationCache:
    """Keep local recurrent states and the transformer's completed-patch cache.

    Incomplete patch features never enter the global cache. This distinction is
    necessary for prefix-consistent predictions when the prompt ends mid-patch.
    """

    global_context: torch.Tensor
    encoder_state: torch.Tensor | None = None
    decoder_state: torch.Tensor | None = None
    global_cache: Any = None
    patch_sum: torch.Tensor | None = None
    patch_length: int = 0
    patches_processed: int = 0
    global_calls: int = 0
