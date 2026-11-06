"""Actionable errors exposed by BytePFT."""


class BytePFTError(Exception):
    """Base error for BytePFT."""


class CompatibilityError(BytePFTError, ValueError):
    """The requested architecture or checkpoint is not supported."""


class NotAdaptedError(BytePFTError, RuntimeError):
    """Generation was requested before adaptation completed."""


class AlignmentError(BytePFTError, ValueError):
    """Source tokens cannot be mapped exactly onto the input bytes."""


class CheckpointError(BytePFTError, ValueError):
    """A checkpoint is incomplete or incompatible."""
