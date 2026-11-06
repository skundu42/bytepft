"""Use installed distribution metadata as the runtime version source."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bytepft")
except PackageNotFoundError:
    __version__ = "0+unknown"
