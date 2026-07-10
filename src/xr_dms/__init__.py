"""xr_dms package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xr_dms")
except PackageNotFoundError:
    __version__ = "unknown"
