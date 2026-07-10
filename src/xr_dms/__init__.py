"""xr_dms: a lazy, chunked, dask-backed Data Mining Sharpener."""

from importlib.metadata import version, PackageNotFoundError

from .sharpener import Sharpener, to_radiance, from_radiance
from .regressors import (
    BaseRegressor,
    DecisionTreeRegressorWithLinearLeafRegression,
    SklearnDMSRegressor,
)

try:
    __version__ = version("xr_dms")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Sharpener",
    "BaseRegressor",
    "DecisionTreeRegressorWithLinearLeafRegression",
    "SklearnDMSRegressor",
    "to_radiance",
    "from_radiance",
    "__version__",
]
