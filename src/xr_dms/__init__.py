"""xr_dms: a lazy, chunked, dask-backed Data Mining Sharpener."""

from importlib.metadata import version, PackageNotFoundError

from .sharpener import Sharpener, SceneState, to_radiance, from_radiance
from .cube import sharpen_cube
from .gridmap import GridMap, RegularGridMap
from .geo import SwathGridMap
from .harmonize import harmonize_features
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
    "SceneState",
    "sharpen_cube",
    "GridMap",
    "RegularGridMap",
    "SwathGridMap",
    "BaseRegressor",
    "DecisionTreeRegressorWithLinearLeafRegression",
    "SklearnDMSRegressor",
    "harmonize_features",
    "to_radiance",
    "from_radiance",
    "__version__",
]
