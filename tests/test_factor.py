"""Tests for infer_factor."""

import numpy as np
import pytest
import xarray as xr

from xr_dms.aggregation import infer_factor


def _grid(ny, nx, band=None):
    dims = ("y", "x")
    shape = (ny, nx)
    if band is not None:
        dims = ("band",) + dims
        shape = (band,) + shape
    return xr.DataArray(np.zeros(shape), dims=dims)


def test_infer_factor_divisible():
    fine = _grid(60, 60, band=3)
    coarse = _grid(6, 6)
    assert infer_factor(fine, coarse) == 10


def test_infer_factor_non_divisible_exact_raises():
    fine = _grid(63, 63)
    coarse = _grid(6, 6)
    with pytest.raises(ValueError, match="integer multiple"):
        infer_factor(fine, coarse, boundary="exact")


def test_infer_factor_non_divisible_trim_ok():
    fine = _grid(63, 63)
    coarse = _grid(6, 6)
    assert infer_factor(fine, coarse, boundary="trim") == 10


def test_infer_factor_anisotropic_raises():
    fine = _grid(60, 30)
    coarse = _grid(6, 6)
    with pytest.raises(ValueError, match="differs between dimensions"):
        infer_factor(fine, coarse)
