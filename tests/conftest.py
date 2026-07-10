"""Shared synthetic-scene fixtures for xr_dms tests.

Builds a small, GRID-ALIGNED scene: fine feature stack, a coarse target
obtained by block-averaging a hidden fine "truth", and the truth itself for
skill evaluation. Ported from ``examples/dms_explained_numpy.py``.
"""

import numpy as np
import pytest
import xarray as xr


def _block_mean(arr, factor):
    h, w = arr.shape
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def make_scene(fine_shape=(60, 60), factor=10, seed=0):
    """Return ``(features_da, target_da, truth_da, factor)`` as xarray objects."""
    rng = np.random.default_rng(seed)
    H, W = fine_shape
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)

    ndvi = np.clip(0.5 + 0.4 * np.sin(6 * xx) * np.cos(5 * yy)
                   + 0.15 * rng.standard_normal((H, W)), 0, 1)
    albedo = np.clip(0.3 - 0.2 * ndvi + 0.1 * np.cos(10 * xx + 3 * yy)
                     + 0.05 * rng.standard_normal((H, W)), 0.05, 0.6)
    builtup = (np.sin(3 * xx) > 0.4).astype(float) + 0.05 * rng.standard_normal((H, W))
    features = np.stack([ndvi, albedo, builtup], axis=0)  # (band, y, x)

    truth = (305.0 - 18.0 * ndvi + 12.0 * albedo + 8.0 * builtup
             + 10.0 * np.sin(8.0 * ndvi) + 4.0 * np.sin(2.5 * xx + 1.5 * yy)
             + 0.3 * rng.standard_normal((H, W)))

    # The coarse sensor sees a block average in radiance space (~T**4).
    lowres = _block_mean(truth ** 4, factor) ** 0.25
    h, w = lowres.shape

    # Coordinates: fine pixel centres and coarse pixel centres on the same axis.
    fine_x = np.arange(W) + 0.5
    fine_y = np.arange(H) + 0.5
    coarse_x = (np.arange(w) + 0.5) * factor
    coarse_y = (np.arange(h) + 0.5) * factor

    features_da = xr.DataArray(
        features,
        dims=("band", "y", "x"),
        coords={"band": ["ndvi", "albedo", "builtup"], "y": fine_y, "x": fine_x},
        name="features",
    )
    target_da = xr.DataArray(
        lowres, dims=("y", "x"), coords={"y": coarse_y, "x": coarse_x}, name="target"
    )
    truth_da = xr.DataArray(
        truth, dims=("y", "x"), coords={"y": fine_y, "x": fine_x}, name="truth"
    )
    return features_da, target_da, truth_da, factor


def make_varying_scene(fine_shape=(160, 160), factor=8, seed=0):
    """A scene whose feature->LST relationship varies across space.

    Same construction as :func:`make_scene`, but the NDVI->LST slope sweeps from
    strongly negative on the left to positive on the right, so a single global
    tree cannot fit the whole scene and moving-window local models help. Returns
    ``(features_da, target_da, truth_da, factor)``.
    """
    rng = np.random.default_rng(seed)
    H, W = fine_shape
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)

    ndvi = np.clip(0.5 + 0.4 * np.sin(6 * xx) * np.cos(5 * yy)
                   + 0.1 * rng.standard_normal((H, W)), 0, 1)
    albedo = np.clip(0.3 - 0.2 * ndvi + 0.1 * np.cos(10 * xx + 3 * yy)
                     + 0.03 * rng.standard_normal((H, W)), 0.05, 0.6)
    builtup = (np.sin(3 * xx) > 0.4).astype(float) + 0.03 * rng.standard_normal((H, W))
    features = np.stack([ndvi, albedo, builtup], axis=0)

    slope = -30.0 + 40.0 * xx  # spatially varying NDVI sensitivity
    truth = (305.0 + slope * ndvi + 12.0 * albedo + 8.0 * builtup
             + 0.2 * rng.standard_normal((H, W)))

    lowres = _block_mean(truth ** 4, factor) ** 0.25
    h, w = lowres.shape

    fine_x = np.arange(W) + 0.5
    fine_y = np.arange(H) + 0.5
    coarse_x = (np.arange(w) + 0.5) * factor
    coarse_y = (np.arange(h) + 0.5) * factor

    features_da = xr.DataArray(
        features,
        dims=("band", "y", "x"),
        coords={"band": ["ndvi", "albedo", "builtup"], "y": fine_y, "x": fine_x},
        name="features",
    )
    target_da = xr.DataArray(
        lowres, dims=("y", "x"), coords={"y": coarse_y, "x": coarse_x}, name="target"
    )
    truth_da = xr.DataArray(
        truth, dims=("y", "x"), coords={"y": fine_y, "x": fine_x}, name="truth"
    )
    return features_da, target_da, truth_da, factor


@pytest.fixture
def scene():
    return make_scene()


@pytest.fixture
def varying_scene():
    return make_varying_scene()
