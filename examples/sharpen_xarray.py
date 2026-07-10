"""Minimal end-to-end example of the ``xr_dms`` package.

Sharpens a coarse (12x12) land-surface-temperature image to the fine (120x120)
grid of three predictor features, using the same arrays as the low-level
``dms_explained_numpy.py`` walk-through -- but here through the high-level
:class:`xr_dms.Sharpener` API on lazy, chunked :class:`xarray.DataArray` inputs.

It shows three things:

1. the classic single global model (``Sharpener().sharpen``);
2. the moving-window local regression (``window_size > 0``), which fits one
   local model per window and blends it with the global model; and
3. that the result stays lazy (dask-backed) and is mass-conserving -- re-
   aggregating the sharpened field reproduces the coarse observation.

Run it from the repo root with the package importable::

    pip install -e .
    python examples/sharpen_xarray.py
"""

import os

import numpy as np
import xarray as xr

from xr_dms import Sharpener

HERE = os.path.dirname(__file__)


def load_scene():
    """Wrap the bundled numpy arrays as co-registered xarray DataArrays."""
    features = np.load(os.path.join(HERE, "highres_features.npy"))  # (H, W, band)
    target = np.load(os.path.join(HERE, "lowres_target.npy"))       # (h, w)

    H, W, _ = features.shape
    h, w = target.shape
    factor = H // h  # 120 / 12 = 10

    # Fine and coarse pixel-centre coordinates on a shared axis: the coarse
    # centres sit at the middle of each factor-sized block of fine pixels.
    fine_x = np.arange(W) + 0.5
    fine_y = np.arange(H) + 0.5
    coarse_x = (np.arange(w) + 0.5) * factor
    coarse_y = (np.arange(h) + 0.5) * factor

    features_da = xr.DataArray(
        np.moveaxis(features, -1, 0),  # -> (band, y, x)
        dims=("band", "y", "x"),
        coords={"band": ["ndvi", "albedo", "builtup"], "y": fine_y, "x": fine_x},
        name="features",
    )
    target_da = xr.DataArray(
        target, dims=("y", "x"), coords={"y": coarse_y, "x": coarse_x},
        name="lst",
    )
    return features_da, target_da, factor


def reaggregate(sharpened, target, factor):
    """Block-average the sharpened field back to the coarse grid (radiance)."""
    coarse = (sharpened ** 4).coarsen(y=factor, x=factor).mean() ** 0.25
    return coarse.assign_coords(y=target.y, x=target.x)


def main():
    features, target, factor = load_scene()
    # Chunk the fine features so the whole pipeline runs lazily on dask.
    features = features.chunk({"y": 60, "x": 60})
    print(f"fine features {dict(features.sizes)}, coarse target {dict(target.sizes)}, "
          f"factor {factor}")

    # 1. Classic global Data Mining Sharpener (land-surface temperature -> use
    #    radiance-space aggregation).
    global_sharp = Sharpener(disaggregating_temperature=True)
    global_out = global_sharp.sharpen(features, target)
    print(f"\nglobal sharpen -> {global_out.dims} {dict(global_out.sizes)}, "
          f"lazy={global_out.chunks is not None}")

    # 2. Moving-window local regression: tile the coarse scene into 4x4-pixel
    #    windows, fit a local model per window, blend with the global model.
    #    smooth_local=True gives a seamless (tent-blended) local field.
    local_sharp = Sharpener(
        disaggregating_temperature=True,
        window_size=4,
        smooth_local=True,
    )
    local_out = local_sharp.sharpen(features, target)
    n_fitted = sum(
        m is not None for row in local_sharp.local_models_ for m in row
    )
    print(f"local sharpen  -> {len(local_sharp.local_models_)}x"
          f"{len(local_sharp.local_models_[0])} windows, {n_fitted} local models fitted")

    # 3. Both results are (approximately) mass-conserving: aggregating the
    #    sharpened field back to the coarse grid returns the observation.
    for name, out in (("global", global_out), ("local", local_out)):
        reagg = reaggregate(out.compute(), target, factor)
        err = float(np.abs(reagg - target).max())
        print(f"{name:>6} max |re-aggregated - target| = {err:.4g}")


if __name__ == "__main__":
    main()
