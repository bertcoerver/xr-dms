"""Synthetic scene builders shared by the xr_dms tests.

Builds a small, GRID-ALIGNED scene: fine feature stack, a coarse target
obtained by block-averaging a hidden fine "truth", and the truth itself for
skill evaluation. Ported from ``examples/dms_explained_numpy.py``.
"""

import numpy as np
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


UTM_CRS = "EPSG:32636"  # Nile delta, the region the real VIIRS/S2 pairing covers


def make_multires_scene(fine_shape=(120, 120), res=10.0, ratios=(1, 2, 6),
                        factor=12, seed=0):
    """Feature bands at several resolutions over one footprint -- the S2 case.

    Every band is derived by block-averaging the same 10 m field, so the coarser
    bands nest *exactly* inside the finest one the way Sentinel-2's 10/20/60 m
    grids nest inside each other. That is what lets a test assert the harmoniser's
    exactness guarantee against a ground truth rather than against a tolerance:
    aggregating the harmonised band must reproduce aggregating the native one.

    ``ratios`` are multiples of ``res`` -- ``(1, 2, 6)`` gives 10 m, 20 m and 60 m.
    ``factor`` is the coarsening from the finest grid to the target, so it must be
    a multiple of every ratio for the coarse pixels to tile all three grids.

    Returns ``(bands, target_da, truth_da, factor)`` where ``bands`` maps
    ``"b10"``/``"b20"``/``"b60"`` to DataArrays on their own grids.
    """
    rng = np.random.default_rng(seed)
    H, W = fine_shape
    x0, y0 = 300000.0, 3220000.0  # upper-left corner, UTM metres
    fine_x = x0 + (np.arange(W) + 0.5) * res
    fine_y = y0 - (np.arange(H) + 0.5) * res  # descending: north-up

    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    ndvi = np.clip(0.5 + 0.4 * np.sin(6 * xx) * np.cos(5 * yy)
                   + 0.1 * rng.standard_normal((H, W)), 0, 1)
    albedo = np.clip(0.3 - 0.2 * ndvi + 0.1 * np.cos(10 * xx + 3 * yy)
                     + 0.03 * rng.standard_normal((H, W)), 0.05, 0.6)
    builtup = (np.sin(3 * xx) > 0.4).astype(float) + 0.03 * rng.standard_normal((H, W))
    truth = (305.0 - 18.0 * ndvi + 12.0 * albedo + 8.0 * builtup
             + 10.0 * np.sin(8.0 * ndvi) + 0.2 * rng.standard_normal((H, W)))

    from pyproj import CRS

    spatial_ref = xr.DataArray(0, attrs={"crs_wkt": CRS.from_user_input(UTM_CRS).to_wkt()})
    coords = {"y": fine_y, "x": fine_x, "spatial_ref": spatial_ref}
    source = {"ndvi": ndvi, "albedo": albedo, "builtup": builtup}

    bands = {}
    for ratio, (name, values) in zip(ratios, source.items()):
        da = xr.DataArray(values, dims=("y", "x"), coords=coords, name=name)
        if ratio > 1:
            da = da.coarsen(y=ratio, x=ratio, boundary="exact").mean()
        bands[f"b{int(res * ratio)}"] = da.rename(f"b{int(res * ratio)}")

    lowres = _block_mean(truth ** 4, factor) ** 0.25
    h, w = lowres.shape
    target_da = xr.DataArray(
        lowres, dims=("y", "x"),
        coords={"y": fine_y.reshape(h, factor).mean(axis=1),
                "x": fine_x.reshape(w, factor).mean(axis=1)},
        name="target",
    )
    truth_da = xr.DataArray(
        truth, dims=("y", "x"),
        coords={"y": fine_y, "x": fine_x}, name="truth",
    )
    return bands, target_da, truth_da, factor


def make_swath_scene(fine_shape=(300, 300), res=20.0, swath_res=375.0, seed=0):
    """A projected fine grid plus a genuinely curvilinear swath over it.

    Mirrors the real pairing: a north-up UTM optical grid (so ``y`` **descends**,
    which the rectilinear code path historically got wrong) and a thermal swath
    whose 2-D lat/lon is skewed and curved, so nothing that assumes a separable
    or rectilinear coarse grid can pass by accident.

    Returns ``(features_ds, target_da, truth_da, grid_map)`` where ``target_da``
    carries the swath's own ``(y, x)`` scan dims, exactly as a granule does.
    """
    from pyproj import CRS, Transformer

    from xr_dms.geo import SwathGridMap

    rng = np.random.default_rng(seed)
    H, W = fine_shape
    x0, y0 = 300000.0, 3220000.0  # upper-left corner
    fine_x = x0 + (np.arange(W) + 0.5) * res
    fine_y = y0 - (np.arange(H) + 0.5) * res  # descending: north-up

    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    ndvi = np.clip(0.5 + 0.4 * np.sin(6 * xx) * np.cos(5 * yy)
                   + 0.1 * rng.standard_normal((H, W)), 0, 1)
    albedo = np.clip(0.3 - 0.2 * ndvi + 0.1 * np.cos(10 * xx + 3 * yy)
                     + 0.03 * rng.standard_normal((H, W)), 0.05, 0.6)
    builtup = (np.sin(3 * xx) > 0.4).astype(float) + 0.03 * rng.standard_normal((H, W))
    truth = (305.0 - 18.0 * ndvi + 12.0 * albedo + 8.0 * builtup
             + 10.0 * np.sin(8.0 * ndvi) + 0.2 * rng.standard_normal((H, W)))

    crs = CRS.from_user_input(UTM_CRS)
    coords = {"y": fine_y, "x": fine_x}
    spatial_ref = xr.DataArray(0, attrs={"crs_wkt": crs.to_wkt()})
    features = xr.Dataset(
        {
            "ndvi": (("y", "x"), ndvi),
            "albedo": (("y", "x"), albedo),
            "builtup": (("y", "x"), builtup),
        },
        coords={**coords, "spatial_ref": spatial_ref},
    )
    truth_da = xr.DataArray(truth, dims=("y", "x"), coords=coords, name="truth")

    # A curvilinear swath: skewed across-track, curved along-track, and rotated
    # relative to the UTM axes -- nothing a rectilinear shortcut can exploit.
    to_ll = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon_c, lat_c = to_ll.transform(fine_x.mean(), fine_y.mean())
    span_y = H * res / swath_res
    span_x = W * res / swath_res
    ny = int(span_y) + 4
    nx = int(span_x) + 4
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    deg = swath_res / 111320.0
    lon = (lon_c + (ii - nx / 2) * deg / np.cos(np.deg2rad(lat_c))
           + 0.15 * (jj - ny / 2) * deg)
    lat = lat_c - (jj - ny / 2) * deg + 0.10 * np.sin(ii / 5.0) * deg
    lon_da = xr.DataArray(lon, dims=("y", "x"), name="longitude")
    lat_da = xr.DataArray(lat, dims=("y", "x"), name="latitude")

    grid_map = SwathGridMap.from_lonlat(lon_da, lat_da, fine=features)

    # The coarse sensor sees the mean over its footprint, in radiance space.
    coarse = grid_map.aggregate_mean(truth_da ** 4) ** 0.25
    target_da = coarse.rename(dict(zip(coarse.dims, ("y", "x")))).rename("target")
    return features, target_da, truth_da, grid_map
