"""Swath-native sharpening: a curvilinear target onto a projected fine grid."""

import numpy as np
import pytest
import xarray as xr

from scenes import UTM_CRS, make_scene, make_swath_scene
from xr_dms import Sharpener, SwathGridMap
from xr_dms.gridmap import RegularGridMap, _axis_geometry


# -- the fine -> swath label map ---------------------------------------------
def test_labels_partition_the_fine_grid(swath_scene):
    _, _, _, gm = swath_scene
    labels = gm.labels

    assert labels.shape == (300, 300)
    assert gm.coverage == pytest.approx(1.0), "the swath should blanket the AOI"
    assert labels.max() < np.prod(gm.coarse_shape)
    # Every fine pixel has exactly one parent, so the cell sizes must add up.
    assert gm._cell_size.sum() == (labels >= 0).sum()


def test_labels_pick_the_nearest_swath_pixel(swath_scene):
    features, _, _, gm = swath_scene
    sx, sy = gm._proj_xy
    pts = np.column_stack([sx.ravel(), sy.ravel()])

    rng = np.random.default_rng(0)
    ys = np.asarray(features["y"].values)
    xs = np.asarray(features["x"].values)
    for iy, ix in zip(rng.integers(0, ys.size, 60), rng.integers(0, xs.size, 60)):
        d = np.hypot(pts[:, 0] - xs[ix], pts[:, 1] - ys[iy])
        assert gm.labels[iy, ix] == int(np.argmin(d))


def test_uncovered_fine_pixels_get_no_parent():
    """A swath that only grazes the grid must leave the rest unparented."""
    features, _, _, gm_full = make_swath_scene()
    lon = xr.DataArray(
        np.linspace(30.9, 31.0, 6)[None, :] * np.ones((5, 1)), dims=("y", "x")
    )
    lat = xr.DataArray(
        np.linspace(29.9, 30.0, 5)[:, None] * np.ones((1, 6)), dims=("y", "x")
    )
    gm = SwathGridMap.from_lonlat(lon, lat, fine=features)
    assert gm.coverage < 0.5
    assert (gm.labels == -1).any()


# -- aggregation --------------------------------------------------------------
def test_aggregate_matches_a_direct_bincount(swath_scene):
    features, _, truth, gm = swath_scene
    mean, std = gm.aggregate(truth, band_dim="band")

    n = int(np.prod(gm.coarse_shape))
    lab = gm.labels.ravel()
    keep = lab >= 0
    idx = np.where(keep, lab, n)
    v = np.asarray(truth.values).ravel()
    cnt = np.bincount(idx, weights=keep.astype(float), minlength=n + 1)[:n]
    s1 = np.bincount(idx, weights=np.where(keep, v, 0.0), minlength=n + 1)[:n]
    s2 = np.bincount(idx, weights=np.where(keep, v * v, 0.0), minlength=n + 1)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_mean = np.where(cnt > 0, s1 / cnt, np.nan)
        ref_std = np.sqrt(np.clip(np.where(cnt > 0, s2 / cnt, np.nan) - ref_mean ** 2, 0, None))

    np.testing.assert_allclose(
        mean.values.ravel(), ref_mean, rtol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(std.values.ravel(), ref_std, rtol=1e-9, equal_nan=True)


def test_aggregate_ignores_nans_but_honours_the_coverage_floor(swath_scene):
    features, _, truth, gm = swath_scene
    holed = truth.where(np.asarray(truth.values) < np.nanpercentile(truth.values, 60))

    mean, _ = gm.aggregate(holed, band_dim="band")
    assert np.isfinite(mean.values).any(), "partly-valid cells should still average"

    strict = SwathGridMap(
        labels=gm.labels, swath_shape=gm.swath_shape, swath_dims=gm.swath_dims,
        fine_coords=gm._fine_coords, x_dim="x", y_dim="y", proj_xy=gm._proj_xy,
        min_fine_fraction=0.95,
    )
    strict_mean, _ = strict.aggregate(holed, band_dim="band")
    assert np.isfinite(strict_mean.values).sum() < np.isfinite(mean.values).sum()


def test_swath_over_a_regular_grid_reproduces_the_regular_map():
    """When the swath centres happen to be a regular block grid, the two maps agree.

    The Voronoi partition of block centres *is* the block partition, so this pins
    SwathGridMap against the long-standing coarsen/interp implementation.
    """
    from pyproj import CRS, Transformer

    factor = 15
    H = W = 300
    res, x0, y0 = 20.0, 300000.0, 3220000.0
    fine_x = x0 + (np.arange(W) + 0.5) * res
    fine_y = y0 - (np.arange(H) + 0.5) * res
    crs = CRS.from_user_input(UTM_CRS)
    field = xr.DataArray(
        np.sin(np.mgrid[0:H, 0:W][1] / 31.0) + np.cos(np.mgrid[0:H, 0:W][0] / 27.0),
        dims=("y", "x"), coords={"y": fine_y, "x": fine_x},
    )
    field = field.assign_coords(
        spatial_ref=xr.DataArray(0, attrs={"crs_wkt": crs.to_wkt()})
    )

    # Block centres of an exact factor-15 coarsening, expressed as lon/lat.
    cx = fine_x.reshape(-1, factor).mean(axis=1)
    cy = fine_y.reshape(-1, factor).mean(axis=1)
    gx, gy = np.meshgrid(cx, cy)
    to_ll = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = to_ll.transform(gx, gy)
    gm = SwathGridMap.from_lonlat(
        xr.DataArray(lon, dims=("y", "x")), xr.DataArray(lat, dims=("y", "x")),
        fine=field, radius_of_influence=res * factor,
    )
    assert gm.coverage == pytest.approx(1.0)

    target = xr.DataArray(np.zeros((H // factor, W // factor)), dims=("y", "x"),
                          coords={"y": cy, "x": cx})
    regular = RegularGridMap.from_grids(field.expand_dims("band"), target)

    swath_mean = gm.aggregate_mean(field).values
    regular_mean = regular.aggregate_mean(field).values
    np.testing.assert_allclose(swath_mean, regular_mean, rtol=1e-10)


# -- the full pipeline --------------------------------------------------------
def test_sharpen_lands_on_the_exact_fine_grid(swath_scene):
    features, target, _, gm = swath_scene
    sharp = Sharpener(grid_map=gm, disaggregating_temperature=True).sharpen(
        features, target
    )

    assert sharp.dims == ("y", "x")
    assert sharp.shape == (features.sizes["y"], features.sizes["x"])
    np.testing.assert_array_equal(sharp["y"].values, features["y"].values)
    np.testing.assert_array_equal(sharp["x"].values, features["x"].values)
    assert np.isfinite(sharp.values).all()


def test_sharpen_conserves_mass_exactly(swath_scene):
    """smooth_residual=False must round-trip through the labelled aggregation."""
    features, target, _, gm = swath_scene
    sharp = Sharpener(
        grid_map=gm, disaggregating_temperature=True, smooth_residual=False
    ).sharpen(features, target)

    back = gm.aggregate_mean(sharp ** 4) ** 0.25
    obs = gm.prepare_target(target)
    both = np.isfinite(back.values) & np.isfinite(obs.values)
    assert both.sum() > 200
    np.testing.assert_allclose(
        back.values[both], obs.values[both], rtol=1e-6
    )


def test_sharpen_beats_a_naive_upsample(swath_scene):
    features, target, truth, gm = swath_scene
    sharp = Sharpener(grid_map=gm, disaggregating_temperature=True).sharpen(
        features, target
    )
    naive = gm.upsample(gm.prepare_target(target), truth, method="nearest")

    err_sharp = float(np.sqrt(((sharp - truth) ** 2).mean()))
    err_naive = float(np.sqrt(((naive - truth) ** 2).mean()))
    assert err_sharp < err_naive


def test_sharpen_stays_lazy_and_chunk_invariant(swath_scene):
    """One fitted model, three chunkings, identical output.

    The regressor is an unseeded bagged ensemble, so the model is fitted once and
    only the *application* is re-chunked -- as the regular-grid tests do.
    """
    features, target, _, gm = swath_scene
    fitted = Sharpener(grid_map=gm, disaggregating_temperature=True).fit(
        features, target
    )

    def run(obj):
        return fitted.residual_correct(fitted.first_guess(obj, target), target)

    lazy = run(features.chunk({"y": 128, "x": 128}))
    assert lazy.chunks is not None, "the swath path must stay dask-backed"

    eager = run(features).values
    a = lazy.compute().values
    b = run(features.chunk({"y": 75, "x": 300})).compute().values
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(eager, a, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("smooth_local", [True, False])
def test_local_windows_work_on_a_swath(swath_scene, smooth_local):
    features, target, truth, gm = swath_scene
    sharp = Sharpener(
        grid_map=gm, disaggregating_temperature=True, window_size=8,
        smooth_local=smooth_local, min_training_samples=10,
    ).sharpen(features, target)

    assert sharp.shape == truth.shape
    assert np.isfinite(sharp.values).all()
    # The blend must not invent values outside the plausible range of the scene.
    assert float(sharp.min()) > float(truth.min()) - 15
    assert float(sharp.max()) < float(truth.max()) + 15


@pytest.mark.parametrize("smooth_local", [True, False])
def test_local_windows_are_chunk_invariant_on_a_swath(swath_scene, smooth_local):
    """The scattered window basis must not depend on how the fine grid is tiled."""
    features, target, _, gm = swath_scene
    fitted = Sharpener(
        grid_map=gm, disaggregating_temperature=True, window_size=8,
        smooth_local=smooth_local,
    ).fit(features, target)

    eager = fitted.predict_local(features).values
    a = fitted.predict_local(features.chunk({"y": 100, "x": 100})).compute().values
    b = fitted.predict_local(features.chunk({"y": 300, "x": 60})).compute().values
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(eager, a, rtol=1e-10, atol=1e-10, equal_nan=True)


# -- guardrails ---------------------------------------------------------------
def test_extra_dimensions_are_rejected(swath_scene):
    features, target, _, gm = swath_scene
    stacked = features.expand_dims(time=2)
    with pytest.raises(ValueError, match="isel"):
        Sharpener(grid_map=gm).fit(stacked, target)


def test_target_shape_must_match_the_swath(swath_scene):
    features, target, _, gm = swath_scene
    with pytest.raises(ValueError, match="does not match the swath"):
        Sharpener(grid_map=gm).fit(features, target.isel(y=slice(0, 5)))


def test_missing_crs_is_reported_clearly():
    features, _, _, _ = make_swath_scene()
    bare = features.drop_vars("spatial_ref")
    lon = xr.DataArray(np.zeros((4, 4)) + 30.9, dims=("y", "x"))
    lat = xr.DataArray(np.zeros((4, 4)) + 29.1, dims=("y", "x"))
    with pytest.raises(ValueError, match="Could not determine the CRS"):
        SwathGridMap.from_lonlat(lon, lat, fine=bare)


def test_single_band_dataarray_features_work(swath_scene):
    """A bare 2-D feature array used to raise KeyError('band') inside predict."""
    features, target, _, gm = swath_scene
    sharp = Sharpener(grid_map=gm).sharpen(features["ndvi"], target)
    assert sharp.shape == (features.sizes["y"], features.sizes["x"])


# -- the rectilinear window geometry, on a north-up axis ----------------------
@pytest.mark.parametrize("descending", [False, True])
def test_window_edges_follow_the_axis_direction(descending):
    """North-up rasters have a descending y; the open cell ends must follow it."""
    factor, window, n_coarse = 5, 1, 3
    coord = np.arange(15, dtype=float) + 0.5
    if descending:
        coord = coord[::-1]

    centers, edges = _axis_geometry(3, n_coarse, coord, window, factor)
    # Each coordinate must fall in exactly one window cell, and in the cell whose
    # centre it is nearest to.
    inside = (coord[None, :] >= edges[:, :1]) & (coord[None, :] < edges[:, 1:])
    np.testing.assert_array_equal(inside.sum(axis=0), np.ones(coord.size))
    np.testing.assert_array_equal(
        inside.argmax(axis=0), np.abs(coord[None, :] - centers[:, None]).argmin(axis=0)
    )


def test_hard_local_windows_survive_a_descending_axis():
    """The same bug, end to end: a north-up scene with smooth_local=False."""
    features, target, truth, factor = make_scene(fine_shape=(60, 60), factor=10)
    flip = {"y": features["y"].values[::-1]}
    features_n = features.isel(y=slice(None, None, -1)).assign_coords(flip)
    truth_n = truth.isel(y=slice(None, None, -1)).assign_coords(flip)
    target_n = target.isel(y=slice(None, None, -1)).assign_coords(
        {"y": target["y"].values[::-1]}
    )

    kw = dict(disaggregating_temperature=True, window_size=2, smooth_local=False,
              min_training_samples=4)
    up = Sharpener(**kw).sharpen(features, target)
    down = Sharpener(**kw).sharpen(features_n, target_n)

    err_up = float(np.sqrt(((up - truth) ** 2).mean()))
    err_down = float(np.sqrt(((down - truth_n) ** 2).mean()))
    assert err_down == pytest.approx(err_up, rel=0.1), (
        "flipping the y axis must not change the sharpening skill"
    )


def test_linear_upsample_is_a_function_of_its_inputs_under_threads():
    """Qhull must not be entered from two threads at once.

    Building a ``LinearNDInterpolator`` while another one is being evaluated
    returns subtly wrong values -- a few tenths of a Kelvin over a fraction of a
    percent of the grid, somewhere different every run. A cube does exactly that:
    one Delaunay per overpass, evaluated block by block. The lock in
    :mod:`xr_dms.geo` is what makes this deterministic; without it this test
    fails most runs.
    """
    from concurrent.futures import ThreadPoolExecutor

    features, _, _, gm = make_swath_scene()
    like = features.to_array("band").isel(band=0, drop=True).chunk({"y": 32, "x": 32})
    n_coarse = int(np.prod(gm.coarse_shape))
    fields = [
        gm._to_coarse_da(np.linspace(0, 1, n_coarse) * scale)
        for scale in (1.0, 2.0, 3.0, 4.0)
    ]

    def up(field):
        return np.asarray(gm.upsample(field, like, method="linear").values)

    expected = [up(f) for f in fields]
    for _ in range(3):
        with ThreadPoolExecutor(len(fields)) as pool:
            got = list(pool.map(up, fields))
        for a, b in zip(expected, got):
            np.testing.assert_array_equal(a, b)
