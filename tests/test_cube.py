"""The whole-time-series path: same numbers, a layer count that does not grow.

:mod:`xr_dms.cube` exists because concatenating per-scene graphs makes the layer
count grow with the series, and that is where dask's culling becomes a lottery.
So the two things worth asserting are that it agrees with the per-scene path it
replaces, and that its graph stays the same size as scenes are added.
"""

import numpy as np
import pytest
import xarray as xr

import dask

from xr_dms import Sharpener, SklearnDMSRegressor
from xr_dms.cube import sharpen_cube

from scenes import make_swath_scene


PINNED = dict(regressor_opt={"random_state": 0}, bagging_opt={"random_state": 0})


def _sharpener(grid_map):
    """A sharpener with the bootstrap pinned, so two fits are comparable.

    The default regressor bags decision trees off the global RNG, which moves the
    result by a couple of Kelvin between runs -- real, and not what these tests
    are about.
    """
    return Sharpener(
        grid_map=grid_map,
        regressor=SklearnDMSRegressor(**PINNED),
        disaggregating_temperature=True,
        smooth_residual=False,
    )


@pytest.fixture(scope="module")
def swath():
    features, target, _, grid_map = make_swath_scene()
    return features, target, grid_map


def _cube(swath, n_scenes, chunk=64):
    """``n_scenes`` copies of one scene, offset so they are not identical."""
    features, target, grid_map = swath
    times = np.arange(n_scenes).astype("datetime64[D]").astype("datetime64[ns]")
    stack = xr.concat(
        [features + 0.01 * i for i in range(n_scenes)], dim="time",
    ).assign_coords(time=times).chunk({"time": 1, "y": chunk, "x": chunk})
    targets = [target + 0.05 * i for i in range(n_scenes)]

    def bundle(i):
        return _sharpener(grid_map).scene_bundle(
            features + 0.01 * i, targets[i],
        )

    return stack, targets, grid_map, bundle


def test_matches_the_per_scene_path(swath):
    """The cube must reproduce what Sharpener.sharpen gives scene by scene."""
    features, target, grid_map = swath
    stack, targets, gm, bundle = _cube(swath, 3)

    cube = sharpen_cube(
        _sharpener(gm), stack, [(i,) for i in range(3)], bundle,
    ).compute()

    for i in range(3):
        expected = _sharpener(gm).sharpen(features + 0.01 * i, targets[i]).compute()
        np.testing.assert_allclose(
            cube.isel(time=i).values, expected.values, rtol=1e-9, atol=1e-9,
        )


def test_layer_count_does_not_grow_with_the_series(swath):
    counts = {}
    for n in (1, 4, 16):
        stack, _, gm, bundle = _cube(swath, n)
        out = sharpen_cube(_sharpener(gm), stack, [(i,) for i in range(n)], bundle)
        counts[n] = len(out.data.__dask_graph__().layers)
    assert len(set(counts.values())) == 1, counts


def test_graph_is_reproducible_across_builds(swath):
    """Two builds of the same cube must produce the same layer names.

    A name that varies between builds reshuffles set iteration inside dask's
    optimiser, and the culled graph then differs run to run -- the failure this
    module was written to end.
    """
    def build():
        stack, _, gm, bundle = _cube(swath, 8)
        out = sharpen_cube(_sharpener(gm), stack, [(i,) for i in range(8)], bundle)
        return set(out.data.__dask_graph__().layers)

    assert build() == build()


def test_culling_is_stable_and_small(swath):
    """One scene, one spatial block, optimised repeatedly from fresh builds."""
    def measure():
        stack, _, gm, bundle = _cube(swath, 16)
        out = sharpen_cube(_sharpener(gm), stack, [(i,) for i in range(16)], bundle)
        sub = out.isel(time=3, y=slice(0, 64), x=slice(0, 64))
        (opt,) = dask.optimize(sub.data)
        return len(opt.__dask_graph__())

    counts = {measure() for _ in range(8)}
    assert len(counts) == 1, counts
    # One of these is the hoisted bundle key (bundled_arrays' shared_arg), which
    # is one key for the whole cube rather than one per scene.
    assert max(counts) <= 20, counts


def test_building_the_cube_computes_nothing(swath):
    calls = []

    def counting_bundle(i):
        calls.append(i)
        return _cube(swath, 1)[3](0)

    stack, _, gm, _ = _cube(swath, 8)
    sharpen_cube(_sharpener(gm), stack, [(i,) for i in range(8)], counting_bundle)
    assert calls == []


def test_one_scene_computes_one_bundle(swath):
    stack, _, gm, bundle = _cube(swath, 8)
    out = sharpen_cube(_sharpener(gm), stack, [(i,) for i in range(8)], bundle)
    (opt,) = dask.optimize(out.isel(time=5).data)
    keys = [k for k in opt.__dask_graph__() if "bundle" in str(k)]
    assert len(keys) <= 1


def test_mass_is_conserved(swath):
    """smooth_residual=False must conserve exactly, scene by scene.

    Re-aggregating the sharpened field through the grid map has to reproduce the
    coarse observation. This is the assertion that caught the impure-fit bug on
    the first lazy implementation, where the residual was measured against one
    model and applied to another.
    """
    features, target, grid_map = swath
    stack, targets, gm, bundle = _cube(swath, 2)
    cube = sharpen_cube(_sharpener(gm), stack, [(i,) for i in range(2)], bundle)

    sharpener = _sharpener(gm)
    for i in range(2):
        sharp = cube.isel(time=i, drop=True).compute()
        back = gm.aggregate_mean(sharp ** 4) ** 0.25
        obs = gm.prepare_target(targets[i])
        finite = np.isfinite(back.values) & np.isfinite(obs.values)
        assert finite.any()
        np.testing.assert_allclose(
            back.values[finite], obs.values[finite], rtol=1e-10,
        )


def test_unfittable_scene_is_all_nan_not_an_exception(swath):
    """A scene with no homogeneous training pixels must be a value, not a raise.

    By the time a bundle runs the graph's shape is fixed, so the failure has to
    travel as data.
    """
    features, target, grid_map = swath
    stack, _, gm, _ = _cube(swath, 1)
    blank = xr.full_like(target, np.nan)

    def bundle(_i):
        return _sharpener(gm).scene_bundle(features, blank)

    out = sharpen_cube(_sharpener(gm), stack, [(0,)], bundle).compute()
    assert np.isnan(out.values).all()


def test_rejects_a_time_axis_chunked_wider_than_one_scene(swath):
    stack, _, gm, bundle = _cube(swath, 4)
    stack = stack.chunk({"time": 2})
    with pytest.raises(ValueError, match="one scene at a time"):
        sharpen_cube(_sharpener(gm), stack, [(i,) for i in range(4)], bundle)


def test_rejects_a_scene_count_mismatch(swath):
    stack, _, gm, bundle = _cube(swath, 4)
    with pytest.raises(ValueError, match="argument tuple"):
        sharpen_cube(_sharpener(gm), stack, [(0,)], bundle)


def test_rejects_smoothed_residual(swath):
    stack, _, gm, bundle = _cube(swath, 1)
    sharpener = Sharpener(grid_map=gm, smooth_residual=True)
    with pytest.raises(NotImplementedError, match="smooth_residual=False"):
        sharpen_cube(sharpener, stack, [(0,)], bundle)
