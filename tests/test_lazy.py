"""The deferred path: build the whole graph without reading a pixel.

The eager path answers "what is the sharpened field?"; this one answers "what
would it cost to ask?". Every test here is really one of two questions: does
deferring change the numbers (it must not), and does deferring actually defer
(nothing may run until ``compute``).

The default regressor bags decision trees without a fixed seed, so two fits of
the same data differ by a couple of Kelvin. Every comparison below therefore
pins ``random_state`` -- without it these tests would compare the regressor's
noise, not the code path.
"""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from scenes import make_swath_lonlat, make_swath_scene
from xr_dms import SceneState, Sharpener, SwathGridMap
from xr_dms.regressors import SklearnDMSRegressor


SHARPEN_OPTS = dict(disaggregating_temperature=True, smooth_residual=False)


def seeded():
    """A regressor whose fit is reproducible, so lazy and eager are comparable."""
    return SklearnDMSRegressor(
        regressor_opt={"random_state": 0}, bagging_opt={"random_state": 0}
    )


class CountTasks(Callback):
    """Counts scheduler invocations, i.e. how many times anything computed."""

    def __init__(self):
        self.runs = []

    def _start(self, dsk):
        self.runs.append(len(dsk))


@pytest.fixture
def swath_lazy():
    """``(features_chunked, target, lon, lat)`` for the deferred path."""
    features, target, _, _ = make_swath_scene()
    lon, lat = make_swath_lonlat(features)
    return features.chunk({"y": 64, "x": 64}), target, lon, lat


def lazy_map(lon, lat, features, **kw):
    return SwathGridMap.from_lonlat(
        lon.chunk(), lat.chunk(), fine=features,
        min_fine_fraction=0.5, lazy=True, **kw,
    )


def eager_map(lon, lat, features, **kw):
    return SwathGridMap.from_lonlat(
        lon, lat, fine=features, min_fine_fraction=0.5, **kw
    )


# -- the grid map -------------------------------------------------------------
def test_lazy_grid_map_reads_no_geolocation(swath_lazy):
    features, _, lon, lat = swath_lazy
    counter = CountTasks()
    with counter:
        gm = lazy_map(lon, lat, features)
    assert gm.lazy
    assert counter.runs == [], "building the map computed something"


def test_lazy_grid_map_equals_the_eager_one(swath_lazy):
    features, _, lon, lat = swath_lazy
    gm_l = lazy_map(lon, lat, features)
    gm_e = eager_map(lon, lat, features)

    np.testing.assert_array_equal(gm_l.labels.compute(), gm_e.labels)
    np.testing.assert_allclose(gm_l._cell_size.compute(), gm_e._cell_size)
    assert gm_l.coverage.compute() == gm_e.coverage
    assert gm_l.coarse_shape == gm_e.coarse_shape


def test_lazy_aggregate_matches_eager(swath_lazy):
    features, _, lon, lat = swath_lazy
    bands = features.to_array(dim="band")
    mean_l, std_l = lazy_map(lon, lat, features).aggregate(bands, band_dim="band")
    mean_e, std_e = eager_map(lon, lat, features).aggregate(
        bands.compute(), band_dim="band"
    )

    assert hasattr(mean_l.data, "dask"), "the lazy map must not force aggregation"
    np.testing.assert_allclose(
        mean_l.compute().values, mean_e.values, rtol=1e-12, equal_nan=True
    )
    # The std is a sum-of-squares difference accumulated per block, so block
    # order costs it a few ulp against the single-pass version.
    np.testing.assert_allclose(
        std_l.compute().values, std_e.values, rtol=1e-9, equal_nan=True
    )


def test_lazy_map_refuses_what_it_cannot_do(swath_lazy):
    features, target, lon, lat = swath_lazy
    gm = lazy_map(lon, lat, features)
    coarse = gm.prepare_target(target)

    with pytest.raises(NotImplementedError, match="swath centres"):
        gm.upsample(coarse, features["ndvi"], method="linear")
    with pytest.raises(NotImplementedError, match="swath centres"):
        gm.build_window_basis(2, 2, 4, False, features)


# -- the sharpener ------------------------------------------------------------
def test_lazy_sharpen_touches_nothing_at_build_time(swath_lazy):
    features, target, lon, lat = swath_lazy
    counter = CountTasks()
    with counter:
        out = Sharpener(
            grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
        ).sharpen(features, target, lazy=True)

    assert counter.runs == [], "building the graph computed something"
    assert out.chunks is not None
    assert out.shape == (features.sizes["y"], features.sizes["x"])


def test_lazy_sharpen_matches_eager_sharpen(swath_lazy):
    features, target, lon, lat = swath_lazy
    lazy = Sharpener(
        grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).sharpen(features, target, lazy=True)
    eager = Sharpener(
        grid_map=eager_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).sharpen(features, target)

    np.testing.assert_allclose(
        lazy.compute().values, eager.compute().values, rtol=1e-8, equal_nan=True
    )


def test_lazy_sharpen_stays_mass_conserving(swath_lazy):
    """``smooth_residual=False`` is exact; deferring must not soften that.

    Deliberately the *default* regressor rather than a seeded one: an impure fit
    breaks conservation without changing anything else observable, and a pinned
    seed would hide it.
    """
    features, target, lon, lat = swath_lazy
    out = Sharpener(
        grid_map=lazy_map(lon, lat, features), **SHARPEN_OPTS
    ).sharpen(features, target, lazy=True).compute()

    check = eager_map(lon, lat, features)
    back = check.aggregate_mean(out ** 4) ** 0.25
    obs = check.prepare_target(target)
    both = np.isfinite(back.values) & np.isfinite(obs.values)
    assert both.any()
    np.testing.assert_allclose(back.values[both], obs.values[both], rtol=1e-12)


def test_unfittable_scene_yields_nan_rather_than_raising(swath_lazy):
    """A lazy graph cannot change its own shape, so a failed fit is a value."""
    features, target, lon, lat = swath_lazy
    blank = features.map(lambda v: v * np.nan)

    eager = Sharpener(
        grid_map=eager_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    )
    with pytest.raises(ValueError, match="No homogeneous training pixels"):
        eager.sharpen(blank, target)

    out = Sharpener(
        grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).sharpen(blank, target, lazy=True)
    assert np.isnan(out.compute().values).all()


def test_fit_delayed_reports_the_failure_on_the_state(swath_lazy):
    features, target, lon, lat = swath_lazy
    blank = features.map(lambda v: v * np.nan)
    state = Sharpener(
        grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).fit_delayed(blank, target).compute()

    assert isinstance(state, SceneState)
    assert not state.fitted
    assert state.band_order == list(features.data_vars)


def test_fit_delayed_refuses_moving_windows(swath_lazy):
    features, target, lon, lat = swath_lazy
    sharpener = Sharpener(
        grid_map=lazy_map(lon, lat, features), window_size=4, **SHARPEN_OPTS
    )
    with pytest.raises(NotImplementedError, match="window_size"):
        sharpener.fit_delayed(features, target)


# -- culling ------------------------------------------------------------------
def test_a_precomputed_state_makes_the_fine_half_blockwise(swath_lazy):
    """The point of :meth:`scene_state`: a spatial subset costs a chunk.

    Without the coarse residual in hand the residual correction is a
    scene-global reduction, so one output chunk drags the whole scene through
    predict. With it, the fine half is predict -> gather -> add and culls.
    """
    import dask

    features, target, lon, lat = swath_lazy
    state = Sharpener(
        grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).scene_state(features, target).compute()
    assert state.fitted
    assert state.residual_coarse.shape == (20, 20)

    out = Sharpener(
        grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).apply(state, features, target)

    full = len(out.data.__dask_graph__())
    (one,) = dask.optimize(out.isel(y=slice(0, 64), x=slice(0, 64)).data)
    assert len(one.__dask_graph__()) < full / 4, (
        f"a single chunk kept {len(one.__dask_graph__())} of {full} tasks"
    )


def test_a_precomputed_state_gives_the_same_numbers(swath_lazy):
    features, target, lon, lat = swath_lazy
    reference = Sharpener(
        grid_map=eager_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).sharpen(features, target).compute()

    state = Sharpener(
        grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).scene_state(features, target).compute()
    out = Sharpener(
        grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
    ).apply(state, features, target)

    np.testing.assert_allclose(
        out.compute().values, reference.values, rtol=1e-8, equal_nan=True
    )


def count_fits(build):
    """How many models get trained while computing whatever ``build`` returns."""
    fits = []
    original = SklearnDMSRegressor.fit

    def counting_fit(self, *args, **kwargs):
        fits.append(1)
        return original(self, *args, **kwargs)

    obj = build()
    assert not fits, "building the graph already fitted something"
    SklearnDMSRegressor.fit = counting_fit
    try:
        obj.compute()
    finally:
        SklearnDMSRegressor.fit = original
    return len(fits)


def sharpened_cube(features, target, lon, lat, n):
    """``n`` per-scene lazy graphs concatenated on a time axis."""
    scenes = []
    for i in range(n):
        sharpener = Sharpener(
            grid_map=lazy_map(lon, lat, features), regressor=seeded(), **SHARPEN_OPTS
        )
        da = sharpener.sharpen(features, target + float(i), lazy=True)
        stamp = np.datetime64("2025-02-01") + np.timedelta64(i, "D")
        scenes.append(da.expand_dims(time=[stamp]))
    return xr.concat(scenes, dim="time")


def test_a_time_series_culls_to_the_selected_scene(swath_lazy):
    """Selecting one scene of a cube must cost one scene, however long the cube.

    Asserted as a ratio rather than an absolute count: the residual correction
    consumes the first guess twice, so dask may evaluate a scene's fit more than
    once. What must not happen is the cost scaling with the *other* scenes.
    """
    features, target, lon, lat = swath_lazy

    one = count_fits(lambda: sharpened_cube(features, target, lon, lat, 1))
    picked = count_fits(
        lambda: sharpened_cube(features, target, lon, lat, 6).isel(time=3)
    )
    assert picked == one, (
        f"one scene of a six-scene cube trained {picked} models, "
        f"but the scene on its own trains {one}"
    )


def test_the_deferred_fit_is_a_pure_function(swath_lazy):
    """The residual is differenced against the first guess, so both must come
    from the *same* model. Dask can evaluate a task twice, so the fit has to be
    reproducible -- otherwise mass conservation silently breaks.

    Uses the default regressor deliberately: its bagging draws on numpy's global
    RNG, which is exactly the thing :meth:`BaseRegressor.seeded` has to tame.
    """
    features, target, lon, lat = swath_lazy
    sharpener = Sharpener(grid_map=lazy_map(lon, lat, features), **SHARPEN_OPTS)
    state = sharpener.fit_delayed(features, target)

    first = state.compute()
    second = state.compute()
    np.testing.assert_allclose(
        first.global_model.predict(np.eye(3)),
        second.global_model.predict(np.eye(3)),
    )


def test_an_explicit_random_state_is_not_overridden():
    """Pinning the seed yourself already makes the fit pure; leave it alone."""
    pinned = SklearnDMSRegressor(bagging_opt={"random_state": 7})
    assert pinned.seeded(999) is pinned

    default = SklearnDMSRegressor()
    assert default.seeded(999).bagging_opt["random_state"] == 999
    assert "random_state" not in default.bagging_opt, "seeded() mutated the original"
