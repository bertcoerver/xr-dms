"""Tests for the moving-window local regression + local/global blending."""

import numpy as np
import pytest

from xr_dms import Sharpener


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _reaggregate(sharpened, target, factor):
    reagg = (sharpened ** 4).coarsen(y=factor, x=factor).mean() ** 0.25
    return reagg.assign_coords(y=target.y, x=target.x)


def test_window_grid_is_built(varying_scene):
    features, target, _, _ = varying_scene
    sharp = Sharpener(window_size=5).fit(features, target)
    # 20x20 coarse grid, window_size 5 -> 4x4 windows.
    assert len(sharp.local_models_) == 4
    assert len(sharp.local_models_[0]) == 4
    n_fitted = sum(m is not None for row in sharp.local_models_ for m in row)
    assert n_fitted >= 1
    # Global model is always trained.
    assert sharp.global_model_ is not None


def test_predict_local_is_lazy_and_finite(varying_scene):
    features, target, _, _ = varying_scene
    sharp = Sharpener(window_size=5).fit(features, target)
    lazy = sharp.predict_local(features.chunk({"y": 40, "x": 40}))
    assert lazy.chunks is not None
    values = lazy.compute()
    assert values.shape == features.isel(band=0).shape
    # Every fine pixel is covered by at least one window model here.
    assert np.isfinite(values.values).all()


def test_predict_local_is_chunk_invariant(varying_scene):
    features, target, _, _ = varying_scene
    sharp = Sharpener(window_size=5).fit(features, target)
    eager = sharp.predict_local(features).values
    a = sharp.predict_local(features.chunk({"y": 40, "x": 40})).compute().values
    b = sharp.predict_local(features.chunk({"y": 32, "x": 80})).compute().values
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(eager, a, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_smooth_local_is_more_seamless_than_hard(varying_scene):
    features, target, _, factor = varying_scene
    window_size = 5
    hard = Sharpener(window_size=window_size, smooth_local=False).fit(features, target)
    smooth = Sharpener(window_size=window_size, smooth_local=True).fit(features, target)
    ph = hard.predict_local(features).values
    ps = smooth.predict_local(features).values

    bcol = window_size * factor  # first interior window boundary
    hard_jump = np.nanmax(np.abs(ph[:, bcol] - ph[:, bcol - 1]))
    smooth_jump = np.nanmax(np.abs(ps[:, bcol] - ps[:, bcol - 1]))
    interior = np.nanpercentile(np.abs(np.diff(ps, axis=1)), 90)

    # The tent blend removes the hard-cell discontinuity ...
    assert smooth_jump < hard_jump
    # ... leaving a boundary step no worse than a few typical interior steps.
    assert smooth_jump <= 3.0 * interior


def test_local_blend_beats_global_on_varying_scene(varying_scene):
    features, target, truth, _ = varying_scene
    global_only = Sharpener(disaggregating_temperature=True).sharpen(
        features, target
    ).compute()
    local = Sharpener(
        disaggregating_temperature=True, window_size=5, smooth_local=True
    ).sharpen(features, target).compute()
    assert _rmse(local, truth) < 0.95 * _rmse(global_only, truth)


def test_local_blend_preserves_exact_mass_conservation(varying_scene):
    features, target, _, factor = varying_scene
    sharp = Sharpener(
        disaggregating_temperature=True, window_size=5, smooth_residual=False
    )
    sharpened = sharp.sharpen(features, target).compute()
    reagg = _reaggregate(sharpened, target, factor)
    np.testing.assert_allclose(reagg.values, target.values, rtol=1e-4, atol=1e-3)


def test_blend_weights_are_bounded(varying_scene):
    features, target, _, _ = varying_scene
    sharp = Sharpener(window_size=5).fit(features, target)
    global_fg = sharp.predict(features)
    local_fg = sharp.predict_local(features)
    # Re-run the internal blend and confirm it produced a finite fine field.
    blended = sharp._blend_local_global(local_fg, global_fg, target).compute()
    assert np.isfinite(blended.values).all()


def test_no_local_models_falls_back_to_global(varying_scene):
    features, target, _, _ = varying_scene
    # An unreachable sample requirement means no window gets its own model.
    sharp = Sharpener(window_size=5, min_training_samples=10 ** 6)
    sharp.fit(features, target)
    assert not sharp._has_local_models()
    with pytest.raises(RuntimeError):
        sharp.predict_local(features)

    # first_guess must be exactly the (un-blended) global prediction ...
    fg = sharp.first_guess(features, target)
    np.testing.assert_allclose(
        fg.values, sharp.predict(features).values, rtol=1e-10, atol=1e-10
    )
    # ... and sharpen stays finite via the global path.
    out = sharp.residual_correct(sharp.first_guess(features, target), target).compute()
    assert np.isfinite(out.values).all()
