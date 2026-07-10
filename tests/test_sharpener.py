"""End-to-end tests for the Sharpener pipeline."""

import numpy as np

from xr_dms import Sharpener


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_predict_is_lazy_when_input_chunked(scene):
    features, target, _, _ = scene
    chunked = features.chunk({"y": 20, "x": 20})
    sharp = Sharpener().fit(features, target)
    first_guess = sharp.predict(chunked)
    # Still dask-backed before compute.
    assert first_guess.chunks is not None
    assert first_guess.compute().shape == features.isel(band=0).shape


def test_chunking_does_not_change_result(scene):
    features, target, _, _ = scene
    sharp = Sharpener().fit(features, target)
    eager = sharp.predict(features).compute()
    lazy = sharp.predict(features.chunk({"y": 15, "x": 15})).compute()
    np.testing.assert_allclose(eager.values, lazy.values, rtol=1e-10, atol=1e-10)


def _reaggregate(sharpened, target, factor):
    """Re-aggregate the fine sharpened result to the coarse grid (radiance)."""
    reagg = (sharpened ** 4).coarsen(y=factor, x=factor).mean() ** 0.25
    return reagg.assign_coords(y=target.y, x=target.x)


def test_exact_residual_correction_reproduces_coarse(scene):
    """smooth_residual=False is exactly mass-conserving."""
    features, target, _, factor = scene
    sharp = Sharpener(disaggregating_temperature=True, smooth_residual=False)
    sharpened = sharp.sharpen(features, target).compute()
    reagg = _reaggregate(sharpened, target, factor)
    np.testing.assert_allclose(reagg.values, target.values, rtol=1e-4, atol=1e-3)


def test_smoothed_residual_correction_reduces_coarse_error(scene):
    """smooth_residual=True substantially reduces the coarse discrepancy."""
    features, target, _, factor = scene
    sharp = Sharpener(disaggregating_temperature=True, smooth_residual=True).fit(
        features, target
    )
    first_guess = sharp.predict(features)
    sharpened = sharp.residual_correct(first_guess, target).compute()

    fg_err = _rmse(_reaggregate(first_guess.compute(), target, factor), target)
    sh_err = _rmse(_reaggregate(sharpened, target, factor), target)
    assert sh_err < 0.75 * fg_err
    bias = float((_reaggregate(sharpened, target, factor) - target).mean())
    assert abs(bias) < 0.5


def test_sharpened_beats_naive_upsample(scene):
    from xr_dms.aggregation import upsample

    features, target, truth, _ = scene
    sharp = Sharpener(disaggregating_temperature=True)
    sharpened = sharp.sharpen(features, target).compute()

    naive = upsample(target, truth)
    assert _rmse(sharpened, truth) < _rmse(naive, truth)


def test_accepts_dataset_input(scene):
    features, target, _, _ = scene
    ds = features.to_dataset(dim="band")
    sharp = Sharpener()
    out = sharp.sharpen(ds, target).compute()
    assert out.shape == target.shape[:0] + features.isel(band=0).shape
    assert np.isfinite(out.values).all()
