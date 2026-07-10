"""Tests for the native aggregation primitives."""

import numpy as np

from xr_dms.aggregation import coarsen_mean_std, homogeneity_cv, upsample


def _np_block(arr, factor, reduce):
    b, h, w = arr.shape
    r = arr.reshape(b, h // factor, factor, w // factor, factor)
    return reduce(r, axis=(2, 4))


def test_coarsen_mean_std_matches_numpy(scene):
    features, _, _, factor = scene
    mean, std = coarsen_mean_std(features, factor)

    arr = features.transpose("band", "y", "x").values
    expected_mean = _np_block(arr, factor, np.mean)
    expected_std = _np_block(arr, factor, np.std)

    np.testing.assert_allclose(
        mean.transpose("band", "y", "x").values, expected_mean, rtol=1e-6
    )
    np.testing.assert_allclose(
        std.transpose("band", "y", "x").values, expected_std, rtol=1e-6
    )


def test_homogeneity_cv_shape_and_positive(scene):
    features, target, _, factor = scene
    mean, std = coarsen_mean_std(features, factor)
    cv = homogeneity_cv(mean, std)
    assert cv.dims == ("y", "x")
    assert cv.shape == target.shape
    assert float(cv.min()) >= 0.0


def test_upsample_shape_and_no_nans(scene):
    features, target, truth, factor = scene
    up = upsample(target, truth)
    assert up.shape == truth.shape
    assert np.isfinite(up.values).all()
    # Upsampled values stay within the coarse data range (+ tiny tolerance).
    assert up.min() >= target.min() - 1e-6
    assert up.max() <= target.max() + 1e-6
