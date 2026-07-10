"""Tests for the pluggable regression backends."""

import numpy as np

from xr_dms.regressors import (
    BaseRegressor,
    DecisionTreeRegressorWithLinearLeafRegression,
    SklearnDMSRegressor,
)
from xr_dms.sharpener import Sharpener


def test_sklearn_regressor_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.standard_normal(200)
    reg = SklearnDMSRegressor(bagging_opt={"n_estimators": 5, "random_state": 0})
    reg.fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (200,)
    # A bagged tree ensemble should track a mostly-linear signal well.
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    assert 1 - ss_res / ss_tot > 0.8


def test_per_leaf_linear_extrapolation_is_clamped():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (100, 1))
    y = (3.0 * X[:, 0]).astype(float)
    reg = DecisionTreeRegressorWithLinearLeafRegression(
        linear_regression_extrapolation_ratio=0.25,
        decision_tree_regressor_opt={"max_leaf_nodes": 4},
    )
    reg.fit(X, y)
    # Predict far outside the training range; output must stay bounded by the
    # per-leaf extrapolation clamp rather than shooting off linearly.
    far = np.array([[100.0], [-100.0]])
    pred = reg.predict(far)
    assert np.all(np.isfinite(pred))
    assert pred.max() <= y.max() + 0.25 * (y.max() - y.min()) + 1e-6
    assert pred.min() >= y.min() - 0.25 * (y.max() - y.min()) - 1e-6


def test_custom_backend_plugs_in(scene):
    """A trivial BaseRegressor subclass can drive the whole Sharpener."""

    class MeanRegressor(BaseRegressor):
        def fit(self, X, y, sample_weight=None):
            self.value_ = float(np.average(y, weights=sample_weight))
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.value_)

    features, target, _, _ = scene
    sharp = Sharpener(regressor=MeanRegressor())
    out = sharp.sharpen(features, target).compute()
    assert out.shape == features.isel(band=0).shape
    assert np.isfinite(out.values).all()
