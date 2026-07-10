"""Pluggable regression backends for the Data Mining Sharpener.

The sharpener trains a regressor on the *coarse* grid (coarse target as a
function of the coarse-aggregated fine features) and then applies it, unchanged,
to the *fine* features to obtain a fine-resolution first guess. The regressor
therefore only ever sees flat 2-D numpy arrays ``X`` of shape ``(n_samples,
n_features)`` and a 1-D target ``y`` -- it knows nothing about xarray, dask or
grids. That keeps the interface tiny and makes alternative backends trivial to
drop in.

The default backend (:class:`SklearnDMSRegressor`) ports the production design
from ``src/pyDMS/pyDMS.py``: a :class:`~sklearn.ensemble.BaggingRegressor` of
regression trees, each tree fitting a per-leaf Ridge regression
(:class:`DecisionTreeRegressorWithLinearLeafRegression`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn import ensemble, linear_model, tree

__all__ = [
    "BaseRegressor",
    "DecisionTreeRegressorWithLinearLeafRegression",
    "SklearnDMSRegressor",
]


class BaseRegressor(ABC):
    """Interface every sharpener regression backend must implement.

    A backend operates purely on flat numpy arrays so it stays decoupled from
    xarray/dask. Implement :meth:`fit` and :meth:`predict` to plug a new
    algorithm into :class:`~xr_dms.sharpener.Sharpener`.
    """

    @abstractmethod
    def fit(self, X, y, sample_weight=None) -> "BaseRegressor":
        """Train on ``X`` (``n_samples, n_features``) and ``y`` (``n_samples,``).

        ``sample_weight`` is an optional per-sample weight (the sharpener passes
        the homogeneity weights here). Return ``self``.
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict for ``X`` (``n_samples, n_features``) -> ``(n_samples,)``."""


class DecisionTreeRegressorWithLinearLeafRegression(tree.DecisionTreeRegressor):
    """Regression tree whose leaves predict via a local Ridge regression.

    Ported from ``src/pyDMS/pyDMS.py``. A standard
    :class:`~sklearn.tree.DecisionTreeRegressor` is fitted first; then, for every
    leaf, a :class:`~sklearn.linear_model.Ridge` is fitted on the training
    samples that fell into that leaf. At predict time each sample's constant leaf
    value is replaced by its leaf's linear prediction, clamped to the leaf's
    observed target range expanded by ``linear_regression_extrapolation_ratio``
    times the range.

    Subclasses :class:`~sklearn.tree.DecisionTreeRegressor` so it remains a valid
    scikit-learn estimator and can be cloned by
    :class:`~sklearn.ensemble.BaggingRegressor`.

    Parameters
    ----------
    linear_regression_extrapolation_ratio : float, default 0.25
        Fraction of each leaf's target range by which the per-leaf linear
        prediction may extrapolate beyond the observed min/max.
    decision_tree_regressor_opt : dict, optional
        Keyword options forwarded to :class:`~sklearn.tree.DecisionTreeRegressor`.
    """

    def __init__(self, linear_regression_extrapolation_ratio=0.25,
                 decision_tree_regressor_opt=None):
        opt = decision_tree_regressor_opt or {}
        super().__init__(**opt)
        self.decision_tree_regressor_opt = opt
        self.linear_regression_extrapolation_ratio = (
            linear_regression_extrapolation_ratio
        )
        self.leaf_parameters = {}

    def fit(self, X, y, sample_weight=None, **fit_opt):
        # Fit a normal regression tree first.
        super().fit(X, y, sample_weight=sample_weight, **fit_opt)

        # Fit one Ridge regression per leaf, keyed by the tree's constant leaf
        # value (unique across leaves for a fitted regression tree).
        predicted = super().predict(X)
        self.leaf_parameters = {}
        for value in np.unique(predicted):
            ind = predicted == value
            leaf_regression = linear_model.Ridge()
            leaf_regression.fit(X[ind, :], y[ind])
            self.leaf_parameters[value] = {
                "linear_regression": leaf_regression,
                "max": float(np.max(y[ind])),
                "min": float(np.min(y[ind])),
            }
        return self

    def predict(self, X, **predict_opt):
        y = super().predict(X, **predict_opt)
        for leaf_value, params in self.leaf_parameters.items():
            ind = y == leaf_value
            if X[ind, :].size == 0:
                continue
            pred = params["linear_regression"].predict(X[ind, :])
            span = self.linear_regression_extrapolation_ratio * (
                params["max"] - params["min"]
            )
            pred = np.maximum(pred, params["min"] - span)
            pred = np.minimum(pred, params["max"] + span)
            y[ind] = pred
        return y


class SklearnDMSRegressor(BaseRegressor):
    """Default backend: a bagged ensemble of per-leaf-linear regression trees.

    Mirrors ``DecisionTreeSharpener._doFit`` in ``src/pyDMS/pyDMS.py``. The
    ensemble averages several trees, each trained on a bootstrap resample, which
    both smooths the output and captures the non-linear feature relationships a
    single linear fit cannot.

    Parameters
    ----------
    local : bool, default False
        Use the tighter ``max_leaf_nodes`` (10) suited to small local training
        sets; ``False`` uses the global value (30). Reserved for the future
        moving-window feature -- the core pipeline fits a single global model.
    per_leaf_linear_regression : bool, default True
        Use :class:`DecisionTreeRegressorWithLinearLeafRegression` as the base
        estimator; if ``False`` a plain
        :class:`~sklearn.tree.DecisionTreeRegressor` is used.
    linear_regression_extrapolation_ratio : float, default 0.25
        Passed to the per-leaf regressor.
    min_samples_number : int, default 10
        Lower bound driving ``min_samples_leaf`` (``min(min_samples_number, 10)``).
    regressor_opt : dict, optional
        Extra options for the base tree (merged with the local/global
        ``max_leaf_nodes`` and ``min_samples_leaf``).
    bagging_opt : dict, optional
        Options for :class:`~sklearn.ensemble.BaggingRegressor` (e.g.
        ``n_estimators``).
    """

    def __init__(self, local=False, per_leaf_linear_regression=True,
                 linear_regression_extrapolation_ratio=0.25,
                 min_samples_number=10, regressor_opt=None, bagging_opt=None):
        self.local = local
        self.per_leaf_linear_regression = per_leaf_linear_regression
        self.linear_regression_extrapolation_ratio = (
            linear_regression_extrapolation_ratio
        )
        self.min_samples_number = min_samples_number
        self.regressor_opt = dict(regressor_opt or {})
        self.bagging_opt = dict(bagging_opt or {})
        self._model = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        opt = dict(self.regressor_opt)
        opt.setdefault("max_leaf_nodes", 10 if self.local else 30)
        opt.setdefault("min_samples_leaf", min(self.min_samples_number, 10))

        if self.per_leaf_linear_regression:
            base = DecisionTreeRegressorWithLinearLeafRegression(
                self.linear_regression_extrapolation_ratio, opt
            )
        else:
            base = tree.DecisionTreeRegressor(**opt)

        model = ensemble.BaggingRegressor(base, **self.bagging_opt)
        # A single training sample cannot be bootstrap-subsampled below 100%.
        if X.shape[0] <= 1:
            model.max_samples = 1.0
        self._model = model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("SklearnDMSRegressor.predict called before fit.")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float)))
