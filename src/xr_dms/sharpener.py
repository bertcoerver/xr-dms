"""The Data Mining Sharpener orchestrator.

:class:`Sharpener` implements the five-step DMS pipeline of Gao et al. (2012) on
native, lazy, dask-backed :class:`xarray.DataArray`/:class:`~xarray.Dataset`
inputs:

    1. Aggregate the fine features to the coarse grid (block mean) and measure
       within-pixel heterogeneity (block std).
    2. Select homogeneous coarse pixels as training samples.
    3. Fit a regression ``coarse_target ~ f(coarse_features)``.
    4. Apply that regression to the FINE features -> a fine first guess.
    5. Residual correction so re-aggregating the sharpened result reproduces the
       coarse observation (mass conservation).

Training (:meth:`fit`) and application (:meth:`predict` / :meth:`residual_correct`)
are kept separate -- as in the reference ``src/pyDMS`` -- so a single fitted
model can later be applied across neighbouring tiles that blend into each other.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .aggregation import (
    EPS,
    binomial_smooth,
    coarsen_mean_std,
    homogeneity_cv,
    infer_factor,
    upsample,
)
from .regressors import BaseRegressor, SklearnDMSRegressor

__all__ = ["Sharpener", "to_radiance", "from_radiance"]


def to_radiance(temperature):
    """Convert temperature to radiance space (``T**4``).

    Thermal sensors integrate energy (~``T**4``), so averaging must happen in
    radiance space, not temperature space.
    """
    return temperature ** 4


def from_radiance(radiance, floor=1.0):
    """Convert radiance back to temperature (``radiance ** 0.25``).

    ``radiance`` is floored at ``floor`` before the fourth root to avoid NaNs
    from tiny negative values introduced by the residual correction.
    """
    if isinstance(radiance, xr.DataArray):
        radiance = radiance.clip(min=floor)
    else:
        radiance = np.maximum(radiance, floor)
    return radiance ** 0.25


class Sharpener:
    """Data Mining Sharpener for lazy, chunked xarray data.

    Parameters
    ----------
    regressor : BaseRegressor, optional
        Regression backend. Defaults to a fresh :class:`SklearnDMSRegressor`
        (bagged per-leaf-linear regression trees). Pass any
        :class:`~xr_dms.regressors.BaseRegressor` to swap the algorithm.
    cv_percentile : float, default 80
        Coarse pixels whose heterogeneity (CV) is at or below this percentile are
        used as training samples (Gao 2012 default).
    disaggregating_temperature : bool, default False
        If ``True`` the residual correction aggregates/differences in radiance
        space (``T**4``), appropriate for land-surface temperature.
    smooth_residual : bool, default True
        If ``True`` the coarse residual is binomially smoothed and bilinearly
        upsampled before being added back (as in pyDMS / the reference example):
        the correction field is gentle but only *approximately* mass-conserving.
        If ``False`` the residual is upsampled block-constant (nearest), which is
        *exactly* mass-conserving -- re-aggregating the sharpened result
        reproduces ``target`` -- at the cost of faint block edges in the
        correction.
    boundary : {"exact", "trim"}, default "exact"
        Coarsening boundary policy (see :func:`~xr_dms.aggregation.infer_factor`).
    x_dim, y_dim : str, default "x", "y"
        Names of the spatial dimensions.
    band_dim : str, default "band"
        Name of the feature dimension used internally.
    """

    def __init__(self, regressor=None, cv_percentile=80,
                 disaggregating_temperature=False, smooth_residual=True,
                 boundary="exact", x_dim="x", y_dim="y", band_dim="band"):
        self.regressor = regressor if regressor is not None else SklearnDMSRegressor()
        if not isinstance(self.regressor, BaseRegressor):
            raise TypeError("regressor must be a BaseRegressor instance.")
        self.cv_percentile = cv_percentile
        self.disaggregating_temperature = disaggregating_temperature
        self.smooth_residual = smooth_residual
        self.boundary = boundary
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.band_dim = band_dim

        # Populated by fit().
        self.factor_ = None
        self.band_order_ = None
        self.fitted_ = False

    # -- input normalisation -------------------------------------------------
    def _as_features(self, features):
        """Normalise features to a ``(band, y, x)`` DataArray."""
        if isinstance(features, xr.Dataset):
            da = features.to_array(dim=self.band_dim)
        elif isinstance(features, xr.DataArray):
            da = features
            if self.band_dim not in da.dims:
                da = da.expand_dims(self.band_dim)
        else:
            raise TypeError("features must be an xarray DataArray or Dataset.")
        return da

    def _match_coarse_coords(self, coarse, target):
        """Assign ``target``'s spatial coords onto a coarsened array.

        Block-averaged coordinates from :meth:`~xarray.DataArray.coarsen` are the
        block centres, which equal the coarse pixel coordinates only up to
        floating-point noise. Overwriting them positionally lets xarray align the
        coarse arrays with the target without spurious NaNs.
        """
        for dim in (self.y_dim, self.x_dim):
            if coarse.sizes[dim] != target.sizes[dim]:
                raise ValueError(
                    f"Coarsened size {coarse.sizes[dim]} along {dim!r} does not "
                    f"match target size {target.sizes[dim]}. Check the grids are "
                    "co-registered and the factor is correct."
                )
        assign = {}
        for dim in (self.y_dim, self.x_dim):
            if dim in target.coords:
                assign[dim] = target[dim].values
        return coarse.assign_coords(assign) if assign else coarse

    # -- STEP 1-3: training --------------------------------------------------
    def fit(self, features, target):
        """Train the regressor on homogeneous coarse pixels (steps 1-3).

        Parameters
        ----------
        features : xarray.DataArray or xarray.Dataset
            Fine-resolution predictors.
        target : xarray.DataArray
            Coarse image to sharpen.

        Returns
        -------
        Sharpener
            ``self``, fitted.
        """
        fine = self._as_features(features)
        self.band_order_ = [str(b) for b in fine[self.band_dim].values]
        self.factor_ = infer_factor(
            fine, target, self.x_dim, self.y_dim, self.boundary
        )

        mean, std = coarsen_mean_std(
            fine, self.factor_, self.x_dim, self.y_dim, self.boundary
        )
        mean = self._match_coarse_coords(mean, target)
        std = self._match_coarse_coords(std, target)
        cv = homogeneity_cv(mean, std, self.band_dim)

        # Bring the (small) coarse training arrays into memory.
        # X: (n_pixels, n_bands) with band as the last axis.
        X = mean.transpose(self.y_dim, self.x_dim, self.band_dim).values
        X = X.reshape(-1, X.shape[-1])
        y = np.asarray(target.transpose(self.y_dim, self.x_dim).values).reshape(-1)
        cv_flat = np.asarray(cv.transpose(self.y_dim, self.x_dim).values).reshape(-1)

        finite = np.isfinite(y) & np.isfinite(cv_flat) & np.all(np.isfinite(X), axis=1)
        if not finite.any():
            raise ValueError("No finite training pixels available.")

        cv_valid = cv_flat[finite]
        threshold = np.percentile(cv_valid, self.cv_percentile)
        homogeneous = finite.copy()
        homogeneous[finite] = cv_valid <= threshold
        if homogeneous.sum() == 0:
            raise ValueError("No homogeneous training pixels below CV threshold.")

        X_train = X[homogeneous]
        y_train = y[homogeneous]
        weights = 1.0 / (cv_flat[homogeneous] + EPS)

        self.regressor.fit(X_train, y_train, sample_weight=weights)
        self.cv_threshold_ = float(threshold)
        self.n_training_samples_ = int(homogeneous.sum())
        self.fitted_ = True
        return self

    # -- STEP 4: apply to fine features -> first guess -----------------------
    def predict(self, features):
        """Apply the fitted regressor to the fine features (step 4).

        Returns a lazy fine-resolution first guess. If ``features`` is
        dask-backed the result stays dask-backed (the regressor is applied per
        chunk via :func:`xarray.apply_ufunc`).
        """
        if not self.fitted_:
            raise RuntimeError("Sharpener.predict called before fit.")
        fine = self._as_features(features)
        # Reorder bands to the training order so feature columns line up.
        fine = fine.sel({self.band_dim: self.band_order_})

        regressor = self.regressor

        def _predict_block(arr):
            # arr has band as the last axis: (..., n_bands).
            spatial = arr.shape[:-1]
            n_bands = arr.shape[-1]
            flat = arr.reshape(-1, n_bands)
            out = np.full(flat.shape[0], np.nan, dtype=float)
            valid = np.all(np.isfinite(flat), axis=1)
            if valid.any():
                out[valid] = regressor.predict(flat[valid])
            return out.reshape(spatial)

        first_guess = xr.apply_ufunc(
            _predict_block,
            fine,
            input_core_dims=[[self.band_dim]],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[float],
        )
        return first_guess.rename("first_guess")

    # -- STEP 5: residual correction -----------------------------------------
    def residual_correct(self, first_guess, target):
        """Make the fine first guess consistent with the coarse observation.

        Aggregates ``first_guess`` back to the coarse grid, differences against
        ``target`` (in radiance space when ``disaggregating_temperature``),
        smooths and upsamples the residual, and adds it back to ``first_guess``.
        The corrected result, re-aggregated, reproduces ``target``.
        """
        if self.factor_ is None:
            raise RuntimeError("residual_correct called before fit.")

        guess = to_radiance(first_guess) if self.disaggregating_temperature else first_guess
        obs = to_radiance(target) if self.disaggregating_temperature else target

        guess_coarse = guess.coarsen(
            {self.y_dim: self.factor_, self.x_dim: self.factor_},
            boundary=self.boundary,
        ).mean()
        guess_coarse = self._match_coarse_coords(guess_coarse, obs)

        residual_coarse = (obs - guess_coarse).compute()
        if self.smooth_residual:
            # Gentle correction: smooth then bilinearly upsample (approximate).
            residual_coarse = binomial_smooth(residual_coarse, self.x_dim, self.y_dim)
            method = "linear"
        else:
            # Block-constant correction: exactly mass-conserving.
            method = "nearest"
        residual_fine = upsample(
            residual_coarse, first_guess, self.x_dim, self.y_dim, method=method
        )

        corrected = guess + residual_fine
        if self.disaggregating_temperature:
            corrected = from_radiance(corrected)
        return corrected.rename("sharpened")

    # -- convenience ---------------------------------------------------------
    def sharpen(self, features, target):
        """Fit, predict and residual-correct in one call (steps 1-5).

        Returns the lazy sharpened fine-resolution :class:`xarray.DataArray`.
        """
        self.fit(features, target)
        first_guess = self.predict(features)
        return self.residual_correct(first_guess, target)
