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
    homogeneity_cv,
)
from .gridmap import RegularGridMap
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
    window_size : int, default 0
        Side length, in *coarse* pixels, of the moving-window local regression
        of Gao (2012) section 2.3. ``0`` (the default) trains only the single
        global model -- the classic pipeline. When ``> 0`` the coarse scene is
        tiled into windows; one local model is trained per window (on the
        homogeneous pixels of an enlarged, overlapping *sampling* extent) in
        addition to the global model, and the two are combined by
        inverse-residual weights (see :meth:`sharpen`). Local models capture
        feature->LST relationships that vary across a large heterogeneous scene.
    window_extension : float, default 0.25
        Fraction of ``window_size`` by which each window's *sampling* extent is
        grown on every side beyond its *prediction* cell, so neighbouring local
        models share training pixels and agree across window boundaries.
    smooth_local : bool, default True
        How per-window local models are combined spatially. ``True``: each fine
        pixel is a piecewise-linear ("tent") blend of the surrounding window
        models -- C0-continuous and seamless by construction. ``False``: hard,
        non-overlapping window cells (faithful pyDMS), with the window seams left
        for the local/global residual blend to soften.
    min_training_samples : int, default 10
        Minimum number of homogeneous pixels a window must contribute to get its
        own local model; windows below this fall back to the global model.
    grid_map : GridMap, optional
        How the fine and coarse grids relate. Defaults to a
        :class:`~xr_dms.RegularGridMap` inferred from the two objects' shapes --
        the classic co-registered, integer-factor case. Pass a
        :class:`~xr_dms.SwathGridMap` to sharpen a curvilinear swath onto a
        projected grid, where no integer factor exists.
    boundary : {"exact", "trim"}, default "exact"
        Coarsening boundary policy (see :func:`~xr_dms.aggregation.infer_factor`).
    x_dim, y_dim : str, default "x", "y"
        Names of the spatial dimensions.
    band_dim : str, default "band"
        Name of the feature dimension used internally.
    """

    def __init__(self, regressor=None, cv_percentile=80,
                 disaggregating_temperature=False, smooth_residual=True,
                 window_size=0, window_extension=0.25, smooth_local=True,
                 min_training_samples=10, grid_map=None,
                 boundary="exact", x_dim="x", y_dim="y", band_dim="band"):
        self.regressor = regressor if regressor is not None else SklearnDMSRegressor()
        if not isinstance(self.regressor, BaseRegressor):
            raise TypeError("regressor must be a BaseRegressor instance.")
        self.cv_percentile = cv_percentile
        self.disaggregating_temperature = disaggregating_temperature
        self.smooth_residual = smooth_residual
        self.window_size = int(window_size)
        self.window_extension = window_extension
        self.smooth_local = smooth_local
        self.min_training_samples = int(min_training_samples)
        self.grid_map = grid_map
        self.boundary = boundary
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.band_dim = band_dim

        # Populated by fit().
        self.grid_map_ = None
        self.factor_ = None
        self.band_order_ = None
        self.fitted_ = False
        self.global_model_ = None
        # Moving-window state (populated by fit when window_size > 0).
        self.local_models_ = None
        self.window_basis_ = None

    # -- input normalisation -------------------------------------------------
    def _as_features(self, features):
        """Normalise features to a ``(band, y, x)`` DataArray."""
        if isinstance(features, xr.Dataset):
            da = features.to_array(dim=self.band_dim)
        elif isinstance(features, xr.DataArray):
            da = features
            if self.band_dim not in da.dims:
                # A bare 2-D field is a single feature. Give the new dimension a
                # label as well, since predict() selects bands by name.
                name = da.name if da.name is not None else "feature"
                da = da.expand_dims(self.band_dim).assign_coords(
                    {self.band_dim: [str(name)]}
                )
        else:
            raise TypeError("features must be an xarray DataArray or Dataset.")

        extra = [
            d for d in da.dims
            if d not in (self.band_dim, self.y_dim, self.x_dim)
        ]
        if extra:
            raise ValueError(
                f"features has unexpected dimension(s) {extra}; the sharpener "
                f"handles a single {self.band_dim}/{self.y_dim}/{self.x_dim} scene. "
                "Select one step first, e.g. features.isel(time=0)."
            )
        da = da.transpose(self.band_dim, self.y_dim, self.x_dim)
        if da.chunks is not None:
            # The regressor consumes all features of a pixel at once, so band is a
            # core dimension and must live in a single chunk. Converting a chunked
            # Dataset gives one band chunk per variable, so this is not optional.
            da = da.chunk({self.band_dim: -1})
        return da

    # -- STEP 1-3: training --------------------------------------------------
    def _fit_region(self, mean_arr, y_arr, cv_arr, rows, cols, local, min_samples):
        """Fit one model on a rectangular sub-region of the coarse grid.

        ``mean_arr`` is ``(ny, nx, n_bands)``; ``y_arr`` and ``cv_arr`` are
        ``(ny, nx)``. Homogeneous pixels (CV at or below the ``cv_percentile``
        percentile *within this region*) are used as training samples, weighted
        by ``1 / (cv + eps)``. Returns ``(model, cv_threshold, n_samples)`` or
        ``None`` if fewer than ``min_samples`` homogeneous pixels are available.
        """
        X = mean_arr[rows, cols, :].reshape(-1, mean_arr.shape[-1])
        y = y_arr[rows, cols].reshape(-1)
        cv = cv_arr[rows, cols].reshape(-1)

        finite = np.isfinite(y) & np.isfinite(cv) & np.all(np.isfinite(X), axis=1)
        if finite.sum() < min_samples:
            return None

        cv_valid = cv[finite]
        threshold = np.percentile(cv_valid, self.cv_percentile)
        homogeneous = finite.copy()
        homogeneous[finite] = cv_valid <= threshold
        if homogeneous.sum() < min_samples:
            return None

        weights = 1.0 / (cv[homogeneous] + EPS)
        model = self.regressor.clone(local=local)
        model.fit(X[homogeneous], y[homogeneous], sample_weight=weights)
        return model, float(threshold), int(homogeneous.sum())

    def fit(self, features, target):
        """Train the regressor on homogeneous coarse pixels (steps 1-3).

        Always trains the global model. When ``window_size > 0`` it additionally
        trains one local model per moving window (Gao 2012 section 2.3).

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

        self.grid_map_ = self.grid_map if self.grid_map is not None else (
            RegularGridMap.from_grids(
                fine, target, self.x_dim, self.y_dim, self.boundary
            )
        )
        self.factor_ = getattr(self.grid_map_, "factor", None)
        cy, cx = self.grid_map_.coarse_dims
        target = self.grid_map_.prepare_target(target)

        mean, std = self.grid_map_.aggregate(fine, self.band_dim)
        cv = homogeneity_cv(mean, std, self.band_dim)

        # Bring the (small) coarse training arrays into memory.
        # mean_arr: (ny, nx, n_bands) with band as the last axis.
        mean_arr = mean.transpose(cy, cx, self.band_dim).values
        y_arr = np.asarray(target.transpose(cy, cx).values)
        cv_arr = np.asarray(cv.transpose(cy, cx).values)
        full = (slice(None), slice(None))

        # -- global model (whole scene) --
        result = self._fit_region(mean_arr, y_arr, cv_arr, *full, local=False,
                                   min_samples=1)
        if result is None:
            raise ValueError(
                "No homogeneous training pixels available for the global model."
            )
        self.global_model_, self.cv_threshold_, self.n_training_samples_ = result

        # -- local (moving-window) models --
        self.local_models_ = None
        if self.window_size > 0:
            self._train_local_models(mean_arr, y_arr, cv_arr, fine)

        self.fitted_ = True
        return self

    def _train_local_models(self, mean_arr, y_arr, cv_arr, fine):
        """Tile the coarse grid into windows and fit one local model each."""
        ny, nx = y_arr.shape
        w = self.window_size
        ext = int(round(self.window_extension * w))
        n_wy = int(np.ceil(ny / w))
        n_wx = int(np.ceil(nx / w))

        # The spatial blend basis is the grid map's business: a separable tent on a
        # regular grid, something scattered on a swath.
        self.window_basis_ = self.grid_map_.build_window_basis(
            n_wy, n_wx, w, self.smooth_local, fine
        )

        models = [[None] * n_wx for _ in range(n_wy)]
        for iy in range(n_wy):
            rows = slice(max(iy * w - ext, 0), min((iy + 1) * w + ext, ny))
            for ix in range(n_wx):
                cols = slice(max(ix * w - ext, 0), min((ix + 1) * w + ext, nx))
                result = self._fit_region(
                    mean_arr, y_arr, cv_arr, rows, cols,
                    local=True, min_samples=self.min_training_samples,
                )
                if result is not None:
                    models[iy][ix] = result[0]
        self.local_models_ = models

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

        regressor = self.global_model_

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

    # -- STEP 4 (local): moving-window first guess ---------------------------
    def _has_local_models(self):
        return self.local_models_ is not None and any(
            m is not None for row in self.local_models_ for m in row
        )

    def predict_local(self, features):
        """Apply the moving-window local models to the fine features.

        Returns a lazy fine-resolution first guess blended from the per-window
        local models. With ``smooth_local=True`` each pixel is a piecewise-linear
        blend of the surrounding window models (seamless); with
        ``smooth_local=False`` each pixel takes its containing window's model
        (hard cells). Pixels not covered by any fitted window model are ``NaN``
        (the local/global blend falls back to the global model there).

        The blend uses only the *global* window geometry captured at
        :meth:`fit`, so the result is independent of how ``features`` is chunked
        (each dask block is computed the same way) and stays dask-backed when the
        input is.
        """
        if not self.fitted_:
            raise RuntimeError("Sharpener.predict_local called before fit.")
        if not self._has_local_models():
            raise RuntimeError(
                "predict_local requires window_size > 0 and fitted local models."
            )
        fine = self._as_features(features).sel({self.band_dim: self.band_order_})

        flat_models = [m for row in self.local_models_ for m in row]
        basis = self.window_basis_
        y_dim, x_dim, band_dim = self.y_dim, self.x_dim, self.band_dim

        def _local_block(block):
            b = block.transpose(band_dim, y_dim, x_dim)
            arr = np.asarray(b.values, dtype=float)  # (n_bands, ny, nx)
            n_bands, nyb, nxb = arr.shape
            # (k, ny, nx) window indices and weights; k is 4 for a separable tent,
            # 3 for barycentric, 1 for hard assignment.
            widx, wgt = basis.weights(b, y_dim, x_dim)

            feat = arr.reshape(n_bands, -1).T  # (npix, n_bands)
            valid = np.all(np.isfinite(feat), axis=1)

            acc = np.zeros((nyb, nxb), dtype=float)
            wsum = np.zeros((nyb, nxb), dtype=float)
            pred_cache = {}
            for slot in range(widx.shape[0]):
                idx_2d = widx[slot]
                w_2d = wgt[slot]
                # Only windows that actually got a model contribute; the weights of
                # the rest are left out of wsum, so the blend renormalises over the
                # models that exist -- as the separable version did.
                for win in np.unique(idx_2d[(idx_2d >= 0) & (w_2d > 0)]):
                    model = flat_models[win]
                    if model is None:
                        continue
                    mask = (idx_2d == win) & (w_2d > 0)
                    if win not in pred_cache:
                        pred = np.full(feat.shape[0], np.nan, dtype=float)
                        if valid.any():
                            pred[valid] = model.predict(feat[valid])
                        pred_cache[win] = pred.reshape(nyb, nxb)
                    pred2d = pred_cache[win]
                    use = mask & np.isfinite(pred2d)
                    acc[use] += w_2d[use] * pred2d[use]
                    wsum[use] += w_2d[use]

            out = np.divide(
                acc, wsum, out=np.full_like(acc, np.nan), where=wsum > 0
            )
            # Derive the result from the block itself so every coordinate the
            # template carries (a CF grid-mapping variable, say) comes along.
            return b.isel({band_dim: 0}, drop=True).astype(float).copy(data=out)

        template = fine.isel({band_dim: 0}, drop=True).astype(float)
        local_guess = xr.map_blocks(_local_block, fine, template=template)
        return local_guess.rename("local_first_guess")

    def _blend_local_global(self, local_fg, global_fg, target):
        """Combine local and global first guesses by inverse-residual weights.

        Following pyDMS section 2.3: each first guess is aggregated back to the
        coarse grid, its absolute residual against ``target`` computed, and the
        two blended with weights ``(1/r)**2`` (so the guess that better matches
        the coarse observation locally dominates). Where the local model has no
        prediction (``NaN``) the global guess is used.
        """
        radiance = self.disaggregating_temperature
        target = self.grid_map_.prepare_target(target)

        def to_coarse(fg):
            r = to_radiance(fg) if radiance else fg
            return self.grid_map_.aggregate_mean(r)

        obs = to_radiance(target) if radiance else target
        res_local = np.abs(obs - to_coarse(local_fg))
        res_global = np.abs(obs - to_coarse(global_fg))

        wl = 1.0 / (res_local + EPS) ** 2
        wg = 1.0 / (res_global + EPS) ** 2
        ww = wl / (wl + wg)
        # Where the local guess is missing, res_local is NaN -> lean fully global.
        ww = ww.where(np.isfinite(ww), 0.0).clip(0.0, 1.0).compute()

        ww_fine = self.grid_map_.upsample(ww, local_fg, method="linear")
        ww_fine = ww_fine.clip(0.0, 1.0)
        fw_fine = 1.0 - ww_fine

        local_filled = local_fg.where(np.isfinite(local_fg), global_fg)
        if radiance:
            blended = from_radiance(
                to_radiance(local_filled) * ww_fine + to_radiance(global_fg) * fw_fine
            )
        else:
            blended = local_filled * ww_fine + global_fg * fw_fine
        return blended.rename("first_guess")

    # -- STEP 5: residual correction -----------------------------------------
    def residual_correct(self, first_guess, target):
        """Make the fine first guess consistent with the coarse observation.

        Aggregates ``first_guess`` back to the coarse grid, differences against
        ``target`` (in radiance space when ``disaggregating_temperature``),
        smooths and upsamples the residual, and adds it back to ``first_guess``.
        The corrected result, re-aggregated, reproduces ``target``.
        """
        if self.grid_map_ is None:
            raise RuntimeError("residual_correct called before fit.")

        target = self.grid_map_.prepare_target(target)
        cy, cx = self.grid_map_.coarse_dims

        guess = to_radiance(first_guess) if self.disaggregating_temperature else first_guess
        obs = to_radiance(target) if self.disaggregating_temperature else target

        guess_coarse = self.grid_map_.aggregate_mean(guess)

        residual_coarse = (obs - guess_coarse).compute()
        if self.smooth_residual:
            # Gentle correction: smooth then bilinearly upsample (approximate).
            # A swath is still a 2-D scan array, so a 3x3 binomial in scan-index
            # space remains a genuine spatial neighbourhood.
            residual_coarse = binomial_smooth(residual_coarse, cx, cy)
            method = "linear"
        else:
            # Block-constant correction: exactly mass-conserving.
            method = "nearest"
        residual_fine = self.grid_map_.upsample(
            residual_coarse, first_guess, method=method
        )

        corrected = guess + residual_fine
        if self.disaggregating_temperature:
            corrected = from_radiance(corrected)
        return corrected.rename("sharpened")

    # -- convenience ---------------------------------------------------------
    def first_guess(self, features, target):
        """Model-based fine first guess (steps 1-4), before residual correction.

        The global model first guess, or -- when local models were fitted --  the
        inverse-residual blend of the local and global first guesses. Requires a
        prior :meth:`fit`.
        """
        global_fg = self.predict(features)
        if self.window_size > 0 and self._has_local_models():
            local_fg = self.predict_local(features)
            return self._blend_local_global(local_fg, global_fg, target)
        return global_fg

    def sharpen(self, features, target):
        """Fit, predict and residual-correct in one call (steps 1-5).

        With ``window_size > 0`` the local and global first guesses are blended
        (Gao 2012 section 2.3) before the residual correction. Returns the lazy
        sharpened fine-resolution :class:`xarray.DataArray`.
        """
        self.fit(features, target)
        return self.residual_correct(self.first_guess(features, target), target)
