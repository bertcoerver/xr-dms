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
    window_basis_1d,
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
                 min_training_samples=10,
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
        self.boundary = boundary
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.band_dim = band_dim

        # Populated by fit().
        self.factor_ = None
        self.band_order_ = None
        self.fitted_ = False
        self.global_model_ = None
        # Moving-window state (populated by fit when window_size > 0).
        self.local_models_ = None
        self.window_centers_x_ = None
        self.window_centers_y_ = None
        self.window_edges_x_ = None
        self.window_edges_y_ = None

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
        # mean_arr: (ny, nx, n_bands) with band as the last axis.
        mean_arr = mean.transpose(self.y_dim, self.x_dim, self.band_dim).values
        y_arr = np.asarray(target.transpose(self.y_dim, self.x_dim).values)
        cv_arr = np.asarray(cv.transpose(self.y_dim, self.x_dim).values)
        ny, nx = y_arr.shape
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
        f = self.factor_
        ext = int(round(self.window_extension * w))
        n_wy = int(np.ceil(ny / w))
        n_wx = int(np.ceil(nx / w))

        fx = np.asarray(fine[self.x_dim].values, dtype=float)
        fy = np.asarray(fine[self.y_dim].values, dtype=float)

        def axis_geometry(n_win, n_coarse, coord):
            """Per-window prediction-cell centres and edges in fine coords."""
            centers = np.empty(n_win, dtype=float)
            edges = np.empty((n_win, 2), dtype=float)
            for i in range(n_win):
                lo_c = i * w
                hi_c = min((i + 1) * w, n_coarse)
                lo_f = lo_c * f
                hi_f = hi_c * f  # exclusive fine index
                centers[i] = coord[lo_f:hi_f].mean()
                lo_edge = -np.inf if i == 0 else 0.5 * (coord[lo_f - 1] + coord[lo_f])
                hi_edge = (
                    np.inf if hi_c >= n_coarse
                    else 0.5 * (coord[hi_f - 1] + coord[hi_f])
                )
                edges[i] = (min(lo_edge, hi_edge), max(lo_edge, hi_edge))
            return centers, edges

        self.window_centers_y_, self.window_edges_y_ = axis_geometry(n_wy, ny, fy)
        self.window_centers_x_, self.window_edges_x_ = axis_geometry(n_wx, nx, fx)

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
        fine = fine.transpose(self.band_dim, self.y_dim, self.x_dim)
        if fine.chunks is not None:
            # A single band chunk keeps every feature together within each block.
            fine = fine.chunk({self.band_dim: -1})

        models = self.local_models_
        n_wy = len(models)
        n_wx = len(models[0])
        centers_y = self.window_centers_y_
        centers_x = self.window_centers_x_
        edges_y = self.window_edges_y_
        edges_x = self.window_edges_x_
        smooth = self.smooth_local
        y_dim, x_dim, band_dim = self.y_dim, self.x_dim, self.band_dim

        def _local_block(block):
            b = block.transpose(band_dim, y_dim, x_dim)
            yc = np.asarray(b[y_dim].values, dtype=float)
            xc = np.asarray(b[x_dim].values, dtype=float)
            arr = np.asarray(b.values, dtype=float)  # (n_bands, ny, nx)
            n_bands, nyb, nxb = arr.shape
            by = window_basis_1d(yc, centers_y, edges_y, smooth)  # (n_wy, nyb)
            bx = window_basis_1d(xc, centers_x, edges_x, smooth)  # (n_wx, nxb)

            feat = arr.reshape(n_bands, -1).T  # (npix, n_bands)
            valid = np.all(np.isfinite(feat), axis=1)

            acc = np.zeros((nyb, nxb), dtype=float)
            wsum = np.zeros((nyb, nxb), dtype=float)
            pred_cache = {}
            for iy in range(n_wy):
                wy = by[iy]
                if not wy.any():
                    continue
                for ix in range(n_wx):
                    model = models[iy][ix]
                    if model is None:
                        continue
                    wx = bx[ix]
                    if not wx.any():
                        continue
                    weight = np.outer(wy, wx)
                    mask = weight > 0
                    if not mask.any():
                        continue
                    if (iy, ix) not in pred_cache:
                        pred = np.full(feat.shape[0], np.nan, dtype=float)
                        if valid.any():
                            pred[valid] = model.predict(feat[valid])
                        pred_cache[(iy, ix)] = pred.reshape(nyb, nxb)
                    pred2d = pred_cache[(iy, ix)]
                    use = mask & np.isfinite(pred2d)
                    acc[use] += weight[use] * pred2d[use]
                    wsum[use] += weight[use]

            out = np.divide(
                acc, wsum, out=np.full_like(acc, np.nan), where=wsum > 0
            )
            return xr.DataArray(
                out, dims=(y_dim, x_dim),
                coords={y_dim: b[y_dim], x_dim: b[x_dim]},
            )

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

        def to_coarse(fg):
            r = to_radiance(fg) if radiance else fg
            c = r.coarsen(
                {self.y_dim: self.factor_, self.x_dim: self.factor_},
                boundary=self.boundary,
            ).mean()
            return self._match_coarse_coords(c, target)

        obs = to_radiance(target) if radiance else target
        res_local = np.abs(obs - to_coarse(local_fg))
        res_global = np.abs(obs - to_coarse(global_fg))

        wl = 1.0 / (res_local + EPS) ** 2
        wg = 1.0 / (res_global + EPS) ** 2
        ww = wl / (wl + wg)
        # Where the local guess is missing, res_local is NaN -> lean fully global.
        ww = ww.where(np.isfinite(ww), 0.0).clip(0.0, 1.0).compute()

        ww_fine = upsample(ww, local_fg, self.x_dim, self.y_dim, method="linear")
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
