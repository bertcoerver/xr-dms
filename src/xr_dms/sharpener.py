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

import copy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import xarray as xr

from .aggregation import (
    EPS,
    binomial_smooth,
    homogeneity_cv,
)
from .geo import one_block
from .gridmap import RegularGridMap
from .regressors import BaseRegressor, SklearnDMSRegressor

__all__ = ["Sharpener", "SceneState", "to_radiance", "from_radiance"]


@dataclass
class SceneState:
    """Everything scene-global that DMS needs, separated from the fine grid.

    DMS has a natural seam: the regression is fitted on *coarse* statistics and
    the residual is a *coarse* field, while applying the model and gathering the
    residual are per-pixel. This holds the coarse half -- a fitted model plus, at
    most, one coarse-grid array. For a VIIRS granule against Sentinel-2 that is
    about a megabyte, against a sharpened field of several hundred.

    Keeping it as a value rather than as mutable attributes on the sharpener is
    what makes deferral possible: a :class:`dask.delayed.Delayed` wrapping one of
    these can be threaded into the per-chunk graph, and a *computed* one can be
    cached to disk so a later run's fine half is pure blockwise -- which is the
    difference between a spatial subset costing one chunk and costing a scene.

    ``global_model=None`` records a scene that could not be fitted (no
    homogeneous training pixels). Applying such a state yields all-NaN rather
    than raising, because a lazy graph cannot change its own shape.
    """

    band_order: list[str]
    global_model: Any = None
    cv_threshold: float | None = None
    n_training_samples: int = 0
    #: Coarse residual, in whichever space the correction is applied. When set,
    #: :meth:`Sharpener.apply` needs no scene-global reduction at all.
    residual_coarse: Any = None

    @property
    def fitted(self):
        return self.global_model is not None


def _seed_from(*arrays):
    """A stable 32-bit seed from the training data itself.

    Content-derived rather than fixed, so two different scenes still draw
    different bootstraps -- it is repeatability that is wanted here, not a
    single shared seed across a whole time series.
    """
    from dask.base import tokenize

    return int(tokenize(*arrays)[:8], 16)


def _is_delayed(obj):
    """True for a ``dask.delayed.Delayed``, without importing dask eagerly."""
    return type(obj).__module__.startswith("dask.") and hasattr(obj, "dask")


def _boxed(state):
    """A :class:`SceneState` as a 0-d argument ``apply_ufunc`` can broadcast.

    A ``Delayed`` state has to reach the per-chunk function as a dask *dependency*
    rather than as a captured constant, or computing any block would drag the fit
    out of the graph. Wrapping it in a 0-d object array is the standard way to say
    that: ``dask="parallelized"`` broadcasts it against every block.

    Note that this makes the fit a shared *dependency*, not a single *execution*.
    The residual correction consumes the first guess on two branches, and dask
    will evaluate the fit on each -- which is why :meth:`BaseRegressor.seeded`
    exists.
    """
    if not _is_delayed(state):
        return xr.DataArray(np.array(state, dtype=object))

    import dask.array as dsa

    return xr.DataArray(dsa.from_delayed(state, shape=(), dtype=object))


def _unbox(state):
    """Recover the :class:`SceneState` inside a 0-d object array."""
    return state.item() if hasattr(state, "item") else state


def _band_order_of(state, fine, band_dim):
    """The training band order, from the state if it is known, else from ``fine``.

    A ``Delayed`` state cannot answer at graph-build time, and the band selection
    has to happen there -- so fall back to the features' own order, which is how
    :meth:`Sharpener.fit_delayed` derived it in the first place. The two are
    cross-checked inside :func:`_predict_block`, where the state is concrete.
    """
    if not _is_delayed(state) and state.band_order is not None:
        return list(state.band_order)
    return [str(b) for b in fine[band_dim].values]


def _predict_block(arr, state, band_order):
    """Apply the fitted regressor to one block of fine features.

    ``arr`` has band as the last axis: ``(..., n_bands)``. An unfitted state
    yields all-NaN -- the lazy counterpart of :meth:`Sharpener.fit` raising.
    """
    state = _unbox(state)
    spatial = arr.shape[:-1]
    if not state.fitted:
        return np.full(spatial, np.nan, dtype=float)
    if list(state.band_order) != list(band_order):
        raise ValueError(
            f"Features are ordered {list(band_order)} but the model was trained "
            f"on {list(state.band_order)}."
        )
    n_bands = arr.shape[-1]
    flat = arr.reshape(-1, n_bands)
    out = np.full(flat.shape[0], np.nan, dtype=float)
    valid = np.all(np.isfinite(flat), axis=1)
    if valid.any():
        out[valid] = state.global_model.predict(flat[valid])
    return out.reshape(spatial)


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
    def _as_features(self, features, extra_dims=()):
        """Normalise features to a ``(band, y, x)`` DataArray.

        ``extra_dims`` names dimensions to tolerate and leave in front of the
        band/y/x ones -- :func:`xr_dms.cube.sharpen_cube` passes the time axis,
        because it sharpens a whole series in one graph. Everything else in this
        class is per-scene and leaves it empty, so an accidental time dimension
        is still caught where it would otherwise become a confusing shape error
        deep in the regressor.
        """
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

        allowed = (self.band_dim, self.y_dim, self.x_dim, *extra_dims)
        extra = [d for d in da.dims if d not in allowed]
        if extra:
            raise ValueError(
                f"features has unexpected dimension(s) {extra}; the sharpener "
                f"handles a single {self.band_dim}/{self.y_dim}/{self.x_dim} scene. "
                "Select one step first, e.g. features.isel(time=0)."
            )
        kept = [d for d in extra_dims if d in da.dims]
        da = da.transpose(*kept, self.band_dim, self.y_dim, self.x_dim)
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

    def _setup_grid_map(self, fine, target):
        """Resolve and record the grid map (and its coarsening factor, if any)."""
        self.grid_map_ = self.grid_map if self.grid_map is not None else (
            RegularGridMap.from_grids(
                fine, target, self.x_dim, self.y_dim, self.boundary
            )
        )
        self.factor_ = getattr(self.grid_map_, "factor", None)
        return self.grid_map_

    def _training_arrays(self, fine, target):
        """The three coarse training fields, oriented but *not* materialised.

        ``(mean, y, cv)`` as ``(ny, nx, n_bands)``, ``(ny, nx)``, ``(ny, nx)``.
        Whether these are numpy- or dask-backed is decided by the grid map: an
        eagerly built one aggregates eagerly, a lazily built one leaves the whole
        reduction in the graph. Splitting this out is what lets :meth:`fit` and
        :meth:`fit_delayed` share one definition of "the training data".
        """
        cy, cx = self.grid_map_.coarse_dims
        mean, std = self.grid_map_.aggregate(fine, self.band_dim)
        cv = homogeneity_cv(mean, std, self.band_dim)
        return (
            mean.transpose(cy, cx, self.band_dim),
            target.transpose(cy, cx),
            cv.transpose(cy, cx),
        )

    def fit_delayed(self, features, target):
        """:meth:`fit` deferred: a ``Delayed`` :class:`SceneState`, nothing computed.

        The counterpart to :meth:`fit` for building a graph over many scenes at
        once. Nothing here reads a pixel -- the aggregation, the homogeneity
        screen and the regression all land in the graph -- so assembling a whole
        time series costs metadata only, and computing one scene computes only
        that scene.

        Where :meth:`fit` raises on a scene with no homogeneous training pixels,
        this returns a state with ``global_model=None``: by the time the
        condition is known the graph's shape is already fixed, so the failure has
        to be a value (an all-NaN slab from :meth:`apply`) rather than an
        exception. Check :attr:`SceneState.fitted` on the computed result.

        Requires ``window_size == 0``; moving-window models need the projected
        swath centres, which a lazily built grid map does not materialise.
        """
        import dask

        if self.window_size > 0:
            raise NotImplementedError(
                "fit_delayed does not support moving-window models "
                "(window_size > 0); use fit()."
            )

        fine = self._as_features(features)
        band_order = [str(b) for b in fine[self.band_dim].values]
        self._setup_grid_map(fine, target)
        target = self.grid_map_.prepare_target(target)

        mean, y, cv = self._training_arrays(fine, target)
        full = (slice(None), slice(None))

        def _fit(mean_arr, y_arr, cv_arr):
            mean_arr = np.asarray(mean_arr)
            y_arr = np.asarray(y_arr)
            cv_arr = np.asarray(cv_arr)
            # Dask may evaluate this task more than once (the first guess feeds
            # both the residual aggregation and the final sum), so it has to be a
            # pure function of the training data or the two evaluations disagree
            # and the residual correction stops conserving mass. See
            # BaseRegressor.seeded.
            pure = copy.copy(self)
            pure.regressor = self.regressor.seeded(
                _seed_from(mean_arr, y_arr, cv_arr)
            )
            result = pure._fit_region(
                mean_arr, y_arr, cv_arr, *full, local=False, min_samples=1,
            )
            if result is None:
                return SceneState(band_order=band_order)
            model, threshold, n = result
            return SceneState(band_order, model, threshold, n)

        return dask.delayed(_fit, pure=True)(
            one_block(mean.data), one_block(y.data), one_block(cv.data),
        )

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
        self._setup_grid_map(fine, target)
        target = self.grid_map_.prepare_target(target)

        mean, y, cv = self._training_arrays(fine, target)

        # Bring the (small) coarse training arrays into memory.
        # mean_arr: (ny, nx, n_bands) with band as the last axis.
        mean_arr = np.asarray(mean.values)
        y_arr = np.asarray(y.values)
        cv_arr = np.asarray(cv.values)
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
    def _resolve_state(self, state):
        """``state`` as given, or the one :meth:`fit` left on ``self``."""
        if state is not None:
            return state
        if not self.fitted_:
            raise RuntimeError("Sharpener.predict called before fit.")
        return SceneState(
            band_order=self.band_order_,
            global_model=self.global_model_,
            cv_threshold=getattr(self, "cv_threshold_", None),
            n_training_samples=getattr(self, "n_training_samples_", 0),
        )

    def predict(self, features, state=None):
        """Apply the fitted regressor to the fine features (step 4).

        Returns a lazy fine-resolution first guess. If ``features`` is
        dask-backed the result stays dask-backed (the regressor is applied per
        chunk via :func:`xarray.apply_ufunc`).

        ``state`` accepts a :class:`SceneState` -- or a ``Delayed`` one from
        :meth:`fit_delayed` -- instead of the model left on ``self`` by
        :meth:`fit`, which is what lets many scenes share one sharpener.
        """
        state = self._resolve_state(state)
        fine = self._as_features(features)
        band_order = _band_order_of(state, fine, self.band_dim)
        # Reorder bands to the training order so feature columns line up.
        fine = fine.sel({self.band_dim: band_order})

        first_guess = xr.apply_ufunc(
            _predict_block,
            fine,
            _boxed(state),
            kwargs={"band_order": band_order},
            input_core_dims=[[self.band_dim], []],
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
    def residual_correct(self, first_guess, target, residual_coarse=None):
        """Make the fine first guess consistent with the coarse observation.

        Aggregates ``first_guess`` back to the coarse grid, differences against
        ``target`` (in radiance space when ``disaggregating_temperature``),
        smooths and upsamples the residual, and adds it back to ``first_guess``.
        The corrected result, re-aggregated, reproduces ``target``.

        Pass ``residual_coarse`` to skip the aggregation entirely -- the residual
        is the only scene-global reduction left in the fine half, so supplying a
        precomputed one (see :attr:`SceneState.residual_coarse`) is what makes a
        spatial subset cost a chunk instead of a whole scene.
        """
        if self.grid_map_ is None:
            raise RuntimeError("residual_correct called before fit.")

        target = self.grid_map_.prepare_target(target)
        cy, cx = self.grid_map_.coarse_dims

        guess = to_radiance(first_guess) if self.disaggregating_temperature else first_guess

        if residual_coarse is None:
            obs = to_radiance(target) if self.disaggregating_temperature else target
            guess_coarse = self.grid_map_.aggregate_mean(guess)
            residual_coarse = obs - guess_coarse
            # An eagerly built grid map has already read the fine grid to get
            # here, so materialising costs nothing and keeps the graph small. A
            # lazy one must stay in the graph -- that is the whole point.
            if not self.grid_map_.lazy:
                residual_coarse = residual_coarse.compute()
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

    def apply(self, state, features, target):
        """Steps 4-5 against a given :class:`SceneState`: the fine half of DMS.

        The counterpart to :meth:`fit_delayed`. ``state`` may be a concrete
        :class:`SceneState` or a ``Delayed`` one; either way nothing is computed
        here.

        How well a subset of the result culls depends on what ``state`` carries:

        * with ``state.residual_coarse`` set, this is pure blockwise -- predict,
          gather, add -- so ``result.isel(y=..., x=...)`` costs only the chunks
          asked for;
        * without it, the residual is derived in-graph, which is a scene-global
          reduction over the first guess, so any output chunk pulls the whole
          scene through :meth:`predict`. That is DMS, not the plumbing: a global
          regression needs global statistics.

        Use :meth:`scene_state` to build the first kind.
        """
        if self.grid_map_ is None:
            self._setup_grid_map(self._as_features(features), target)
        guess = self.predict(features, state=state)
        residual = None if _is_delayed(state) else state.residual_coarse
        return self.residual_correct(guess, target, residual_coarse=residual)

    def scene_state(self, features, target):
        """A ``Delayed`` :class:`SceneState` carrying the coarse residual too.

        :meth:`fit_delayed` plus the residual, so the whole scene-global half of
        DMS is one deferred value. Computing and caching these turns :meth:`apply`
        into a pure blockwise graph -- which is the only way a spatial subset of
        the sharpened field costs less than a full scene.
        """
        import dask

        state = self.fit_delayed(features, target)
        guess = self.predict(features, state=state)

        prepared = self.grid_map_.prepare_target(target)
        guess_r = to_radiance(guess) if self.disaggregating_temperature else guess
        obs = to_radiance(prepared) if self.disaggregating_temperature else prepared
        residual = obs - self.grid_map_.aggregate_mean(guess_r)

        @dask.delayed(pure=True)
        def _attach(st, resid):
            return replace(st, residual_coarse=resid)

        return _attach(state, residual)

    def _smoothed_residual_fine(self, residual_coarse, labels, like):
        """The coarse residual smoothed, interpolated and put on the fine grid.

        The :meth:`scene_bundle` counterpart of the ``smooth_residual=True``
        branch of :meth:`residual_correct`, materialised: pyDMS's binomial
        smoother followed by a bilinear (here: scattered-linear) upsample, which
        is what keeps the correction from carrying the coarse cell geometry into
        the sharpened field.

        The observation footprint is *preserved*. Interpolating over the swath
        centres fills the gaps the coarse grid has -- a cloudy cell, a cell the
        fine grid barely covers -- and a filled cloud gap is an invented surface
        temperature, not a sharpened one. So the interpolated field is masked
        back to the cells a block-constant gather would have given a value,
        leaving the result with exactly the coverage the block-constant path had
        and differing from it only in the values.
        """
        cy, cx = self.grid_map_.coarse_dims
        smoothed = binomial_smooth(residual_coarse, cx, cy)
        fine = self.grid_map_.upsample(smoothed, like, method="linear")
        fine = np.asarray(fine.values, dtype=float)

        flat = np.asarray(residual_coarse.values, dtype=float).ravel()
        observed = np.isfinite(np.append(flat, np.nan)[labels])
        return np.where(observed, fine, np.nan)

    def scene_bundle(self, features, target):
        """One scene's whole coarse half, computed now: ``(gather, state)``.

        The eager twin of :meth:`scene_state`, and the piece
        :func:`xr_dms.cube.sharpen_cube` defers one task at a time. Where
        :meth:`scene_state` leaves the fit in the graph and lets dask decide when
        to run it, this runs it here and hands back plain arrays -- which is what
        lets the *caller's* graph be one task per scene rather than the thirty-odd
        layers of ``delayed``/``from_delayed`` scaffolding that deferring each
        piece separately costs.

        ``gather`` is how the fine half is to get at the residual, and which of
        the two forms it takes is :attr:`smooth_residual`'s doing:

        * ``smooth_residual=False`` -- the fine-grid **label** array, an index
          into the 1-D ``state.residual_coarse``. Block-constant, and so exactly
          mass-conserving, at the cost of a step at every coarse cell edge.
        * ``smooth_residual=True`` -- the smoothed, interpolated **residual
          itself**, already on the fine grid (see
          :meth:`_smoothed_residual_fine`). The pyDMS correction: gentle, only
          approximately conserving, and free of the coarse cell geometry.

        Either way the state's ``residual_coarse`` is raveled to 1-D. An
        unfittable scene comes back with ``global_model=None`` and an all-NaN
        residual rather than raising: by the time this runs the graph's shape is
        fixed, so the failure has to be a value.

        Requires ``window_size == 0``.
        """
        if self.window_size > 0:
            raise NotImplementedError(
                "scene_bundle does not support moving-window models "
                "(window_size > 0); use fit()."
            )

        fine = self._as_features(features)
        band_order = [str(b) for b in fine[self.band_dim].values]
        self._setup_grid_map(fine, target)
        labels = np.asarray(self.grid_map_.labels)
        n_coarse = int(labels.max()) + 1 if labels.size else 0

        prepared = self.grid_map_.prepare_target(target)
        mean, y, cv = self._training_arrays(fine, prepared)
        mean_arr = np.asarray(mean.values)
        y_arr = np.asarray(y.values)
        cv_arr = np.asarray(cv.values)

        # Seeded for the same reason fit_delayed seeds: a bundle may be recomputed
        # (a lost worker, a second .compute() of a neighbouring block), and the
        # residual only conserves mass against the model it was measured from.
        pure = copy.copy(self)
        pure.regressor = self.regressor.seeded(_seed_from(mean_arr, y_arr, cv_arr))
        result = pure._fit_region(
            mean_arr, y_arr, cv_arr, slice(None), slice(None),
            local=False, min_samples=1,
        )
        if result is None:
            # An unfittable scene's gather has to have the dtype the caller's
            # graph declared for it, which follows smooth_residual either way.
            gather = (
                np.full(labels.shape, np.nan) if self.smooth_residual else labels
            )
            return gather, SceneState(
                band_order=band_order,
                residual_coarse=np.full(max(n_coarse, 1), np.nan),
            )

        model, threshold, n = result
        state = SceneState(band_order, model, threshold, n)

        guess = self.predict(fine, state=state)
        guess_r = to_radiance(guess) if self.disaggregating_temperature else guess
        obs = to_radiance(prepared) if self.disaggregating_temperature else prepared
        residual = obs - self.grid_map_.aggregate_mean(guess_r)

        gather = (
            self._smoothed_residual_fine(residual, labels, fine)
            if self.smooth_residual else labels
        )
        residual = np.asarray(residual.values, dtype=float).ravel()

        return gather, replace(state, residual_coarse=residual)

    def sharpen(self, features, target, lazy=False):
        """Fit, predict and residual-correct in one call (steps 1-5).

        With ``window_size > 0`` the local and global first guesses are blended
        (Gao 2012 section 2.3) before the residual correction. Returns the lazy
        sharpened fine-resolution :class:`xarray.DataArray`.

        ``lazy=True`` defers the training as well, via :meth:`fit_delayed` and
        :meth:`apply`, so the call touches no pixel at all and a scene that
        cannot be fitted comes back all-NaN instead of raising. It needs
        ``window_size == 0``, and leaves ``self`` unfitted -- the model lives in
        the returned graph, not on the sharpener.
        """
        if lazy:
            return self.apply(self.fit_delayed(features, target), features, target)
        self.fit(features, target)
        return self.residual_correct(self.first_guess(features, target), target)
