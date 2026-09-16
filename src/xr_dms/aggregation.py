"""Native xarray/dask grid operations for the Data Mining Sharpener.

These functions replace the GDAL resampling primitives used by the reference
``src/pyDMS`` implementation. They assume the fine and coarse grids are
*co-registered* and related by an integer coarsening ``factor`` (see
:func:`infer_factor`), so aggregation is a plain block reduction
(:meth:`xarray.DataArray.coarsen`) rather than a reprojecting warp. Everything
is lazy-friendly: ``coarsen`` and ``interp`` preserve dask backing.

They are the machinery behind :class:`~xr_dms.RegularGridMap`; a pairing that is
*not* co-registered -- a curvilinear swath over a projected grid -- goes through
:class:`~xr_dms.SwathGridMap` instead. :func:`homogeneity_cv` and
:func:`binomial_smooth` are grid-agnostic and serve both.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "infer_factor",
    "coarsen_mean_std",
    "homogeneity_cv",
    "upsample",
    "binomial_smooth",
    "window_basis_1d",
]

# Small constant that guards divisions by (near-)zero means / std.
EPS = 1e-6


def infer_factor(fine, coarse, x_dim="x", y_dim="y", boundary="exact"):
    """Derive the integer coarsening factor relating ``fine`` to ``coarse``.

    The factor ``f`` is such that each coarse pixel covers an ``f x f`` block of
    fine pixels, i.e. ``fine_size == coarse_size * f`` along both spatial dims.

    Parameters
    ----------
    fine, coarse : xarray.DataArray or xarray.Dataset
        Objects carrying the fine and coarse spatial dimensions.
    x_dim, y_dim : str
        Names of the spatial dimensions.
    boundary : {"exact", "trim"}
        ``"exact"`` requires the fine size to be an exact multiple of the coarse
        size and raises otherwise. ``"trim"`` allows a remainder (the leftover
        fine edge pixels are dropped later by :func:`coarsen_mean_std`).

    Returns
    -------
    int
        The coarsening factor, identical along both dimensions.
    """
    factors = {}
    for dim in (y_dim, x_dim):
        n_fine = fine.sizes[dim]
        n_coarse = coarse.sizes[dim]
        if n_coarse == 0:
            raise ValueError(f"Coarse dimension {dim!r} has size 0.")
        f = n_fine // n_coarse
        if f < 1:
            raise ValueError(
                f"Fine grid is not finer than coarse grid along {dim!r} "
                f"({n_fine} < {n_coarse})."
            )
        if boundary == "exact" and n_fine != n_coarse * f:
            raise ValueError(
                f"Fine size {n_fine} along {dim!r} is not an integer multiple "
                f"of coarse size {n_coarse}. Use boundary='trim' to allow "
                f"dropping the leftover edge, or re-align the grids."
            )
        factors[dim] = f

    if factors[y_dim] != factors[x_dim]:
        raise ValueError(
            f"Coarsening factor differs between dimensions: "
            f"{y_dim}={factors[y_dim]}, {x_dim}={factors[x_dim]}. "
            "xr_dms requires a single isotropic integer factor."
        )
    return int(factors[x_dim])


def coarsen_mean_std(fine, factor, x_dim="x", y_dim="y", boundary="exact"):
    """Block mean and standard deviation of ``fine`` over ``factor`` blocks.

    Replaces pyDMS ``resampleHighResToLowRes``: it brings fine features onto the
    coarse grid (block mean) while also measuring within-block heterogeneity
    (block std, ``ddof=0`` to match numpy's default).

    Parameters
    ----------
    fine : xarray.DataArray
        Fine-resolution data with spatial dims ``x_dim``/``y_dim`` (and,
        typically, a ``band`` dim).
    factor : int
        Coarsening factor from :func:`infer_factor`.
    boundary : {"exact", "trim"}
        Forwarded to :meth:`xarray.DataArray.coarsen`.

    Returns
    -------
    (xarray.DataArray, xarray.DataArray)
        ``(mean, std)`` on the coarse grid.
    """
    coarsen = fine.coarsen({y_dim: factor, x_dim: factor}, boundary=boundary)
    mean = coarsen.mean()
    std = coarsen.std()
    return mean, std


def homogeneity_cv(mean, std, band_dim="band", eps=EPS):
    """Per-coarse-pixel coefficient of variation, averaged over features.

    The CV (``std / |mean|``) quantifies how internally mixed each coarse pixel
    is; averaging over the feature (``band``) dimension gives a single
    homogeneity score per pixel (Gao 2012, section 2.2). Low CV = homogeneous =
    a trustworthy training sample.

    If there is no ``band`` dimension the ratio is returned as-is.
    """
    cv = std / (np.abs(mean) + eps)
    if band_dim in cv.dims:
        cv = cv.mean(dim=band_dim)
    return cv


def upsample(coarse, like, x_dim="x", y_dim="y", method="linear"):
    """Bilinearly resample a coarse 2-D field onto a fine grid.

    Replaces pyDMS ``resampleLowResToHighRes``: used to spread the smoothed
    coarse residual back over the fine grid. Coarse pixel centres are inset from
    the fine grid edges, so linear interpolation leaves NaNs on the outermost
    fine rows/columns; these are filled by nearest-edge extension
    (``ffill``/``bfill``), mirroring pyDMS's edge padding.

    Parameters
    ----------
    coarse : xarray.DataArray
        2-D coarse field (spatial dims only).
    like : xarray.DataArray or xarray.Dataset
        Provides the target fine coordinates along ``x_dim``/``y_dim``.
    """
    up = coarse.interp(
        {y_dim: like[y_dim], x_dim: like[x_dim]},
        method=method,
    )
    # Fill the thin band of edge NaNs left by interpolation with nearest values.
    up = up.ffill(x_dim).bfill(x_dim).ffill(y_dim).bfill(y_dim)
    return up


def binomial_smooth(coarse, x_dim="x", y_dim="y"):
    """3x3 binomial (1-2-1)(1-2-1) smoothing with edge padding.

    Ports pyDMS ``binomialSmoother``, including its NaN handling: a missing
    neighbour is *left out* of the average and the remaining weights are
    renormalised, and a missing centre stays missing. A plain convolution would
    instead let every gap in the coarse residual -- a cloudy cell, a cell the
    fine grid barely covers -- eat its eight neighbours, so one run of the
    smoother would widen every hole by a pixel on all sides.

    The renormalisation is exact rather than an approximation of the 3x3 filter:
    the 9-point kernel is the outer product of ``[1, 2, 1]`` with itself, so
    smoothing the masked values and the mask separably and dividing gives the
    same answer as the 2-D weighted filter.

    The coarse grid is small, so this runs eagerly on the underlying array.
    """
    from scipy.ndimage import correlate1d

    weights = np.array([0.25, 0.5, 0.25])
    axes = {d: i for i, d in enumerate(coarse.dims)}
    values = np.asarray(coarse.values, dtype=float)

    present = np.isfinite(values)
    numerator = np.where(present, values, 0.0)
    denominator = present.astype(float)
    for dim in (y_dim, x_dim):
        numerator = correlate1d(numerator, weights, axis=axes[dim], mode="nearest")
        denominator = correlate1d(
            denominator, weights, axis=axes[dim], mode="nearest"
        )

    smoothed = np.divide(
        numerator, denominator,
        out=np.full_like(numerator, np.nan), where=denominator > 0,
    )
    return coarse.copy(data=np.where(present, smoothed, np.nan))


def window_basis_1d(coords, centers, edges, smooth):
    """Per-window 1-D blending weights for a moving-window sharpener.

    Returns a ``(n_windows, n_coords)`` matrix giving, along a single spatial
    axis, how much each window contributes to each fine coordinate. The 2-D
    window weight field used by the local prediction is the outer product of the
    y- and x-axis results, so the blend stays separable and cheap.

    Parameters
    ----------
    coords : array-like, shape (n_coords,)
        Fine pixel-centre coordinates along the axis (any orientation).
    centers : array-like, shape (n_windows,)
        Coordinate of each window prediction cell's centre along the axis.
    edges : array-like, shape (n_windows, 2)
        ``[lo, hi]`` bounds (``lo < hi``) of each window's prediction cell; the
        outermost bounds should be ``-inf``/``+inf`` so the cells partition the
        whole axis. Used only when ``smooth`` is ``False``.
    smooth : bool
        ``True``: piecewise-linear ("tent") basis over ``centers`` -- a partition
        of unity that interpolates linearly between neighbouring window centres
        and clamps to the nearest window past the end centres, giving a
        C0-continuous (seamless) local field. ``False``: hard indicator of
        ``coords`` falling inside each window's ``edges`` (nearest / blocky).

    Returns
    -------
    numpy.ndarray, shape (n_windows, n_coords)
        Non-negative weights. For ``smooth`` they sum to 1 across windows at
        every coordinate; for the hard case exactly one window is 1 per
        coordinate.
    """
    coords = np.asarray(coords, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n_win = centers.shape[0]

    if not smooth:
        edges = np.asarray(edges, dtype=float)
        lo = edges[:, 0][:, None]
        hi = edges[:, 1][:, None]
        return ((coords[None, :] >= lo) & (coords[None, :] < hi)).astype(float)

    # Piecewise-linear basis. np.interp needs ascending sample points, so work in
    # the sorted-centre order and scatter the rows back to the original order.
    order = np.argsort(centers)
    cs = centers[order]
    basis_sorted = np.empty((n_win, coords.shape[0]), dtype=float)
    for k in range(n_win):
        unit = np.zeros(n_win, dtype=float)
        unit[k] = 1.0
        # np.interp clamps to the end values outside [cs[0], cs[-1]], which is
        # exactly the "nearest window past the ends" behaviour we want.
        basis_sorted[k] = np.interp(coords, cs, unit)
    basis = np.empty_like(basis_sorted)
    basis[order] = basis_sorted
    return basis
