"""Native xarray/dask grid operations for the Data Mining Sharpener.

These functions replace the GDAL resampling primitives used by the reference
``src/pyDMS`` implementation. They assume the fine and coarse grids are
*co-registered* and related by an integer coarsening ``factor`` (see
:func:`infer_factor`), so aggregation is a plain block reduction
(:meth:`xarray.DataArray.coarsen`) rather than a reprojecting warp. Everything
is lazy-friendly: ``coarsen`` and ``interp`` preserve dask backing.
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

    Ports pyDMS ``binomialSmoother``: the coarse residual is smoothed before
    being upsampled so the correction field is gentle rather than blocky. The
    coarse grid is small, so this runs eagerly on the underlying array.
    """
    from scipy.ndimage import correlate1d

    weights = np.array([0.25, 0.5, 0.25])
    axes = {d: i for i, d in enumerate(coarse.dims)}
    values = np.asarray(coarse.values, dtype=float)
    for dim in (y_dim, x_dim):
        values = correlate1d(values, weights, axis=axes[dim], mode="nearest")
    return coarse.copy(data=values)
