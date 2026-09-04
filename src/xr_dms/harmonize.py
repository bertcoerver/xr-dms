"""Put high-resolution features that live on different grids onto one grid.

:class:`~xr_dms.Sharpener` requires every feature to share a single ``y``/``x``
grid: ``_as_features`` puts them through :meth:`xarray.Dataset.to_array` and does
no regridding at all. That rules out the common case of Sentinel-2 bands at 10 m,
20 m and 60 m in one model. :func:`harmonize_features` is the missing pre-step.

Why this is a *pre-step* and not a change to the sharpener
---------------------------------------------------------
DMS trains entirely on the **coarse** grid, so feature resolution enters in only
two places: aggregating the features down to the coarse grid, and evaluating the
regression at every fine pixel. The second unavoidably needs all bands
co-located. The first turns out not to care, because

    **nearest-neighbour upsampling before aggregation is exactly equivalent to
    aggregating the band on its own native grid.**

Replicating a value ``k`` times leaves both the block mean and the block standard
deviation bit-for-bit unchanged, and mean and std are the *only* things
:meth:`~xr_dms.gridmap.GridMap.aggregate` produces. So harmonising up costs
nothing in accuracy -- it is not an approximation -- and a grid map that
aggregated each band natively would buy memory alone. On a swath it would in fact
be slightly *worse*: :class:`~xr_dms.SwathGridMap`'s Voronoi cells cut across 20 m
pixel boundaries, so one label array per resolution would aggregate different
bands over subtly different footprints, whereas harmonising first puts every band
on one partition.

That equivalence is specific to ``"nearest"``. Bilinear upsampling smooths, which
moves the block std substantially, and with it the CV that
:func:`~xr_dms.aggregation.homogeneity_cv` uses to choose training pixels --
which is why ``"nearest"`` is the default and ``"linear"`` is offered only as an
explicit opt-out.

Two things to know before mixing in a coarse band
-------------------------------------------------
* **One NaN voids the pixel.** :meth:`~xr_dms.Sharpener.predict` keeps a pixel
  only where *every* feature is finite, and the swath aggregation does the same.
  One cloudy 60 m pixel therefore voids the whole 6x6 block of 10 m output under
  it, so the output's valid fraction is capped by the *coarsest* band's -- and a
  60 m cloud mask is blunter than a 10 m one, flagging whole 60 m pixels where a
  finer mask would have kept most of the area. Add coarse bands deliberately.
* **Downsampling discards the heterogeneity signal.** ``grid="coarsest"``
  block-averages the finer bands, which preserves the coarse mean exactly but
  shrinks the within-block std -- the very quantity the training-sample selection
  reads. Prefer ``grid="finest"`` unless memory forces the issue.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import xarray as xr

from ._grids import Grid, describe, locate, nesting
from .aggregation import upsample as _interp_upsample

__all__ = ["harmonize_features"]

#: Names accepted for each direction, mapped to what the fallback warp calls them.
_UPSAMPLE_METHODS = {"nearest": "nearest", "linear": "bilinear", "bilinear": "bilinear"}
_DOWNSAMPLE_METHODS = {"average": "average", "mean": "average", "nearest": "nearest"}

#: CF grid-mapping coordinates, stripped per band and re-attached once from the
#: reference. Left in place they collide on merge: two bands read from different
#: stores carry WKT that differs by whitespace and xarray refuses the union.
_CRS_COORDS = ("spatial_ref", "crs")


def _as_mapping(bands, band_dim="band"):
    """Normalise the many shapes of ``bands`` into ``{name: DataArray}``."""
    out = {}

    def add(name, da):
        name = str(name)
        if name in out:
            raise ValueError(
                f"Duplicate feature name {name!r}. Pass a mapping with explicit "
                "names to disambiguate bands that share one."
            )
        out[name] = da

    def take(obj, fallback=None):
        if isinstance(obj, xr.Dataset):
            for key in obj.data_vars:
                add(key, obj[key])
        elif isinstance(obj, xr.DataArray):
            if band_dim in obj.dims:
                for label in obj[band_dim].values:
                    add(label, obj.sel({band_dim: label}, drop=True))
            elif fallback is not None:
                add(fallback, obj)
            elif obj.name is not None:
                add(obj.name, obj)
            else:
                raise ValueError(
                    "Unnamed DataArray in `bands`. Name it, or pass a mapping "
                    "from name to DataArray."
                )
        else:
            raise TypeError(
                f"`bands` entries must be DataArrays or Datasets; got {type(obj)}."
            )

    # Dataset is itself a Mapping, so it has to be tested before Mapping.
    if isinstance(bands, (xr.Dataset, xr.DataArray)):
        take(bands)
    elif isinstance(bands, Mapping):
        for key, value in bands.items():
            take(value, fallback=key)
    else:
        for item in bands:
            take(item)

    if not out:
        raise ValueError("`bands` is empty; nothing to harmonise.")
    return out


def _axis_coverage(dst, edge, step, n, tol):
    """Which ``dst`` centres fall inside a source axis's ``[edge, edge + n*step]``."""
    lo, hi = sorted((edge, edge + n * step))
    return (dst >= lo - tol) & (dst <= hi + tol)


def _coverage(src, ref_y, ref_x, tol):
    """``(ny, nx)`` mask of reference pixels that fall inside a source's extent."""
    return (
        _axis_coverage(ref_y, src.y_edge, src.dy, src.ny, tol)[:, None]
        & _axis_coverage(ref_x, src.x_edge, src.dx, src.nx, tol)[None, :]
    )


def _gather_onto(da, src, dst_y, dst_x, x_dim, y_dim):
    """Block-constant resample of ``da`` onto ``(dst_y, dst_x)``.

    On grids that nest, the *nearest* source centre is always the source pixel
    that *contains* the destination centre, so a plain gather is exact
    block-constant resampling -- and, when the two resolutions are equal, an
    alignment that trims or pads without touching a value. Destination centres
    outside the source footprint become ``NaN`` rather than being clamped to the
    edge, so a band that does not cover the whole reference does not silently
    extrapolate across it.
    """
    sy = np.asarray(da[y_dim].values, dtype=float)
    sx = np.asarray(da[x_dim].values, dtype=float)
    out = da.isel({y_dim: locate(sy, dst_y), x_dim: locate(sx, dst_x)})
    out = out.assign_coords({y_dim: dst_y, x_dim: dst_x})

    in_y = _axis_coverage(dst_y, src.y_edge, src.dy, src.ny, abs(src.dy) * 1e-6)
    in_x = _axis_coverage(dst_x, src.x_edge, src.dx, src.nx, abs(src.dx) * 1e-6)
    if in_y.all() and in_x.all():
        # Skipping .where keeps an integer band integral and the graph one node
        # shorter; the mask would be a no-op anyway.
        return out
    mask = (
        xr.DataArray(in_y, dims=y_dim, coords={y_dim: dst_y})
        & xr.DataArray(in_x, dims=x_dim, coords={x_dim: dst_x})
    )
    return out.where(mask)


def _subgrid_coords(ref, factors):
    """Pixel centres that tile the reference grid at ``1 / factor`` its step."""
    fy, fx = factors
    sub_dy, sub_dx = ref.dy / fy, ref.dx / fx
    return (
        ref.y_edge + (np.arange(ref.ny * fy) + 0.5) * sub_dy,
        ref.x_edge + (np.arange(ref.nx * fx) + 0.5) * sub_dx,
    )


def _block_mean_onto(da, src, ref, factors, x_dim, y_dim):
    """Block-average a finer band onto the reference grid.

    The band is first gathered onto the exact sub-grid that tiles the reference at
    its own resolution -- a 1:1 selection, since the two steps are equal by
    construction -- which handles any extent or phase difference, and only then
    reduced. Partly covered reference pixels average what is actually there,
    matching how the swath aggregation treats a partly-cloudy cell.
    """
    sub_y, sub_x = _subgrid_coords(ref, factors)
    aligned = _gather_onto(da, src, sub_y, sub_x, x_dim, y_dim)
    fy, fx = factors
    out = aligned.coarsen({y_dim: fy, x_dim: fx}, boundary="exact").mean()
    return out


def _warp_onto(da, name, ref_da, resampling, reason):
    """Fallback for grids that do not nest: a real warp via odc.geo."""
    warnings.warn(
        f"Feature {name!r} does not nest inside the reference grid ({reason}); "
        f"falling back to a {resampling} warp. The exactness guarantee that "
        "aggregating the harmonised band matches aggregating it natively holds "
        "only for nested grids, so training samples may shift slightly.",
        UserWarning,
        stacklevel=3,
    )
    try:
        from xr_utils import reproject_like
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"Feature {name!r} needs a reprojecting warp, which requires xr_utils "
            "(and odc-geo). Install it, or pass bands on grids that nest inside "
            "the reference."
        ) from err
    return reproject_like(da, ref_da, resampling=resampling)


def _reference(entries, grids, like, grid, x_dim, y_dim):
    """Resolve the grid everything is put onto, plus its exact coordinates."""
    if like is not None:
        ref_obj = like
    else:
        if grid not in ("finest", "coarsest"):
            raise ValueError(
                f"grid must be 'finest' or 'coarsest', got {grid!r}. Pass like=... "
                "for an explicit reference grid."
            )
        pick = min if grid == "finest" else max
        # Ties keep insertion order, so the reference is reproducible.
        name = pick(entries, key=lambda n: grids[n].pixel_area)
        ref_obj = entries[name]
    return (
        describe(ref_obj, x_dim, y_dim),
        np.asarray(ref_obj[y_dim].values, dtype=float),
        np.asarray(ref_obj[x_dim].values, dtype=float),
        next(
            (ref_obj[c] for c in _CRS_COORDS if c in ref_obj.coords),
            None,
        ),
    )


def harmonize_features(
    bands,
    *,
    like=None,
    grid="finest",
    upsample="nearest",
    downsample="average",
    trim=False,
    x_dim="x",
    y_dim="y",
    band_dim="band",
):
    """Put features from several grids onto one, ready for :class:`~xr_dms.Sharpener`.

    Parameters
    ----------
    bands : Mapping, Dataset, DataArray or iterable
        The features, each on its own grid. A mapping of ``{name: DataArray}``,
        a single ``Dataset`` (one feature per data variable), a ``DataArray`` with
        a ``band_dim`` (one feature per label), or an iterable mixing those.
        Extra non-spatial dimensions (``time``, say) are carried through
        untouched.
    like : DataArray or Dataset, optional
        Explicit reference grid. Overrides ``grid``. Its ``y``/``x`` coordinates
        become the output's, verbatim.
    grid : {"finest", "coarsest"}, default "finest"
        Which input grid to use as the reference when ``like`` is not given,
        by pixel area. ``"finest"`` upsamples everything and keeps the full
        resolution; ``"coarsest"`` block-averages the finer bands down, which is
        cheaper but discards the within-pixel variance that training-sample
        selection reads.
    upsample : {"nearest", "linear"}, default "nearest"
        How to bring a coarser band up. ``"nearest"`` is exact under aggregation
        (see the module docstring); ``"linear"`` is not, and is offered only for
        continuous fields where blockiness in the *first guess* matters more.
    downsample : {"average", "nearest"}, default "average"
        How to bring a finer band down. ``"average"`` is a block mean;
        ``"nearest"`` decimates.
    trim : bool, default False
        Crop the output to the footprint every band actually covers. Off by
        default because cropping changes the fine grid's *size*, and
        :func:`~xr_dms.aggregation.infer_factor` derives the coarsening factor
        from sizes -- so a trim can silently break a
        :class:`~xr_dms.RegularGridMap` built against the untrimmed grid.
        (:meth:`~xr_dms.SwathGridMap.from_lonlat` reads its geometry off the fine
        grid it is handed, so it is unaffected either way.)
    x_dim, y_dim, band_dim : str
        Dimension names.

    Returns
    -------
    xarray.Dataset
        One variable per feature, all on the reference grid with bit-identical
        coordinates, carrying the reference's CRS coordinate if it had one.
        Dask-backed inputs stay dask-backed. Each variable gains an
        ``xr_dms_resampling`` attribute recording the path it took.

    Examples
    --------
    Sentinel-2 at three resolutions, sharpened onto the 10 m grid::

        features = harmonize_features({
            "blue": b02_10m, "green": b03_10m, "red": b04_10m, "nir": b08_10m,
            "swir": b11_20m, "water_vapour": b09_60m,
        })
        Sharpener(grid_map=grid_map).sharpen(features, thermal)

    Note that ``lazy_dino`` refuses a mixed-resolution request at load time (its
    variables must share a grid), so load one group per resolution and combine
    them here.
    """
    if upsample not in _UPSAMPLE_METHODS:
        raise ValueError(
            f"upsample must be one of {sorted(_UPSAMPLE_METHODS)}, got {upsample!r}."
        )
    if downsample not in _DOWNSAMPLE_METHODS:
        raise ValueError(
            f"downsample must be one of {sorted(_DOWNSAMPLE_METHODS)}, "
            f"got {downsample!r}."
        )

    entries = _as_mapping(bands, band_dim)
    grids = {name: describe(da, x_dim, y_dim) for name, da in entries.items()}
    ref, ref_y, ref_x, crs_coord = _reference(
        entries, grids, like, grid, x_dim, y_dim
    )

    # Built lazily: only the fallback warp needs it, and constructing it demands a
    # CRS that a purely synthetic grid may legitimately not have.
    ref_da = None
    tol = min(abs(ref.dy), abs(ref.dx)) * 1e-6

    out = {}
    covered = np.ones((ref.ny, ref.nx), dtype=bool)
    cross_crs = []  # bands whose footprint `trim` cannot reason about
    for name, da in entries.items():
        src = grids[name]
        relation = nesting(src, ref)

        if relation is None:
            if ref_da is None:
                if crs_coord is None:
                    raise ValueError(
                        f"Feature {name!r} does not nest inside the reference grid "
                        "and the reference carries no CRS, so it cannot be warped "
                        "either. Attach a CRS to the reference, or supply bands on "
                        "grids that nest."
                    )
                ref_da = xr.DataArray(
                    np.zeros((ref.ny, ref.nx), dtype="float32"),
                    dims=(y_dim, x_dim), coords={y_dim: ref_y, x_dim: ref_x},
                ).assign_coords(spatial_ref=crs_coord)
            coarser = src.pixel_area > ref.pixel_area
            method = (
                _UPSAMPLE_METHODS[upsample] if coarser
                else _DOWNSAMPLE_METHODS[downsample]
            )
            reason = (
                "different CRS" if (src.crs is not None and ref.crs is not None
                                    and src.crs != ref.crs)
                else "non-integer pixel-size ratio or misaligned pixel edges"
            )
            # odc.geo reads the source's own grid-mapping coord. A band that
            # carries none is taken to be in the reference's CRS -- which is the
            # only reading consistent with nesting() having compared their
            # geometry at all.
            source = da if src.crs is not None else da.assign_coords(
                spatial_ref=crs_coord
            )
            result = _warp_onto(source, name, ref_da, method, reason)
            result = result.drop_vars(_CRS_COORDS, errors="ignore")
            result.attrs = {**da.attrs, "xr_dms_resampling": f"warp/{method}"}
            out[name] = result
            if reason == "different CRS":
                # The band's extent is in another projection, so comparing it to
                # the reference's axes would be meaningless. Leave `covered`
                # untouched and say so, rather than trim to a bogus footprint.
                cross_crs.append(name)
            else:
                covered &= _coverage(src, ref_y, ref_x, tol)
            continue

        kind, factors = relation
        if kind == "down":
            method = _DOWNSAMPLE_METHODS[downsample]
            if method == "average":
                result = _block_mean_onto(da, src, ref, factors, x_dim, y_dim)
            else:
                result = _gather_onto(da, src, ref_y, ref_x, x_dim, y_dim)
            label = f"{method} /{factors[0]}x{factors[1]}"
        elif kind == "up" and upsample != "nearest":
            # interp needs the source's own coordinates, so this one path does not
            # go through the gather. aggregation.upsample already handles the band
            # of edge NaNs that interpolation leaves behind.
            result = _interp_upsample(
                da, xr.Dataset(coords={y_dim: ref_y, x_dim: ref_x}),
                x_dim, y_dim, method="linear",
            )
            label = f"linear x{factors[0]}x{factors[1]}"
        else:
            result = _gather_onto(da, src, ref_y, ref_x, x_dim, y_dim)
            label = (
                "identity" if kind == "same"
                else f"nearest x{factors[0]}x{factors[1]}"
            )

        # Dropped from the result rather than the input: the warp path above needs
        # the source's own grid-mapping coord, and leaving it on here would collide
        # on merge with a sibling band whose WKT differs only in whitespace.
        result = result.drop_vars(_CRS_COORDS, errors="ignore")
        result = result.assign_coords({y_dim: ref_y, x_dim: ref_x})
        result.attrs = {**da.attrs, "xr_dms_resampling": label}
        out[name] = result
        covered &= _coverage(src, ref_y, ref_x, tol)

    result = xr.Dataset(out)
    if crs_coord is not None:
        result = result.assign_coords({crs_coord.name or "spatial_ref": crs_coord})

    if trim:
        if cross_crs:
            warnings.warn(
                f"trim=True ignores the footprint of {sorted(cross_crs)}: their "
                "extent is in another CRS, so it cannot be compared against the "
                "reference axes. The trimmed result may still contain NaN there.",
                UserWarning,
                stacklevel=2,
            )
        rows = np.flatnonzero(covered.any(axis=1))
        cols = np.flatnonzero(covered.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            raise ValueError(
                "trim=True left nothing: the supplied bands share no common "
                "footprint on the reference grid."
            )
        result = result.isel(
            {y_dim: slice(rows[0], rows[-1] + 1),
             x_dim: slice(cols[0], cols[-1] + 1)}
        )
    return result
