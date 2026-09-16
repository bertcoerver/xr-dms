"""Sharpening a curvilinear swath onto a projected grid.

A thermal swath (VIIRS, MODIS, SLSTR) carries a 2-D ``latitude``/``longitude``
pair per scan pixel and has no integer relationship to an optical grid in a
projected CRS: the footprint grows off-nadir, the scan is skewed, and the two
objects do not even share a coordinate system. :class:`SwathGridMap` supplies the
three primitives :class:`~xr_dms.gridmap.GridMap` requires for that pairing,
**without ever resampling the observations**.

The mechanism is a nearest-neighbour map from every fine pixel to its parent
swath pixel -- a Voronoi tessellation of the swath pixel centres. Because that is
a genuine partition (each fine pixel has exactly one parent), the three
primitives fall out as pure index arithmetic:

    * aggregation is a labelled reduction (:func:`numpy.bincount`),
    * ``"nearest"`` upsampling is a gather, which round-trips through the
      aggregation exactly and so keeps ``smooth_residual=False`` mass-conserving,
    * ``"linear"`` upsampling is scattered interpolation over the swath centres
      projected into the fine CRS.

The sharpened result lands on the fine grid unchanged -- same dims, coords, shape
and CRS -- with ``NaN`` wherever the swath does not reach.
"""

from __future__ import annotations

import threading

import numpy as np
import xarray as xr

from ._grids import axis_step as _axis_step
from ._grids import crs_of, locate as _locate
from .aggregation import EPS
from .gridmap import GridMap, WindowBasis

__all__ = ["SwathGridMap"]

#: Serialises every entry into Qhull, across the whole process.
#:
#: ``scipy.interpolate.LinearNDInterpolator`` is *not* safe to construct in one
#: thread while another is evaluating a different one: Qhull keeps process-global
#: scratch state, so the two interleave and the evaluation comes back subtly
#: wrong -- measured at a few tenths of a Kelvin over a fraction of a percent of
#: the fine grid, in different places every run. Constructing them concurrently
#: is fine, and evaluating them concurrently is fine; it is only the mixture that
#: corrupts, which is exactly what a cube of scenes does, one Delaunay per
#: overpass against a fine grid tiled into blocks.
#:
#: The cost is that the scattered interpolation runs one thread at a time. That
#: is a small share of DMS -- the regression and the aggregation, which are the
#: expensive halves, stay parallel -- and the alternative is a sharpened field
#: that is not a function of its inputs.
_QHULL_LOCK = threading.Lock()

#: Private coarse-namespace dim names. The swath's own ``y``/``x`` are scan row
#: and column while the fine grid's are projected northing and easting; letting
#: xarray align those would silently produce an empty intersection.
COARSE_Y = "_dms_cy"
COARSE_X = "_dms_cx"


# -- fine grid description ---------------------------------------------------
#: A swath always needs a CRS on the fine side -- there is nothing to project the
#: geolocation into without one -- so the shared reader is pinned to required.
def _crs_from(obj):
    """Find a CRS on an xarray object, or raise. See :func:`xr_dms._grids.crs_of`."""
    return crs_of(obj, required=True)


#: What a non-uniform fine coordinate breaks, in SwathGridMap's terms.
_SWATH_CONTEXT = "a regular projected grid on the fine side"


def _area_from_fine(fine, x_dim, y_dim, crs):
    """Build a pyresample AreaDefinition from the fine grid's 1-D coordinates.

    Returns ``(area, flip_y, flip_x)``. pyresample lays an area out north-up and
    x-ascending; the flags say how to bring its output back into the *data's* own
    row/column order, which for a north-up raster has ``y`` descending.
    """
    from pyresample.geometry import AreaDefinition

    xs = np.asarray(fine[x_dim].values, dtype=float)
    ys = np.asarray(fine[y_dim].values, dtype=float)
    dx = _axis_step(xs, x_dim, _SWATH_CONTEXT)
    dy = _axis_step(ys, y_dim, _SWATH_CONTEXT)

    # Pixel-edge extent from centre coordinates.
    x_lo, x_hi = xs.min() - abs(dx) / 2, xs.max() + abs(dx) / 2
    y_lo, y_hi = ys.min() - abs(dy) / 2, ys.max() + abs(dy) / 2

    area = AreaDefinition(
        "xr_dms_fine", "xr_dms fine grid", "xr_dms_fine",
        crs, xs.size, ys.size, (x_lo, y_lo, x_hi, y_hi),
    )
    # pyresample row 0 is the northernmost; flip if the data is stored the other way.
    return area, dy > 0, dx < 0


# -- labelled reductions -----------------------------------------------------
def _block_stats(values, labels, n_coarse):
    """Per-coarse-cell ``(count, sum, sumsq)`` for one block.

    ``values`` is ``(n_bands, ny, nx)``, ``labels`` is ``(ny, nx)`` with ``-1``
    meaning "no parent". NaNs are excluded from every accumulator so a coarse cell
    partly covered by cloud still yields the mean of what was seen.
    """
    n_bands = values.shape[0]
    flat = labels.ravel()
    keep = flat >= 0
    # Everything unparented goes to a sentinel bucket that is sliced off.
    lab = np.where(keep, flat, n_coarse)
    out = np.zeros((n_bands, n_coarse, 3), dtype=float)
    for b in range(n_bands):
        v = values[b].ravel()
        w = keep & np.isfinite(v)
        vv = np.where(w, v, 0.0)
        out[b, :, 0] = np.bincount(lab, weights=w.astype(float), minlength=n_coarse + 1)[:n_coarse]
        out[b, :, 1] = np.bincount(lab, weights=vv, minlength=n_coarse + 1)[:n_coarse]
        out[b, :, 2] = np.bincount(lab, weights=vv * vv, minlength=n_coarse + 1)[:n_coarse]
    return out


def _block_stats_boxed(values, labels, n_coarse):
    """:func:`_block_stats` with two leading length-1 block axes.

    ``blockwise`` needs every output index to name a real axis, so the per-block
    histogram is boxed as ``(1, 1, n_bands, n_coarse, 3)`` and the two block axes
    are summed away afterwards.
    """
    return _block_stats(values, labels, n_coarse)[None, None]


def _accumulate(values, labels, n_coarse):
    """``(count, sum, sumsq)`` over the whole fine grid, streaming if dask-backed.

    Each dask block contributes a small ``(n_bands, n_coarse, 3)`` histogram which
    are summed, so peak memory stays per-block rather than whole-array. The result
    is *lazy* when ``values`` is dask-backed -- it is a scene-global reduction, but
    a small one, and leaving it in the graph is what lets the caller defer the fit.

    ``labels`` may itself be dask-backed (a lazily built :class:`SwathGridMap`); it
    is rechunked onto the fine array's spatial blocks either way, so the pairing is
    dask's problem rather than a hand-rolled offset loop. That also keeps the whole
    reduction in one ``Blockwise`` layer instead of one ``MaterializedLayer`` per
    block, which is what makes graph optimisation stable.
    """
    if not hasattr(values, "dask"):
        if hasattr(labels, "dask"):
            labels = labels.compute()
        return _block_stats(np.asarray(values), labels, n_coarse)

    import dask.array as dsa

    spatial = values.chunks[1:]
    lab = labels if hasattr(labels, "dask") else dsa.from_array(labels, chunks=spatial)
    lab = lab.rechunk(spatial)

    boxed = dsa.blockwise(
        _block_stats_boxed, "yxbnk",
        values, "byx",
        lab, "yx",
        n_coarse=n_coarse,
        new_axes={"n": n_coarse, "k": 3},
        adjust_chunks={"y": 1, "x": 1},
        dtype=float,
    )
    return boxed.sum(axis=(0, 1))


def _gather_block(lab, vals):
    """One fine block's worth of the block-constant upsample."""
    return np.append(vals, np.nan)[lab]


def _mean_std_np(stats, min_count, min_fraction, cell_size):
    """Turn ``(count, sum, sumsq)`` accumulators into mean and ``ddof=0`` std."""
    count, total, total_sq = stats[..., 0], stats[..., 1], stats[..., 2]
    enough = count >= np.maximum(min_count, min_fraction * np.maximum(cell_size, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(enough, total / np.where(count > 0, count, np.nan), np.nan)
        var = np.where(enough, total_sq / np.where(count > 0, count, np.nan) - mean ** 2, np.nan)
    return mean, np.sqrt(np.clip(var, 0.0, None))


def _mean_std(stats, min_count, min_fraction, cell_size):
    """:func:`_mean_std_np`, deferred whole when either input is dask-backed.

    One task rather than a blockwise expression: both operands are coarse-grid
    sized (a few MB at most), and keeping the numpy body intact keeps the
    ``errstate`` suppression of the empty-cell divides working -- an ``errstate``
    context around graph *construction* would not be in scope when the blocks
    actually run.
    """
    if not (hasattr(stats, "dask") or hasattr(cell_size, "dask")):
        return _mean_std_np(stats, min_count, min_fraction, cell_size)

    import dask
    import dask.array as dsa

    pair = dask.delayed(_mean_std_np, nout=2, pure=True)(
        one_block(stats), min_count, min_fraction, one_block(cell_size),
    )
    shape = stats.shape[:-1]
    return (
        dsa.from_delayed(pair[0], shape=shape, dtype=float),
        dsa.from_delayed(pair[1], shape=shape, dtype=float),
    )


def one_block(x):
    """A dask array as a single ``Delayed`` block, with a *deterministic* key.

    Handing a dask array straight to ``dask.delayed`` wraps it in a
    ``finalize-hlgfinalizecompute-<uuid>`` layer whose name is freshly random on
    every call -- even under ``pure=True``, and even with ``PYTHONHASHSEED``
    pinned. Graph optimisation iterates over sets of layer names, so those random
    names make slice pushdown and culling come out differently run to run: the
    same selection off the same pipeline was measured at 26 tasks once and 40554
    the next time. Going through ``to_delayed`` instead keeps the key derived
    from the array's own name, so the graph is reproducible and culling is
    stable.

    Everything this module defers is coarse-grid sized, so collapsing to one
    block costs nothing.
    """
    if not hasattr(x, "dask"):
        return x
    return x.rechunk(-1).to_delayed().ravel()[0]


def _cell_stats(labels, n_coarse):
    """``(cell_size, coverage)`` from a label array: fine pixels per coarse cell."""
    flat = labels.ravel()
    cell_size = np.bincount(flat[flat >= 0], minlength=n_coarse)[:n_coarse].astype(float)
    return cell_size, float((labels >= 0).mean())


def _labels_from_lonlat(lon_v, lat_v, area, flip_y, flip_x, crs,
                        radius_of_influence, n_coarse):
    """The whole numpy body of :meth:`SwathGridMap.from_lonlat`, as a pure function.

    Kept separate so it can be handed to ``dask.delayed`` unchanged: it depends on
    nothing but the two geolocation arrays and metadata already known from the fine
    grid. Returns ``(labels, cell_size, coverage, sx, sy)``.
    """
    from pyproj import CRS, Transformer
    from pyresample import kd_tree
    from pyresample.geometry import SwathDefinition

    lon_v = np.asarray(lon_v, dtype=float)
    lat_v = np.asarray(lat_v, dtype=float)

    # Swath centres in the fine CRS: needed for the default radius, for the
    # "linear" upsample, and for window centres.
    fwd = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    sx, sy = fwd.transform(lon_v, lat_v)

    if radius_of_influence is None:
        radius_of_influence = 1.1 * _median_spacing(sx, sy)

    swath = SwathDefinition(lons=lon_v, lats=lat_v)
    valid_in, valid_out, index_array, _ = kd_tree.get_neighbour_info(
        swath, area, float(radius_of_influence), neighbours=1,
    )

    # index_array indexes the *reduced* source set and uses tree.n as the
    # out-of-range sentinel for targets past the cut-off.
    src_flat = np.flatnonzero(valid_in)
    idx = np.asarray(index_array).ravel()
    hit = idx < src_flat.size
    labels = np.full(area.size, -1, dtype=np.int64)
    out_pos = np.flatnonzero(valid_out) if valid_out.dtype == bool else valid_out
    labels[out_pos[hit]] = src_flat[idx[hit]]
    labels = labels.reshape(area.shape)

    # Bring pyresample's north-up / x-ascending layout into the data's order.
    if flip_y:
        labels = labels[::-1]
    if flip_x:
        labels = labels[:, ::-1]

    labels = np.ascontiguousarray(labels)
    cell_size, coverage = _cell_stats(labels, n_coarse)
    return labels, cell_size, coverage, sx, sy


class SwathGridMap(GridMap):
    """Maps a projected fine grid onto a curvilinear swath's pixels.

    Build it with :meth:`from_lonlat` rather than calling the constructor.

    Attributes
    ----------
    labels : numpy.ndarray
        ``(ny_fine, nx_fine)`` flat index of each fine pixel's parent swath pixel,
        or ``-1`` where the swath does not reach.
    coverage : float
        Fraction of fine pixels that have a parent.
    """

    def __init__(self, labels, swath_shape, swath_dims, fine_coords, x_dim, y_dim,
                 proj_xy, min_fine_fraction=0.0, min_fine_pixels=1,
                 cell_size=None, coverage=None):
        self.labels = labels
        self.swath_shape = tuple(swath_shape)
        self.swath_dims = tuple(swath_dims)
        self.coarse_shape = tuple(swath_shape)
        self.coarse_dims = (COARSE_Y, COARSE_X)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._fine_coords = fine_coords
        self._proj_xy = proj_xy  # (sx, sy) swath centres in the fine CRS, or None if lazy
        self.min_fine_fraction = float(min_fine_fraction)
        self.min_fine_pixels = int(min_fine_pixels)

        # A lazily built map hands these in already-derived (they come out of the
        # same delayed body as the labels, since bincount has no dask equivalent).
        if cell_size is None or coverage is None:
            n = int(np.prod(self.swath_shape))
            cell_size, coverage = _cell_stats(labels, n)
        self._cell_size = cell_size
        self.coverage = coverage

    @property
    def lazy(self):
        """True when ``labels`` is dask-backed, i.e. no geolocation has been read."""
        return hasattr(self.labels, "dask")

    def _require_eager(self, what):
        if self._proj_xy is None:
            raise NotImplementedError(
                f"{what} needs the swath centres projected into the fine CRS, which "
                "a lazily built SwathGridMap does not materialise. Rebuild with "
                "SwathGridMap.from_lonlat(..., lazy=False)."
            )

    # -- construction --------------------------------------------------------
    @classmethod
    def from_lonlat(cls, lon, lat, fine, crs=None, radius_of_influence=None,
                    x_dim="x", y_dim="y", min_fine_fraction=0.0, min_fine_pixels=1,
                    lazy=False):
        """Build the map from a swath's 2-D geolocation and a projected fine grid.

        Parameters
        ----------
        lon, lat : xarray.DataArray
            2-D geolocation of the swath, in degrees (WGS84), sharing the swath's
            ``(y, x)`` scan dims.
        fine : xarray.DataArray or xarray.Dataset
            The fine grid to sharpen onto. Supplies the 1-D projected ``x``/``y``
            coordinates and, unless ``crs`` is given, the CRS.
        crs : optional
            Anything :meth:`pyproj.CRS.from_user_input` accepts. Overrides the CRS
            detected on ``fine``.
        radius_of_influence : float, optional
            Cut-off distance in metres for the neighbour search. Defaults to 1.1x
            the median spacing between adjacent swath centres, which covers the
            corners of a Voronoi cell with a little margin. A single global radius
            is a compromise on a scan whose footprint grows off-nadir -- raise it
            to close gaps at the scan edge, lower it to tighten coverage.
        min_fine_fraction : float, default 0.0
            Discard a coarse cell whose valid (non-NaN) fine pixels fall below this
            fraction of the cell's size. Use it to drop thermal pixels that are
            mostly cloud on the optical side.
        min_fine_pixels : int, default 1
            Discard a coarse cell backed by fewer than this many valid fine pixels.
        lazy : bool, default False
            Defer the neighbour search instead of running it here. ``labels`` comes
            back as a dask array and no geolocation is read until something is
            computed -- which is what lets a whole time series of overpasses be
            assembled without touching a pixel. The trade-off is ``_proj_xy``: the
            projected swath centres are not kept, so ``upsample(method="linear")``
            and moving-window models are unavailable (see :meth:`_require_eager`).
        """
        from pyproj import CRS

        if lon.shape != lat.shape or lon.ndim != 2:
            raise ValueError(
                f"lon/lat must be matching 2-D arrays; got {lon.shape} and {lat.shape}."
            )

        crs = CRS.from_user_input(crs) if crs is not None else _crs_from(fine)
        area, flip_y, flip_x = _area_from_fine(fine, x_dim, y_dim, crs)
        n_coarse = int(np.prod(lon.shape))

        fine_coords = {
            y_dim: np.asarray(fine[y_dim].values, dtype=float),
            x_dim: np.asarray(fine[x_dim].values, dtype=float),
        }
        common = dict(
            swath_shape=lon.shape,
            swath_dims=lon.dims,
            fine_coords=fine_coords,
            x_dim=x_dim, y_dim=y_dim,
            min_fine_fraction=min_fine_fraction,
            min_fine_pixels=min_fine_pixels,
        )

        if not lazy:
            labels, cell_size, coverage, sx, sy = _labels_from_lonlat(
                lon.values, lat.values, area, flip_y, flip_x, crs,
                radius_of_influence, n_coarse,
            )
            return cls(labels=labels, proj_xy=(sx, sy),
                       cell_size=cell_size, coverage=coverage, **common)

        import dask
        import dask.array as dsa

        # One delayed call, five outputs: the neighbour search, the label assembly
        # and the per-cell counts are one indivisible piece of work over the same
        # two arrays, so splitting them would only duplicate the KDTree.
        parts = dask.delayed(_labels_from_lonlat, nout=5, pure=True)(
            one_block(lon.data), one_block(lat.data), area, flip_y, flip_x, crs,
            radius_of_influence, n_coarse,
        )
        labels = dsa.from_delayed(parts[0], shape=area.shape, dtype=np.int64)
        cell_size = dsa.from_delayed(parts[1], shape=(n_coarse,), dtype=float)
        return cls(labels=labels, proj_xy=None,
                   cell_size=cell_size, coverage=parts[2], **common)

    # -- GridMap interface ---------------------------------------------------
    def prepare_target(self, target):
        if tuple(target.shape) != self.swath_shape:
            raise ValueError(
                f"Target shape {tuple(target.shape)} does not match the swath "
                f"geolocation shape {self.swath_shape}."
            )
        renamed = target.rename(dict(zip(target.dims, self.coarse_dims)))
        # Scan-space coords (row/column indices) would be meaningless downstream.
        return renamed.drop_vars(
            [c for c in renamed.coords if c not in self.coarse_dims], errors="ignore"
        ).drop_vars(list(self.coarse_dims), errors="ignore")

    def _to_coarse_da(self, values, band_dim=None, bands=None):
        """Wrap a ``(n_coarse,)`` or ``(n_bands, n_coarse)`` result as a DataArray."""
        if values.ndim == 1:
            return xr.DataArray(
                values.reshape(self.coarse_shape), dims=self.coarse_dims
            )
        arr = values.reshape((values.shape[0],) + self.coarse_shape)
        da = xr.DataArray(arr, dims=(band_dim,) + self.coarse_dims)
        return da.assign_coords({band_dim: bands}) if bands is not None else da

    def _stats(self, fine, band_dim):
        """``(mean, std, band_dim_or_None, band_labels_or_None)`` on the coarse grid.

        ``band_dim=None`` (or a dim the array lacks) means a plain 2-D field, which
        is temporarily given a length-1 band axis so one code path serves both.
        """
        squeeze = band_dim is None or band_dim not in fine.dims
        if squeeze:
            band_dim = band_dim or "__band__"
            ordered = fine.expand_dims(band_dim)
        else:
            ordered = fine
        ordered = ordered.transpose(band_dim, self.y_dim, self.x_dim)
        data = ordered.data
        if hasattr(data, "dask"):
            data = data.rechunk({0: -1})
        stats = _accumulate(data, self.labels, int(np.prod(self.coarse_shape)))
        mean, std = _mean_std(
            stats, self.min_fine_pixels, self.min_fine_fraction, self._cell_size
        )
        if squeeze:
            return mean[0], std[0], None, None
        bands = ordered[band_dim].values if band_dim in ordered.coords else None
        return mean, std, band_dim, bands

    def aggregate(self, fine, band_dim="band"):
        mean, std, bd, bands = self._stats(fine, band_dim)
        return (
            self._to_coarse_da(mean, bd, bands),
            self._to_coarse_da(std, bd, bands),
        )

    def aggregate_mean(self, fine):
        mean, _, bd, bands = self._stats(fine, None)
        return self._to_coarse_da(mean, bd, bands)

    def upsample(self, coarse, like, method="linear"):
        data = getattr(coarse, "data", coarse)
        if hasattr(data, "dask"):
            # A deferred residual: keep it in the graph rather than forcing the
            # scene-global reduction that produced it.
            values = data.reshape(-1).astype(float)
        else:
            values = np.asarray(coarse.values, dtype=float).ravel()
        if method == "nearest":
            out = self._gather(values, like)
        else:
            out = self._interpolate(values, like)
        coords = {
            d: like[d] for d in (self.y_dim, self.x_dim) if d in like.coords
        }
        return xr.DataArray(out, dims=(self.y_dim, self.x_dim), coords=coords)

    def _gather(self, values, like):
        """Block-constant upsample: the exact inverse of the labelled aggregation."""
        lab = self.labels
        chunks = _fine_chunks(like, self.y_dim, self.x_dim)

        if not hasattr(values, "dask") and not hasattr(lab, "dask"):
            padded = np.append(values, np.nan)  # index -1 lands on the NaN pad
            if chunks is None:
                return padded[lab]
            import dask.array as dsa

            return dsa.map_blocks(
                lambda block: padded[block],
                dsa.from_array(lab, chunks=chunks),
                dtype=float,
            )

        import dask.array as dsa

        if chunks is None:
            raise ValueError(
                "A lazy residual or grid map needs a chunked fine grid to gather "
                "onto; pass a dask-backed `like`."
            )
        lab_da = lab if hasattr(lab, "dask") else dsa.from_array(lab, chunks=chunks)
        lab_da = lab_da.rechunk(chunks)
        vals = values if hasattr(values, "dask") else dsa.from_array(values, chunks=-1)
        # `values` is one coarse grid (~100k floats), so handing every fine block the
        # whole of it costs nothing and keeps this a single Blockwise layer.
        return dsa.blockwise(
            _gather_block, "yx", lab_da, "yx", vals.rechunk(-1), "c",
            concatenate=True, dtype=float,
        )

    def _interpolate(self, values, like):
        """Scattered linear interpolation over the projected swath centres.

        Fine pixels outside the convex hull of the swath centres -- the outermost
        ~1% -- get the nearest coarse value instead, mirroring the ``ffill``/``bfill``
        edge extension of the regular-grid upsampler.

        Qhull is entered under :data:`_QHULL_LOCK`; see there for why.
        """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        self._require_eager('upsample(method="linear")')
        sx, sy = self._proj_xy
        pts = np.column_stack([sx.ravel(), sy.ravel()])
        ok = np.isfinite(pts).all(axis=1) & np.isfinite(values)
        if not ok.any():
            raise ValueError("No finite coarse values to upsample.")
        with _QHULL_LOCK:
            lin = LinearNDInterpolator(pts[ok], values[ok])
        near = NearestNDInterpolator(pts[ok], values[ok])

        ys = self._fine_coords[self.y_dim]
        xs = self._fine_coords[self.x_dim]

        def _eval(y_block, x_block):
            xx, yy = np.meshgrid(x_block, y_block)
            q = np.column_stack([xx.ravel(), yy.ravel()])
            with _QHULL_LOCK:
                out = lin(q)
            gap = ~np.isfinite(out)
            if gap.any():
                out[gap] = near(q[gap])
            return out.reshape(xx.shape)

        chunks = _fine_chunks(like, self.y_dim, self.x_dim)
        if chunks is None:
            return _eval(ys, xs)

        import dask.array as dsa

        y_da = dsa.from_array(ys, chunks=chunks[0])
        x_da = dsa.from_array(xs, chunks=chunks[1])
        return dsa.blockwise(
            _eval, "yx", y_da, "y", x_da, "x", dtype=float,
        )

    def build_window_basis(self, n_wy, n_wx, window_size, smooth, fine):
        self._require_eager("A moving-window (window_size > 0) model")
        ny, nx = self.coarse_shape
        sx, sy = self._proj_xy
        centers = np.full((n_wy * n_wx, 2), np.nan)
        for iy in range(n_wy):
            rows = slice(iy * window_size, min((iy + 1) * window_size, ny))
            for ix in range(n_wx):
                cols = slice(ix * window_size, min((ix + 1) * window_size, nx))
                bx, by = sx[rows, cols], sy[rows, cols]
                good = np.isfinite(bx) & np.isfinite(by)
                if good.any():
                    centers[iy * n_wx + ix] = (bx[good].mean(), by[good].mean())
        return SwathWindowBasis(
            centers=centers, n_wy=n_wy, n_wx=n_wx, window_size=window_size,
            smooth=smooth, labels=self.labels, swath_nx=nx,
            fine_coords=self._fine_coords, y_dim=self.y_dim, x_dim=self.x_dim,
        )


def _median_spacing(sx, sy):
    """Median distance between adjacent swath centres, in projected units."""
    dists = []
    for arr_x, arr_y in ((np.diff(sx, axis=1), np.diff(sy, axis=1)),
                         (np.diff(sx, axis=0), np.diff(sy, axis=0))):
        d = np.hypot(arr_x, arr_y)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            dists.append(np.median(d))
    if not dists:
        raise ValueError("Could not derive a swath pixel spacing; pass radius_of_influence.")
    return float(np.median(dists))


def _fine_chunks(like, y_dim, x_dim):
    """``(y_chunks, x_chunks)`` of a dask-backed fine array, or ``None``."""
    if getattr(like, "chunks", None) is None:
        return None
    sizes = dict(zip(like.dims, like.chunks))
    if y_dim not in sizes or x_dim not in sizes:
        return None
    return sizes[y_dim], sizes[x_dim]


class SwathWindowBasis(WindowBasis):
    """Window weights over a swath, where the cells are not a rectilinear product.

    ``smooth=False`` is exact and free: a fine pixel already knows its parent swath
    pixel, so its window is that pixel's scan-index block. ``smooth=True`` uses
    barycentric coordinates on a Delaunay triangulation of the window centres --
    the 2-D generalisation of the 1-D tent basis, and likewise a C0-continuous
    partition of unity.
    """

    def __init__(self, centers, n_wy, n_wx, window_size, smooth, labels, swath_nx,
                 fine_coords, y_dim, x_dim):
        self.centers = centers
        self.n_wy = n_wy
        self.n_wx = n_wx
        self.window_size = window_size
        self.smooth = smooth
        self.labels = labels
        self.swath_nx = swath_nx
        self.fine_coords = fine_coords
        self.y_dim = y_dim
        self.x_dim = x_dim

        self._tri = None
        if smooth:
            good = np.isfinite(centers).all(axis=1)
            self._valid = np.flatnonzero(good)
            if self._valid.size >= 3:
                from scipy.spatial import Delaunay, cKDTree

                try:
                    self._tri = Delaunay(centers[good])
                    self._tree = cKDTree(centers[good])
                except Exception:  # noqa: BLE001 - degenerate (collinear) centres
                    self._tri = None
            if self._tri is None and self._valid.size:
                from scipy.spatial import cKDTree

                self._tree = cKDTree(centers[good])

    def weights(self, block, y_dim, x_dim):
        yb = np.asarray(block[y_dim].values, dtype=float)
        xb = np.asarray(block[x_dim].values, dtype=float)
        ny, nx = yb.size, xb.size

        if not self.smooth:
            iy = _locate(self.fine_coords[self.y_dim], yb)
            ix = _locate(self.fine_coords[self.x_dim], xb)
            lab = self.labels[np.ix_(iy, ix)]
            sy_i, sx_i = np.divmod(np.maximum(lab, 0), self.swath_nx)
            win = (sy_i // self.window_size) * self.n_wx + (sx_i // self.window_size)
            idx = np.where(lab >= 0, win, -1)[None, :, :]
            return idx, (idx >= 0).astype(float)

        xx, yy = np.meshgrid(xb, yb)
        q = np.column_stack([xx.ravel(), yy.ravel()])
        idx = np.full((3, ny * nx), -1, dtype=np.int64)
        w = np.zeros((3, ny * nx), dtype=float)
        if self._valid.size == 0:
            return idx.reshape(3, ny, nx), w.reshape(3, ny, nx)

        if self._tri is not None:
            simplex = self._tri.find_simplex(q)
            inside = simplex >= 0
            if inside.any():
                s = simplex[inside]
                trans = self._tri.transform[s]
                delta = q[inside] - trans[:, 2]
                bary = np.einsum("ijk,ik->ij", trans[:, :2], delta)
                bary = np.column_stack([bary, 1.0 - bary.sum(axis=1)])
                verts = self._tri.simplices[s]
                idx[:, inside] = self._valid[verts].T
                w[:, inside] = np.clip(bary, 0.0, None).T
        else:
            inside = np.zeros(q.shape[0], dtype=bool)

        # Outside the hull (and when the triangulation degenerates): nearest window.
        if (~inside).any():
            _, near = self._tree.query(q[~inside], k=1)
            idx[0, ~inside] = self._valid[near]
            w[0, ~inside] = 1.0

        total = w.sum(axis=0)
        np.divide(w, np.where(total > EPS, total, 1.0), out=w)
        return idx.reshape(3, ny, nx), w.reshape(3, ny, nx)
