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

import numpy as np
import xarray as xr

from .aggregation import EPS
from .gridmap import GridMap, WindowBasis

__all__ = ["SwathGridMap"]

#: Private coarse-namespace dim names. The swath's own ``y``/``x`` are scan row
#: and column while the fine grid's are projected northing and easting; letting
#: xarray align those would silently produce an empty intersection.
COARSE_Y = "_dms_cy"
COARSE_X = "_dms_cx"


# -- fine grid description ---------------------------------------------------
def _crs_from(obj):
    """Find a CRS on an xarray object, without requiring rioxarray.

    Looks for the CF grid-mapping variable that rioxarray writes (``spatial_ref``
    or ``crs``) and reads the WKT off its attributes.
    """
    from pyproj import CRS

    for name in ("spatial_ref", "crs"):
        if name in getattr(obj, "coords", {}) or name in getattr(obj, "variables", {}):
            var = obj[name]
            for key in ("crs_wkt", "spatial_ref"):
                wkt = var.attrs.get(key)
                if wkt:
                    return CRS.from_user_input(wkt)
            if var.attrs:
                try:
                    return CRS.from_cf(var.attrs)
                except Exception:  # noqa: BLE001 - fall through to the error below
                    pass
    raise ValueError(
        "Could not determine the CRS of the fine grid. Pass crs=... explicitly, "
        "or attach a CF grid-mapping variable (rioxarray's .rio.write_crs())."
    )


def _axis_step(values, name):
    """Uniform step of a 1-D coordinate, or raise."""
    if values.size < 2:
        raise ValueError(f"Fine coordinate {name!r} needs at least 2 points.")
    steps = np.diff(values)
    step = steps[0]
    if not np.allclose(steps, step, rtol=1e-6, atol=abs(step) * 1e-6):
        raise ValueError(
            f"Fine coordinate {name!r} is not evenly spaced; SwathGridMap needs a "
            "regular projected grid on the fine side."
        )
    return float(step)


def _area_from_fine(fine, x_dim, y_dim, crs):
    """Build a pyresample AreaDefinition from the fine grid's 1-D coordinates.

    Returns ``(area, flip_y, flip_x)``. pyresample lays an area out north-up and
    x-ascending; the flags say how to bring its output back into the *data's* own
    row/column order, which for a north-up raster has ``y`` descending.
    """
    from pyresample.geometry import AreaDefinition

    xs = np.asarray(fine[x_dim].values, dtype=float)
    ys = np.asarray(fine[y_dim].values, dtype=float)
    dx = _axis_step(xs, x_dim)
    dy = _axis_step(ys, y_dim)

    # Pixel-edge extent from centre coordinates.
    x_lo, x_hi = xs.min() - abs(dx) / 2, xs.max() + abs(dx) / 2
    y_lo, y_hi = ys.min() - abs(dy) / 2, ys.max() + abs(dy) / 2

    area = AreaDefinition(
        "xr_dms_fine", "xr_dms fine grid", "xr_dms_fine",
        crs, xs.size, ys.size, (x_lo, y_lo, x_hi, y_hi),
    )
    # pyresample row 0 is the northernmost; flip if the data is stored the other way.
    return area, dy > 0, dx < 0


def _locate(global_coord, block_coord):
    """Integer positions of ``block_coord`` within ``global_coord``.

    Lets a per-block computation recover its offset into the global grid from its
    coordinates alone, which is what keeps the window basis chunk-invariant.
    """
    ascending = global_coord.size < 2 or global_coord[-1] >= global_coord[0]
    ref = global_coord if ascending else global_coord[::-1]
    pos = np.searchsorted(ref, block_coord)
    pos = np.clip(pos, 0, ref.size - 1)
    # searchsorted lands left of an exact hit only under floating-point noise;
    # pick whichever neighbour is actually closest.
    left = np.clip(pos - 1, 0, ref.size - 1)
    pick = np.where(
        np.abs(ref[left] - block_coord) <= np.abs(ref[pos] - block_coord), left, pos
    )
    return pick if ascending else global_coord.size - 1 - pick


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


def _accumulate(values, labels, n_coarse):
    """``(count, sum, sumsq)`` over the whole fine grid, streaming if dask-backed.

    Each dask block contributes a small ``(n_bands, n_coarse, 3)`` histogram which
    are summed, so peak memory stays per-block rather than whole-array.
    """
    if not hasattr(values, "dask"):
        return _block_stats(np.asarray(values), labels, n_coarse)

    import dask
    import dask.array as dsa

    blocks = values.to_delayed()  # (nb_band, nb_y, nb_x) object array
    y_off = np.cumsum((0,) + values.chunks[1][:-1])
    x_off = np.cumsum((0,) + values.chunks[2][:-1])
    parts = []
    for iy, (y0, ylen) in enumerate(zip(y_off, values.chunks[1])):
        for ix, (x0, xlen) in enumerate(zip(x_off, values.chunks[2])):
            lab = labels[y0:y0 + ylen, x0:x0 + xlen]
            # Bands are kept whole (the caller rechunks), so index 0 on that axis.
            part = dask.delayed(_block_stats)(blocks[0, iy, ix], lab, n_coarse)
            parts.append(
                dsa.from_delayed(
                    part, shape=(values.shape[0], n_coarse, 3), dtype=float
                )
            )
    return sum(parts).compute()


def _mean_std(stats, min_count, min_fraction, cell_size):
    """Turn ``(count, sum, sumsq)`` accumulators into mean and ``ddof=0`` std."""
    count, total, total_sq = stats[..., 0], stats[..., 1], stats[..., 2]
    enough = count >= np.maximum(min_count, min_fraction * np.maximum(cell_size, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(enough, total / np.where(count > 0, count, np.nan), np.nan)
        var = np.where(enough, total_sq / np.where(count > 0, count, np.nan) - mean ** 2, np.nan)
    return mean, np.sqrt(np.clip(var, 0.0, None))


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
                 proj_xy, min_fine_fraction=0.0, min_fine_pixels=1):
        self.labels = labels
        self.swath_shape = tuple(swath_shape)
        self.swath_dims = tuple(swath_dims)
        self.coarse_shape = tuple(swath_shape)
        self.coarse_dims = (COARSE_Y, COARSE_X)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._fine_coords = fine_coords
        self._proj_xy = proj_xy  # (sx, sy) swath centres in the fine CRS
        self.min_fine_fraction = float(min_fine_fraction)
        self.min_fine_pixels = int(min_fine_pixels)

        n = int(np.prod(self.swath_shape))
        flat = labels.ravel()
        self._cell_size = np.bincount(flat[flat >= 0], minlength=n)[:n].astype(float)
        self.coverage = float((labels >= 0).mean())

    # -- construction --------------------------------------------------------
    @classmethod
    def from_lonlat(cls, lon, lat, fine, crs=None, radius_of_influence=None,
                    x_dim="x", y_dim="y", min_fine_fraction=0.0, min_fine_pixels=1):
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
        """
        from pyproj import CRS, Transformer
        from pyresample import kd_tree
        from pyresample.geometry import SwathDefinition

        if lon.shape != lat.shape or lon.ndim != 2:
            raise ValueError(
                f"lon/lat must be matching 2-D arrays; got {lon.shape} and {lat.shape}."
            )

        crs = CRS.from_user_input(crs) if crs is not None else _crs_from(fine)
        area, flip_y, flip_x = _area_from_fine(fine, x_dim, y_dim, crs)

        lon_v = np.asarray(lon.values, dtype=float)
        lat_v = np.asarray(lat.values, dtype=float)

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

        fine_coords = {
            y_dim: np.asarray(fine[y_dim].values, dtype=float),
            x_dim: np.asarray(fine[x_dim].values, dtype=float),
        }
        return cls(
            labels=np.ascontiguousarray(labels),
            swath_shape=lon.shape,
            swath_dims=lon.dims,
            fine_coords=fine_coords,
            x_dim=x_dim, y_dim=y_dim,
            proj_xy=(sx, sy),
            min_fine_fraction=min_fine_fraction,
            min_fine_pixels=min_fine_pixels,
        )

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
        padded = np.append(values, np.nan)  # index -1 lands on the NaN pad
        lab = self.labels
        chunks = _fine_chunks(like, self.y_dim, self.x_dim)
        if chunks is None:
            return padded[lab]
        import dask.array as dsa

        return dsa.map_blocks(
            lambda block: padded[block],
            dsa.from_array(lab, chunks=chunks),
            dtype=float,
        )

    def _interpolate(self, values, like):
        """Scattered linear interpolation over the projected swath centres.

        Fine pixels outside the convex hull of the swath centres -- the outermost
        ~1% -- get the nearest coarse value instead, mirroring the ``ffill``/``bfill``
        edge extension of the regular-grid upsampler.
        """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        sx, sy = self._proj_xy
        pts = np.column_stack([sx.ravel(), sy.ravel()])
        ok = np.isfinite(pts).all(axis=1) & np.isfinite(values)
        if not ok.any():
            raise ValueError("No finite coarse values to upsample.")
        lin = LinearNDInterpolator(pts[ok], values[ok])
        near = NearestNDInterpolator(pts[ok], values[ok])

        ys = self._fine_coords[self.y_dim]
        xs = self._fine_coords[self.x_dim]

        def _eval(y_block, x_block):
            xx, yy = np.meshgrid(x_block, y_block)
            q = np.column_stack([xx.ravel(), yy.ravel()])
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
