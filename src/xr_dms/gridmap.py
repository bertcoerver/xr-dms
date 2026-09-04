"""The fine<->coarse grid relationship, as a swappable object.

:class:`Sharpener` needs exactly three geometric primitives, and nothing else
about how the two grids relate:

    * **aggregate** fine data onto the coarse grid (block mean and std),
    * **upsample** a coarse field back onto the fine grid,
    * a **window basis** that says, for each fine pixel, which moving-window
      models cover it and with what weight.

A :class:`GridMap` supplies those three. :class:`RegularGridMap` implements them
for the classic co-registered case -- an integer coarsening factor, so
aggregation is :meth:`xarray.DataArray.coarsen` and upsampling is
:meth:`xarray.DataArray.interp` on 1-D coordinates. :class:`~xr_dms.geo.SwathGridMap`
implements the same three for a curvilinear swath observed over a projected fine
grid, where no integer factor exists.

The coarse side lives in its own **dimension namespace** (``coarse_dims``). For
the regular map that is just the fine dim names, since the grids are
co-registered. For a swath it must not be: the swath's ``y``/``x`` are scan row
and column while the fine grid's ``y``/``x`` are projected northing and easting,
and letting xarray align those two would silently yield an empty intersection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .aggregation import (
    coarsen_mean_std,
    infer_factor,
    upsample,
    window_basis_1d,
)

__all__ = ["GridMap", "RegularGridMap", "WindowBasis", "SeparableWindowBasis"]


class WindowBasis(ABC):
    """Per-fine-pixel weights over the moving-window models.

    Built once at :meth:`Sharpener.fit` from *global* window geometry, then
    evaluated per dask block, which is what makes ``predict_local``
    chunk-invariant: a block's weights depend only on its own coordinates and the
    global geometry, never on how the array happens to be tiled.
    """

    @abstractmethod
    def weights(self, block, y_dim, x_dim):
        """Return ``(window_index, weight)``, both shaped ``(k, ny, nx)``.

        ``window_index`` holds flat window indices (``iy * n_wx + ix``), or ``-1``
        where the corresponding weight is meaningless. ``k`` is the small maximum
        number of windows that can cover one pixel (4 for a separable tent, 3 for
        barycentric, 1 for hard assignment).
        """


class SeparableWindowBasis(WindowBasis):
    """Outer product of two 1-D bases -- the rectilinear case.

    Valid only when the window cells tile the plane as a product of x- and
    y-intervals, which is true for a regular coarse grid and false for a swath.
    """

    def __init__(self, centers_y, edges_y, centers_x, edges_x, smooth):
        self.centers_y = centers_y
        self.edges_y = edges_y
        self.centers_x = centers_x
        self.edges_x = edges_x
        self.smooth = smooth
        self.n_wy = len(centers_y)
        self.n_wx = len(centers_x)

    def weights(self, block, y_dim, x_dim):
        yc = np.asarray(block[y_dim].values, dtype=float)
        xc = np.asarray(block[x_dim].values, dtype=float)
        by = window_basis_1d(yc, self.centers_y, self.edges_y, self.smooth)
        bx = window_basis_1d(xc, self.centers_x, self.edges_x, self.smooth)

        # A tent basis is non-zero for at most 2 windows per axis, so at most 4
        # windows cover any pixel; the hard basis reduces to 1. Rather than
        # materialise (n_wy * n_wx, ny, nx), keep the k largest contributions.
        ny, nx = yc.size, xc.size
        ky = min(2, self.n_wy)
        kx = min(2, self.n_wx)
        top_y = np.argsort(by, axis=0)[-ky:]  # (ky, ny)
        top_x = np.argsort(bx, axis=0)[-kx:]  # (kx, nx)

        idx = np.full((ky * kx, ny, nx), -1, dtype=np.int64)
        w = np.zeros((ky * kx, ny, nx), dtype=float)
        for a in range(ky):
            iy = top_y[a]  # (ny,)
            wy = by[iy, np.arange(ny)]  # (ny,)
            for b in range(kx):
                ix = top_x[b]  # (nx,)
                wx = bx[ix, np.arange(nx)]  # (nx,)
                slot = a * kx + b
                idx[slot] = iy[:, None] * self.n_wx + ix[None, :]
                w[slot] = wy[:, None] * wx[None, :]
        idx = np.where(w > 0, idx, -1)
        return idx, w


class GridMap(ABC):
    """How a fine grid and a coarse observation relate to each other."""

    #: ``(y, x)`` dimension names of the coarse namespace.
    coarse_dims: tuple

    #: ``(ny, nx)`` shape of the coarse grid.
    coarse_shape: tuple

    #: True when the map's own geometry is still in the dask graph, so callers
    #: know not to force it (see :meth:`SwathGridMap.from_lonlat`'s ``lazy``).
    lazy = False

    @abstractmethod
    def prepare_target(self, target):
        """Validate the coarse observation and put it in the coarse namespace."""

    @abstractmethod
    def aggregate(self, fine, band_dim="band"):
        """Fine -> coarse block mean and standard deviation (``ddof=0``)."""

    @abstractmethod
    def aggregate_mean(self, fine):
        """Fine -> coarse block mean only (the residual and blend paths)."""

    @abstractmethod
    def upsample(self, coarse, like, method="linear"):
        """Coarse -> fine.

        ``method="nearest"`` must reproduce the aggregation partition exactly, so
        that aggregating an upsampled field returns it unchanged -- that identity
        is what makes ``smooth_residual=False`` exactly mass-conserving.
        """

    @abstractmethod
    def build_window_basis(self, n_wy, n_wx, window_size, smooth, fine):
        """Geometry for the moving-window blend (see :class:`WindowBasis`)."""


class RegularGridMap(GridMap):
    """Co-registered grids related by an integer coarsening factor.

    The classic case, and the default: each coarse pixel covers a ``factor x
    factor`` block of fine pixels, aggregation is a plain block reduction and
    upsampling is interpolation on 1-D coordinates.
    """

    def __init__(self, factor, coarse_shape, coarse_coords, x_dim="x", y_dim="y",
                 boundary="exact"):
        self.factor = int(factor)
        self.coarse_shape = tuple(coarse_shape)
        self._coarse_coords = coarse_coords
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.boundary = boundary
        self.coarse_dims = (y_dim, x_dim)

    @classmethod
    def from_grids(cls, fine, target, x_dim="x", y_dim="y", boundary="exact"):
        factor = infer_factor(fine, target, x_dim, y_dim, boundary)
        coarse_coords = {
            dim: target[dim].values
            for dim in (y_dim, x_dim)
            if dim in target.coords
        }
        return cls(
            factor=factor,
            coarse_shape=(target.sizes[y_dim], target.sizes[x_dim]),
            coarse_coords=coarse_coords,
            x_dim=x_dim, y_dim=y_dim, boundary=boundary,
        )

    def _match_coarse_coords(self, coarse):
        """Assign the target's spatial coords onto a coarsened array.

        Block-averaged coordinates from :meth:`~xarray.DataArray.coarsen` are the
        block centres, which equal the coarse pixel coordinates only up to
        floating-point noise. Overwriting them positionally lets xarray align the
        coarse arrays with the target without spurious NaNs.
        """
        for dim, size in zip((self.y_dim, self.x_dim), self.coarse_shape):
            if coarse.sizes[dim] != size:
                raise ValueError(
                    f"Coarsened size {coarse.sizes[dim]} along {dim!r} does not "
                    f"match target size {size}. Check the grids are "
                    "co-registered and the factor is correct."
                )
        return (
            coarse.assign_coords(self._coarse_coords)
            if self._coarse_coords else coarse
        )

    def prepare_target(self, target):
        for dim, size in zip((self.y_dim, self.x_dim), self.coarse_shape):
            if target.sizes[dim] != size:
                raise ValueError(
                    f"Target size {target.sizes[dim]} along {dim!r} does not match "
                    f"the coarse grid size {size}."
                )
        return target

    def aggregate(self, fine, band_dim="band"):
        mean, std = coarsen_mean_std(
            fine, self.factor, self.x_dim, self.y_dim, self.boundary
        )
        return self._match_coarse_coords(mean), self._match_coarse_coords(std)

    def aggregate_mean(self, fine):
        coarse = fine.coarsen(
            {self.y_dim: self.factor, self.x_dim: self.factor},
            boundary=self.boundary,
        ).mean()
        return self._match_coarse_coords(coarse)

    def upsample(self, coarse, like, method="linear"):
        return upsample(coarse, like, self.x_dim, self.y_dim, method=method)

    def build_window_basis(self, n_wy, n_wx, window_size, smooth, fine):
        ny, nx = self.coarse_shape
        fy = np.asarray(fine[self.y_dim].values, dtype=float)
        fx = np.asarray(fine[self.x_dim].values, dtype=float)
        centers_y, edges_y = _axis_geometry(n_wy, ny, fy, window_size, self.factor)
        centers_x, edges_x = _axis_geometry(n_wx, nx, fx, window_size, self.factor)
        return SeparableWindowBasis(centers_y, edges_y, centers_x, edges_x, smooth)


def _axis_geometry(n_win, n_coarse, coord, window_size, factor):
    """Per-window prediction-cell centres and edges, in fine coordinates.

    The outermost cells extend to infinity so that the cells partition the whole
    axis. Which *end* gets ``+inf`` depends on the axis direction: on a north-up
    raster ``y`` descends, so window 0 sits at the largest coordinate and its open
    side is the upper one. Orienting by the sign of the step keeps the hard
    (``smooth_local=False``) basis correct for both orientations.
    """
    w, f = window_size, factor
    ascending = coord.size < 2 or coord[-1] >= coord[0]
    centers = np.empty(n_win, dtype=float)
    edges = np.empty((n_win, 2), dtype=float)
    for i in range(n_win):
        lo_c = i * w
        hi_c = min((i + 1) * w, n_coarse)
        lo_f = lo_c * f
        hi_f = hi_c * f  # exclusive fine index
        centers[i] = coord[lo_f:hi_f].mean()
        first, last = i == 0, hi_c >= n_coarse
        # The cell's boundary shared with the previous / next window, at the
        # midpoint between the two adjacent fine pixel centres.
        start = -np.inf if first else 0.5 * (coord[lo_f - 1] + coord[lo_f])
        stop = np.inf if last else 0.5 * (coord[hi_f - 1] + coord[hi_f])
        if not ascending:
            # Descending axis: window 0 is at the high end, so the open side flips.
            start = np.inf if first else start
            stop = -np.inf if last else stop
        edges[i] = (min(start, stop), max(start, stop))
    return centers, edges
