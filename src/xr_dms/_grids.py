"""Shared 1-D grid arithmetic.

Both the swath map (:mod:`xr_dms.geo`) and the feature harmoniser
(:mod:`xr_dms.harmonize`) need the same three things: read a CRS off an xarray
object, describe a regular grid from its 1-D coordinates, and locate one set of
coordinates inside another. Keeping that arithmetic here is what lets the two
agree on when two grids are "the same grid" -- a question they would otherwise
each answer slightly differently.

Grids are described by their first pixel's *outer edge* rather than its centre.
Nesting is a statement about pixel boundaries -- a 20 m band tiles a 10 m band
only if their edges coincide -- and edges make that check one subtraction
instead of a half-pixel bookkeeping exercise at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Grid", "axis_step", "crs_of", "describe", "locate", "nesting"]


def crs_of(obj, required=True):
    """Find a CRS on an xarray object, without requiring rioxarray.

    Looks for the CF grid-mapping variable that rioxarray writes (``spatial_ref``
    or ``crs``) and reads the WKT off its attributes. Returns ``None`` when
    ``required`` is ``False`` and no CRS is present, which is what lets the
    harmoniser work on a bare synthetic grid that carries no projection at all.
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
    if not required:
        return None
    raise ValueError(
        "Could not determine the CRS of the fine grid. Pass crs=... explicitly, "
        "or attach a CF grid-mapping variable (rioxarray's .rio.write_crs())."
    )


def axis_step(values, name, context="a regular grid"):
    """Uniform (signed) step of a 1-D coordinate, or raise.

    The sign is kept: a north-up raster's ``y`` descends, and every caller here
    needs to know that to place pixel edges on the correct side.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"Coordinate {name!r} needs at least 2 points.")
    steps = np.diff(values)
    step = steps[0]
    if not np.allclose(steps, step, rtol=1e-6, atol=abs(step) * 1e-6):
        raise ValueError(
            f"Coordinate {name!r} is not evenly spaced; {context} is required."
        )
    return float(step)


def locate(global_coord, block_coord):
    """Integer positions of ``block_coord`` within ``global_coord``.

    Lets a per-block computation recover its offset into the global grid from its
    coordinates alone, which is what keeps the window basis chunk-invariant. The
    harmoniser uses the same call for a different reason: on grids that nest, the
    nearest coarse *centre* is always the coarse pixel that *contains* the fine
    centre, so this doubles as the block-constant upsampling index.

    Positions are clipped to the range of ``global_coord``; callers that care
    about coordinates falling outside it must mask separately.
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


@dataclass(frozen=True)
class Grid:
    """A regular 2-D grid, described by pixel edges rather than centres."""

    crs: object | None
    #: Outer edge of the first row/column -- ``coord[0] - step / 2``.
    y_edge: float
    x_edge: float
    #: Signed pixel size. ``dy`` is negative on a north-up raster.
    dy: float
    dx: float
    ny: int
    nx: int

    @property
    def pixel_area(self):
        """``|dy * dx|`` -- the ordering key for "finest" and "coarsest"."""
        return abs(self.dy * self.dx)


def describe(obj, x_dim="x", y_dim="y", crs=None):
    """Build a :class:`Grid` from an xarray object's 1-D spatial coordinates."""
    for dim in (y_dim, x_dim):
        if dim not in obj.coords:
            raise ValueError(
                f"Object has no {dim!r} coordinate; the harmoniser needs 1-D "
                f"{y_dim!r}/{x_dim!r} pixel-centre coordinates on every band."
            )
    ys = np.asarray(obj[y_dim].values, dtype=float)
    xs = np.asarray(obj[x_dim].values, dtype=float)
    dy = axis_step(ys, y_dim)
    dx = axis_step(xs, x_dim)
    return Grid(
        crs=crs if crs is not None else crs_of(obj, required=False),
        y_edge=ys[0] - dy / 2, x_edge=xs[0] - dx / 2,
        dy=dy, dx=dx, ny=ys.size, nx=xs.size,
    )


def _axis_nesting(src_d, src_edge, ref_d, ref_edge, tol=1e-6):
    """Nesting of one axis: ``("same" | "up" | "down", factor)``, or ``None``.

    ``"up"`` means ``src`` is the *coarser* of the two and has to be upsampled to
    reach ``ref``; ``"down"`` means it is finer and has to be block-averaged.
    ``None`` means the two do not nest and the caller must fall back to a warp.
    """
    if np.sign(src_d) != np.sign(ref_d):
        return None  # flipped axis; let the general path sort it out
    ratio = abs(src_d) / abs(ref_d)
    if abs(ratio - 1.0) <= tol:
        kind, factor, fine_d = "same", 1, abs(ref_d)
    elif ratio > 1.0:
        factor = int(round(ratio))
        if factor < 2 or abs(ratio - factor) > tol:
            return None
        kind, fine_d = "up", abs(ref_d)
    else:
        inverse = 1.0 / ratio
        factor = int(round(inverse))
        if factor < 2 or abs(inverse - factor) > tol:
            return None
        kind, fine_d = "down", abs(src_d)

    # Pixel edges must coincide, measured in whole pixels of the finer grid.
    offset = (src_edge - ref_edge) / fine_d
    if abs(offset - round(offset)) > tol:
        return None
    return kind, factor


def nesting(src, ref, tol=1e-6):
    """How ``src`` nests inside ``ref``: ``(kind, (factor_y, factor_x))``.

    ``kind`` is ``"same"``, ``"up"`` or ``"down"``. Returns ``None`` when the two
    grids do not nest -- different CRS, a non-integer pixel-size ratio, misaligned
    pixel edges, or one axis coarser while the other is finer. Anisotropic factors
    are allowed: nothing downstream of the harmoniser requires ``factor_y ==
    factor_x`` (that constraint belongs to
    :func:`~xr_dms.aggregation.infer_factor`, which relates the *fine* grid to the
    *coarse target*, a different pairing entirely).
    """
    if src.crs is not None and ref.crs is not None and src.crs != ref.crs:
        return None
    y = _axis_nesting(src.dy, src.y_edge, ref.dy, ref.y_edge, tol)
    x = _axis_nesting(src.dx, src.x_edge, ref.dx, ref.x_edge, tol)
    if y is None or x is None or y[0] != x[0]:
        return None
    return y[0], (y[1], x[1])
