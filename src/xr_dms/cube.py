"""Sharpening a whole time series as one graph, with one task per scene.

:class:`~xr_dms.sharpener.Sharpener` is deliberately per-scene: one grid map, one
model, one coarse observation. Building a time series out of it by concatenating
per-scene results works, but the graph then grows with the series -- measured at
34 layers per overpass, so 1429 for a season -- and that is where dask's culling
stops being dependable. The same selection off the same pipeline, rebuilt in
fresh processes, optimised to 27 tasks on a good draw and 822 on a bad one, while
the identical selection off a single-scene graph came back at exactly 23 every
time. See :mod:`xr_dms._graph` for why.

This module builds the same cube with a layer count that does not depend on how
many scenes are in it. The split is the one DMS already has:

* **one deferred task per scene** covers everything scene-global -- reading the
  coarse observation, building the grid map, fitting the regression, measuring
  the coarse residual (:meth:`~xr_dms.sharpener.Sharpener.scene_bundle`). None of
  this ever culled spatially in the first place; a global regression needs global
  statistics.
* **three blockwise operations over the whole cube** cover the fine half --
  predict, gather the residual, add. One layer each, for every scene at once.

The consequence worth knowing about is that the scene-global half is now opaque
to the graph: the swath read happens inside the task, so nothing outside it needs
to know the swath's shape. That is what lets a caller assemble the cube without
having read a single granule header -- the fine grid comes from the optical
store, and the only swath-shaped object, the 1-D coarse residual, is carried as a
dask array of *unknown* length, which the gather indexes into without ever
needing to know how long it is.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._graph import bundled_arrays
from .sharpener import from_radiance, to_radiance, _predict_block, _unbox

__all__ = ["sharpen_cube"]


def _scene_bundle_block(bundle, *args):
    """One scene's coarse half, shaped as one block of each output array.

    Module level so it pickles by reference. ``bundle`` returns
    ``(labels, state)``; this adds the leading scene axis and splits the coarse
    residual out of the state, because the three travel as arrays of different
    shape (see :func:`~xr_dms._graph.bundled_arrays`).
    """
    labels, state = bundle(*args)
    residual = np.asarray(state.residual_coarse, dtype=float).ravel()
    return (
        np.asarray(labels)[None],
        np.array([state], dtype=object),
        residual[None],
    )


def _predict_scene_block(arr, states, band_order):
    """:func:`~xr_dms.sharpener._predict_block` over a leading scene axis.

    ``arr`` is ``(n_scenes, ..., n_bands)`` and ``states`` is ``(n_scenes,)``.
    Each scene has its own model, so they cannot share one call; looping is the
    honest expression of that, and the loop is over dask chunks of the time axis,
    which are one scene long in practice.
    """
    states = np.atleast_1d(np.asarray(states, dtype=object))
    return np.stack([
        _predict_block(arr[i], _unbox(states[i]), band_order)
        for i in range(arr.shape[0])
    ])


def _gather_scene_block(labels, residual):
    """Block-constant upsample of each scene's coarse residual onto its labels.

    ``residual`` is ``(n_scenes, n_coarse)`` with ``n_coarse`` unknown to the
    graph -- which is fine, because a label is only ever an index into it. Label
    ``-1`` marks a fine pixel no coarse cell claims and lands on the appended NaN.
    """
    return np.stack([
        np.append(residual[i], np.nan)[labels[i]]
        for i in range(labels.shape[0])
    ])


def sharpen_cube(sharpener, features, scene_args, bundle,
                 time_dim="time", task_group="dms_scene"):
    """The sharpened cube for a whole time series, as one lazy DataArray.

    Parameters
    ----------
    sharpener : Sharpener
        Supplies the options only -- ``band_dim``, ``disaggregating_temperature``
        and the feature handling. It is neither fitted nor mutated here; the
        models live in the deferred bundles.
    features : xarray.DataArray or xarray.Dataset
        The fine cube, dims ``(time_dim, y, x)`` (plus a band dim, or one
        variable per band). Must be dask-backed and chunked one scene at a time
        along ``time_dim``.
    scene_args : sequence of tuple
        One entry per scene, in ``features[time_dim]`` order, holding the
        positional arguments for ``bundle``.
    bundle : callable
        ``bundle(*args) -> (labels, state)`` for one scene, as
        :meth:`~xr_dms.sharpener.Sharpener.scene_bundle` returns. Called inside
        the graph, once per scene, and must be picklable and pure -- dask offers
        no guarantee that a task runs only once, and a residual measured against
        one model but applied to another silently stops conserving mass.

    Returns
    -------
    xarray.DataArray
        Dims ``(time_dim, y, x)``, carrying ``features``' coordinates. Nothing is
        computed: selecting a scene costs that scene's bundle and the blocks
        asked for.
    """
    if sharpener.window_size > 0:
        raise NotImplementedError(
            "sharpen_cube does not support moving-window models "
            "(window_size > 0); use Sharpener.sharpen()."
        )
    if sharpener.smooth_residual:
        raise NotImplementedError(
            "sharpen_cube applies the residual block-constant; it needs "
            "smooth_residual=False."
        )

    import dask.array as dsa

    fine = sharpener._as_features(features, extra_dims=(time_dim,))
    band_dim = sharpener.band_dim
    band_order = [str(b) for b in fine[band_dim].values]
    fine = fine.sel({band_dim: band_order})

    y_dim, x_dim = sharpener.y_dim, sharpener.x_dim
    n_scenes = fine.sizes[time_dim]
    if len(scene_args) != n_scenes:
        raise ValueError(
            f"{len(scene_args)} scene argument tuple(s) for {n_scenes} step(s) "
            f"along {time_dim!r}."
        )
    if fine.chunks is None:
        raise ValueError("sharpen_cube needs a dask-backed `features`.")
    time_chunks = fine.chunks[fine.dims.index(time_dim)]
    if set(time_chunks) != {1}:
        raise ValueError(
            f"`features` must be chunked one scene at a time along {time_dim!r} "
            f"(got chunks {time_chunks}); each scene has its own model."
        )

    ny, nx = fine.sizes[y_dim], fine.sizes[x_dim]
    labels, states, residual = bundled_arrays(
        task_group, _scene_bundle_block,
        {(t,): tuple(args) for t, args in enumerate(scene_args)},
        [
            (((1,) * n_scenes, (ny,), (nx,)), np.int64),
            (((1,) * n_scenes,), object),
            (((1,) * n_scenes, (np.nan,)), float),
        ],
        # Hoisted rather than repeated per scene: a bundle typically closes over
        # the fine cube, so embedding it in each task would ship that cube's
        # whole graph n_scenes times to the scheduler.
        shared_arg=bundle,
    )

    guess = xr.apply_ufunc(
        _predict_scene_block,
        fine,
        xr.DataArray(states, dims=[time_dim]),
        kwargs={"band_order": band_order},
        input_core_dims=[[band_dim], []],
        dask="parallelized",
        output_dtypes=[float],
    )

    # The labels come out of one task per scene, so they arrive whole; rechunking
    # to the fine grid is what makes the gather blockwise rather than a single
    # 110 MB block travelling to every worker.
    fine_chunks = (
        (1,) * n_scenes,
        fine.chunks[fine.dims.index(y_dim)],
        fine.chunks[fine.dims.index(x_dim)],
    )
    residual_fine = dsa.blockwise(
        _gather_scene_block, "tyx",
        labels.rechunk(fine_chunks), "tyx",
        residual, "tc",
        concatenate=True, dtype=float,
    )
    residual_fine = xr.DataArray(
        residual_fine, dims=(time_dim, y_dim, x_dim),
        coords={d: guess[d] for d in (time_dim, y_dim, x_dim) if d in guess.coords},
    )

    disagg = sharpener.disaggregating_temperature
    corrected = (to_radiance(guess) if disagg else guess) + residual_fine
    if disagg:
        corrected = from_radiance(corrected)
    return corrected.rename("sharpened")
