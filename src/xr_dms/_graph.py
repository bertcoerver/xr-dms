"""Single-layer dask graph construction for the deferred per-scene work.

The idiom this replaces -- one ``dask.delayed`` plus a ``da.from_delayed`` per
output per scene -- costs about fourteen single-task ``MaterializedLayer``s per
scene to deliver two arrays, and the layer count then grows with the length of
the time series. Measured on a 46-overpass cube: 1429 layers, of which roughly
40% were that scaffolding.

That is not merely untidy. Graph *construction* is where dask's culling becomes
unreliable: the same selection off the same pipeline, rebuilt in fresh
processes, optimised to 27 tasks on a good draw and to 822 on a bad one, while
the identical selection off a *one-scene* graph (34 layers) came back at exactly
23 tasks in twenty consecutive runs. Layer names feed set iteration inside the
optimiser, so any name that is not a pure function of the data -- and
``dask.base.tokenize`` of a fitted sklearn model is not, it differs between two
unpickles of the same bytes -- reshuffles that iteration and changes what slice
pushdown manages to prove. Pinning ``PYTHONHASHSEED`` does not help, because the
randomness is in the names themselves, not in the hashing of them.

So the fix is structural rather than a hunt for the offending tokens: build one
hand-made layer whose keys *are* the block lattice, and let the whole time
series live in it. A cube then contributes O(1) layers however many scenes it
holds, and because a key is its own block index, culling a selection is exact by
construction rather than by inference.

The technique is lifted from ``lazy_dino.readers._graph``, which arrived at it
for the same reason from the other end of the pipeline (100k+ layers over
windowed reads). It is reimplemented here rather than imported: ``xr_dms``
depends on neither ``lazy_dino`` nor anything that reads a granule.

- :func:`block_array` is the core: one task per block, one layer.
- :func:`bundled_arrays` covers the case this package actually needs -- one task
  per scene whose tuple output feeds several arrays *of different shape*
  (``(t, y, x)`` labels, ``(t,)`` states, ``(t, ...)`` coarse residuals of
  unknown length). One shared layer plus one thin getter layer per output.

``dask._task_spec.Task``/``TaskRef`` and ``MaterializedLayer`` are semi-private
dask API; the imports are isolated in this module and canaried by
``tests/test_graph.py``.
"""

from __future__ import annotations

import operator
from math import prod
from typing import Callable, Sequence

import numpy as np
import dask.array as da
from dask._task_spec import DataNode, Task, TaskRef
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer

__all__ = ["block_array", "bundled_arrays"]


def _check_blocks(args_by_block: dict, chunks: tuple) -> None:
    n_expected = prod(len(c) for c in chunks) if chunks else 0
    if len(args_by_block) != n_expected:
        raise ValueError(
            f"args_by_block has {len(args_by_block)} entries but chunks imply "
            f"{n_expected} blocks"
        )


def _meta(dtype, ndim):
    return np.empty((0,) * ndim, dtype=np.dtype(dtype))


def block_array(task_group, func: Callable, args_by_block: dict, chunks, dtype):
    """A lazy N-D dask array from one task per block, in a single layer.

    ``func`` returns the full-dimensionality block for its index;
    ``args_by_block`` maps every block index implied by ``chunks`` to that
    call's positional arguments.

    The layer name is tokenised over the arguments themselves, so two builds of
    the same work produce the same name -- which is the whole point, see the
    module docstring.
    """
    _check_blocks(args_by_block, chunks)

    token = tokenize(task_group, sorted(args_by_block.items()), chunks, str(dtype))
    name = f"{task_group or 'block'}-{token}"

    dsk = {
        (name, *idx): Task((name, *idx), func, *args)
        for idx, args in args_by_block.items()
    }
    graph = HighLevelGraph.from_collections(
        name, MaterializedLayer(dsk), dependencies=()
    )
    return da.Array(graph, name, chunks, meta=_meta(dtype, len(chunks)))


_NO_SHARED = object()


def bundled_arrays(
    task_group,
    func: Callable,
    args_by_block: dict[tuple[int, ...], tuple],
    outputs: Sequence[tuple],
    shared_arg=_NO_SHARED,
) -> list[da.Array]:
    """Several lazy arrays of differing shape, fed by one shared task per block.

    ``func`` returns a tuple of ``len(outputs)`` values for each entry of
    ``args_by_block``. Each output is declared as ``(chunks, dtype)``; its
    leading axes must be the shared block lattice (matching the keys of
    ``args_by_block``, one chunk per entry) and every axis beyond those must be
    a single chunk, since one task produces one whole block of each output.

    That last rule is what lets the three outputs disagree about shape. The
    per-scene bundle returns a ``(y, x)`` label array, a scalar state object and
    a coarse residual whose length is not known until the swath has been read;
    as arrays over the time axis those are ``(t, y, x)``, ``(t,)`` and
    ``(t, nan)``, sharing only the ``t``.

    ``shared_arg``, when given, is one value passed to ``func`` ahead of each
    block's own arguments -- and it is stored **once**, in a layer of its own,
    which every block task refers to. That is a size decision, not a
    convenience: an argument embedded in ``args_by_block`` is serialised
    separately for every task on the way to the scheduler, so a per-scene
    bundle closing over a lazy cube would ship that whole cube's graph once per
    scene (measured at ~120 MB over 46 overpasses, which is what dask's "large
    graph" warning was about). As a single key it is serialised once and the
    scheduler hands the same object to each task.

    Returns one array per output: ``1 + len(outputs)`` layers in total (plus one
    for ``shared_arg``). Culling any output's blocks culls the shared tasks with
    them, and dask's fusion usually inlines the getter back into the shared task.
    """
    lead = len(next(iter(args_by_block)))
    token = tokenize(
        task_group, sorted(args_by_block.items()),
        [(chunks, str(dtype)) for chunks, dtype in outputs],
        *([] if shared_arg is _NO_SHARED else [shared_arg]),
    )
    group = task_group or "block"
    shared = f"{group}_bundle-{token}"

    layers: dict = {}
    dependencies: dict = {}

    # The hoisted constant, as its own single-key layer the block tasks depend
    # on (a DataNode is the task-spec way to put a plain value in a graph).
    if shared_arg is _NO_SHARED:
        prefix: tuple = ()
        shared_deps: set = set()
    else:
        const = f"{group}_shared-{token}"
        layers[const] = MaterializedLayer({const: DataNode(const, shared_arg)})
        dependencies[const] = set()
        prefix = (TaskRef(const),)
        shared_deps = {const}

    layers[shared] = MaterializedLayer({
        (shared, *idx): Task((shared, *idx), func, *prefix, *args)
        for idx, args in args_by_block.items()
    })
    dependencies[shared] = shared_deps

    arrays = []
    for j, (chunks, dtype) in enumerate(outputs):
        _check_blocks(args_by_block, chunks[:lead])
        trailing = chunks[lead:]
        if any(len(c) != 1 for c in trailing):
            raise ValueError(
                f"output {j} is chunked beyond the shared block lattice "
                f"({trailing}); one task produces one whole block of each "
                f"output, so only the leading {lead} axis/axes may be split."
            )
        tail = (0,) * len(trailing)
        name = f"{group}_{j}-{token}"
        layers[name] = MaterializedLayer({
            (name, *idx, *tail): Task(
                (name, *idx, *tail), operator.getitem, TaskRef((shared, *idx)), j,
            )
            for idx in args_by_block
        })
        dependencies[name] = {shared}
        arrays.append((name, chunks, dtype))

    graph = HighLevelGraph(layers, dependencies)
    return [
        da.Array(graph, name, chunks, meta=_meta(dtype, len(chunks)))
        for name, chunks, dtype in arrays
    ]
