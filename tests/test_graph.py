"""The graph helper, and a canary on the semi-private dask API it stands on.

``_graph`` exists to keep the layer count independent of the length of the time
series -- see its module docstring for the culling instability that motivated
it. These tests hold it to that: one layer per shared task group however many
scenes, exact culling, and reproducible layer names across builds.
"""

import numpy as np
import pytest

import dask
import dask.array as dsa

from xr_dms._graph import block_array, bundled_arrays


def test_dask_private_api_still_present():
    """Canary: ``Task``/``TaskRef``/``MaterializedLayer`` are semi-private.

    If dask moves or renames them this fails here, with a name to search for,
    rather than somewhere deep in a sharpening graph.
    """
    from dask._task_spec import Task, TaskRef  # noqa: F401
    from dask.highlevelgraph import HighLevelGraph, MaterializedLayer  # noqa: F401

    name = "canary"
    task = Task((name, 0), np.zeros, 3)
    layer = MaterializedLayer({(name, 0): task})
    graph = HighLevelGraph.from_collections(name, layer, dependencies=())
    array = dsa.Array(graph, name, ((3,),), meta=np.empty((0,), dtype=float))
    np.testing.assert_array_equal(array.compute(), np.zeros(3))


def _tile(t, value):
    # One task produces one whole block, so the leading (scene) axis is part of
    # what it returns -- the array is (n, 4, 4) chunked one scene at a time.
    return np.full((1, 4, 4), value + t, dtype=float)


def test_block_array_is_one_layer_per_group():
    for n in (1, 8, 64):
        arr = block_array(
            "tile", _tile,
            {(t, 0, 0): (t, 10.0) for t in range(n)},
            ((1,) * n, (4,), (4,)), float,
        )
        assert len(arr.__dask_graph__().layers) == 1
        assert arr.shape == (n, 4, 4)


def test_block_array_culls_exactly():
    n = 64
    arr = block_array(
        "tile", _tile,
        {(t, 0, 0): (t, 10.0) for t in range(n)},
        ((1,) * n, (4,), (4,)), float,
    )
    (opt,) = dask.optimize(arr[3])
    assert len(opt.__dask_graph__()) <= 2  # the block itself, plus at most a getter
    np.testing.assert_array_equal(arr[3].compute(), np.full((4, 4), 13.0))


def test_block_array_name_is_reproducible():
    """Two builds of the same work must produce the same layer names.

    This is the property the whole module exists for: a name that varies
    between builds reshuffles set iteration inside dask's optimiser and makes
    culling come out differently run to run.
    """
    def build():
        return set(block_array(
            "tile", _tile,
            {(t, 0, 0): (t, 10.0) for t in range(8)},
            ((1,) * 8, (4,), (4,)), float,
        ).__dask_graph__().layers)

    assert build() == build()


def test_block_array_rejects_missing_blocks():
    with pytest.raises(ValueError, match="args_by_block has 1 entries"):
        block_array("tile", _tile, {(0, 0, 0): (0, 1.0)},
                    ((1, 1), (4,), (4,)), float)


# -- bundled_arrays ----------------------------------------------------------

def _bundle(t):
    """One scene's outputs, deliberately of three different shapes.

    Stands in for the real per-scene bundle: a fine-grid label array, a scalar
    state object, and a coarse residual whose length is not knowable until the
    swath has been read.
    """
    labels = np.full((1, 4, 4), t, dtype=np.int64)
    state = np.array([{"scene": t}], dtype=object)
    residual = np.arange(t + 1, dtype=float)[None]
    return labels, state, residual


def _bundled(n):
    return bundled_arrays(
        "scene", _bundle, {(t,): (t,) for t in range(n)},
        [
            (((1,) * n, (4,), (4,)), np.int64),
            (((1,) * n,), object),
            (((1,) * n, (np.nan,)), float),
        ],
    )


def test_bundled_arrays_layer_count_is_independent_of_scene_count():
    for n in (1, 8, 64):
        labels, states, residual = _bundled(n)
        # One shared layer plus one getter per output, whatever n is.
        assert len(labels.__dask_graph__().layers) == 4
        assert labels.shape == (n, 4, 4)
        assert states.shape == (n,)
        assert np.isnan(residual.shape[1])


def test_bundled_arrays_values():
    labels, states, residual = _bundled(3)
    np.testing.assert_array_equal(labels[2].compute(), np.full((4, 4), 2))
    assert states[1].compute() == {"scene": 1}
    # .blocks, not [], because the residual's length is deliberately unknown:
    # ordinary slicing needs chunk sizes, block selection does not.
    np.testing.assert_array_equal(
        residual.blocks[2].compute(), np.arange(3.0)[None],
    )


def test_bundled_arrays_share_one_task_per_block():
    """Two outputs of the same scene must not run the bundle twice.

    The scene bundle fits a regressor; running it twice would give the residual
    correction a different model from the one it is correcting -- the failure
    that made mass conservation break on the first lazy implementation. What
    guarantees it here is that both getters reference the *same* shared key.
    """
    labels, _, residual = _bundled(4)
    keys = set()
    for array in (labels, residual):
        keys |= {k for k in array.__dask_graph__()
                 if isinstance(k, tuple) and "_bundle-" in k[0] and k[1] == 1}
    assert len(keys) == 1


def test_bundled_arrays_cull_exactly():
    labels, states, residual = _bundled(64)
    (opt,) = dask.optimize(labels[7])
    assert len(opt.__dask_graph__()) <= 2  # the shared task, plus at most a getter


def test_bundled_arrays_reject_chunking_past_the_lattice():
    with pytest.raises(ValueError, match="chunked beyond the shared block lattice"):
        bundled_arrays(
            "scene", _bundle, {(t,): (t,) for t in range(2)},
            [(((1, 1), (2, 2), (4,)), np.int64)],
        )


def test_unknown_length_residual_gathers_and_culls():
    """The fine half must work against a coarse axis of unknown length.

    This is what lets the swath read be deferred at all: the label array is
    shaped by the *fine* grid, which is known from the optical store, and the
    only swath-shaped object is the 1-D residual -- which the gather indexes
    into without needing to know how long it is.
    """
    def gather(lab, vals):
        return np.append(vals[0], np.nan)[lab]

    # Scene 0's residual holds one cell, so label 0 gathers it and label 1 is
    # the out-of-range sentinel that lands as NaN.
    fine = (np.arange(16).reshape(1, 4, 4) % 2)
    labels = dsa.from_array(fine, chunks=(1, 2, 2))
    (_, _, residual) = _bundled(1)
    assert np.isnan(residual.shape[1])

    out = dsa.blockwise(gather, "tyx", labels, "tyx", residual, "tc",
                        concatenate=True, dtype=float)
    assert out.shape == (1, 4, 4)
    assert out.chunks == ((1,), (2, 2), (2, 2))

    (opt,) = dask.optimize(out[:, :2, :2])
    assert len(opt.__dask_graph__()) < len(out.__dask_graph__())

    np.testing.assert_array_equal(
        out.compute(), np.append(np.array([0.0]), np.nan)[fine],
    )
