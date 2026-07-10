# xr_dms

A lazy, chunked implementation of the **Data Mining Sharpener** (Gao, Kustas &
Anderson 2012) for `xarray`/`dask`. It sharpens a coarse image (e.g. a
land-surface-temperature product) to the resolution of co-registered fine
predictor features by fitting a regression `coarse_target ~ f(coarse_features)`,
applying it to the fine features, and residual-correcting so the result stays
consistent with the coarse observation (mass conservation).

Everything is dask-friendly: pass chunked `DataArray`s and the pipeline stays
lazy end to end.

## Installation

```bash
pip install xr_dms
```

## Quickstart

```python
import xarray as xr
from xr_dms import Sharpener

features = xr.open_dataarray("fine_features.nc").chunk({"y": 512, "x": 512})
target = xr.open_dataarray("coarse_lst.nc")

sharpened = Sharpener(disaggregating_temperature=True).sharpen(features, target)
sharpened.compute()  # or .to_netcdf(...)
```

- `features` is a fine-resolution `DataArray` with a `band` dimension (or a
  `Dataset` of feature variables) and spatial dims `y`/`x`.
- `target` is the coarse `DataArray` to sharpen, on a grid co-registered with
  `features` by an integer coarsening factor.
- `disaggregating_temperature=True` aggregates/differences in radiance space
  (`T**4`), appropriate for thermal data.

### Moving-window local regression

Over a large, heterogeneous scene the feature-to-target relationship varies in
space, and a single global model underfits it. Set `window_size` (in coarse
pixels) to also fit one **local** model per window and blend it with the global
model by inverse-residual weights (Gao 2012 §2.3):

```python
sharpened = Sharpener(
    disaggregating_temperature=True,
    window_size=20,        # coarse pixels per window; 0 (default) = global only
    smooth_local=True,     # tent-blend neighbouring windows -> seamless field
).sharpen(features, target)
```

`smooth_local=True` blends neighbouring window models with a piecewise-linear
(tent) weighting so the local field is seamless by construction; `False` uses
hard, non-overlapping window cells (faithful to pyDMS). The windowing is handled
internally via `xr.map_blocks`, so the result is **chunk-invariant** — it does
not depend on how you tile the input.

See [`examples/sharpen_xarray.py`](examples/sharpen_xarray.py) for a complete,
runnable end-to-end example on the bundled sample scene, and
[`examples/dms_explained_numpy.py`](examples/dms_explained_numpy.py) for a
low-level, GDAL-free walk-through of the algorithm.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Release

```bash
./tag_release.sh patch   # or minor / major
```
