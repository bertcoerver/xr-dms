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

> **In a conda environment, take `pyresample` from conda-forge:**
> ```bash
> conda install -c conda-forge pyresample pykdtree
> ```
> The pip wheel of `pykdtree` (which `pyresample` pulls in) vendors its own
> `libomp`, which collides with the OpenMP runtime a conda environment already
> has. The result is `OMP: Error #15` and an aborted process the first time
> `SwathGridMap` runs its neighbour search.

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
  `features` by an integer coarsening factor — or on a curvilinear swath, via a
  `SwathGridMap` (see below).
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

### Sharpening a swath onto a projected grid

A thermal swath (VIIRS, MODIS, SLSTR) carries a 2-D `latitude`/`longitude` pair
per scan pixel and has no integer relationship to an optical grid in a projected
CRS: the footprint grows off-nadir, the scan is skewed, and the two do not share
a coordinate system. Pass a `SwathGridMap` and the pairing works anyway:

```python
from xr_dms import Sharpener, SwathGridMap

optical = optical_out.isel(time=k)          # (y, x) on a UTM grid, with spatial_ref
thermal = thermal_out["A2025077_1036"].ds   # (y, x) scan geometry + 2-D lat/lon

grid_map = SwathGridMap.from_lonlat(
    thermal["longitude"], thermal["latitude"], fine=optical,
    min_fine_fraction=0.5,   # drop thermal pixels that are mostly cloud
)
sharpened = Sharpener(grid_map=grid_map).sharpen(optical, thermal["I05"])
```

The result lands on **exactly** the optical grid — same dims, coords, shape and
CRS — with `NaN` wherever the swath does not reach.

The observations are never resampled. `SwathGridMap` maps each fine pixel to its
parent swath pixel (a Voronoi tessellation of the swath pixel centres), which
makes aggregation a labelled reduction and block-constant upsampling a gather, so
`smooth_residual=False` stays *exactly* mass-conserving — re-aggregating the
sharpened field through the same map reproduces the raw observations to ~1e-14.

Set `disaggregating_temperature=True` only when the target really is in Kelvin;
a target already in radiance averages linearly and needs no `T**4`.

See [`examples/sharpen_xarray.py`](examples/sharpen_xarray.py) for a complete,
runnable end-to-end example on the bundled sample scene,
[`examples/sharpen_swath.py`](examples/sharpen_swath.py) for the VIIRS-onto-Sentinel-2
pairing above, and [`examples/dms_explained_numpy.py`](examples/dms_explained_numpy.py)
for a low-level, GDAL-free walk-through of the algorithm.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Release

```bash
./tag_release.sh patch   # or minor / major
```
