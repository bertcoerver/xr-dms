"""Sharpen a VIIRS thermal swath onto a Sentinel-2 UTM grid.

The real pairing this feature exists for, using the two Zarr stores that
``lazy-dino``'s ``examples/combine_clms_laads.py`` writes:

* **thermal** -- a ``DataTree`` with one node per VIIRS overpass. Each node is a
  375 m brightness-temperature scene in the granule's own scan geometry, carrying
  2-D ``latitude``/``longitude`` and no projection at all.
* **optical** -- a Sentinel-2 L2A ``Dataset`` on a 20 m UTM zone 36N grid, with
  ``(time, y, x)`` dims and a CF grid mapping in ``spatial_ref``.

:class:`~xr_dms.SwathGridMap` maps every optical pixel to its parent swath pixel,
so the sharpened field lands on the optical grid untouched -- same dims, coords
and CRS -- while the thermal observations are never resampled.

Choosing which optical acquisition goes with which overpass is a data question,
not a sharpening one, so it lives here rather than in ``xr_dms``.
"""

import os

import numpy as np
import xarray as xr

from xr_dms import Sharpener, SwathGridMap

WORKDIR = "/Users/hmcoerver/Local/lazy_dino_tests"
EXTENT = "30.3_31.0_29.0_29.7"
PERIOD = "2025-02-01_2025-05-01"

S2_STORE = os.path.join(WORKDIR, f"S2_{EXTENT}_{PERIOD}.zarr")
VIIRS_STORE = os.path.join(WORKDIR, f"VIIRS_{EXTENT}_{PERIOD}.zarr")

# lazy-dino prefixes each variable with its source collection.
I05 = "VNP02IMG_2.observation_data_I05"
CLOUD = "CLDMSK_L2_VIIRS_SNPP_1.geophysical_data_Integer_Cloud_Mask"
LAT = "VNP03IMG_2.geolocation_data_latitude"
LON = "VNP03IMG_2.geolocation_data_longitude"
BANDS = {
    "blue": "sentinel-2-l2a.B02_20m",
    "green": "sentinel-2-l2a.B03_20m",
    "red": "sentinel-2-l2a.B04_20m",
    "nir": "sentinel-2-l2a.B8A_20m",
}
SCL = "sentinel-2-l2a.SCL_20m"
# Sentinel-2 scene classes worth keeping: dark, vegetation, bare, water,
# unclassified. Dropped: nodata, saturated, shadow, cloud (medium/high), cirrus.
SCL_KEEP = (2, 4, 5, 6, 7)
# VIIRS cloud mask: 0 cloudy, 1 probably cloudy, 2 probably clear, 3 confident clear.
CLOUD_KEEP = 2

# A window of the full 3949x3478 grid keeps the demo quick; set to None for the
# whole scene.
WINDOW = dict(y=slice(1200, 2600), x=slice(900, 2300))


def scene_time(node_name):
    """``A2025032_1124`` -> a numpy datetime64 (year, day-of-year, HHMM UTC)."""
    date_part, hhmm = node_name.lstrip("A").split("_")
    year, doy = int(date_part[:4]), int(date_part[4:])
    day = np.datetime64(f"{year}-01-01") + np.timedelta64(doy - 1, "D")
    return day + np.timedelta64(int(hhmm[:2]), "h") + np.timedelta64(int(hhmm[2:]), "m")


def main():
    optical_all = xr.open_zarr(S2_STORE, consolidated=True)
    thermal_all = xr.open_datatree(
        VIIRS_STORE, engine="zarr", consolidated=True, chunks={}
    )

    # -- pair one overpass with the nearest optical acquisition --------------
    nodes = list(thermal_all.children)
    times = np.array([scene_time(n) for n in nodes])
    opt_times = optical_all["time"].values
    gaps = np.abs(times[:, None] - opt_times[None, :]).min(axis=1)
    pick = int(np.argmin(gaps))
    node = nodes[pick]
    k = int(np.argmin(np.abs(opt_times - times[pick])))
    print(f"thermal {node} ({times[pick]}) <- optical {opt_times[k]} "
          f"(gap {gaps[pick] / np.timedelta64(1, 'h'):.1f} h)")

    thermal = thermal_all[node].ds
    optical = optical_all.isel(time=k, drop=True)
    if WINDOW:
        optical = optical.isel(**WINDOW)

    # -- masking -------------------------------------------------------------
    clear = optical[SCL].isin(SCL_KEEP)
    features = xr.Dataset(
        {name: optical[var].where(clear) for name, var in BANDS.items()},
        coords={"y": optical["y"], "x": optical["x"],
                "spatial_ref": optical["spatial_ref"]},
    )
    total = features["nir"] + features["red"]
    features["ndvi"] = (features["nir"] - features["red"]) / total.where(total != 0)

    lst = thermal[I05].where(thermal[CLOUD] >= CLOUD_KEEP)

    # -- the map, and the sharpening ----------------------------------------
    grid_map = SwathGridMap.from_lonlat(
        thermal[LON], thermal[LAT], fine=features,
        # Drop thermal pixels whose optical footprint is mostly cloud.
        min_fine_fraction=0.5,
    )
    print(f"swath {grid_map.coarse_shape} -> fine "
          f"{features.sizes['y']}x{features.sizes['x']}, "
          f"coverage {grid_map.coverage:.1%}")

    # lazy-dino stores I05 already scaled to radiance (W/m^2/sr/um), and the
    # brightness-temperature LUT is indexed by the raw DN it has consumed. Radiance
    # averages linearly, so leave disaggregating_temperature off -- turn it on only
    # for a target already in Kelvin, where averaging must happen as T**4.
    sharpened = Sharpener(
        grid_map=grid_map,
        disaggregating_temperature=False,
        smooth_residual=False,  # exactly mass-conserving
    ).sharpen(features, lst).compute()

    # -- checks --------------------------------------------------------------
    assert sharpened.dims == ("y", "x")
    np.testing.assert_array_equal(sharpened["y"].values, features["y"].values)
    np.testing.assert_array_equal(sharpened["x"].values, features["x"].values)

    back = grid_map.aggregate_mean(sharpened)
    obs = grid_map.prepare_target(lst)
    both = np.isfinite(back.values) & np.isfinite(obs.values)
    err = np.abs(back.values[both] - obs.values[both]).max()

    valid = np.isfinite(sharpened.values)
    unit = "W/m2/sr/um"
    print(f"output   : {sharpened.shape}, {valid.mean():.1%} finite")
    print(f"range    : {np.nanmin(sharpened.values):.2f} .. "
          f"{np.nanmax(sharpened.values):.2f} {unit} "
          f"(input {float(obs.min()):.2f} .. {float(obs.max()):.2f})")
    print(f"mass cons: max |re-aggregated - observed| = {err:.2e} {unit} "
          f"over {both.sum()} thermal pixels")
    return sharpened


if __name__ == "__main__":
    main()
