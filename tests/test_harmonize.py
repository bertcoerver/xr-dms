"""Feature harmonisation across resolutions.

The suite is organised around one load-bearing claim: nearest-neighbour
upsampling before aggregation is *exactly* equivalent to aggregating a band on
its own native grid. Everything the harmoniser does rests on that, so it is
pinned to floating-point noise rather than to a tolerance -- and the companion
test shows what goes wrong when the interpolation is linear instead.
"""

import numpy as np
import pytest
import xarray as xr

from scenes import make_multires_scene, make_swath_scene
from xr_dms import RegularGridMap, Sharpener, harmonize_features
from xr_dms._grids import describe, nesting


def _native_aggregate(band, factor):
    """Block mean and std of a band on its own grid, over ``factor`` blocks."""
    blocks = band.coarsen(y=factor, x=factor, boundary="exact")
    return blocks.mean(), blocks.std()


# -- the exactness guarantee -------------------------------------------------
def test_harmonized_aggregation_matches_native_aggregation(multires_scene):
    """Nearest-upsample then aggregate == aggregate natively, to the last bit.

    This is the whole justification for harmonising at the input instead of
    teaching the sharpener to aggregate each resolution on its own grid, so it is
    asserted at ``atol=1e-12`` rather than at a physical tolerance.
    """
    bands, target, _, factor = multires_scene
    features = harmonize_features(bands)

    grid_map = RegularGridMap.from_grids(features, target)
    mean, std = grid_map.aggregate(features.to_array(dim="band"))

    for name, band in bands.items():
        # How many of this band's own pixels a coarse target pixel covers.
        ratio = int(round(abs(describe(band).dy) / 10.0))
        native_mean, native_std = _native_aggregate(band, factor // ratio)
        np.testing.assert_allclose(
            mean.sel(band=name).values, native_mean.values, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            std.sel(band=name).values, native_std.values, rtol=0, atol=1e-12
        )


def test_linear_upsampling_breaks_the_exactness_guarantee(multires_scene):
    """Why "nearest" is the default: bilinear smooths, and the std moves with it.

    The block *mean* survives interpolation almost intact; the block *std* does
    not, and the std is what feeds homogeneity_cv's choice of training pixels.
    """
    bands, target, _, factor = multires_scene
    features = harmonize_features(bands, upsample="linear")

    grid_map = RegularGridMap.from_grids(features, target)
    _, std = grid_map.aggregate(features.to_array(dim="band"))

    # b60 is upsampled 6x, so it is where smoothing bites hardest.
    _, native_std = _native_aggregate(bands["b60"], factor // 6)
    assert not np.allclose(
        std.sel(band="b60").values, native_std.values, rtol=0, atol=1e-12
    )


# -- laziness ----------------------------------------------------------------
def test_stays_lazy_when_input_is_chunked(multires_scene):
    bands, _, _, _ = multires_scene
    chunked = {k: v.chunk({"y": 20, "x": 20}) for k, v in bands.items()}
    features = harmonize_features(chunked)
    assert all(features[v].chunks is not None for v in features.data_vars)


def test_chunking_does_not_change_the_result(multires_scene):
    bands, _, _, _ = multires_scene
    eager = harmonize_features(bands)
    for chunks in ({"y": 20, "x": 20}, {"y": 60, "x": 30}):
        lazy = harmonize_features({k: v.chunk(chunks) for k, v in bands.items()})
        for name in eager.data_vars:
            np.testing.assert_array_equal(
                eager[name].values, lazy[name].compute().values
            )


# -- reference-grid selection ------------------------------------------------
def test_finest_and_coarsest_pick_the_expected_reference(multires_scene):
    bands, _, _, _ = multires_scene
    finest = harmonize_features(bands, grid="finest")
    coarsest = harmonize_features(bands, grid="coarsest")

    assert finest.sizes == {"y": 120, "x": 120}      # the 10 m grid
    assert coarsest.sizes == {"y": 20, "x": 20}      # the 60 m grid
    assert finest["b60"].attrs["xr_dms_resampling"] == "nearest x6x6"
    assert coarsest["b10"].attrs["xr_dms_resampling"] == "average /6x6"


def test_like_overrides_the_grid_choice(multires_scene):
    bands, _, _, _ = multires_scene
    features = harmonize_features(bands, like=bands["b20"])
    assert features.sizes == {"y": 60, "x": 60}
    np.testing.assert_array_equal(features["y"].values, bands["b20"]["y"].values)


def test_downsampling_preserves_the_mean_but_shrinks_the_std(multires_scene):
    """The documented cost of grid="coarsest", asserted rather than asserted-in-prose."""
    bands, _, _, _ = multires_scene
    coarsest = harmonize_features(bands, grid="coarsest")
    # b10 block-averaged to 60 m keeps the scene mean and loses within-pixel spread.
    assert coarsest["b10"].mean().item() == pytest.approx(
        bands["b10"].mean().item(), abs=1e-12
    )
    assert coarsest["b10"].std().item() < bands["b10"].std().item()


# -- input shapes ------------------------------------------------------------
def test_accepts_datasets_dataarrays_and_mappings(multires_scene):
    bands, _, _, _ = multires_scene
    as_dataset = xr.Dataset({"a": bands["b10"], "b": bands["b10"] * 2})
    features = harmonize_features([as_dataset, bands["b20"], bands["b60"]])
    assert set(features.data_vars) == {"a", "b", "b20", "b60"}


def test_extra_dimensions_are_carried_through(multires_scene):
    """A time cube harmonises fine; only the Sharpener insists on a single scene."""
    bands, _, _, _ = multires_scene
    cube = bands["b20"].expand_dims(time=[0, 1])
    features = harmonize_features({"b10": bands["b10"], "cube": cube})
    assert features["cube"].sizes == {"time": 2, "y": 120, "x": 120}


def test_duplicate_names_are_rejected(multires_scene):
    bands, _, _, _ = multires_scene
    with pytest.raises(ValueError, match="Duplicate feature name"):
        harmonize_features([bands["b10"], bands["b10"]])


# -- coverage and trimming ---------------------------------------------------
def test_partial_coverage_becomes_nan_not_edge_extrapolation(multires_scene):
    """A band that stops short leaves NaN, rather than smearing its edge across."""
    bands, _, _, _ = multires_scene
    clipped = bands["b20"].isel(y=slice(0, 30), x=slice(0, 30))  # covers a quarter
    features = harmonize_features({"b10": bands["b10"], "b20": clipped})

    values = features["b20"].values
    assert np.isfinite(values[:60, :60]).all()
    assert np.isnan(values[60:, :]).all()
    assert np.isnan(values[:, 60:]).all()


def test_trim_crops_to_the_common_footprint(multires_scene):
    bands, _, _, _ = multires_scene
    clipped = bands["b20"].isel(y=slice(0, 30), x=slice(0, 30))
    features = harmonize_features(
        {"b10": bands["b10"], "b20": clipped}, trim=True
    )
    assert features.sizes == {"y": 60, "x": 60}
    assert np.isfinite(features["b20"].values).all()


# -- the nesting predicate ---------------------------------------------------
@pytest.mark.parametrize(
    "step, offset, nests",
    [
        (20.0, 0.0, True),    # a clean 2x
        (60.0, 0.0, True),    # a clean 6x
        (15.0, 0.0, False),   # non-integer ratio
        (20.0, 5.0, False),   # right ratio, misaligned pixel edges
    ],
)
def test_nesting_predicate(step, offset, nests):
    ref = xr.DataArray(
        np.zeros((120, 120)), dims=("y", "x"),
        coords={"y": 3220000.0 - (np.arange(120) + 0.5) * 10.0,
                "x": 300000.0 + (np.arange(120) + 0.5) * 10.0},
    )
    n = int(1200 // step)
    src = xr.DataArray(
        np.zeros((n, n)), dims=("y", "x"),
        coords={"y": 3220000.0 + offset - (np.arange(n) + 0.5) * step,
                "x": 300000.0 + offset + (np.arange(n) + 0.5) * step},
    )
    assert (nesting(describe(src), describe(ref)) is not None) is nests


def test_non_nesting_band_warns_and_warps(multires_scene):
    """The fallback path: a 15 m band on a 10 m reference has no integer ratio."""
    pytest.importorskip("xr_utils")
    bands, _, _, _ = multires_scene
    odd = bands["b10"].coarsen(y=3, x=3, boundary="exact").mean()
    odd = odd.interp(
        y=bands["b10"]["y"].values[::3][:40], x=bands["b10"]["x"].values[::3][:40],
    )
    # Rebuild on a genuine 15 m grid so the ratio really is 1.5.
    fifteen = xr.DataArray(
        np.asarray(odd.values)[:40, :40], dims=("y", "x"),
        coords={"y": 3220000.0 - (np.arange(40) + 0.5) * 15.0,
                "x": 300000.0 + (np.arange(40) + 0.5) * 15.0,
                "spatial_ref": bands["b10"]["spatial_ref"]},
    )
    with pytest.warns(UserWarning, match="does not nest"):
        features = harmonize_features({"b10": bands["b10"], "odd": fifteen})
    assert features.sizes == {"y": 120, "x": 120}
    assert features["odd"].attrs["xr_dms_resampling"].startswith("warp/")


def test_trim_accounts_for_a_warped_band(multires_scene):
    """A warped band still constrains the common footprint, same as a nested one."""
    pytest.importorskip("xr_utils")
    bands, _, _, _ = multires_scene
    # 15 m, and covering only the north-west quarter of the 10 m reference.
    partial = xr.DataArray(
        np.zeros((40, 40)), dims=("y", "x"),
        coords={"y": 3220000.0 - (np.arange(40) + 0.5) * 15.0,
                "x": 300000.0 + (np.arange(40) + 0.5) * 15.0,
                "spatial_ref": bands["b10"]["spatial_ref"]},
        name="odd",
    )
    with pytest.warns(UserWarning, match="does not nest"):
        features = harmonize_features(
            {"b10": bands["b10"], "odd": partial}, trim=True
        )
    # 40 pixels of 15 m = 600 m = 60 pixels of 10 m.
    assert features.sizes == {"y": 60, "x": 60}


def test_non_nesting_band_without_a_crs_raises_clearly():
    """A warp needs a projection; say so instead of failing inside odc.geo."""
    ref = xr.DataArray(
        np.zeros((12, 12)), dims=("y", "x"),
        coords={"y": np.arange(12) + 0.5, "x": np.arange(12) + 0.5}, name="ref",
    )
    odd = xr.DataArray(
        np.zeros((8, 8)), dims=("y", "x"),
        coords={"y": (np.arange(8) + 0.5) * 1.5, "x": (np.arange(8) + 0.5) * 1.5},
        name="odd",
    )
    with pytest.raises(ValueError, match="carries no CRS"):
        harmonize_features([ref, odd])


def test_invalid_method_names_are_rejected(multires_scene):
    bands, _, _, _ = multires_scene
    with pytest.raises(ValueError, match="upsample must be"):
        harmonize_features(bands, upsample="cubic")
    with pytest.raises(ValueError, match="downsample must be"):
        harmonize_features(bands, downsample="mode")


# -- end to end --------------------------------------------------------------
def test_regular_sharpen_through_harmonized_features(multires_scene):
    bands, target, truth, _ = multires_scene
    features = harmonize_features(bands)
    sharpened = Sharpener(smooth_residual=False).sharpen(features, target)

    assert sharpened.sizes == {"y": 120, "x": 120}
    assert np.isfinite(sharpened.values).all()
    # It has to beat the trivial answer, or the features are doing no work.
    naive = target.interp(y=truth["y"], x=truth["x"]).ffill("x").bfill("x")
    naive = naive.ffill("y").bfill("y")
    assert (np.abs(sharpened - truth).mean().item()
            < np.abs(naive - truth).mean().item())


def test_swath_sharpen_through_harmonized_features_conserves_mass():
    """A 60 m band mixed into the 20 m swath scene keeps mass conservation exact.

    Reproduces test_geo.py's conservation assertion through the new input path:
    harmonisation must not perturb the property that makes smooth_residual=False
    worth having.
    """
    features, target, truth, grid_map = make_swath_scene()
    mixed = harmonize_features({
        "ndvi": features["ndvi"],
        "albedo": features["albedo"],
        # A genuinely coarser band, as a 60 m Sentinel-2 band would be.
        "builtup": features["builtup"].coarsen(y=3, x=3, boundary="exact").mean(),
    })
    assert mixed["builtup"].attrs["xr_dms_resampling"] == "nearest x3x3"
    assert mixed.sizes == {"y": 300, "x": 300}

    sharpener = Sharpener(
        grid_map=grid_map, disaggregating_temperature=True, smooth_residual=False,
    )
    sharpened = sharpener.sharpen(mixed, target).compute()

    back = grid_map.aggregate_mean(sharpened ** 4) ** 0.25
    prepared = grid_map.prepare_target(target)
    valid = np.isfinite(back.values) & np.isfinite(prepared.values)
    np.testing.assert_allclose(
        back.values[valid], prepared.values[valid], rtol=0, atol=1e-6
    )
