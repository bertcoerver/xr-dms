"""
Thermal sharpening explained, step by step, in (mostly) numpy.
=================================================================

This is an *educational* re-implementation of the Data Mining Sharpener
(DMS) of Gao et al. (2012), the same algorithm that pyWaPOR uses in
`pywapor/enhancers/dms/pyDMS.py` and `thermal_sharpener.py`.

The real code uses GDAL for the raster I/O / reprojection. To keep the
*ideas* visible we strip that away -- inputs are plain numpy arrays (no
GDAL, no files, no projections) -- but we keep the two pieces that matter
most for the result and re-implement them in numpy:

    * the regressor can be a single multivariate linear least-squares fit
      (np.linalg.lstsq, easy to read) OR a small bagged ensemble of
      regression trees with per-leaf linear regression -- the same design
      as the production `DecisionTreeRegressorWithLinearLeafRegression` +
      `BaggingRegressor` in pyDMS, written from scratch in numpy.
    * temperature is averaged in RADIANCE space (T**4), exactly as pyDMS
      does when `disaggregatingTemperature=True`, because sensors integrate
      energy and T is non-linear so averaging T directly is unphysical.

The pipeline is the genuine DMS pipeline:

    INPUT
      - a COARSE image we want to sharpen        (e.g. land surface temperature)
      - several FINE feature images               (e.g. NDVI, albedo, ...)

    STEP 1  Aggregate the fine features to the coarse grid (block mean),
            and measure within-pixel heterogeneity (block std).
    STEP 2  Decide which coarse pixels are "homogeneous" enough to learn from.
    STEP 3  Train a regression:  coarse_target ~ f(coarse_features).
    STEP 4  Apply that regression to the FINE features -> a fine first guess.
    STEP 5  Residual correction: the fine guess, re-aggregated, will not
            exactly reproduce the observed coarse image. Spread that
            difference back over the fine guess so the sharpened result is
            "consistent": averaging it back to the coarse grid reproduces
            the original coarse observation.

The script also generates dummy inputs (`highres_features.npy` and
`lowres_target.npy`) so it runs out of the box:

    python dms_explained_numpy.py            # run the demo, save .npy inputs
    python dms_explained_numpy.py --plot     # also draw a figure (needs matplotlib)

Reference
---------
Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data Mining Approach for
Sharpening Thermal Satellite Imagery over Land. Remote Sensing, 4(11),
3287-3319. https://doi.org/10.3390/rs4113287
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Small numpy helpers (these replace the GDAL resampling used in the real code)
# ---------------------------------------------------------------------------

def block_mean(arr, factor):
    """Average each `factor` x `factor` block -> coarse grid (= sensor averaging).

    Works on 2-D arrays (H, W) or 3-D feature stacks (H, W, B).
    `H` and `W` must be divisible by `factor`.
    """
    h, w = arr.shape[:2]
    assert h % factor == 0 and w % factor == 0, "shape must be divisible by factor"
    if arr.ndim == 2:
        return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    b = arr.shape[2]
    return arr.reshape(h // factor, factor, w // factor, factor, b).mean(axis=(1, 3))


def block_std(arr, factor):
    """Standard deviation within each `factor` x `factor` block (2-D or 3-D)."""
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return arr.reshape(h // factor, factor, w // factor, factor).std(axis=(1, 3))
    b = arr.shape[2]
    return arr.reshape(h // factor, factor, w // factor, factor, b).std(axis=(1, 3))


def bilinear_upsample(coarse, out_shape):
    """Bilinearly resample a coarse 2-D array up to `out_shape` (pure numpy).

    Uses centre-aligned coordinates and separable 1-D interpolation
    (np.interp along columns, then rows) which is exactly bilinear for
    up-sampling. This stands in for GDAL's bilinear warp.
    """
    in_h, in_w = coarse.shape
    out_h, out_w = out_shape

    # Output pixel centres expressed in the coarse grid's index space.
    col_src = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    row_src = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5

    # 1) interpolate along columns
    tmp = np.empty((in_h, out_w))
    xcols = np.arange(in_w)
    for i in range(in_h):
        tmp[i] = np.interp(col_src, xcols, coarse[i])

    # 2) interpolate along rows
    out = np.empty((out_h, out_w))
    yrows = np.arange(in_h)
    for j in range(out_w):
        out[:, j] = np.interp(row_src, yrows, tmp[:, j])
    return out


def binomial_smooth(arr):
    """3x3 binomial (1-2-1)x(1-2-1) smoothing, edge-padded. Pure numpy.

    Mirrors pyDMS `binomialSmoother`: residuals are smoothed before being
    up-sampled so the correction field is gentle, not blocky.
    """
    k = np.array([1.0, 2.0, 1.0])
    k = k / k.sum()
    p = np.pad(arr, 1, mode="edge")
    # convolve along rows then columns (separable)
    tmp = (k[0] * p[:, :-2] + k[1] * p[:, 1:-1] + k[2] * p[:, 2:])
    out = (k[0] * tmp[:-2, :] + k[1] * tmp[1:-1, :] + k[2] * tmp[2:, :])
    return out


# ---------------------------------------------------------------------------
# A pure-numpy regression tree (the production regressor, in miniature)
# ---------------------------------------------------------------------------
#
# pyDMS uses sklearn's BaggingRegressor wrapped around a custom
# DecisionTreeRegressorWithLinearLeafRegression. Below is the same idea in
# numpy: a CART regression tree (recursive weighted-variance splits) where
# each leaf, instead of predicting a constant, fits a *linear* regression on
# the samples that reached it -- with extrapolation clamped to the leaf's
# observed range (pyDMS `linearRegressionExtrapolationRatio`, default 0.25).

def _weighted_sse(y, w):
    """Weighted sum of squared deviations from the weighted mean."""
    if y.size == 0:
        return 0.0
    mu = np.average(y, weights=w)
    return float(np.sum(w * (y - mu) ** 2))


class RegressionTree:
    """Minimal CART regression tree with optional per-leaf linear regression."""

    def __init__(self, max_depth=4, min_samples_leaf=10,
                 per_leaf_linear=True, extrapolation_ratio=0.25):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.per_leaf_linear = per_leaf_linear
        self.extrapolation_ratio = extrapolation_ratio
        self.root = None

    def fit(self, X, y, w):
        self.root = self._build(X, y, w, depth=0)
        return self

    # -- training --------------------------------------------------------
    def _build(self, X, y, w, depth):
        # Stop splitting: too deep, too few samples to split, or ~constant y.
        if (depth >= self.max_depth
                or y.size < 2 * self.min_samples_leaf
                or np.ptp(y) < 1e-9):
            return self._make_leaf(X, y, w)

        feature, threshold = self._best_split(X, y, w)
        if feature is None:                       # no split improved fit
            return self._make_leaf(X, y, w)

        left = X[:, feature] <= threshold
        return {
            "leaf": False, "feature": feature, "threshold": threshold,
            "left": self._build(X[left], y[left], w[left], depth + 1),
            "right": self._build(X[~left], y[~left], w[~left], depth + 1),
        }

    def _best_split(self, X, y, w):
        """Exhaustive search for the split that minimises child weighted SSE.

        O(features * n^2) -- written for clarity, not speed; fine for the
        small training sets (hundreds of pixels) DMS produces.
        """
        n, n_features = X.shape
        parent_sse = _weighted_sse(y, w)
        best_sse, best = np.inf, (None, None)
        m = self.min_samples_leaf

        for f in range(n_features):
            order = np.argsort(X[:, f], kind="mergesort")
            sv, sy, sw = X[order, f], y[order], w[order]
            # candidate split points keep >= min_samples_leaf on each side
            for i in range(m, n - m + 1):
                if sv[i] == sv[i - 1]:            # can't split equal values
                    continue
                sse = _weighted_sse(sy[:i], sw[:i]) + _weighted_sse(sy[i:], sw[i:])
                if sse < best_sse:
                    best_sse = sse
                    best = (f, 0.5 * (sv[i] + sv[i - 1]))

        # Only accept the split if it actually reduces the error.
        return best if best_sse < parent_sse - 1e-12 else (None, None)

    def _make_leaf(self, X, y, w):
        node = {"leaf": True, "value": float(np.average(y, weights=w))}
        # Per-leaf linear regression (needs more samples than features).
        if self.per_leaf_linear and y.size > X.shape[1] + 1 and np.ptp(y) > 0:
            coef, intercept = _weighted_linear_fit(X, y, w)
            span = self.extrapolation_ratio * (y.max() - y.min())
            node["linear"] = (coef, intercept, y.min() - span, y.max() + span)
        return node

    # -- prediction (vectorised tree walk) -------------------------------
    def predict(self, X):
        out = np.empty(X.shape[0])
        self._predict(self.root, X, np.arange(X.shape[0]), out)
        return out

    def _predict(self, node, X, idx, out):
        if idx.size == 0:
            return
        if node["leaf"]:
            if "linear" in node:
                coef, intercept, lo, hi = node["linear"]
                # clamp to the leaf's range (+ allowed extrapolation)
                out[idx] = np.clip(X[idx] @ coef + intercept, lo, hi)
            else:
                out[idx] = node["value"]
            return
        left = X[idx, node["feature"]] <= node["threshold"]
        self._predict(node["left"], X, idx[left], out)
        self._predict(node["right"], X, idx[~left], out)


def fit_bagged_trees(X, y, w, n_trees=10, max_depth=4, min_samples_leaf=10,
                     per_leaf_linear=True, seed=0):
    """Bagging: train each tree on a bootstrap resample; predict = average.

    Returns a `predict(X)` function, matching how `sharpen` consumes the
    regressor regardless of which kind it is.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    trees = []
    for _ in range(n_trees):
        b = rng.integers(0, n, n)                 # bootstrap sample (with repl.)
        trees.append(RegressionTree(max_depth, min_samples_leaf,
                                    per_leaf_linear).fit(X[b], y[b], w[b]))

    def predict(Xf):
        return np.mean([t.predict(Xf) for t in trees], axis=0)

    return predict


# ---------------------------------------------------------------------------
# Dummy input generation
# ---------------------------------------------------------------------------

def make_dummy_inputs(fine_shape=(120, 120), factor=10, seed=0):
    """Create a synthetic scene with a known fine-resolution "truth".

    We build a fine temperature field that genuinely depends on the fine
    features (plus some structure the features cannot explain), then average
    it down to make the coarse observation. Because we kept the truth, the
    demo can show how close the sharpened result gets to it.

    Returns
    -------
    highres_features : (H, W, B) float array   -- known at FINE resolution
    lowres_target    : (h, w)   float array    -- the COARSE image to sharpen
    fine_truth       : (H, W)   float array    -- ground truth (for evaluation)
    """
    rng = np.random.default_rng(seed)
    H, W = fine_shape

    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)

    # Feature 1: a vegetation-like index (smooth gradient + blobs), 0..1
    ndvi = 0.5 + 0.4 * np.sin(6 * xx) * np.cos(5 * yy)
    ndvi += 0.15 * rng.standard_normal((H, W))
    ndvi = np.clip(ndvi, 0, 1)

    # Feature 2: an albedo-like field, anti-correlated with ndvi + own pattern
    albedo = 0.3 - 0.2 * ndvi + 0.1 * np.cos(10 * xx + 3 * yy)
    albedo += 0.05 * rng.standard_normal((H, W))
    albedo = np.clip(albedo, 0.05, 0.6)

    # Feature 3: a "built-up / bare soil" index with sharp edges (blocks)
    builtup = (np.sin(3 * xx) > 0.4).astype(float)
    builtup += 0.05 * rng.standard_normal((H, W))

    features = np.stack([ndvi, albedo, builtup], axis=-1)

    # Fine "truth" temperature (K). Cooler over vegetation, hotter over bare /
    # built-up & high-albedo. It has three kinds of structure on purpose:
    #   * a LINEAR-in-features part (any regressor can learn it),
    #   * a NON-LINEAR-in-features part (sin of ndvi) -- a single linear fit
    #     cannot capture this, but the regression-tree ensemble can; this is
    #     why production DMS uses trees,
    #   * a slowly-varying part the features do NOT contain at all -- which is
    #     exactly what the residual-correction step (step 5) exists to recover.
    truth = (305.0
             - 18.0 * ndvi
             + 12.0 * albedo
             + 8.0 * builtup
             + 10.0 * np.sin(8.0 * ndvi)               # non-linear in a feature
             + 4.0 * np.sin(2.5 * xx + 1.5 * yy))       # unexplained structure
    truth += 0.3 * rng.standard_normal((H, W))          # sensor-ish noise

    # The coarse sensor sees only a block average of the truth. For a thermal
    # sensor that average happens in RADIANCE space (energy, ~T**4), so the
    # coarse pixel is the 4th-root of the mean of T**4, not the mean of T.
    lowres_target = block_mean(truth ** 4, factor) ** 0.25

    return features, lowres_target, truth


# ---------------------------------------------------------------------------
# The DMS pipeline, step by step
# ---------------------------------------------------------------------------

def sharpen(highres_features, lowres_target, factor,
            cv_percentile=80, regressor="tree", n_trees=10,
            disaggregating_temperature=True, verbose=True):
    """Sharpen `lowres_target` using `highres_features`, DMS-style.

    Parameters
    ----------
    highres_features : (H, W, B) array, fine resolution predictors.
    lowres_target    : (h, w) array, coarse image to sharpen (H = h*factor).
    factor           : int, resolution ratio between fine and coarse grids.
    cv_percentile    : keep coarse pixels whose heterogeneity is below this
                       percentile as training samples (pyDMS default = 80).
    regressor        : "tree" (bagged regression trees with per-leaf linear
                       regression, like production) or "linear" (a single
                       multivariate least-squares fit).
    n_trees          : number of trees when `regressor="tree"`.
    disaggregating_temperature : if True, aggregate in radiance space (T**4)
                       during residual analysis, as pyDMS does for LST.

    Returns
    -------
    dict with the key intermediate products, so each step can be inspected
    / plotted.
    """
    def say(*a):
        if verbose:
            print(*a)

    H, W, B = highres_features.shape
    h, w = lowres_target.shape
    say(f"Fine grid : {H} x {W}  ({B} features)")
    say(f"Coarse grid: {h} x {w}   (factor {factor})")

    # -- STEP 1 -------------------------------------------------------------
    # Bring the fine features onto the coarse grid so they line up with the
    # coarse target. We also keep the within-block std as a heterogeneity
    # measure. (Real DMS: utils.resampleHighResToLowRes returns mean + std.)
    say("\nSTEP 1  Aggregate fine features to coarse grid (block mean + std)")
    feat_coarse_mean = block_mean(highres_features, factor)   # (h, w, B)
    feat_coarse_std = block_std(highres_features, factor)     # (h, w, B)

    # -- STEP 2 -------------------------------------------------------------
    # Homogeneity test. A coarse pixel that is internally very mixed is a poor
    # teacher: we don't know how its single coarse target value maps onto the
    # varied fine features inside it. Coefficient of variation (std/mean),
    # averaged over features, quantifies that mixing. Low CV = homogeneous.
    say("STEP 2  Select homogeneous coarse pixels as training samples")
    eps = 1e-6
    cv = np.mean(feat_coarse_std / (np.abs(feat_coarse_mean) + eps), axis=-1)  # (h, w)
    # pyDMS sets the threshold automatically at the 80th percentile of CV.
    cv_threshold = np.percentile(cv, cv_percentile)
    homogeneous = cv <= cv_threshold
    # weight homogeneous samples by inverse heterogeneity (more homogeneous ->
    # more trustworthy -> higher weight), as in Gao 2012 section 2.2.
    sample_weight = 1.0 / (cv[homogeneous] + eps)
    say(f"         CV threshold (p{cv_percentile}) = {cv_threshold:.3f}; "
        f"{homogeneous.sum()} / {h * w} pixels used for training")

    # -- STEP 3 -------------------------------------------------------------
    # Train the regression on the coarse grid: target ~ features.
    # "tree"   -> bagged regression-tree ensemble with per-leaf linear
    #             regression (the production design).
    # "linear" -> one weighted multivariate least-squares fit (simplest case).
    say(f"STEP 3  Fit regression  target ~ features  (regressor={regressor!r})")
    X_train = feat_coarse_mean[homogeneous]          # (n, B)
    y_train = lowres_target[homogeneous]             # (n,)
    if regressor == "linear":
        coef, intercept = _weighted_linear_fit(X_train, y_train, sample_weight)
        def predict(Xf):
            return Xf @ coef + intercept
        say(f"         intercept = {intercept:.2f}, coefficients = "
            + ", ".join(f"{c:+.2f}" for c in coef))
    elif regressor == "tree":
        predict = fit_bagged_trees(X_train, y_train, sample_weight,
                                   n_trees=n_trees, min_samples_leaf=10,
                                   per_leaf_linear=True, seed=0)
        say(f"         bagged ensemble of {n_trees} trees, per-leaf linear "
            "regression")
    else:
        raise ValueError("regressor must be 'tree' or 'linear'")

    # -- STEP 4 -------------------------------------------------------------
    # Apply the *same* relationship to the FINE features. Because the features
    # exist at fine resolution, the prediction is now at fine resolution: this
    # is the sharpened first guess.
    say("STEP 4  Apply regression to FINE features -> fine first guess")
    fine_flat = highres_features.reshape(-1, B)
    first_guess = predict(fine_flat).reshape(H, W)

    # -- STEP 5 -------------------------------------------------------------
    # Residual correction (Gao 2012 section 2.4). The regression captures the
    # feature-driven part of the signal but not what the features cannot
    # explain. Enforce consistency: average the fine guess back to the coarse
    # grid, compare with the observed coarse image, smooth that residual,
    # upsample it, and add it back. The corrected result, re-aggregated,
    # reproduces the coarse observation exactly (mass conservation).
    say("STEP 5  Residual correction (make it consistent with the coarse obs)")
    if disaggregating_temperature:
        # Aggregate and difference in RADIANCE space (T**4); add the correction
        # there too, then convert the 4th root back to temperature. Mirrors
        # pyDMS: corrected = (residual_HR + scene_HR**4)**0.25.
        guess_coarse = block_mean(first_guess ** 4, factor)            # radiance
        residual_coarse = lowres_target ** 4 - guess_coarse            # radiance
        residual_coarse_smooth = binomial_smooth(residual_coarse)
        residual_fine = bilinear_upsample(residual_coarse_smooth, (H, W))
        base = first_guess ** 4 + residual_fine
        sharpened = np.maximum(base, 1.0) ** 0.25                       # -> K
        # report the residual converted back to temperature (interpretable)
        res_T = (residual_coarse + 273.15 ** 4) ** 0.25 - 273.15
        bias, rmsd = np.nanmean(res_T), np.sqrt(np.nanmean(res_T ** 2))
    else:
        guess_coarse = block_mean(first_guess, factor)                 # (h, w)
        residual_coarse = lowres_target - guess_coarse                 # (h, w)
        residual_coarse_smooth = binomial_smooth(residual_coarse)
        residual_fine = bilinear_upsample(residual_coarse_smooth, (H, W))
        sharpened = first_guess + residual_fine
        bias = np.nanmean(residual_coarse)
        rmsd = np.sqrt(np.nanmean(residual_coarse ** 2))
    say(f"         coarse residual bias = {bias:+.3f} K, RMSD = {rmsd:.3f} K")

    return {
        "feat_coarse_mean": feat_coarse_mean,
        "cv": cv,
        "homogeneous": homogeneous,
        "first_guess": first_guess,
        "residual_coarse": residual_coarse,
        "residual_fine": residual_fine,
        "sharpened": sharpened,
    }


def _weighted_linear_fit(X, y, w):
    """Weighted least squares with intercept, via np.linalg.lstsq.

    Returns (coef[B], intercept). Solving the normal equations on the
    sqrt(weight)-scaled, intercept-augmented design matrix.
    """
    A = np.hstack([X, np.ones((X.shape[0], 1))])     # add intercept column
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(A * sw, y * sw[:, 0], rcond=None)
    return beta[:-1], beta[-1]


# ---------------------------------------------------------------------------
# Demo / entry point
# ---------------------------------------------------------------------------

def _evaluate(name, pred, truth):
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    print(f"   {name:<28s} RMSE vs truth = {rmse:.3f} K")
    return rmse


def main(plot=False):
    here = os.path.dirname(os.path.abspath(__file__))
    factor = 10

    # ---- create & save dummy inputs --------------------------------------
    features, lowres_target, truth = make_dummy_inputs(
        fine_shape=(120, 120), factor=factor, seed=0)

    feat_path = os.path.join(here, "highres_features.npy")
    targ_path = os.path.join(here, "lowres_target.npy")
    np.save(feat_path, features)
    np.save(targ_path, lowres_target)
    print(f"Saved dummy inputs:\n   {feat_path}   shape {features.shape}\n"
          f"   {targ_path}   shape {lowres_target.shape}\n")

    # ---- run the pipeline with each regressor ----------------------------
    print("=" * 60, "\nLINEAR regressor\n" + "=" * 60)
    out_lin = sharpen(features, lowres_target, factor=factor,
                      regressor="linear", disaggregating_temperature=True)
    print("\n" + "=" * 60, "\nBAGGED DECISION-TREE regressor (production-style)\n"
          + "=" * 60)
    out_tree = sharpen(features, lowres_target, factor=factor,
                       regressor="tree", disaggregating_temperature=True)

    # ---- evaluate against the known truth --------------------------------
    print("\nHow good is it? (RMSE vs hidden truth, lower is better)")
    # Naive baseline: just upsample the coarse image (no sharpening at all).
    naive = bilinear_upsample(lowres_target, truth.shape)
    _evaluate("naive bilinear upsample", naive, truth)
    _evaluate("linear: first guess (step 4)", out_lin["first_guess"], truth)
    _evaluate("linear: sharpened (step 5)", out_lin["sharpened"], truth)
    _evaluate("tree:   first guess (step 4)", out_tree["first_guess"], truth)
    _evaluate("tree:   sharpened (step 5)", out_tree["sharpened"], truth)

    if plot:
        _plot(features, lowres_target, truth, out_tree)


def _plot(features, lowres_target, truth, out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure.")
        return

    vmin, vmax = truth.min(), truth.max()
    tkw = dict(vmin=vmin, vmax=vmax, cmap="inferno")

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))

    axs[0, 0].imshow(lowres_target, **tkw)
    axs[0, 0].set_title("Coarse input (to sharpen)")

    axs[0, 1].imshow(features[..., 0], cmap="YlGn")
    axs[0, 1].set_title("A fine feature (NDVI-like)")

    axs[0, 2].imshow(out["cv"], cmap="viridis")
    axs[0, 2].set_title("Step 2: heterogeneity (CV)")

    axs[1, 0].imshow(out["first_guess"], **tkw)
    axs[1, 0].set_title("Step 4: first guess (tree)")

    axs[1, 1].imshow(out["sharpened"], **tkw)
    axs[1, 1].set_title("Step 5: sharpened (final)")

    axs[1, 2].imshow(truth, **tkw)
    axs[1, 2].set_title("Ground truth (hidden)")

    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Data Mining Sharpener (tree ensemble), step by step",
                 fontsize=14)
    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(here, "dms_explained_numpy.png")
    fig.savefig(out_png, dpi=120)
    print(f"\nFigure written to {out_png}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true",
                   help="draw a figure of the steps (needs matplotlib)")
    args = p.parse_args()
    main(plot=args.plot)
