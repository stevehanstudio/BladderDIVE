"""Image-level QC metrics."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from scripts.qc.io import QCContext, free_memory, level_scale
from scripts.qc.registry import register_qc
from scripts.qc.thresholds import metric

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    phase_cross_correlation = None


def _laplacian_var(tile: np.ndarray) -> float:
    t = tile.astype(np.float64)
    if cv2 is not None:
        try:
            return float(cv2.Laplacian(t, cv2.CV_64F).var())
        except cv2.error:
            pass
    gy, gx = np.gradient(t)
    return float((gx**2 + gy**2).mean())


def _tenengrad(tile: np.ndarray) -> float:
    t = tile.astype(np.float64)
    if cv2 is not None:
        try:
            gx = cv2.Sobel(t, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(t, cv2.CV_64F, 0, 1, ksize=3)
            return float((gx**2 + gy**2).mean())
        except cv2.error:
            pass
    gy, gx = np.gradient(t)
    return float((gx**2 + gy**2).mean())


@register_qc("image", "focus")
def focus(ctx: QCContext) -> dict:
    rows = []
    for tid, y0, x0, tile in ctx.iter_tiles(channel=ctx.dapi_index, mask=False):
        lv = _laplacian_var(tile)
        tg = _tenengrad(tile)
        rows.append(
            {
                "tile_id": tid,
                "y": y0,
                "x": x0,
                "laplacian_var": lv,
                "tenengrad": tg,
                "mean_intensity": float(tile.mean()),
            }
        )
    df = pd.DataFrame(rows)
    ctx.write_tiles("image_focus", df)
    med = float(df["laplacian_var"].median()) if len(df) else 0.0
    frac_bad = float((df["laplacian_var"] < 0.5 * med).mean()) if len(df) else 0.0

    if plt is not None and len(df):
        fig_path = ctx.figure_path("focus_heatmap.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df["x"], df["y"], c=df["laplacian_var"], cmap="viridis", s=30)
        ax.set_title("Focus (Laplacian variance) per tile")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax, label="Laplacian var")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    del df
    free_memory()

    return {
        "focus_frac_bad": metric(frac_bad, ctx.thr("image", "focus_frac_bad")),
        "focus_median_laplacian": metric(med),
    }


@register_qc("image", "illumination")
def illumination(ctx: QCContext) -> dict:
    rows = []
    for tid, y0, x0, tile in ctx.iter_tiles(channel=ctx.dapi_index, mask=False):
        rows.append({"tile_id": tid, "y": y0, "x": x0, "mean_intensity": float(tile.mean())})
    df = pd.DataFrame(rows)
    means = df["mean_intensity"].values
    cv = float(np.std(means) / (np.mean(means) + 1e-9)) if len(means) else 0.0

    if len(df) >= 4:
        cx = df["x"].median()
        cy = df["y"].median()
        dist = np.hypot(df["x"] - cx, df["y"] - cy)
        corner = df.loc[dist >= dist.quantile(0.75), "mean_intensity"].mean()
        center = df.loc[dist <= dist.quantile(0.25), "mean_intensity"].mean()
        vignetting = float(corner / (center + 1e-9))
    else:
        vignetting = 1.0

    if plt is not None and len(df):
        fig_path = ctx.figure_path("illumination_heatmap.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df["x"], df["y"], c=df["mean_intensity"], cmap="magma", s=30)
        ax.set_title("Mean DAPI intensity per tile")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    del df
    free_memory()

    return {
        "illumination_cv": metric(cv, ctx.thr("image", "illumination_cv")),
        "vignetting_index": metric(vignetting, ctx.thr("image", "vignetting_index")),
    }


@register_qc("image", "registration_drift")
def registration_drift(ctx: QCContext) -> dict:
    level = ctx._resolve_level(ctx.cfg.pyramid_level)
    ref = ctx.get_channel_2d(ctx.dapi_index, level=level)
    mov = ctx.get_channel_2d(ctx.dapi_r06_index, level=level)
    if phase_cross_correlation is None:
        return {"max_drift_px": metric(float("nan"), ctx.thr("image", "max_drift_px"))}

    shift, error, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
    dy, dx = float(shift[0]), float(shift[1])
    scale = level_scale(ctx.celldive_zarr, level)
    disp_px = float(np.hypot(dy, dx) * scale)

    if plt is not None:
        fig_path = ctx.figure_path("drift_dapi_rounds.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["dy", "dx"], [dy * scale, dx * scale], color=["steelblue", "sandybrown"])
        ax.set_ylabel("Displacement (px at full res)")
        ax.set_title(f"DAPI R01 vs R06 drift: {disp_px:.2f} px")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    del ref, mov
    free_memory()

    return {
        "max_drift_px": metric(disp_px, ctx.thr("image", "max_drift_px")),
        "drift_dy_px": metric(dy * scale),
        "drift_dx_px": metric(dx * scale),
        "phase_corr_error": metric(float(error)),
    }


@register_qc("image", "bleedthrough")
def bleedthrough(ctx: QCContext) -> dict:
    """Within-fluorophore spillover on a small set of channel pairs (cached tile samples)."""
    max_corr = 0.0
    pairs_tested = 0
    n_samples = min(ctx.cfg.max_pixels, 50_000)
    pair_limit = int(ctx.cfg.raw.get("max_bleedthrough_pairs", 8))

    pairs: list[tuple[int, int]] = []
    for _fluor, channels in ctx.cfg.fluorophore_groups.items():
        if len(channels) < 2:
            continue
        # one adjacent pair per fluorophore group is enough for QC
        pairs.append((channels[0], channels[1]))
        if len(channels) > 2:
            pairs.append((channels[-2], channels[-1]))
    pairs = pairs[:pair_limit]

    for ch_a, ch_b in pairs:
        samples = ctx.sample_pixels_from_channels([ch_a, ch_b], n_samples=n_samples, seed=0)
        a, b = samples[ch_a], samples[ch_b]
        bright = (a > np.percentile(a, 90)) | (b > np.percentile(b, 90))
        if bright.sum() < 500:
            continue
        c = float(np.corrcoef(a[bright], b[bright])[0, 1])
        max_corr = max(max_corr, abs(c))
        pairs_tested += 1

    ctx.clear_caches()

    return {
        "bleedthrough_corr": metric(max_corr, ctx.thr("image", "bleedthrough_corr")),
        "pairs_tested": metric(float(pairs_tested)),
    }


@register_qc("image", "autofluorescence")
def autofluorescence(ctx: QCContext) -> dict:
    """AF contamination: correlate markers with DAPI_AF on shared in-memory subsample."""
    max_corr = 0.0
    n_samples = min(ctx.cfg.max_pixels, 50_000)
    level = ctx._resolve_level(ctx.cfg.pyramid_level)

    af_plane = ctx.get_channel_2d(ctx.dapi_index, level=level)
    rng = np.random.default_rng(0)
    pick = rng.choice(af_plane.size, size=min(n_samples, af_plane.size), replace=False)
    af = af_plane.ravel()[pick]

    for marker, info in ctx.cfg.markers.items():
        if marker in ("DAPI", "DAPI2"):
            continue
        ch = int(info["channel"])
        m_plane = ctx.get_channel_2d(ch, level=level)
        m = m_plane.ravel()[pick]
        bright = m > np.percentile(m, 75)
        if bright.sum() < 500:
            del m_plane
            continue
        c = float(np.corrcoef(m[bright], af[bright])[0, 1])
        max_corr = max(max_corr, abs(c))
        del m_plane, m

    del af_plane, af
    free_memory()

    return {
        "autofluorescence_corr": metric(max_corr, ctx.thr("image", "autofluorescence_corr")),
    }
