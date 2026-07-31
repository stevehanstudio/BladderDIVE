"""Spatial QC metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.qc.io import QCContext, free_memory
from scripts.qc.registry import register_qc
from scripts.qc.thresholds import metric

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None


def _mad_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - med) / mad


def _get_xy(ctx: QCContext) -> np.ndarray:
    if "spatial" in ctx.adata.obsm:
        return np.asarray(ctx.adata.obsm["spatial"])
    return ctx.adata.obs[["centroid_x", "centroid_y"]].values


@register_qc("spatial", "coverage_density")
def coverage_density(ctx: QCContext) -> dict:
    xy = _get_xy(ctx)
    if xy.shape[0] > ctx.max_cells_sample:
        rng = np.random.default_rng(0)
        xy = xy[rng.choice(xy.shape[0], size=ctx.max_cells_sample, replace=False)]

    n_bins = 50
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=n_bins)
    occupied = H > 0
    coverage_frac = float(occupied.mean())
    nonzero = H[H > 0]
    density_cv = float(nonzero.std() / (nonzero.mean() + 1e-9)) if nonzero.size else 0.0

    if plt is not None:
        fig_path = ctx.figure_path("density_map.png")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(H.T, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title("Cell density (binned)")
        ax.set_xlabel("x bin")
        ax.set_ylabel("y bin")
        fig.colorbar(im, ax=ax, label="cells/bin")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    del xy, H
    free_memory()

    return {
        "coverage_frac": metric(coverage_frac, ctx.thr("spatial", "coverage_frac")),
        "density_cv": metric(density_cv, ctx.thr("spatial", "density_cv")),
    }


@register_qc("spatial", "local_outliers")
def local_outliers(ctx: QCContext) -> dict:
    if NearestNeighbors is None:
        return {"spatial_outlier_frac": metric(float("nan"), ctx.thr("spatial", "spatial_outlier_frac"))}

    xy_full = _get_xy(ctx)
    n = xy_full.shape[0]
    k = min(15, n - 1)
    if k < 3:
        return {"spatial_outlier_frac": metric(0.0, ctx.thr("spatial", "spatial_outlier_frac"))}

    sample_n = min(n, ctx.max_cells_sample)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=sample_n, replace=False) if n > sample_n else np.arange(n)
    xy_sub = xy_full[idx]

    nn = NearestNeighbors(n_neighbors=k).fit(xy_sub)
    dist_sub = nn.kneighbors(xy_sub)[0].mean(axis=1)
    z_sub = np.abs(_mad_z(dist_sub))
    outlier_sub = z_sub > 5.0
    outlier_frac = float(outlier_sub.mean())

    outlier_full = np.zeros(n, dtype=bool)
    outlier_full[idx] = outlier_sub

    ctx.add_cell_flags(
        pd.DataFrame({"qc_spatial_outlier": outlier_full}, index=ctx.adata.obs["cell_id"].values)
    )

    if plt is not None and sample_n <= 200_000:
        fig_path = ctx.figure_path("spatial_outliers.png")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(xy_sub[~outlier_sub, 0], xy_sub[~outlier_sub, 1], s=0.1, c="lightgray", alpha=0.3, rasterized=True)
        ax.scatter(xy_sub[outlier_sub, 0], xy_sub[outlier_sub, 1], s=0.5, c="crimson", alpha=0.5, rasterized=True)
        ax.set_title(f"Spatial outliers ({outlier_frac*100:.2f}%, sampled)")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    del xy_full, xy_sub, nn, dist_sub
    free_memory()

    return {
        "spatial_outlier_frac": metric(outlier_frac, ctx.thr("spatial", "spatial_outlier_frac")),
    }


@register_qc("spatial", "stripe_seam_detection")
def stripe_seam_detection(ctx: QCContext) -> dict:
    xy = _get_xy(ctx)
    if xy.shape[0] > ctx.max_cells_sample:
        rng = np.random.default_rng(0)
        xy = xy[rng.choice(xy.shape[0], size=ctx.max_cells_sample, replace=False)]

    x_bins = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 100)
    y_bins = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 100)
    x_hist, _ = np.histogram(xy[:, 0], bins=x_bins)
    y_hist, _ = np.histogram(xy[:, 1], bins=y_bins)
    x_cv = float(x_hist.std() / (x_hist.mean() + 1e-9))
    y_cv = float(y_hist.std() / (y_hist.mean() + 1e-9))
    empty_frac_x = float((x_hist == 0).mean())
    empty_frac_y = float((y_hist == 0).mean())

    return {
        "row_density_cv": metric(x_cv),
        "col_density_cv": metric(y_cv),
        "empty_bin_frac_x": metric(empty_frac_x),
        "empty_bin_frac_y": metric(empty_frac_y),
    }


@register_qc("spatial", "edge_effects")
def edge_effects(ctx: QCContext) -> dict:
    xy_full = _get_xy(ctx)
    dapi_full = ctx.marker_vector("DAPI", subsample=False)

    n = min(xy_full.shape[0], ctx.max_cells_sample)
    if xy_full.shape[0] > n:
        rng = np.random.default_rng(0)
        sel = rng.choice(xy_full.shape[0], size=n, replace=False)
        xy = xy_full[sel]
        dapi = dapi_full[sel]
    else:
        xy = xy_full
        dapi = dapi_full

    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    margin_x = 0.05 * (xmax - xmin)
    margin_y = 0.05 * (ymax - ymin)
    edge = (
        (xy[:, 0] <= xmin + margin_x)
        | (xy[:, 0] >= xmax - margin_x)
        | (xy[:, 1] <= ymin + margin_y)
        | (xy[:, 1] >= ymax - margin_y)
    )
    edge_frac = float(edge.mean())
    edge_mean = float(dapi[edge].mean()) if edge.any() else float("nan")
    center_mean = float(dapi[~edge].mean()) if (~edge).any() else float("nan")
    edge_ratio = edge_mean / (center_mean + 1e-9)

    del xy_full, dapi_full, xy, dapi
    free_memory()

    return {
        "edge_cell_frac": metric(edge_frac),
        "edge_center_dapi_ratio": metric(edge_ratio),
    }
