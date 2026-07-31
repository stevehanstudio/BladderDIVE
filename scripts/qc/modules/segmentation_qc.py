"""Segmentation QC metrics."""

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
    from skimage.measure import regionprops_table
except ImportError:
    regionprops_table = None


def _mad_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - med) / mad


@register_qc("segmentation", "mask_vs_dapi")
def mask_vs_dapi(ctx: QCContext) -> dict:
    """DAPI inside mask vs background using sparse pixel sampling (memory-safe)."""
    d_flat, inside = ctx.sample_mask_dapi_pixels()
    i_flat = inside
    fg = d_flat[i_flat]
    bg = d_flat[~i_flat]
    mean_in = float(fg.mean()) if fg.size else float("nan")
    mean_out = float(bg.mean()) if bg.size else float("nan")
    ratio = mean_in / mean_out if mean_out > 0 else float("inf")
    p5 = float(np.percentile(d_flat, 5))
    frac_low = float((fg < p5).mean()) if fg.size else float("nan")

    if plt is not None:
        fig_path = ctx.figure_path("mask_vs_dapi_qc.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        hmax = np.percentile(d_flat, 99.5)
        for ax, data, title, color in [
            (axes[0], fg, f"Inside mask (n={len(fg):,})", "steelblue"),
            (axes[1], bg, f"Background (n={len(bg):,})", "sandybrown"),
        ]:
            if data.size == 0:
                ax.set_title(f"{title} (empty)")
                continue
            ax.hist(data, bins=80, range=(0, hmax), color=color, alpha=0.8, edgecolor="none", density=True)
            ax.axvline(p5, color="crimson", ls="--", lw=1.5, label=f"Global 5% pctl={p5:,.0f}")
            ax.set_xlim(0, hmax * 1.02)
            ax.set_xlabel("DAPI intensity")
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Mask vs DAPI (sampled) ratio={ratio:.2f} frac_low={frac_low*100:.2f}%",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "cell_id" in ctx.adata.obs.columns and fg.size:
        cell_dapi = ctx.marker_vector("DAPI", subsample=False)
        low = cell_dapi < p5
        ctx.add_cell_flags(pd.DataFrame({"qc_low_dapi": low}, index=ctx.adata.obs["cell_id"].values))

    del d_flat, inside, fg, bg
    free_memory()

    return {
        "frac_mask_fg": metric(float(i_flat.mean())),
        "mean_dapi_inside": metric(mean_in),
        "mean_dapi_outside": metric(mean_out),
        "dapi_in_mask_ratio": metric(ratio, ctx.thr("segmentation", "dapi_in_mask_ratio")),
        "frac_fg_below_global_p5": metric(frac_low, ctx.thr("segmentation", "frac_fg_below_global_p5")),
    }


@register_qc("segmentation", "area_shape")
def area_shape(ctx: QCContext) -> dict:
    obs = ctx.adata.obs
    area = obs["area"].astype(float).values
    lo = float(ctx.cfg.raw.get("min_cell_area", 50))
    hi = float(ctx.cfg.raw.get("max_cell_area", 5000))
    area_outlier = (area < lo) | (area > hi)
    median_area = float(np.median(area))
    doublet = area > 2.0 * median_area

    flags = pd.DataFrame(
        {
            "qc_area_outlier": area_outlier,
            "qc_doublet": doublet,
        },
        index=obs["cell_id"].values,
    )
    ctx.add_cell_flags(flags)

    solidity_median = float("nan")
    skip_regionprops = bool(ctx.cfg.raw.get("skip_regionprops", ctx.low_memory))

    if not skip_regionprops and regionprops_table is not None:
        level = ctx._resolve_level(ctx.cfg.pyramid_level)
        mask = ctx.get_mask_2d(level=level)
        try:
            props = regionprops_table(
                mask,
                properties=("label", "area", "eccentricity", "solidity", "extent"),
            )
            prop_df = pd.DataFrame(props).set_index("label")
            mapped_solidity = prop_df["solidity"].reindex(obs["cell_id"].values)
            solidity_median = float(np.nanmedian(mapped_solidity))
            shape_outlier = (mapped_solidity < 0.75).fillna(False).values
            ctx.add_cell_flags(
                pd.DataFrame({"qc_shape_outlier": shape_outlier}, index=obs["cell_id"].values)
            )
        except Exception:
            pass
        finally:
            del mask
            free_memory()

    if plt is not None:
        fig_path = ctx.figure_path("area_distribution.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(np.log10(area + 1), bins=80, color="steelblue", alpha=0.85, edgecolor="none")
        ax.axvline(np.log10(lo + 1), color="crimson", ls="--", label=f"min={lo}")
        ax.axvline(np.log10(hi + 1), color="darkred", ls="--", label=f"max={hi}")
        ax.set_xlabel("log10(cell area + 1)")
        ax.set_ylabel("Count")
        ax.set_title("Cell area distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "area_outlier_frac": metric(float(area_outlier.mean()), ctx.thr("segmentation", "area_outlier_frac")),
        "doublet_frac": metric(float(doublet.mean())),
        "solidity_median": metric(solidity_median, ctx.thr("segmentation", "solidity_median")),
        "median_area": metric(median_area),
        "regionprops_skipped": metric(float(skip_regionprops)),
    }


@register_qc("segmentation", "tile_outliers")
def tile_outliers(ctx: QCContext) -> dict:
    obs = ctx.adata.obs
    level = ctx._resolve_level(ctx.cfg.pyramid_level)
    scale = 2**level
    tile_px = max(1, ctx.cfg.tile_size // scale)

    tile_y = (obs["centroid_y"] // tile_px).astype(int)
    tile_x = (obs["centroid_x"] // tile_px).astype(int)
    tile_key = tile_y.astype(str) + "_" + tile_x.astype(str)
    counts = tile_key.value_counts()
    density = tile_key.map(counts).astype(float).values
    z = np.abs(_mad_z(density))

    tile_df = pd.DataFrame(
        {
            "tile_key": tile_key.values,
            "centroid_x": obs["centroid_x"].values,
            "centroid_y": obs["centroid_y"].values,
            "density": density,
            "density_mad_z": z,
        }
    )
    ctx.write_tiles("segmentation_tile_density", tile_df.drop_duplicates("tile_key"))

    frac_outlier_tiles = float((z > 5).sum() / max(len(obs), 1))
    max_z = float(np.max(z)) if len(z) else 0.0

    return {
        "tile_density_mad_z_max": metric(max_z, ctx.thr("segmentation", "tile_density_mad_z")),
        "tile_density_outlier_cell_frac": metric(frac_outlier_tiles),
    }


@register_qc("segmentation", "coverage")
def coverage(ctx: QCContext) -> dict:
    coverage_frac = ctx.mask_coverage_fraction()
    n_cells = float(ctx.adata.n_obs)

    return {
        "mask_coverage_frac": metric(coverage_frac, ctx.thr("segmentation", "mask_coverage_frac")),
        "n_cells_in_adata": metric(n_cells),
    }
