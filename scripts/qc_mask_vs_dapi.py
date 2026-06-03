#!/usr/bin/env python3
"""
Quantitative QC: Cellpose label mask vs DAPI channel in the CellDIVE image zarr.

- Loads the same pyramid level for image + mask (spatial dimensions must match).
- DAPI index is resolved from .zattrs OME-JSON (label containing "DAPI" and not "DAPI2" / "R06"),
  with fallback to channel 0.
- Reports mean DAPI inside mask (label > 0) vs background, ratio, and optional percentiles.
- Can save a figure (histograms + optional crop) without opening Napari.

Usage:
  python scripts/qc_mask_vs_dapi.py
  python scripts/qc_mask_vs_dapi.py --level 4 --max-pixels 5000000
  python scripts/qc_mask_vs_dapi.py --no-figure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _get_project_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    return p


def dapi_channel_index(celldive_zarr: Path) -> int:
    """Index of the primary DAPI channel (DAPI1 / DAPI_AF_R01), not DAPI2 / DAPI_R06."""
    zattrs = celldive_zarr / ".zattrs"
    if not zattrs.exists():
        return 0
    with open(zattrs) as f:
        meta = json.load(f)
    channels = meta.get("omero", {}).get("channels", [])
    if not channels and "multiscales" in meta:
        channels = meta["multiscales"][0].get("omero", {}).get("channels", [])

    for i, ch in enumerate(channels):
        lab = (ch.get("label") or f"Ch{i}").upper()
        if "DAPI" in lab and "DAPI2" not in lab and "R06" not in lab:
            return i
    # If only "DAPI_R06" is named DAPI, prefer first DAPI* match
    for i, ch in enumerate(channels):
        lab = (ch.get("label") or f"Ch{i}").upper()
        if "DAPI_AF" in lab or lab.startswith("DAPI_") and "R06" not in lab:
            return i
    for i, ch in enumerate(channels):
        if "DAPI" in (ch.get("label") or "").upper():
            return i
    return 0


def run_qc(
    celldive_zarr: Path,
    mask_zarr: Path,
    level: int = 4,
    max_pixels: int = 0,
    seed: int = 0,
    out_figure: Path | None = None,
) -> dict:
    img_g = zarr.open_group(str(celldive_zarr), mode="r")
    mask_g = zarr.open_group(str(mask_zarr), mode="r")
    k = str(level)
    if k not in img_g or k not in mask_g:
        raise KeyError(f"Level {k} not in both zarrs. Keys: img={list(img_g.keys())} mask={list(mask_g.keys())}")

    dapi_i = dapi_channel_index(celldive_zarr)
    dapi = img_g[k][dapi_i]
    mask = mask_g[k]

    sh_img = dapi.shape
    sh_m = mask.shape
    if sh_img != sh_m:
        raise ValueError(f"Shape mismatch at L{level}: DAPI 2D {sh_img} vs mask {sh_m}")

    dapi = np.asarray(dapi, dtype=np.float32)
    m = np.asarray(mask, dtype=np.int32)
    inside = m > 0

    rng = np.random.default_rng(seed)
    n_pix = dapi.size
    use_subsample = max_pixels and n_pix > max_pixels
    if use_subsample:
        idx = rng.choice(n_pix, size=max_pixels, replace=False)
        d_flat = dapi.ravel()[idx]
        i_flat = inside.ravel()[idx]
    else:
        d_flat = dapi.ravel()
        i_flat = inside.ravel()

    fg = d_flat[i_flat]
    bg = d_flat[~i_flat]

    stats: dict = {
        "level": level,
        "dapi_channel_index": dapi_i,
        "shape_2d": sh_img,
        "n_pixels_total": int(n_pix),
        "subsampled": use_subsample,
        "n_sample_used": int(len(d_flat)),
        "frac_mask_fg": float(i_flat.mean()),
        "mean_dapi_inside": float(fg.mean()) if fg.size else float("nan"),
        "mean_dapi_outside": float(bg.mean()) if bg.size else float("nan"),
        "median_dapi_inside": float(np.median(fg)) if fg.size else float("nan"),
        "median_dapi_outside": float(np.median(bg)) if bg.size else float("nan"),
    }
    if stats["mean_dapi_outside"] and stats["mean_dapi_outside"] > 0:
        stats["ratio_fg_over_bg"] = float(stats["mean_dapi_inside"] / stats["mean_dapi_outside"])
    else:
        stats["ratio_fg_over_bg"] = float("inf") if stats["mean_dapi_inside"] > 0 else float("nan")

    p = [1, 5, 10, 25, 50, 75, 90, 99]
    stats["percentile_inside"] = {f"p{q}": float(np.percentile(fg, q)) for q in p} if fg.size else {}
    stats["percentile_outside"] = {f"p{q}": float(np.percentile(bg, q)) for q in p} if bg.size else {}

    # Pixels that are in mask but very dim: fraction of fg below global p5 of all pixels (or subsample)
    p5 = float(np.percentile(d_flat, 5))
    if fg.size:
        stats["frac_fg_below_global_p5"] = float((fg < p5).mean())
    else:
        stats["frac_fg_below_global_p5"] = float("nan")

    if out_figure is not None and plt is not None:
        out_figure.parent.mkdir(parents=True, exist_ok=True)
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

        supt = (
            f"Mask vs DAPI (pyramid L{level}, ch {dapi_i})   "
            f"mean_in/mean_out={stats.get('ratio_fg_over_bg', float('nan')):.2f}   "
            f"frac(mask px & below global p5)={stats['frac_fg_below_global_p5']*100:.2f}%"
        )
        fig.suptitle(supt, fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(out_figure, dpi=150, bbox_inches="tight")
        plt.close(fig)
        stats["figure"] = str(out_figure)

    return stats


def main() -> None:
    root = _get_project_root()
    p = argparse.ArgumentParser(description="QC Cellpose mask vs DAPI in CellDIVE zarr")
    p.add_argument(
        "--celldive",
        type=Path,
        default=root / "data" / "CellDIVE_SLIDE-045.zarr",
        help="CellDIVE image OME zarr",
    )
    p.add_argument(
        "--mask",
        type=Path,
        default=root / "output" / "cellpose_output" / "cellpose_masks_dapi_only_9tiles.zarr",
        help="Cellpose label mask zarr (pyramid 0/1/...)",
    )
    p.add_argument(
        "--level", type=int, default=4, help="Pyramid level (0=full res; 4 is fast, ~4k×4.6k)"
    )
    p.add_argument(
        "--max-pixels", type=int, default=0,
        help="If >0, subsample this many pixel pairs for speed (0 = all pixels at level)"
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=root / "output" / "figures" / "mask_vs_dapi_qc.png",
        help="Output PNG (set --no-figure to skip)"
    )
    p.add_argument("--no-figure", action="store_true", help="Do not write figure")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.celldive.exists():
        print(f"ERROR: image zarr not found: {args.celldive}", file=sys.stderr)
        sys.exit(1)
    if not args.mask.exists():
        print(f"ERROR: mask zarr not found: {args.mask}", file=sys.stderr)
        sys.exit(1)
    if plt is None and not args.no_figure:
        print("Warning: matplotlib not found; use --no-figure or install matplotlib", file=sys.stderr)
        args.no_figure = True

    out_fig = None if args.no_figure else args.figure
    s = run_qc(
        args.celldive, args.mask,
        level=args.level,
        max_pixels=args.max_pixels,
        seed=args.seed,
        out_figure=out_fig,
    )

    print("=== Cellpose mask vs DAPI ===")
    print(f"  CellDIVE : {args.celldive}")
    print(f"  Mask     : {args.mask}")
    print(f"  Level    : {s['level']}")
    print(f"  Shape    : {s['shape_2d']}")
    print(f"  DAPI ch  : {s['dapi_channel_index']}")
    print(f"  Foreground (mask) fraction of pixels: {s['frac_mask_fg']*100:.2f}%")
    print(f"  Mean DAPI inside  mask: {s['mean_dapi_inside']:,.1f}")
    print(f"  Mean DAPI outside mask: {s['mean_dapi_outside']:,.1f}")
    print(f"  Ratio (in/out):         {s['ratio_fg_over_bg']}")
    print(f"  Fraction of mask pixels with DAPI < global 5% pctl: {s['frac_fg_below_global_p5']*100:.2f}%")
    if out_fig and "figure" in s:
        print(f"  Figure:   {s['figure']}")


if __name__ == "__main__":
    main()
