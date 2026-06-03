#!/usr/bin/env python3
"""
Export 5 full-resolution RGB TIF composites for Biographic Image Contest 2026.

Each composite combines 3–5 channels with custom colors, gamma, and opacity.
Additive blending; channels summed and clipped. See CONTEST_EXPORT.md for full docs.

Usage:
  python scripts/export_biographic_contest_composites.py
  python scripts/export_biographic_contest_composites.py --photoshop
  python scripts/export_biographic_contest_composites.py --photoshop --format jpeg

Output:
  Invasive_Front.tif, Immune_Surveillance.tif, Vascular_Architecture.tif,
  Proliferation_Zone.tif, Tertiary_Lymphoid_Structures.tif

Tuning (edit COMPOSITES in this script):
  Gamma: 0.4–0.5 = brighter; 0.6–0.7 = balanced; 0.7–0.8 = more contrast
  Opacity: 0.3–0.5 = background; 0.6–0.8 = supporting; 0.9–1.0 = primary marker
  Blending: additive — lower opacity or gamma to reduce saturation where channels overlap

Photoshop 64 GB: --photoshop (PNG ~3 GB) or --photoshop --format jpeg (~200 MB)
"""

import argparse
import json
import numpy as np
import tifffile
import zarr
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Channel label -> index in CellDIVE zarr (from .zattrs)
CHANNEL_INDEX = {
    "DAPI_AF_R01": 0,
    "CD45-AF488-CST": 1,
    "CD3E-AF555-CST": 2,
    "Ki67-AF647-CST": 3,
    "CD8a-AF750-CST": 4,
    "Vim-AF488-CST": 5,
    "CD68-AF555-CST": 6,
    "HLA-DRA-AF4647": 7,
    "CD31-AF750": 8,
    "SMA-AF488": 9,   # ACTA2
    "CD20-AF555": 10,
    "CD163-AF647": 11,
    "CD44-AF750-Nov": 12,
    "PanCK-AF488": 13,
    "CD38-AF555": 14,
    "CD11c-AF647": 15,
    "PDGFRA-AF488": 16,
    "COL1A1-AF555": 17,
    "CD14-AF647": 18,
    "EPCAM-AF488": 19,
    "CD56-AF555": 20,
    "CD45RO-AF647-BioT": 21,
    "DAPI_R06": 22,
}

# Short names -> full zarr labels
CHANNEL_ALIAS = {
    "DAPI": "DAPI_AF_R01",
    "PANCK": "PanCK-AF488",
    "ACTA2": "SMA-AF488",
    "SMA": "SMA-AF488",
    "VIM": "Vim-AF488-CST",
    "CD8a": "CD8a-AF750-CST",
    "CD68": "CD68-AF555-CST",
    "CD163": "CD163-AF647",
    "CD31": "CD31-AF750",
    "PDGFRA": "PDGFRA-AF488",
    "COL1A1": "COL1A1-AF555",
    "CD45": "CD45-AF488-CST",
    "Ki67": "Ki67-AF647-CST",
    "CD44": "CD44-AF750-Nov",
    "CD20": "CD20-AF555",
    "CD3E": "CD3E-AF555-CST",
    "CD38": "CD38-AF555",
}


def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert #RRGGBB to (r, g, b) in 0–1."""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)


# Original spectral colors from CellDIVE zarr .zattrs (rainbow: blue -> cyan -> green -> yellow -> red)
ORIGINAL_COLORS = {
    "DAPI": "#000083",
    "CD45": "#0000B1",
    "CD3E": "#0000E0",
    "Ki67": "#000FFF",
    "CD8a": "#003DFF",
    "VIM": "#006CFF",
    "CD68": "#0099FF",
    "CD31": "#00F6FF",
    "ACTA2": "#25FFDA",
    "CD20": "#54FFAB",
    "CD163": "#82FF7E",
    "CD44": "#AFFF50",
    "PANCK": "#DEFF21",
    "CD38": "#FFF200",
    "PDGFRA": "#FF9500",
    "COL1A1": "#FF6800",
}

# --- Palette: "contest" (artistic, curated) ---
COMPOSITES_CONTEST = [
    (
        "Invasive_Front",
        [
            ("PANCK", "#DEFF21", 1.0, 0.55),    # Yellow-green (orig) — epithelial tumor, pops against blue
            ("VIM", "#006CFF", 0.85, 0.45),      # Blue (orig) — mesenchymal stroma
            ("ACTA2", "#25FFDA", 0.9, 0.55),     # Teal (orig) — smooth muscle
            ("Ki67", "#FFD700", 0.6, 0.55),      # Gold — proliferating cells (warm accent)
            ("DAPI", "#FFFFFF", 0.4, 0.5),       # White — nuclei (dim)
        ],
    ),
    (
        "Immune_Surveillance",
        [
            ("CD8a", "#FAFA33", 1.0, 0.5),      # Bright yellow — cytotoxic T cells (pops against blue)
            ("CD68", "#0099FF", 0.95, 0.45),     # Cyan (orig) — pan-macrophage
            ("CD163", "#82FF7E", 0.8, 0.55),     # Green (orig) — M2 macrophage
            ("DAPI", "#1F51FF", 0.6, 0.5),       # Neon blue — nuclei
        ],
    ),
    (
        "Vascular_Architecture",
        [
            ("CD31", "#40FFFF", 1.0, 0.4),       # Light cyan — endothelial / vessels (brighter than orig)
            ("COL1A1", "#FF6800", 0.8, 0.55),    # Orange (orig) — collagen ECM
            ("PDGFRA", "#FF0000", 0.5, 0.55),    # Bright red — fibroblasts (accent, distinct from orange)
            ("DAPI", "#1F51FF", 0.5, 0.5),        # Blue — nuclei
        ],
    ),
    (
        "Proliferation_Zone",
        [
            ("Ki67", "#CCFF00", 0.7, 0.85),      # Chartreuse — proliferating cells (yellow-green pop)
            ("PANCK", "#1F51FF", 1.0, 0.45),     # Neon blue — epithelium
            ("CD44", "#FF1493", 0.85, 0.5),      # Deep pink — adhesion / stemness
            ("DAPI", "#FFFFFF", 0.55, 0.5),      # White — nuclei
        ],
    ),
    (
        "Tertiary_Lymphoid_Structures",
        [
            ("CD20", "#2EB86A", 0.9, 0.5),       # Darkened green — B cells (dimmed so yellow pops)
            ("CD3E", "#0000E0", 0.95, 0.4),      # Blue (orig) — T cells
            ("CD38", "#FFF200", 1.0, 0.4),       # Bright yellow (orig) — plasma cells (star)
            ("PANCK", "#DEFF21", 0.4, 0.65),     # Yellow-green — epithelial context (dim)
            ("DAPI", "#1F51FF", 0.5, 0.5),       # Blue — nuclei
        ],
    ),
]


def _make_original_palette(contest_composites):
    """Build composites using original CellDIVE spectral colors, keeping opacity/gamma from contest palette."""
    result = []
    for title, channels in contest_composites:
        new_channels = []
        for ch_name, _hex, opacity, gamma in channels:
            color = ORIGINAL_COLORS.get(ch_name, _hex)
            new_channels.append((ch_name, color, opacity, gamma))
        result.append((title, new_channels))
    return result


COMPOSITES_ORIGINAL = _make_original_palette(COMPOSITES_CONTEST)

# Default palette
COMPOSITES = COMPOSITES_CONTEST


def resolve_channel(ch: str) -> int:
    label = CHANNEL_ALIAS.get(ch, ch)
    if label in CHANNEL_INDEX:
        return CHANNEL_INDEX[label]
    raise KeyError(f"Unknown channel: {ch}")


def compute_tissue_mask(store, dapi_idx: int = 0, threshold_percentile: float = 5) -> np.ndarray:
    """Build a binary tissue mask from a mid-resolution pyramid level using DAPI.

    Returns a boolean array at chosen resolution. Pixels below the threshold
    are considered background (outside tissue / between acquisition tiles).
    Does NOT fill holes -- missing acquisition tiles should stay masked out.
    """
    from scipy import ndimage

    level_keys = sorted([k for k in store.keys() if k.isdigit()], key=int)
    # Use level 3 (~8k px) for better tile-gap detection than level 4
    mask_level = min(3, int(level_keys[-1])) if level_keys else 0
    mask_key = str(mask_level)
    level_data = store[mask_key]
    dapi = np.array(level_data[dapi_idx]).astype(np.float32)

    thresh = np.percentile(dapi[dapi > 0], threshold_percentile)
    mask = dapi > thresh

    # Light cleanup: close small noise gaps, then erode to tighten edges.
    # Do NOT fill holes -- missing tiles must remain masked.
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_erosion(mask, iterations=1)

    print(f"  Tissue mask from level {mask_key} ({level_data.shape[1]}x{level_data.shape[2]}): "
          f"{mask.sum()}/{mask.size} px ({100 * mask.sum() / mask.size:.1f}% tissue)")
    return mask


def compute_global_percentiles(
    store,
    channels_config: list,
    p_low: float = 1,
    p_high: float = 99.5,
) -> dict[int, tuple[float, float]]:
    """Compute percentile limits from the coarsest pyramid level for consistent normalization."""
    level_keys = sorted([k for k in store.keys() if k.isdigit()], key=int)
    coarse_key = level_keys[-1] if level_keys else "0"
    coarse = store[coarse_key]
    print(f"  Computing global percentiles from pyramid level {coarse_key} ({coarse.shape[1]}x{coarse.shape[2]})")

    limits = {}
    for ch_name, _hex, _opacity, _gamma in channels_config:
        idx = resolve_channel(ch_name)
        if idx not in limits:
            arr = np.array(coarse[idx])
            lo, hi = np.percentile(arr, [p_low, p_high])
            if hi <= lo:
                hi = lo + 1
            limits[idx] = (float(lo), float(hi))
    return limits


def normalize_and_gamma(
    arr: np.ndarray,
    gamma: float = 0.6,
    limits: tuple[float, float] | None = None,
    p_low: float = 1,
    p_high: float = 99.5,
) -> np.ndarray:
    """Percentile-based contrast + gamma. Returns float 0-1.

    When `limits` is provided, uses pre-computed (lo, hi) for consistent
    normalization across chunks (avoids banding artifacts).
    """
    if limits is not None:
        lo, hi = limits
    else:
        lo, hi = np.percentile(arr, [p_low, p_high])
        if hi <= lo:
            hi = lo + 1
    norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return np.power(norm, gamma)


def get_mask_region(
    tissue_mask: np.ndarray,
    full_shape: tuple[int, int],
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Upscale a region of the coarse tissue mask to match the requested chunk size."""
    mask_h, mask_w = tissue_mask.shape
    full_h, full_w = full_shape
    my0 = int(y_start * mask_h / full_h)
    my1 = min(int(np.ceil(y_end * mask_h / full_h)) + 1, mask_h)
    mx0 = int(x_start * mask_w / full_w)
    mx1 = min(int(np.ceil(x_end * mask_w / full_w)) + 1, mask_w)
    region = tissue_mask[my0:my1, mx0:mx1]

    if HAS_PIL:
        pil = Image.fromarray(region.astype(np.uint8) * 255, mode="L")
        resized = pil.resize((x_end - x_start, y_end - y_start), Image.Resampling.NEAREST)
        return np.array(resized) > 127
    else:
        from scipy.ndimage import zoom
        sy = (y_end - y_start) / region.shape[0]
        sx = (x_end - x_start) / region.shape[1]
        return zoom(region.astype(np.float32), (sy, sx), order=0) > 0.5


def build_composite_chunk(
    zarr_data,
    channels_config: list,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    global_limits: dict[int, tuple[float, float]] | None = None,
    tissue_mask_region: np.ndarray | None = None,
    blend: str = "additive",
) -> np.ndarray:
    """Build one chunk of the RGB composite.

    Blend modes:
      additive: sum channels (classic fluorescence, can wash out to white)
      screen:   1-(1-a)*(1-b), like Photoshop Screen (saturates gracefully)
      max:      each pixel takes the color of the strongest channel (purest colors)
    """
    h, w = y_end - y_start, x_end - x_start
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for ch_name, hex_color, opacity, gamma in channels_config:
        idx = resolve_channel(ch_name)
        block = np.array(zarr_data[idx, y_start:y_end, x_start:x_end])
        ch_limits = global_limits.get(idx) if global_limits else None
        norm = normalize_and_gamma(block, gamma=gamma, limits=ch_limits)
        r, g, b = hex_to_rgb(hex_color)
        layer = np.zeros((h, w, 3), dtype=np.float32)
        layer[:, :, 0] = norm * r * opacity
        layer[:, :, 1] = norm * g * opacity
        layer[:, :, 2] = norm * b * opacity

        if blend == "screen":
            rgb = 1.0 - (1.0 - rgb) * (1.0 - layer)
        elif blend == "max":
            rgb = np.maximum(rgb, layer)
        else:
            rgb += layer

    rgb = np.clip(rgb, 0, 1)

    if tissue_mask_region is not None:
        rgb[~tissue_mask_region] = 0

    return (rgb * 255).astype(np.uint8)


def _pick_pyramid_level(store, max_size: int) -> tuple[str, float]:
    """Choose the smallest pyramid level whose longest side >= max_size.

    Returns (level_key, remaining_scale) where remaining_scale < 1 means
    we still need to downsample slightly after reading from that level.
    """
    level_keys = sorted([k for k in store.keys() if k.isdigit()], key=int)
    for key in reversed(level_keys):
        h, w = store[key].shape[1], store[key].shape[2]
        if max(h, w) >= max_size:
            scale = min(max_size / h, max_size / w)
            return key, scale
    finest = level_keys[0]
    h, w = store[finest].shape[1], store[finest].shape[2]
    return finest, min(max_size / h, max_size / w, 1.0)


def export_composite(
    zarr_path: Path,
    output_path: Path,
    title: str,
    channels_config: list,
    chunk_rows: int = 2048,
    test_crop: tuple[int, int] | None = None,
    fmt: str = "tiff",
    max_size: int | None = None,
    blend: str = "additive",
) -> None:
    """Export one composite to RGB (TIF, PNG, or JPEG)."""
    store = zarr.open(str(zarr_path), mode="r")

    # Pick an appropriate pyramid level to avoid loading massive data
    if max_size and not test_crop:
        src_level, residual_scale = _pick_pyramid_level(store, max_size)
    else:
        src_level, residual_scale = "0", 1.0

    data = store[src_level]
    full0_h, full0_w = store["0"].shape[1], store["0"].shape[2]
    src_h, src_w = data.shape[1], data.shape[2]

    y_off, x_off = 0, 0
    if test_crop:
        crop_h, crop_w = test_crop
        y_off = max(0, (src_h - crop_h) // 2)
        x_off = max(0, (src_w - crop_w) // 2)
        src_h = min(crop_h, src_h - y_off)
        src_w = min(crop_w, src_w - x_off)
        print(f"  {title}: {src_w}×{src_h} (center crop)")
    else:
        print(f"  {title}: {full0_w}×{full0_h} (reading from level {src_level}: {src_w}×{src_h})")

    global_limits = compute_global_percentiles(store, channels_config)
    tissue_mask = compute_tissue_mask(store, dapi_idx=resolve_channel("DAPI"))

    need_resize = residual_scale < 0.99
    if need_resize:
        out_h = int(src_h * residual_scale)
        out_w = int(src_w * residual_scale)
        print(f"  Final output: {out_w}×{out_h} (residual scale {residual_scale:.3f})")
    else:
        out_h, out_w = src_h, src_w

    # Build composite in chunks directly from chosen pyramid level
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mem_per_chunk = out_w * chunk_rows * 3 * 4 / 1e6  # float32 RGB estimate
    print(f"  Chunk size: {chunk_rows} rows (~{mem_per_chunk:.0f} MB per chunk)")

    for y0 in range(0, out_h, chunk_rows):
        y1 = min(y0 + chunk_rows, out_h)

        if need_resize:
            iy0 = int(y0 / residual_scale)
            iy1 = min(int(np.ceil(y1 / residual_scale)) + 1, src_h)

            # Mask region must match the chunk being read (src level coords)
            mask_region = get_mask_region(
                tissue_mask, (data.shape[1], data.shape[2]),
                y_off + iy0, y_off + iy1, x_off, x_off + src_w,
            )

            chunk = build_composite_chunk(
                data, channels_config,
                y_off + iy0, y_off + iy1, x_off, x_off + src_w,
                global_limits=global_limits, tissue_mask_region=mask_region,
                blend=blend,
            )
            if HAS_PIL:
                pil = Image.fromarray(chunk, mode="RGB")
                resized = pil.resize((out_w, y1 - y0), Image.Resampling.LANCZOS)
                rgb[y0:y1, :, :] = np.array(resized)
            else:
                step_y = max(1, chunk.shape[0] // (y1 - y0))
                step_x = max(1, chunk.shape[1] // out_w)
                rgb[y0:y1, :, :] = chunk[::step_y, ::step_x, :][: (y1 - y0), :out_w, :]
        else:
            mask_region = get_mask_region(
                tissue_mask, (data.shape[1], data.shape[2]),
                y_off + y0, y_off + y1, x_off, x_off + out_w,
            )

            chunk = build_composite_chunk(
                data, channels_config,
                y_off + y0, y_off + y1, x_off, x_off + out_w,
                global_limits=global_limits, tissue_mask_region=mask_region,
                blend=blend,
            )
            rgb[y0:y1, :, :] = chunk

        print(f"    rows {y0}-{y1} / {out_h}")

    # Ensure output path has correct extension for format
    out = output_path
    if fmt == "png" and out.suffix.lower() != ".png":
        out = out.with_suffix(".png")
    elif fmt == "jpeg" and out.suffix.lower() not in (".jpg", ".jpeg"):
        out = out.with_suffix(".jpg")

    if fmt == "tiff":
        # Tiled for partial loading. LZW for Fiji compatibility (deflate not supported).
        write_kw = dict(
            photometric="rgb",
            planarconfig="contig",
            tile=(512, 512),
        )
        try:
            tifffile.imwrite(str(out), rgb, **write_kw, compression="lzw")
        except Exception:
            tifffile.imwrite(str(out), rgb, **write_kw)  # uncompressed fallback
    elif fmt in ("png", "jpeg") and HAS_PIL:
        JPEG_MAX = 65500
        if fmt == "jpeg" and (rgb.shape[0] > JPEG_MAX or rgb.shape[1] > JPEG_MAX):
            scale = min(JPEG_MAX / rgb.shape[0], JPEG_MAX / rgb.shape[1])
            new_h, new_w = int(rgb.shape[0] * scale), int(rgb.shape[1] * scale)
            print(f"  JPEG max is {JPEG_MAX} px; resizing to {new_w}x{new_h}")
            pil = Image.fromarray(rgb, mode="RGB")
            pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            pil = Image.fromarray(rgb, mode="RGB")
        if fmt == "jpeg":
            pil.save(str(out), "JPEG", quality=95)
        else:
            pil.save(str(out), "PNG")
    else:
        if fmt in ("png", "jpeg") and not HAS_PIL:
            raise RuntimeError("PNG/JPEG export requires Pillow: pip install Pillow")
        raise ValueError(f"Unknown format: {fmt}")

    print(f"  Saved: {out}")

    # Preview for full-res TIF: save as JPEG at 20000px longest side, quality 70
    if fmt == "tiff" and not max_size and (out_h > 8192 or out_w > 8192) and HAS_PIL:
        max_preview = 20000
        scale = min(max_preview / out_h, max_preview / out_w, 1.0)
        new_w = int(out_w * scale)
        new_h = int(out_h * scale)
        pil = Image.fromarray(rgb, mode="RGB")
        preview_pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        preview_path = out.with_stem(out.stem + "_preview").with_suffix(".jpg")
        preview_pil.save(str(preview_path), "JPEG", quality=70)
        print(f"  Preview: {preview_path} ({new_w}×{new_h})")


def main():
    parser = argparse.ArgumentParser(
        description="Export Biographic Image Contest 2026 composites from CellDIVE zarr.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        default="data/CellDIVE_SLIDE-045.zarr",
        help="Path to CellDIVE zarr (default: data/CellDIVE_SLIDE-045.zarr)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output/contest_2026",
        help="Output directory for TIF files (default: output/contest_2026)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=2048,
        help="Rows per chunk for memory efficiency (default: 2048)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: export only first 2000×2000 px of each composite",
    )
    parser.add_argument(
        "--format",
        choices=["tiff", "png", "jpeg"],
        default="tiff",
        help="Output format (default: tiff). PNG=lossless, JPEG=smaller with quality 95",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="N",
        help="Max dimension in pixels. Downsamples to fit (safe for Photoshop 64 GB)",
    )
    parser.add_argument(
        "--photoshop",
        action="store_true",
        help="Optimize for Photoshop on 64 GB: max-size 32768, PNG (override with --format jpeg)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        nargs="?",
        const=2000,
        default=None,
        metavar="PX",
        help="Quick preview: cap longest side at PX (default 2000) and output JPEG",
    )
    parser.add_argument(
        "--palette",
        choices=["contest", "original"],
        default="contest",
        help="Color palette: 'contest' (artistic, curated) or 'original' (CellDIVE spectral rainbow)",
    )
    parser.add_argument(
        "--blend",
        choices=["additive", "screen", "max"],
        default="additive",
        help="Blending mode: additive (classic), screen (no washout), max (purest colors)",
    )
    args = parser.parse_args()

    if args.preview:
        args.max_size = args.preview
        args.format = "jpeg"
    elif args.photoshop:
        if args.max_size is None:
            args.max_size = 32768
        if args.format == "tiff":
            args.format = "png"

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    zarr_path = project_root / args.input
    out_dir = project_root / args.output_dir

    if not zarr_path.exists():
        parser.error(f"Zarr not found: {zarr_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {zarr_path}")
    print(f"Output: {out_dir}")
    print()

    composites = COMPOSITES_ORIGINAL if args.palette == "original" else COMPOSITES_CONTEST
    print(f"Palette: {args.palette}  |  Blend: {args.blend}\n")

    test_crop = (2000, 2000) if args.test else None
    if args.test:
        print("TEST MODE: exporting 2000×2000 px crops only\n")
    if args.preview:
        print(f"PREVIEW MODE: {args.preview} px max, JPEG\n")
    elif args.max_size:
        print(f"Max dimension {args.max_size} px ({args.format.upper()})\n")

    ext = {"tiff": ".tif", "png": ".png", "jpeg": ".jpg"}[args.format]
    for title, channels_config in composites:
        out_path = out_dir / f"{title}{ext}"
        export_composite(
            zarr_path,
            out_path,
            title,
            channels_config,
            chunk_rows=args.chunk_rows,
            test_crop=test_crop,
            fmt=args.format,
            max_size=args.max_size,
            blend=args.blend,
        )

    print("\nDone. Tuning: edit COMPOSITES in this script.")
    print("  Gamma: 0.4–0.5 brighter | 0.6–0.7 balanced | 0.7–0.8 more contrast")
    print("  Opacity: 0.3–0.5 background | 0.9–1.0 primary marker")
    print("  Full guide: CONTEST_EXPORT.md")


if __name__ == "__main__":
    main()
