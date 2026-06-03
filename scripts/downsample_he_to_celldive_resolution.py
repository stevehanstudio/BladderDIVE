#!/usr/bin/env python3
"""
Downsample H&E SVS to match CellDIVE spatial resolution.

The H&E SVS images are at 0.263 µm/pixel (40×). CellDIVE is at 0.325 µm/pixel.
This script resamples the H&E to 0.325 µm/px so both images share the same
resolution for easier registration/alignment.

Uses tiled processing to avoid loading the full slide into memory.

For QuPath and other whole-slide viewers that show "Image too large", use
--pyramidal to write a pyramidal OME-TIFF with multiple resolution levels.

Usage:
    # Flat TIFF (for downstream processing)
    python scripts/downsample_he_to_celldive_resolution.py "PH 001 C13_085117.svs" -o HE_celldive_resolution.tif

    # Pyramidal OME-TIFF (for QuPath, Napari, etc.)
    python scripts/downsample_he_to_celldive_resolution.py "PH 001 C13_085117.svs" -o HE_celldive_resolution.ome.tif --pyramidal
"""

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

try:
    import openslide
except ImportError:
    openslide = None

try:
    from skimage.transform import resize
except ImportError:
    resize = None

try:
    import pyvips
except ImportError:
    pyvips = None

# Known resolutions (µm/pixel)
HE_SVS_MPP = 0.263012
CELLDIVE_MPP = 0.325002437518281


def get_svs_mpp(slide) -> float:
    """Get microns-per-pixel from SVS, fallback to default if missing."""
    mpp = slide.properties.get("openslide.mpp-x") or slide.properties.get("aperio.MPP")
    if mpp is not None:
        return float(mpp)
    return HE_SVS_MPP


def downsample_svs_pyvips(
    svs_path: Path,
    output_path: Path,
    target_mpp: float,
    source_mpp: float,
    max_rows: Optional[int] = None,
) -> None:
    """
    Downsample SVS to target resolution using pyvips; writes pyramidal OME-TIFF.
    Uses lazy streaming - does not load full image into memory.
    """
    scale = source_mpp / target_mpp
    img = pyvips.Image.new_from_file(str(svs_path), access="sequential")
    img_resized = img.resize(scale)
    if max_rows is not None and img_resized.height > max_rows:
        img_resized = img_resized.crop(0, 0, img_resized.width, max_rows)
    img_resized.tiffsave(
        str(output_path),
        tile=True,
        pyramid=True,
        bigtiff=True,
        compression="lzw",
        tile_width=256,
        tile_height=256,
    )
    print(f"  Done: {output_path}")


def downsample_svs_to_target_mpp(
    svs_path: Path,
    output_path: Path,
    target_mpp: float = CELLDIVE_MPP,
    tile_height: int = 1024,
    max_rows: Optional[int] = None,
    pyramidal: bool = False,
) -> None:
    """
    Downsample SVS to target spatial resolution using tiled processing.

    Parameters
    ----------
    svs_path : Path
        Path to input SVS file
    output_path : Path
        Path to output TIFF file
    target_mpp : float
        Target microns per pixel (default: CellDIVE 0.325 µm/px)
    tile_height : int
        Number of output rows per tile (trade-off: smaller = less RAM, more I/O)
    pyramidal : bool
        If True, use pyvips to write pyramidal OME-TIFF (for QuPath, etc.)
    """
    # Pyramidal path: use pyvips (lazy streaming, no full load)
    if pyramidal:
        if pyvips is None:
            raise ImportError(
                "pyvips required for --pyramidal. Install: conda install -c conda-forge pyvips"
            )
        source_mpp = HE_SVS_MPP
        if openslide:
            with openslide.OpenSlide(str(svs_path)) as s:
                w0, h0 = s.dimensions
                source_mpp = get_svs_mpp(s)
        else:
            with pyvips.Image.new_from_file(str(svs_path), access="sequential") as probe:
                w0, h0 = probe.width, probe.height
        scale = source_mpp / target_mpp
        w_out = int(round(w0 * scale))
        h_out = int(round(h0 * scale))
        if max_rows is not None:
            h_out = min(h_out, max_rows)
        print(f"SVS: {svs_path.name}")
        print(f"  Source: {w0}×{h0} px @ {source_mpp:.4f} µm/px")
        print(f"  Target: {w_out}×{h_out} px @ {target_mpp:.4f} µm/px")
        print(f"  Writing pyramidal OME-TIFF (QuPath-compatible)...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        downsample_svs_pyvips(
            svs_path, output_path, target_mpp, source_mpp, max_rows=max_rows
        )
        return

    if openslide is None:
        raise ImportError("openslide-python required. Install: conda install -c conda-forge openslide-python")
    if resize is None:
        raise ImportError("scikit-image required. Install: pip install scikit-image")

    slide = openslide.OpenSlide(str(svs_path))
    source_mpp = get_svs_mpp(slide)
    w0, h0 = slide.dimensions

    scale = source_mpp / target_mpp  # < 1 since we're downsampling
    w_out = int(round(w0 * scale))
    h_out = int(round(h0 * scale))
    if max_rows is not None:
        h_out = min(h_out, max_rows)
        print(f"(Test mode: limiting to {h_out} rows)")

    print(f"SVS: {svs_path.name}")
    print(f"  Source: {w0}×{h0} px @ {source_mpp:.4f} µm/px")
    print(f"  Target: {w_out}×{h_out} px @ {target_mpp:.4f} µm/px")
    print(f"  Scale factor: {scale:.4f}")
    print(f"  Tile height: {tile_height} rows")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use memmap to avoid holding full image in RAM
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        out = np.memmap(tmp_path, dtype=np.uint8, mode="w+", shape=(h_out, w_out, 3))

        y = 0
        while y < h_out:
            h_tile = min(tile_height, h_out - y)
            y_in_start = int(y / scale)
            h_in = int(np.ceil((y + h_tile) / scale)) - y_in_start
            h_in = min(h_in, h0 - y_in_start)

            # Read from SVS (level 0, full width)
            patch = slide.read_region((0, y_in_start), 0, (w0, h_in))
            if patch.mode == "RGBA":
                patch = patch.convert("RGB")
            arr = np.array(patch)

            # Resize to output tile size (preserve RGB channels)
            resized = resize(
                arr,
                (h_tile, w_out, 3),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.uint8)

            out[y : y + h_tile, :, :] = resized
            y += h_tile

            if y % (tile_height * 5) == 0 or y >= h_out:
                print(f"  Progress: {min(y, h_out)}/{h_out} rows ({100 * min(y, h_out) / h_out:.1f}%)")

        out.flush()
        del out

        # Write to TIFF (BigTIFF for large files)
        size_gb = h_out * w_out * 3 / (1024**3)
        use_bigtiff = size_gb > 4

        print(f"\n  Writing TIFF (BigTIFF={use_bigtiff})...")
        memmap_arr = np.memmap(tmp_path, dtype=np.uint8, mode="r", shape=(h_out, w_out, 3))
        tifffile.imwrite(
            str(output_path),
            memmap_arr,
            photometric="rgb",
            compression="lzw",
            bigtiff=use_bigtiff,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    slide.close()
    print(f"  Done: {output_path}")


def resolve_svs_path(name: str, workspace: Path) -> Path:
    """
    Resolve SVS path, trying common name variants (e.g. 85117 vs 085117).
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    # Try in workspace
    q = workspace / name
    if q.exists():
        return q
    # Try variant with leading zero (85117 -> 085117)
    base = name
    if "85117" in base and "085117" not in base:
        alt = base.replace("85117", "085117")
        q = workspace / alt
        if q.exists():
            return q
    # Try vice versa
    if "085117" in base:
        alt = base.replace("085117", "85117")
        q = workspace / alt
        if q.exists():
            return q
    return Path(name)  # Return as-is, will fail on OpenSlide if missing


def main():
    parser = argparse.ArgumentParser(
        description="Downsample H&E SVS to CellDIVE spatial resolution (0.325 µm/px)"
    )
    parser.add_argument(
        "svs_file",
        type=str,
        nargs="?",
        default="raw/PH 001 C13_085117.svs",
        help='SVS file path (default: "PH 001 C13_085117.svs")',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/HE_celldive_resolution.tif",
        help="Output TIFF path (default: HE_celldive_resolution.tif)",
    )
    parser.add_argument(
        "--target-mpp",
        type=float,
        default=CELLDIVE_MPP,
        help=f"Target µm/pixel (default: {CELLDIVE_MPP} for CellDIVE)",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=1024,
        help="Rows per tile for memory efficiency (default: 1024)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only process first 2048 rows (for quick verification)",
    )
    parser.add_argument(
        "--pyramidal",
        action="store_true",
        help="Write pyramidal OME-TIFF for QuPath/whole-slide viewers (requires pyvips)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    workspace = script_dir.parent

    svs_path = resolve_svs_path(args.svs_file, workspace)
    if not svs_path.exists():
        print(f"Error: SVS file not found: {svs_path}")
        return 1

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = workspace / output_path

    downsample_svs_to_target_mpp(
        svs_path,
        output_path,
        target_mpp=args.target_mpp,
        tile_height=args.tile_height,
        max_rows=2048 if args.test else None,
        pyramidal=args.pyramidal,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
