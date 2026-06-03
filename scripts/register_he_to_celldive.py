#!/usr/bin/env python3
"""
Register H&E image to CellDIVE zarr for aligned viewing in Napari.

Extracts DAPI from CellDIVE, crops H&E to a region ~10-15% larger than CellDIVE,
registers using phase cross-correlation, and saves the aligned H&E.

Outputs:
  - HE_registered_to_celldive.ome.tif (or .tif): aligned H&E, 10-15% larger than CellDIVE
  - he_celldive_transform.json: transform params for Napari positioning

Usage:
  python scripts/register_he_to_celldive.py \\
      --he-image HE_celldive_resolution.ome.tif \\
      --celldive-zarr data/CellDIVE_SLIDE-045.zarr \\
      --output-dir data/he_celldive_registration \\
      --margin-percent 12
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile

try:
    import zarr
except ImportError:
    zarr = None

try:
    from skimage.registration import phase_cross_correlation
    from skimage.transform import resize
    from scipy.ndimage import shift as ndshift
except ImportError:
    phase_cross_correlation = resize = ndshift = None

try:
    import pyvips
except ImportError:
    pyvips = None


def get_celldive_dapi(zarr_path: Path) -> np.ndarray:
    """Extract DAPI channel (index 0) from CellDIVE zarr level 0."""
    if zarr is None:
        raise ImportError("zarr required. Install: pip install zarr")
    group = zarr.open(str(zarr_path), mode="r")
    arr = group["0"]
    # Shape: (C, Y, X); DAPI is first channel
    dapi = np.array(arr[0, :, :])
    return dapi


def get_celldive_shape(zarr_path: Path) -> Tuple[int, int]:
    """Get (height, width) of CellDIVE level 0."""
    group = zarr.open(str(zarr_path), mode="r")
    arr = group["0"]
    return (int(arr.shape[1]), int(arr.shape[2]))


def load_he_region(
    he_path: Path, x0: int, y0: int, width: int, height: int, scale: float = 1.0
) -> np.ndarray:
    """Load a region from H&E. Uses pyvips for large pyramidal TIFFs if available."""
    if pyvips is not None:
        img = pyvips.Image.new_from_file(str(he_path), access="sequential")
        w = min(width, img.width - x0)
        h = min(height, img.height - y0)
        if w <= 0 or h <= 0:
            raise ValueError(f"Crop region ({x0},{y0})+{width}x{height} outside image {img.width}x{img.height}")
        region = img.crop(x0, y0, w, h)
        if scale != 1.0:
            region = region.resize(scale)
        arr = np.ndarray(
            buffer=region.write_to_memory(),
            dtype=np.uint8,
            shape=(region.height, region.width, 3) if region.bands >= 3 else (region.height, region.width),
        )
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return np.ascontiguousarray(arr)

    with tifffile.TiffFile(he_path) as tif:
        img = tif.asarray(out="memmap")
        if img.ndim == 3 and img.shape[2] >= 3:
            crop = np.array(img[y0 : y0 + height, x0 : x0 + width, :3])
        else:
            crop = np.array(img[y0 : y0 + height, x0 : x0 + width])
        if scale != 1.0 and resize is not None:
            h, w = crop.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            if crop.ndim == 3:
                crop = resize(crop, (new_h, new_w, 3), preserve_range=True, anti_aliasing=True).astype(np.uint8)
            else:
                crop = resize(crop, (new_h, new_w), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        return crop


def get_he_shape(he_path: Path) -> Tuple[int, int]:
    """Get (height, width) of H&E level 0."""
    with tifffile.TiffFile(he_path) as tif:
        if hasattr(tif, "series") and tif.series:
            s = tif.series[0]
            shape = s.shape
            if len(shape) >= 2:
                return (int(shape[-2]), int(shape[-1]))
        page = tif.pages[0]
        return (int(page.imagelength), int(page.imagewidth))


def find_rough_offset(
    ref_small: np.ndarray,
    moving_small: np.ndarray,
    downsample: int,
) -> Tuple[float, float]:
    """Find translation: shift aligns moving to ref. Returns (dx, dy) in full-res coords."""
    shift, _, _ = phase_cross_correlation(ref_small, moving_small, upsample_factor=2)
    # shift = (dy, dx): move moving by this to align with ref. ref(0,0) ≈ moving(shift[0], shift[1])
    # In full-res coords: multiply by downsample
    dx = shift[1] * downsample
    dy = shift[0] * downsample
    return float(dx), float(dy)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale."""
    if rgb.ndim == 2:
        return rgb
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)


def register_he_to_celldive(
    he_path: Path,
    celldive_zarr_path: Path,
    output_dir: Path,
    margin_percent: float = 12.0,
    coarse_downsample: int = 8,
    fine_downsample: int = 2,
    output_scale: float = 0.25,
) -> dict:
    """
    Register H&E to CellDIVE. Crops H&E to CellDIVE + margin, finds alignment, saves result.

    Returns transform dict for Napari (translate_yx, scale, he_shape, celldive_shape).
    """
    if phase_cross_correlation is None or resize is None or ndshift is None:
        raise ImportError("scikit-image and scipy required")

    output_dir.mkdir(parents=True, exist_ok=True)

    cd_h, cd_w = get_celldive_shape(celldive_zarr_path)
    he_h, he_w = get_he_shape(he_path)

    # Target H&E crop size: CellDIVE + margin (10-15%)
    scale_margin = 1.0 + margin_percent / 100.0
    target_h = int(round(cd_h * scale_margin))
    target_w = int(round(cd_w * scale_margin))

    if he_h < target_h or he_w < target_w:
        raise ValueError(
            f"H&E ({he_w}×{he_h}) is smaller than target ({target_w}×{target_h}). "
            "Use full-resolution H&E or reduce --margin-percent."
        )

    # Load DAPI
    print("Loading DAPI from CellDIVE...")
    dapi = get_celldive_dapi(celldive_zarr_path)
    print(f"  DAPI: {dapi.shape}")

    # Coarse registration: downsample both, find approximate offset
    print(f"Coarse registration (downsample={coarse_downsample})...")
    dapi_small = resize(
        (dapi.astype(np.float32) / max(dapi.max(), 1) * 255).astype(np.uint8),
        (cd_h // coarse_downsample, cd_w // coarse_downsample),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    # Center crop from H&E for coarse search (assume overlap is roughly centered)
    crop_margin = max(0, (he_h - target_h) // 2)
    y_cand = max(0, crop_margin)
    x_cand = max(0, (he_w - target_w) // 2)
    he_crop = load_he_region(he_path, x_cand, y_cand, target_w, target_h)
    he_gray = rgb_to_gray(he_crop)
    he_small = resize(
        he_gray,
        (target_h // coarse_downsample, target_w // coarse_downsample),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    # Match sizes for phase_cross_correlation
    min_h = min(dapi_small.shape[0], he_small.shape[0])
    min_w = min(dapi_small.shape[1], he_small.shape[1])
    dapi_patch = dapi_small[:min_h, :min_w]
    he_patch = he_small[:min_h, :min_w]

    # ref=he, moving=dapi: (dx, dy) = offset of dapi(0,0) within he_crop
    dx0, dy0 = find_rough_offset(he_patch, dapi_patch, coarse_downsample)
    margin_x = (target_w - cd_w) / 2.0
    margin_y = (target_h - cd_h) / 2.0
    # Crop top-left so dapi sits at (margin_x, margin_y) in crop
    x0 = int(max(0, x_cand + dx0 - margin_x))
    y0 = int(max(0, y_cand + dy0 - margin_y))
    x0 = min(x0, he_w - target_w)
    y0 = min(y0, he_h - target_h)
    x0 = max(0, x0)
    y0 = max(0, y0)

    print(f"  Coarse crop offset: (x={x0}, y={y0})")

    # Load the refined H&E crop (at reduced scale for registration if very large)
    reg_scale = 1.0
    if target_h * target_w > 4000 * 4000:
        reg_scale = max(0.25, 4000 / (target_h * target_w) ** 0.5)
        print(f"  Using registration scale {reg_scale:.2f} for memory")
    he_crop = load_he_region(he_path, x0, y0, target_w, target_h, scale=reg_scale)
    reg_h, reg_w = he_crop.shape[:2]
    he_gray = rgb_to_gray(he_crop)

    # Fine registration
    print(f"Fine registration (downsample={fine_downsample})...")
    dapi_fine = resize(
        (dapi.astype(np.float32) / max(dapi.max(), 1) * 255).astype(np.uint8),
        (cd_h // fine_downsample, cd_w // fine_downsample),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    he_fine = resize(
        he_gray,
        (reg_h // fine_downsample, reg_w // fine_downsample),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    min_h = min(dapi_fine.shape[0], he_fine.shape[0])
    min_w = min(dapi_fine.shape[1], he_fine.shape[1])
    dapi_patch = dapi_fine[:min_h, :min_w]
    he_patch = he_fine[:min_h, :min_w]

    dx_fine, dy_fine = find_rough_offset(he_patch, dapi_patch, fine_downsample)
    shift_x_fine = dx_fine
    shift_y_fine = dy_fine
    print(f"  Fine shift: (dx={shift_x_fine:.1f}, dy={shift_y_fine:.1f})")

    # Apply shift to H&E crop (subpixel)
    registered = np.zeros_like(he_crop, dtype=he_crop.dtype)
    for c in range(3):
        registered[:, :, c] = ndshift(
            he_crop[:, :, c],
            (shift_y_fine, shift_x_fine),
            order=1,
            mode="constant",
            cval=0,
        )

    # Save registered H&E
    out_he = output_dir / "HE_registered_to_celldive.tif"
    print(f"Saving: {out_he}")
    tifffile.imwrite(
        str(out_he),
        registered,
        photometric="rgb",
        compression="lzw",
        bigtiff=registered.nbytes > 4 * 1024**3,
    )

    # Napari positioning: CellDIVE is at (0,0). H&E is larger; we want the overlapping
    # region to align. The overlap is the CellDIVE region, centered in the H&E crop.
    # H&E crop size: target_w x target_h. CellDIVE: cd_w x cd_h.
    # Margin on each side: (target_w - cd_w)/2, (target_h - cd_h)/2
    margin_x = (target_w - cd_w) / 2.0
    margin_y = (target_h - cd_h) / 2.0
    # So in Napari data coords (y,x): H&E's (0,0) should appear at (-margin_y, -margin_x)
    # so that H&E's (margin_y, margin_x) aligns with CellDIVE's (0,0).
    # Plus the fine shift we applied (we shifted he crop, so the "content" moved).
    # The fine shift was applied to align he to dapi. So we might need to adjust.
    # Actually: we want CellDIVE at (0,0) and H&E such that when you overlay them, they align.
    # H&E is target_w x target_h. The DAPI-equivalent region in H&E is centered at
    # (target_w/2, target_h/2) roughly, with size cd_w x cd_h. So the top-left of that
    # region in H&E is at (margin_x, margin_y). In Napari, CellDIVE (0,0) corresponds to
    # H&E (margin_y, margin_x). So H&E layer's translate should be (-margin_y, -margin_x)
    # so that H&E pixel (margin_y, margin_x) maps to data coords (0, 0) = CellDIVE (0,0).
    translate_y = -margin_y
    translate_x = -margin_x

    # H&E pixel size: 0.325 µm/px at full res; if we saved at reg_scale, scale increases
    he_scale_um = 0.325 / reg_scale
    transform = {
        "he_path": str(out_he),
        "celldive_shape": [cd_h, cd_w],
        "he_shape": list(registered.shape[:2]),
        "translate": [translate_y, translate_x],
        "scale": [he_scale_um, he_scale_um],
        "celldive_scale": [0.325, 0.325],
        "margin_percent": margin_percent,
        "reg_scale": reg_scale,
    }
    transform_path = output_dir / "he_celldive_transform.json"
    with open(transform_path, "w") as f:
        json.dump(transform, f, indent=2)
    print(f"Saved transform: {transform_path}")

    return transform


def main():
    parser = argparse.ArgumentParser(
        description="Register H&E to CellDIVE for aligned viewing in Napari"
    )
    parser.add_argument("--he-image", type=str, required=True, help="H&E TIFF (CellDIVE resolution)")
    parser.add_argument(
        "--celldive-zarr",
        type=str,
        default="data/CellDIVE_SLIDE-045.zarr",
        help="CellDIVE zarr path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/he_celldive_registration",
        help="Output directory",
    )
    parser.add_argument(
        "--margin-percent",
        type=float,
        default=12.0,
        help="H&E crop margin around CellDIVE FOV (default: 12, so ~10-15%% larger)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    workspace = script_dir.parent

    he_path = Path(args.he_image)
    if not he_path.is_absolute():
        he_path = workspace / he_path
    if not he_path.exists():
        print(f"Error: H&E not found: {he_path}")
        return 1

    zarr_path = Path(args.celldive_zarr)
    if not zarr_path.is_absolute():
        zarr_path = workspace / zarr_path
    if not zarr_path.exists():
        print(f"Error: CellDIVE zarr not found: {zarr_path}")
        return 1

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir

    register_he_to_celldive(
        he_path,
        zarr_path,
        output_dir,
        margin_percent=args.margin_percent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
