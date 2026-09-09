"""
Extract validation ROIs from the full slide for the segmentation bake-off.

Every candidate segmenter must see byte-identical input, so each ROI is written once as a
multi-channel OME-TIFF (all 23 channels, level 0, uint16) plus the derived single-channel
inputs that two-channel models need:

    <roi>.ome.tif            23 x H x W, all channels, channel names in OME metadata
    <roi>_nuclear.tif        DAPI only
    <roi>_membrane.tif       PanCK + CD45 max-composite (Mesmer / Cellpose cyto input)
    <roi>_mask_dapi_only.tif the current production mask, for reference scoring

ROIs are chosen to isolate the two failure modes driving Artifact_Spillover:
  lymphoid  - densest CD20, targets CD3E+/CD20+ and maximal axial crowding
  vascular  - densest CD31, targets CD45+/CD31+ and dim nuclei
  invasive  - PanCK and VIM co-dense, tests epithelial boundary quality

Usage:
    python scripts/extract_validation_rois.py
    python scripts/extract_validation_rois.py --size 4096 --rois lymphoid vascular
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ZARR = PROJECT_ROOT / "data" / "CellDIVE_SLIDE-045.zarr"
MASK_ZARR = PROJECT_ROOT / "output" / "cellpose_output" / "cellpose_masks_dapi_only_9tiles.zarr"
REFERENCE_H5AD = PROJECT_ROOT / "output" / "celldive_protein_matrix.h5ad"
OUT_DIR = PROJECT_ROOT / "output" / "validation_rois"

PIXEL_UM = 0.325
SEARCH_LEVEL = 4

# marker(s) whose local density defines each ROI
ROI_MARKERS = {
    "lymphoid": ["CD20"],
    "vascular": ["CD31"],
    "invasive": ["PANCK", "VIM"],
}


def channel_names() -> list[str]:
    import anndata as ad

    return list(ad.read_h5ad(REFERENCE_H5AD, backed="r").var_names)


def find_roi(image, col: dict, markers: list[str], size: int) -> tuple[tuple[int, int], float]:
    """Locate the window maximising joint high-signal density for the given markers."""
    factor = 2**SEARCH_LEVEL
    win = size // factor
    half = win // 2
    joint = None
    for m in markers:
        plane = np.asarray(image[str(SEARCH_LEVEL)][col[m]]).astype(np.float32)
        nz = plane[plane > 0]
        cut = np.percentile(nz, 99) if nz.size else 0.0
        dens = ndimage.uniform_filter((plane > cut).astype(np.float32), size=win)
        joint = dens if joint is None else np.minimum(joint, dens)
    joint[:half, :] = 0
    joint[-half:, :] = 0
    joint[:, :half] = 0
    joint[:, -half:] = 0
    cy, cx = np.unravel_index(int(np.argmax(joint)), joint.shape)
    return (int((cy - half) * factor), int((cx - half) * factor)), float(joint[cy, cx])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--rois", nargs="+", default=list(ROI_MARKERS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    names = channel_names()
    col = {m: i for i, m in enumerate(names)}
    image = zarr.open(str(IMAGE_ZARR), mode="r")
    masks = zarr.open(str(MASK_ZARR), mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for roi in args.rois:
        markers = ROI_MARKERS[roi]
        (y0, x0), score = find_roi(image, col, markers, args.size)
        s = args.size
        sl = (slice(y0, y0 + s), slice(x0, x0 + s))

        stack = np.asarray(image["0"][(slice(None),) + sl])
        mask = np.asarray(masks["0"][sl])
        n_cells = int(len(np.unique(mask)) - 1)

        base = args.out_dir / roi
        tifffile.imwrite(
            f"{base}.ome.tif",
            stack,
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": PIXEL_UM,
                "PhysicalSizeY": PIXEL_UM,
                "PhysicalSizeXUnit": "\u00b5m",
                "PhysicalSizeYUnit": "\u00b5m",
                "Channel": {"Name": names},
            },
            ome=True,
        )
        tifffile.imwrite(f"{base}_nuclear.tif", stack[col["DAPI"]])
        membrane = np.maximum(stack[col["PANCK"]], stack[col["CD45"]])
        tifffile.imwrite(f"{base}_membrane.tif", membrane)
        tifffile.imwrite(f"{base}_mask_dapi_only.tif", mask.astype(np.uint32))

        manifest[roi] = {
            "markers": markers,
            "origin_yx_level0": [y0, x0],
            "size_px": s,
            "size_um": round(s * PIXEL_UM, 1),
            "density_score": round(score, 4),
            "n_cells_dapi_only_mask": n_cells,
            "pixel_size_um": PIXEL_UM,
            "channels": names,
        }
        print(
            f"{roi:9s} origin=({y0},{x0}) score={score:.3f} "
            f"{n_cells:,} cells in production mask -> {base.name}.ome.tif"
        )
        del stack, mask

    mpath = args.out_dir / "rois.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {mpath}")
    print("channel order:", ", ".join(names))


if __name__ == "__main__":
    main()
