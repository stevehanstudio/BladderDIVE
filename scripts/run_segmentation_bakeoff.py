"""
Run a candidate segmenter over the validation ROIs and write a label image per ROI.

Each model reads the same OME-TIFF written by scripts/extract_validation_rois.py, so any
difference in the downstream spillover score is attributable to the segmenter rather than
to preprocessing.

Models live in different containers, so imports are deliberately lazy:

    # InstanSeg (all 23 channels, ChannelNet)
    docker run --rm --gpus all --ipc=host -v "$PWD":/workspace instanseg-nvidia:latest \
        python scripts/run_segmentation_bakeoff.py --model instanseg

    # Cellpose-SAM (DAPI + PanCK/CD45 membrane composite), the incumbent
    docker run --rm --gpus all --ipc=host -v "$PWD":/workspace cellpose-nvidia:latest \
        python scripts/run_segmentation_bakeoff.py --model cellpose

Outputs, per ROI, into output/bakeoff_masks/:
    <roi>__<variant>.tif        uint32 label image (whole cells)
    <roi>__<variant>_nuclei.tif uint32 nuclei labels, when the model emits them
    <roi>__<variant>.json       runtime, cell count, model configuration
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROI_DIR = PROJECT_ROOT / "output" / "validation_rois"
OUT_DIR = PROJECT_ROOT / "output" / "bakeoff_masks"
PIXEL_UM = 0.325

# Channel order is fixed by output/validation_rois/rois.json.
NUCLEAR = "DAPI"
MEMBRANE = ["PANCK", "CD45"]


def load_roi(roi: str) -> tuple[np.ndarray, list[str]]:
    manifest = json.loads((ROI_DIR / "rois.json").read_text())
    names = manifest[roi]["channels"]
    stack = tifffile.imread(ROI_DIR / f"{roi}.ome.tif")
    assert stack.shape[0] == len(names), f"{stack.shape[0]} planes vs {len(names)} names"
    return stack, names


def run_instanseg(stack, names, args):
    import torch
    from instanseg import InstanSeg

    model = InstanSeg(args.instanseg_model, verbosity=1)
    # InstanSeg's fluorescence model is channel-invariant (ChannelNet), so it consumes all
    # 23 markers directly. That is the whole point of testing it: unlike a two-channel
    # model, it can see CD3E and CD20 as distinct evidence when separating adjacent
    # T and B cells in a lymphoid region.
    img = torch.from_numpy(stack.astype(np.float32))
    out = model.eval_medium_image(
        img,
        pixel_size=PIXEL_UM,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        target="all_outputs",
        return_image_tensor=False,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 2:
        nuclei, cells = arr[0], arr[1]
    elif arr.ndim == 2:
        nuclei, cells = None, arr
    else:
        raise RuntimeError(f"unexpected InstanSeg output shape {arr.shape}")
    cfg = {
        "model": args.instanseg_model,
        "channels_used": names,
        "n_channels": len(names),
        "pixel_size_um": PIXEL_UM,
        "tile_size": args.tile_size,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return cells, nuclei, cfg


def run_cellpose(stack, names, args):
    import torch
    from cellpose import models

    col = {m: i for i, m in enumerate(names)}
    nuc = stack[col[NUCLEAR]].astype(np.float32)
    mem = np.maximum.reduce([stack[col[m]] for m in MEMBRANE]).astype(np.float32)
    # Cellpose expects (cytoplasm, nucleus); this mirrors the production 2-channel input.
    img = np.stack([mem, nuc], axis=-1)

    # Cellpose 4.x ignores model_type and always loads cpsam (Cellpose-SAM); the
    # production runs labelled "cyto3" in segmentation_summary*.txt were in fact cpsam.
    model = models.CellposeModel(gpu=True)
    result = model.eval(img, diameter=args.diameter)
    cells = result[0] if isinstance(result, tuple) else result
    cfg = {
        "model": "cpsam (Cellpose-SAM)",
        "channels_used": [MEMBRANE, NUCLEAR],
        "n_channels": 2,
        "diameter": args.diameter,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return np.asarray(cells), None, cfg


RUNNERS = {"instanseg": run_instanseg, "cellpose": run_cellpose}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--rois", nargs="+", default=["lymphoid", "vascular", "invasive"])
    ap.add_argument("--variant", default=None, help="output label, defaults to --model")
    ap.add_argument("--instanseg-model", default="fluorescence_nuclei_and_cells")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--diameter", type=float, default=30.0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    variant = args.variant or args.model
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = RUNNERS[args.model]

    for roi in args.rois:
        stack, names = load_roi(roi)
        print(f"\n=== {roi} :: {variant} === input {stack.shape} {stack.dtype}", flush=True)
        t0 = time.perf_counter()
        cells, nuclei, cfg = runner(stack, names, args)
        elapsed = time.perf_counter() - t0

        cells = cells.astype(np.uint32)
        n_cells = int(cells.max())
        areas = np.bincount(cells.ravel())[1:]
        areas = areas[areas > 0]

        base = args.out_dir / f"{roi}__{variant}"
        tifffile.imwrite(f"{base}.tif", cells)
        if nuclei is not None:
            tifffile.imwrite(f"{base}_nuclei.tif", nuclei.astype(np.uint32))

        meta = {
            "roi": roi,
            "variant": variant,
            "runner": args.model,
            "runtime_s": round(elapsed, 1),
            "n_cells": n_cells,
            "mean_area_px": float(areas.mean()) if areas.size else 0.0,
            "median_area_px": float(np.median(areas)) if areas.size else 0.0,
            "has_nuclei": nuclei is not None,
            "config": cfg,
        }
        Path(f"{base}.json").write_text(json.dumps(meta, indent=2))
        print(
            f"    {n_cells:,} cells, mean area {meta['mean_area_px']:.1f} px, "
            f"{elapsed:.1f}s -> {base.name}.tif",
            flush=True,
        )
        del stack, cells


if __name__ == "__main__":
    main()
