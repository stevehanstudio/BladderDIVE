#!/usr/bin/env python3
"""Segment one nuclear TIFF with Cellpose 4 and write a label mask.

Cellpose's own 256px network tiles are not optional. ``run_net`` allocates
every patch, and a second array of the same size, before the forward pass.
On a ~65k x 74k image that pair is about 154 GiB, which does not fit in the
121 GiB unified memory on this machine even for a single channel (Cellpose
also pads grayscale to 3 channels). This script therefore picks the smallest
n x n outer grid whose patch buffers fit, then stitches with the same overlap
crop used by ``cellpose_segmentation.py``.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from tifffile import TiffFile, imwrite, memmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cellpose_segmentation import (  # noqa: E402
    calculate_overlap,
    generate_grid_tiles,
    stitch_tiles,
)

BSIZE = 256
TILE_OVERLAP = 0.1
NCHAN = 3  # Cellpose 4 pads grayscale to 3 channels
NOUT = 3   # dy, dx, cell probability


def patch_buffer_bytes(height, width):
    """Bytes for the all-patches input array plus the all-patches output array."""
    ny = 1 if height <= BSIZE else int(np.ceil((1.0 + 2 * TILE_OVERLAP) * height / BSIZE))
    nx = 1 if width <= BSIZE else int(np.ceil((1.0 + 2 * TILE_OVERLAP) * width / BSIZE))
    ntiles = ny * nx
    one = ntiles * BSIZE * BSIZE * 4
    return one * (NCHAN + NOUT), ntiles


def choose_grid(image_shape, available_bytes):
    """Smallest n such that every outer tile's patch buffers stay under the budget."""
    # Leave room for the float image, the model, and the stitched mask.
    budget = int(available_bytes * 0.45)
    height, width = image_shape
    for n in range(1, 11):
        base_h = int(np.ceil(height / n))
        base_w = int(np.ceil(width / n))
        overlap = calculate_overlap(min(base_h, base_w))
        tiles = generate_grid_tiles(image_shape, n, n, overlap)
        worst = 0
        worst_ntiles = 0
        for y0, y1, x0, x1 in tiles:
            nbytes, ntiles = patch_buffer_bytes(y1 - y0, x1 - x0)
            if nbytes > worst:
                worst = nbytes
                worst_ntiles = ntiles
        print(
            f"  {n}x{n}: worst patch buffers {worst / 1024**3:.1f} GiB "
            f"({worst_ntiles} patches), budget {budget / 1024**3:.1f} GiB",
            flush=True,
        )
        if worst <= budget:
            return n, overlap, tiles, worst
    raise RuntimeError("No grid up to 10x10 fits in available memory")


def read_tile(path, y0, y1, x0, x1):
    mapped = memmap(path)
    tile = np.array(mapped[y0:y1, x0:x1], copy=True)
    del mapped
    return tile


def eval_tile(model, tile):
    result = model.eval(tile, diameter=None, batch_size=8, bsize=BSIZE, tile_overlap=TILE_OVERLAP)
    if isinstance(result, tuple):
        masks = result[0]
    else:
        masks = result
    # Drop flows immediately. They are several times the tile size.
    del result
    gc.collect()
    return masks


def _is_oom(exc):
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or isinstance(exc, MemoryError)


def main():
    parser = argparse.ArgumentParser(description="Cellpose mask from one nuclear channel")
    parser.add_argument("nuclear_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output mask TIFF")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    from cellpose import models

    nuclear_file = args.nuclear_file.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with TiffFile(nuclear_file) as tif:
        image_shape = tif.pages[0].shape
        dtype = tif.pages[0].dtype
    print(f"Image: {nuclear_file}", flush=True)
    print(f"  shape {image_shape} dtype {dtype}", flush=True)

    available = psutil.virtual_memory().available
    print(f"  RAM available {available / 1024**3:.1f} GiB", flush=True)
    print("Grid search (patch buffers must fit; 1x1 is tried first):", flush=True)
    n, overlap, tiles, worst = choose_grid(image_shape, available)
    print(
        f"Starting at {n}x{n} ({len(tiles)} tiles), overlap {overlap} px, "
        f"worst patch buffers {worst / 1024**3:.1f} GiB",
        flush=True,
    )

    print("Loading Cellpose model...", flush=True)
    model = models.CellposeModel(gpu=args.gpu)

    cache = output.parent / ".dapi2_tile_cache"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        paths = []
        try:
            for i, (y0, y1, x0, x1) in enumerate(tiles, start=1):
                print(
                    f"\nTile {i}/{len(tiles)} y={y0}:{y1} x={x0}:{x1} "
                    f"({y1 - y0} x {x1 - x0})",
                    flush=True,
                )
                tile = read_tile(nuclear_file, y0, y1, x0, x1)
                started = time.time()
                masks = eval_tile(model, tile)
                del tile
                n_cells_tile = int(np.unique(masks).size - 1)
                print(
                    f"  {n_cells_tile} cells in {time.time() - started:.0f}s, "
                    f"mask {masks.dtype} {masks.shape}",
                    flush=True,
                )
                path = cache / f"tile_{i:04d}.npz"
                np.savez(path, masks=masks, coords=np.array([y0, y1, x0, x1]))
                paths.append(str(path))
                del masks
                gc.collect()
            break
        except Exception as exc:
            if not _is_oom(exc) or n >= 10:
                raise
            print(f"\nOut of memory on {n}x{n}: {exc}", flush=True)
            n += 1
            for path in cache.glob("tile_*.npz"):
                path.unlink()
            gc.collect()
            base = int(np.ceil(min(image_shape) / n))
            overlap = calculate_overlap(base)
            tiles = generate_grid_tiles(image_shape, n, n, overlap)
            print(f"Retrying with {n}x{n} ({len(tiles)} tiles), overlap {overlap} px", flush=True)

    print("\nStitching...", flush=True)
    full = stitch_tiles(paths, tiles, image_shape, overlap, use_disk_storage=True, temp_dir=str(cache))
    n_cells = int(np.unique(full).size - 1)
    print(f"Full mask {full.shape} {full.dtype}, {n_cells} cells", flush=True)
    output = output.with_name(f"{output.stem}_{len(tiles)}tiles{output.suffix}")
    print(f"Writing {output}", flush=True)
    imwrite(output, full)

    tag = output.stem.removeprefix("cellpose_masks_")
    summary = output.parent / f"segmentation_summary_{tag}.txt"
    y0, y1, x0, x1 = tiles[0]
    with summary.open("w") as fh:
        fh.write("Cellpose Segmentation Summary\n")
        fh.write("=" * 50 + "\n")
        fh.write("Model: CellposeModel (cpsam, cellpose 4)\n")
        fh.write(f"GPU used: {args.gpu}\n")
        fh.write(f"Nuclear file: {nuclear_file}\n")
        fh.write(f"Tiling used: {len(tiles) > 1}\n")
        fh.write(f"Number of tiles: {len(tiles)}\n")
        fh.write(f"Tile size: {y1 - y0} × {x1 - x0}\n")
        fh.write(f"Tile overlap: {overlap} pixels\n")
        fh.write(f"Number of cells detected: {n_cells}\n")
        fh.write(f"Image shape: {tuple(image_shape)}\n")
        fh.write(f"Masks saved to: {output}\n")
        fh.write(f"Elapsed seconds: {time.time() - t0:.0f}\n")
    print(f"Summary: {summary}", flush=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
