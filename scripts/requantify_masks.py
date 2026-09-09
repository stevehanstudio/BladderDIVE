"""
Requantify per-cell mean intensities for several candidate masks in ONE pass over the image.

The image Zarr is 107 GB and a full read costs ~6 minutes, while the masks are ~1 GB each.
Quantifying masks sequentially would re-read the image once per mask; this reads each tile
once and fans it out to every mask accumulator, so N masks cost the same I/O as one.

Writes one h5ad per mask, matching the schema produced by
scripts/extract_protein_matrix_tiled.py (X = mean intensity, obs = cell_id/area/centroid).

Usage:
    python scripts/requantify_masks.py --gpu
    python scripts/requantify_masks.py --masks dapi_only dapi+panck --tile-size 8192
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:  # pragma: no cover
    HAS_CUPY = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ZARR = PROJECT_ROOT / "data" / "CellDIVE_SLIDE-045.zarr"
MASK_DIR = PROJECT_ROOT / "output" / "cellpose_output"
OUT_DIR = PROJECT_ROOT / "output" / "requant"
REFERENCE_H5AD = PROJECT_ROOT / "output" / "celldive_protein_matrix.h5ad"

DEFAULT_MASKS = ["dapi_only", "dapi+panck", "dapi+panck+cd45", "dapi+vim"]


def mask_path(tag: str) -> Path:
    return MASK_DIR / f"cellpose_masks_{tag}_9tiles.zarr"


def channel_names() -> list[str]:
    # Take the canonical marker order from the production h5ad so that every downstream
    # comparison indexes channels identically.
    import anndata as ad

    return list(ad.read_h5ad(REFERENCE_H5AD, backed="r").var_names)


class MaskAccumulator:
    """Running per-label sums for one mask: pixel count, centroid, and per-channel intensity."""

    def __init__(self, tag: str, arr, n_channels: int, use_gpu: bool):
        self.tag = tag
        self.arr = arr
        self.use_gpu = use_gpu
        self.n_channels = n_channels
        self.n = None  # set by scan()

    def scan(self, tile_size: int) -> int:
        h, w = self.arr.shape
        mx = 0
        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                tile = np.asarray(self.arr[y0 : y0 + tile_size, x0 : x0 + tile_size])
                if tile.size:
                    mx = max(mx, int(tile.max()))
        self.n = mx + 1
        xp = cp if self.use_gpu else np
        self.counts = xp.zeros(self.n, dtype=xp.float64)
        self.sum_y = xp.zeros(self.n, dtype=xp.float64)
        self.sum_x = xp.zeros(self.n, dtype=xp.float64)
        self.sum_i = xp.zeros((self.n_channels, self.n), dtype=xp.float64)
        return mx

    def load_tile(self, y0, y1, x0, x1):
        tile = np.asarray(self.arr[y0:y1, x0:x1]).astype(np.int32).ravel()
        self._flat = cp.asarray(tile) if self.use_gpu else tile

    def add_geometry(self, y_flat, x_flat):
        xp = cp if self.use_gpu else np
        self.counts += xp.bincount(self._flat, minlength=self.n)[: self.n]
        self.sum_y += xp.bincount(self._flat, weights=y_flat, minlength=self.n)[: self.n]
        self.sum_x += xp.bincount(self._flat, weights=x_flat, minlength=self.n)[: self.n]

    def add_channel(self, c, values):
        xp = cp if self.use_gpu else np
        self.sum_i[c] += xp.bincount(self._flat, weights=values, minlength=self.n)[: self.n]

    def release_tile(self):
        self._flat = None

    def finalize(self):
        to_np = cp.asnumpy if self.use_gpu else np.asarray
        counts = to_np(self.counts)
        sum_y, sum_x, sum_i = to_np(self.sum_y), to_np(self.sum_x), to_np(self.sum_i)
        ids = np.flatnonzero(counts > 0)
        ids = ids[ids > 0]  # label 0 is background
        area = counts[ids]
        obs = pd.DataFrame(
            {
                "cell_id": ids.astype(np.int64),
                "area": area,
                "centroid_x": sum_x[ids] / area,
                "centroid_y": sum_y[ids] / area,
            }
        )
        X = (sum_i[:, ids] / area).T.astype(np.float32)
        return X, obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--masks", nargs="+", default=DEFAULT_MASKS)
    ap.add_argument("--tile-size", type=int, default=8192)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    use_gpu = args.gpu and HAS_CUPY
    if args.gpu and not HAS_CUPY:
        print("cupy not importable; falling back to CPU")
    print(f"backend: {'GPU (cupy)' if use_gpu else 'CPU (numpy)'}, tile={args.tile_size}")

    import anndata as ad

    names = channel_names()
    image = zarr.open(str(IMAGE_ZARR), mode="r")["0"]
    n_channels, h, w = image.shape
    assert len(names) == n_channels, f"{len(names)} names vs {n_channels} channels"

    accs = []
    for tag in args.masks:
        p = mask_path(tag)
        if not p.exists():
            print(f"  SKIP {tag}: {p} not found")
            continue
        arr = zarr.open(str(p), mode="r")["0"]
        assert arr.shape == (h, w), f"{tag} shape {arr.shape} != image {(h, w)}"
        accs.append(MaskAccumulator(tag, arr, n_channels, use_gpu))
    if not accs:
        raise SystemExit("no masks to process")

    t0 = time.perf_counter()
    print(f"\n[1/3] scanning {len(accs)} masks for max label")
    for a in accs:
        mx = a.scan(args.tile_size)
        print(f"      {a.tag:22s} max_label={mx:,}  ({time.perf_counter()-t0:.0f}s)")

    ny = (h + args.tile_size - 1) // args.tile_size
    nx = (w + args.tile_size - 1) // args.tile_size
    total = ny * nx
    print(f"\n[2/3] one pass over {total} tiles x {n_channels} channels")
    done = 0
    for y0 in range(0, h, args.tile_size):
        for x0 in range(0, w, args.tile_size):
            y1, x1 = min(y0 + args.tile_size, h), min(x0 + args.tile_size, w)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            y_flat = yy.ravel().astype(np.float64)
            x_flat = xx.ravel().astype(np.float64)
            if use_gpu:
                y_flat, x_flat = cp.asarray(y_flat), cp.asarray(x_flat)
            for a in accs:
                a.load_tile(y0, y1, x0, x1)
                a.add_geometry(y_flat, x_flat)
            del y_flat, x_flat
            for c in range(n_channels):
                vals = np.asarray(image[c, y0:y1, x0:x1]).ravel().astype(np.float64)
                if use_gpu:
                    vals = cp.asarray(vals)
                for a in accs:
                    a.add_channel(c, vals)
                del vals
            for a in accs:
                a.release_tile()
            if use_gpu:
                cp.get_default_memory_pool().free_all_blocks()
            done += 1
            el = time.perf_counter() - t0
            print(f"      tile {done}/{total}  {el:.0f}s elapsed, eta {el/done*(total-done):.0f}s", flush=True)

    print(f"\n[3/3] writing h5ad to {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for a in accs:
        X, obs = a.finalize()
        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=names))
        adata.uns["mask_tag"] = a.tag
        adata.uns["mask_zarr"] = str(mask_path(a.tag))
        adata.uns["data_source"] = "scripts/requantify_masks.py"
        out = args.out_dir / f"protein_matrix_{a.tag}.h5ad"
        adata.write_h5ad(out)
        print(f"      {a.tag:22s} {adata.n_obs:>9,} cells  mean area {obs['area'].mean():7.1f} px  -> {out.name}")
    print(f"\ndone in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
