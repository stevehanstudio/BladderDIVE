"""
Tile-based per-cell intensity extraction. Memory-safe: only one tile in RAM at a time.
Use when full-image loading causes OOM (e.g. 125GB+ on 128GB system).

With 128 GB RAM: tile_size=16384 (~4 GB/tile) or 32768 (~16 GB/tile) is fine.
GPU (CuPy+CUDA) optional: speeds up bincount per tile. Requires NVIDIA GPU.
"""
import time
import numpy as np
import pandas as pd
import zarr
from pathlib import Path

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def extract_protein_matrix_from_zarr(
    celldive_zarr: Path,
    mask_zarr: Path,
    channel_names: list | None = None,
    tile_size: int = 16384,
    use_gpu: bool = False,
    get_channel_names_fn=None,
) -> tuple[np.ndarray, pd.DataFrame, list]:
    """
    Extract per-cell mean intensity via tile-based bincount. Memory-safe: only one
    tile in RAM at a time (~tile_size² × 16 bytes for mask+channel).
    Returns: (X matrix, obs DataFrame, var names).
    """
    use_gpu = use_gpu and HAS_CUPY
    if use_gpu:
        print("Using GPU (CuPy) for bincount", flush=True)
    t0 = time.perf_counter()
    mask_store = zarr.open(str(mask_zarr), mode="r")
    img_store = zarr.open(str(celldive_zarr), mode="r")
    mask_arr = mask_store["0"]
    img_level0 = img_store["0"]
    n_channels = img_level0.shape[0]
    h, w = mask_arr.shape

    if channel_names is None or len(channel_names) != n_channels:
        if get_channel_names_fn:
            channel_names = get_channel_names_fn(celldive_zarr) or [f"Ch{i}" for i in range(n_channels)]
        else:
            channel_names = [f"Ch{i}" for i in range(n_channels)]

    # 1) Scan tiles to get max_label (no full load)
    print("[1/5] Scanning mask for max label...", flush=True)
    max_label = 0
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            tile = np.array(mask_arr[y0 : y0 + tile_size, x0 : x0 + tile_size])
            max_label = max(max_label, int(tile.max()))
    print(f"      max_label={max_label:,} in {time.perf_counter()-t0:.1f}s", flush=True)

    # 2) Accumulate counts, sum_x, sum_y (geometry)
    n = max_label + 1
    counts = np.zeros(n, dtype=np.float64)
    sum_x = np.zeros(n, dtype=np.float64)
    sum_y = np.zeros(n, dtype=np.float64)
    sum_intensity = [np.zeros(n, dtype=np.float64) for _ in range(n_channels)]

    n_tiles_y = (h + tile_size - 1) // tile_size
    n_tiles_x = (w + tile_size - 1) // tile_size
    total_tiles = n_tiles_y * n_tiles_x

    print("[2/5] Accumulating geometry (counts, centroids)...", flush=True)
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
            mask_tile = np.array(mask_arr[y0:y1, x0:x1], dtype=np.int32)
            m_flat = mask_tile.ravel()
            yy, xx = np.mgrid[y0:y1, x0:x1]
            y_flat = yy.ravel().astype(np.float64)
            x_flat = xx.ravel().astype(np.float64)
            if use_gpu:
                m_gpu = cp.asarray(m_flat)
                y_gpu = cp.asarray(y_flat)
                x_gpu = cp.asarray(x_flat)
                counts += cp.asnumpy(cp.bincount(m_gpu, minlength=n))
                sum_x += cp.asnumpy(cp.bincount(m_gpu, weights=x_gpu, minlength=n))
                sum_y += cp.asnumpy(cp.bincount(m_gpu, weights=y_gpu, minlength=n))
            else:
                counts += np.bincount(m_flat, minlength=n)
                sum_x += np.bincount(m_flat, weights=x_flat, minlength=n)
                sum_y += np.bincount(m_flat, weights=y_flat, minlength=n)

    # 3) Accumulate intensity per channel (one mask load per tile, shared across channels)
    print("[3/5] Extracting intensities per channel...", flush=True)
    tiles_done = 0
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
            mask_tile = np.array(mask_arr[y0:y1, x0:x1], dtype=np.int32)
            m_flat = mask_tile.ravel()
            m_gpu = cp.asarray(m_flat) if use_gpu else None
            for c in range(n_channels):
                ch_tile = np.array(img_level0[c, y0:y1, x0:x1], dtype=np.float64)
                ch_flat = ch_tile.ravel()
                if use_gpu:
                    sum_intensity[c] += cp.asnumpy(cp.bincount(m_gpu, weights=cp.asarray(ch_flat), minlength=n))
                else:
                    sum_intensity[c] += np.bincount(m_flat, weights=ch_flat, minlength=n)
            tiles_done += 1
            if tiles_done % 20 == 0 or tiles_done == total_tiles:
                print(f"      tiles {tiles_done}/{total_tiles}...", flush=True)

    # 4) Build obs and X (cells with count > 0)
    print("[4/5] Building obs and X...", flush=True)
    valid = counts > 0
    cell_ids = np.where(valid)[0]
    n_cells = len(cell_ids)

    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "area": counts[cell_ids],
            "centroid_x": np.divide(sum_x[cell_ids], counts[cell_ids]),
            "centroid_y": np.divide(sum_y[cell_ids], counts[cell_ids]),
        }
    )

    X = np.zeros((n_cells, n_channels), dtype=np.float32)
    for c in range(n_channels):
        X[:, c] = np.divide(sum_intensity[c][cell_ids], counts[cell_ids])

    print(f"[5/5] Done. {n_cells:,} cells in {time.perf_counter()-t0:.1f}s", flush=True)
    return X, obs, channel_names
