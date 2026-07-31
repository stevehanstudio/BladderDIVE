"""I/O helpers: zarr, AnnData, tiled iteration, channel resolution."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from scripts.qc.config import QCConfig, SlideConfig


def dapi_channel_index(celldive_zarr: Path, exclude_r06: bool = True) -> int:
    """Index of the primary DAPI channel (DAPI_AF_R01), not DAPI_R06."""
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
        if "DAPI" in lab and (not exclude_r06 or ("R06" not in lab and "DAPI2" not in lab)):
            return i
    for i, ch in enumerate(channels):
        lab = (ch.get("label") or f"Ch{i}").upper()
        if "DAPI_AF" in lab or (lab.startswith("DAPI_") and "R06" not in lab):
            return i
    for i, ch in enumerate(channels):
        if "DAPI" in (ch.get("label") or "").upper():
            return i
    return 0


def get_channel_labels(celldive_zarr: Path) -> list[str]:
    zattrs = celldive_zarr / ".zattrs"
    if not zattrs.exists():
        return []
    with open(zattrs) as f:
        meta = json.load(f)
    channels = meta.get("omero", {}).get("channels", [])
    if not channels and "multiscales" in meta:
        channels = meta["multiscales"][0].get("omero", {}).get("channels", [])
    return [ch.get("label", f"Ch{i}") for i, ch in enumerate(channels)]


def level_scale(celldive_zarr: Path, level: int) -> float:
    """Pixel scale factor from pyramid level to full resolution."""
    zattrs = celldive_zarr / ".zattrs"
    if not zattrs.exists():
        return float(2**level)
    with open(zattrs) as f:
        meta = json.load(f)
    scales = meta.get("multiscales", [{}])[0].get("datasets", [])
    if level < len(scales):
        transform = scales[level].get("coordinateTransformations", [{}])[0]
        scale = transform.get("scale", [1, 1, 1])
        if len(scale) >= 2:
            return float(scale[0])
    return float(2**level)


def free_memory() -> None:
    gc.collect()


def _plane_nbytes(h: int, w: int, dtype_bytes: int = 4) -> int:
    return h * w * dtype_bytes


@dataclass
class QCContext:
    cfg: QCConfig
    slide: SlideConfig
    adata: ad.AnnData
    celldive_zarr: Path
    mask_zarr: Path
    qc_slide_dir: Path
    dapi_index: int = 0
    dapi_r06_index: int = 22
    channel_labels: list[str] = field(default_factory=list)
    _cell_flags: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _tile_frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _img_group: Any = field(default=None, repr=False)
    _mask_group: Any = field(default=None, repr=False)
    _channel_sample_cache: dict[tuple[int, int, int], np.ndarray] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def from_slide(cls, cfg: QCConfig, slide_id: str, load_adata: bool = True) -> QCContext:
        slide = cfg.slide(slide_id)
        if load_adata:
            low_memory = bool(cfg.raw.get("low_memory", True))
            adata = ad.read_h5ad(slide.adata_in, backed="r" if low_memory else None)
        else:
            adata = None
        qc_slide_dir = cfg.qc_dir / slide.id
        qc_slide_dir.mkdir(parents=True, exist_ok=True)
        (qc_slide_dir / "figures").mkdir(exist_ok=True)
        (qc_slide_dir / "tiles").mkdir(exist_ok=True)
        (qc_slide_dir / "multiqc").mkdir(exist_ok=True)

        dapi_idx = cfg.dapi_rounds.get("R01", dapi_channel_index(slide.celldive_zarr))
        dapi_r06 = cfg.dapi_rounds.get("R06", 22)
        labels = get_channel_labels(slide.celldive_zarr)

        return cls(
            cfg=cfg,
            slide=slide,
            adata=adata,
            celldive_zarr=slide.celldive_zarr,
            mask_zarr=slide.mask_zarr,
            qc_slide_dir=qc_slide_dir,
            dapi_index=dapi_idx,
            dapi_r06_index=dapi_r06,
            channel_labels=labels,
        )

    @property
    def low_memory(self) -> bool:
        return bool(self.cfg.raw.get("low_memory", True))

    @property
    def qc_tile_size(self) -> int:
        return int(self.cfg.raw.get("qc_tile_size", 2048))

    @property
    def max_cells_sample(self) -> int:
        return int(self.cfg.raw.get("max_cells_sample", 200_000))

    @property
    def max_array_mb(self) -> int:
        return int(self.cfg.raw.get("max_array_mb", 512))

    def thr(self, module: str, name: str):
        return self.cfg.thr(module, name)

    def markers(self) -> list[str]:
        return [m for m in self.cfg.marker_names() if m in self.adata.var_names]

    def open_image_group(self):
        if self._img_group is None:
            self._img_group = zarr.open_group(str(self.celldive_zarr), mode="r")
        return self._img_group

    def open_mask_group(self):
        if self._mask_group is None:
            self._mask_group = zarr.open_group(str(self.mask_zarr), mode="r")
        return self._mask_group

    def _resolve_level(self, level: int | None) -> int:
        level = self.cfg.pyramid_level if level is None else level
        if self.low_memory and level < 3:
            level = max(level, 3)
        return level

    def _check_plane_size(self, h: int, w: int, dtype_bytes: int = 4) -> None:
        nbytes = _plane_nbytes(h, w, dtype_bytes)
        limit = self.max_array_mb * 1024 * 1024
        if nbytes > limit:
            raise MemoryError(
                f"QC plane {h}x{w} ({nbytes / 1e9:.2f} GB) exceeds max_array_mb={self.max_array_mb}. "
                "Increase pyramid_level or enable low_memory in qc_config.yaml."
            )

    def get_channel_2d(self, channel: int, level: int | None = None) -> np.ndarray:
        level = self._resolve_level(level)
        img_g = self.open_image_group()
        arr = img_g[str(level)][channel]
        self._check_plane_size(arr.shape[0], arr.shape[1])
        return np.asarray(arr, dtype=np.float32)

    def get_mask_2d(self, level: int | None = None) -> np.ndarray:
        level = self._resolve_level(level)
        mask_g = self.open_mask_group()
        arr = mask_g[str(level)]
        self._check_plane_size(arr.shape[0], arr.shape[1], dtype_bytes=4)
        return np.asarray(arr, dtype=np.int32)

    def iter_tiles(
        self,
        channel: int | None = None,
        level: int | None = None,
        tile_size: int | None = None,
        mask: bool = False,
    ) -> Iterator[tuple[int, int, int, np.ndarray]]:
        """Yield (tile_id, y0, x0, tile_array)."""
        level = self._resolve_level(level)
        tile_size = self.qc_tile_size if tile_size is None else tile_size
        if mask:
            arr = self.open_mask_group()[str(level)]
        else:
            arr = self.open_image_group()[str(level)][channel if channel is not None else self.dapi_index]
        h, w = arr.shape
        tid = 0
        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                tile = np.asarray(arr[y0 : y0 + tile_size, x0 : x0 + tile_size])
                yield tid, y0, x0, tile
                tid += 1

    def sample_pixels_from_channels(
        self,
        channels: list[int],
        n_samples: int | None = None,
        level: int | None = None,
        seed: int = 0,
    ) -> dict[int, np.ndarray]:
        """Sample pixels using contiguous tile reads (fast on zarr; avoids random I/O)."""
        level = self._resolve_level(level)
        n_samples = n_samples or min(self.cfg.max_pixels, 100_000)
        cache_key = (level, n_samples, seed)
        missing = [ch for ch in channels if (ch, *cache_key) not in self._channel_sample_cache]

        if missing:
            img_arr = self.open_image_group()[str(level)]
            h, w = img_arr.shape[1], img_arr.shape[2]
            tile_size = self.qc_tile_size
            rng = np.random.default_rng(seed)
            buffers: dict[int, list[np.ndarray]] = {ch: [] for ch in missing}

            while min(sum(x.size for x in buffers[ch]) for ch in missing) < n_samples:
                y0 = int(rng.integers(0, max(1, h - tile_size)))
                x0 = int(rng.integers(0, max(1, w - tile_size)))
                y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
                for ch in missing:
                    tile = np.asarray(img_arr[ch][y0:y1, x0:x1], dtype=np.float32).ravel()
                    buffers[ch].append(tile)

            for ch in missing:
                vals = np.concatenate(buffers[ch])
                if vals.size > n_samples:
                    pick = rng.choice(vals.size, size=n_samples, replace=False)
                    vals = vals[pick]
                self._channel_sample_cache[(ch, *cache_key)] = vals
                del buffers[ch]

        return {ch: self._channel_sample_cache[(ch, *cache_key)] for ch in channels}

    def sample_mask_dapi_pixels(
        self,
        n_samples: int | None = None,
        level: int | None = None,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (dapi_values, inside_mask_bool) via in-memory subsample of L4 planes."""
        level = self._resolve_level(level)
        n_samples = n_samples or self.cfg.max_pixels
        dapi = self.get_channel_2d(self.dapi_index, level=level)
        mask = self.get_mask_2d(level=level)
        rng = np.random.default_rng(seed)
        flat_dapi = dapi.ravel()
        flat_mask = mask.ravel()
        pick = rng.choice(flat_dapi.size, size=min(n_samples, flat_dapi.size), replace=False)
        dapi_vals = flat_dapi[pick]
        inside = flat_mask[pick] > 0
        del dapi, mask, flat_dapi, flat_mask
        free_memory()
        return dapi_vals, inside

    def mask_coverage_fraction(self, level: int | None = None, n_tiles: int = 20) -> float:
        """Estimate mask coverage from random tiles without loading the full mask."""
        level = self._resolve_level(level)
        mask = self.open_mask_group()[str(level)]
        h, w = mask.shape
        tile_size = self.qc_tile_size
        rng = np.random.default_rng(0)
        fracs = []
        for _ in range(n_tiles):
            y0 = int(rng.integers(0, max(1, h - tile_size)))
            x0 = int(rng.integers(0, max(1, w - tile_size)))
            tile = np.asarray(mask[y0 : y0 + tile_size, x0 : x0 + tile_size])
            fracs.append(float((tile > 0).mean()))
        return float(np.mean(fracs))

    def marker_vector(self, marker: str, subsample: bool = True) -> np.ndarray:
        """Read one marker column via sequential h5ad access, then subsample in memory."""
        col_idx = list(self.adata.var_names).index(marker)
        x = self.adata.X[:, col_idx]
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float64).ravel()
        if subsample and x.size > self.max_cells_sample:
            rng = np.random.default_rng(0)
            x = x[rng.choice(x.size, size=self.max_cells_sample, replace=False)]
        return x

    def marker_matrix(
        self,
        markers: list[str],
        n_samples: int | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Read marker columns in one sequential pass, then subsample rows in memory."""
        n_samples = n_samples or self.max_cells_sample
        col_idx = [list(self.adata.var_names).index(m) for m in markers]
        # Sequential column read — fast on backed h5ad (avoid per-column fancy row indexing)
        X = np.asarray(self.adata.X[:, col_idx], dtype=np.float64)
        if X.shape[0] > n_samples:
            rng = np.random.default_rng(seed)
            sel = rng.choice(X.shape[0], size=n_samples, replace=False)
            X = X[sel]
        return X

    def write_tiles(self, name: str, df: pd.DataFrame) -> Path:
        out = self.qc_slide_dir / "tiles" / f"{name}.parquet"
        df.to_parquet(out, index=False)
        self._tile_frames[name] = df
        return out

    def add_cell_flags(self, flags: pd.DataFrame) -> None:
        """flags indexed by cell_id matching adata.obs['cell_id']."""
        for col in flags.columns:
            if col not in self._cell_flags:
                self._cell_flags[col] = flags[col]
            else:
                self._cell_flags[col] = self._cell_flags[col] | flags[col]

    def apply_cell_flags_to_adata(self) -> None:
        if self.adata is None or not self._cell_flags:
            return
        obs = self.adata.obs
        if "cell_id" not in obs.columns:
            raise ValueError("adata.obs must contain 'cell_id' for QC flag mapping")
        cell_ids = obs["cell_id"].values
        for col, series in self._cell_flags.items():
            mapped = series.reindex(cell_ids).fillna(False).astype(bool).values
            self.adata.obs[col] = mapped

    def figure_path(self, name: str) -> Path:
        return self.qc_slide_dir / "figures" / name

    def marker_background(self, marker: str) -> float:
        bg = self.adata.uns.get("background_estimate", {})
        if marker in bg:
            return float(bg[marker])
        thr = self.adata.uns.get("binary_thresholds", {})
        if marker in thr:
            return float(thr[marker])
        return 0.0

    def gating_threshold(self, marker: str) -> float | None:
        gating = self.cfg.raw.get("gating_thresholds", {})
        if marker in gating:
            return float(gating[marker])
        return None

    def materialize_adata_for_write(self) -> ad.AnnData:
        """Return AnnData ready to write. Avoids copying X when only obs changed."""
        if self.adata.isbacked:
            # obs QC columns were updated in place; write without loading X into RAM
            return self.adata
        return self.adata

    def clear_caches(self) -> None:
        self._channel_sample_cache.clear()
        self._img_group = None
        self._mask_group = None
        free_memory()
