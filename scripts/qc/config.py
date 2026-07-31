"""Load and validate QC configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scripts.qc.thresholds import Threshold


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class SlideConfig:
    id: str
    batch: str
    acquisition_date: str
    celldive_zarr: Path
    mask_zarr: Path
    adata_in: Path
    adata_out: Path | None = None

    def __post_init__(self) -> None:
        root = project_root()
        self.celldive_zarr = _resolve_path(self.celldive_zarr, root)
        self.mask_zarr = _resolve_path(self.mask_zarr, root)
        self.adata_in = _resolve_path(self.adata_in, root)
        if self.adata_out is not None:
            self.adata_out = _resolve_path(self.adata_out, root)


@dataclass
class QCConfig:
    schema_version: str
    qc_dir: Path
    tile_size: int
    pyramid_level: int
    max_pixels: int
    markers: dict[str, dict[str, Any]]
    dapi_rounds: dict[str, int]
    fluorophore_groups: dict[str, list[int]]
    thresholds: dict[str, dict[str, dict[str, Any]]]
    slides: list[SlideConfig]
    config_hash: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def thr(self, module: str, name: str) -> Threshold | None:
        spec = self.thresholds.get(module, {}).get(name)
        if spec is None:
            return None
        return Threshold(
            warn=float(spec["warn"]),
            fail=float(spec["fail"]),
            direction=str(spec.get("direction", "high")),
        )

    def slide(self, slide_id: str) -> SlideConfig:
        for s in self.slides:
            if s.id == slide_id:
                return s
        raise KeyError(f"Slide not found in config: {slide_id}")

    def marker_names(self) -> list[str]:
        return list(self.markers.keys())

    def marker_channel_index(self, marker: str) -> int | None:
        info = self.markers.get(marker)
        if info is None:
            return None
        return int(info.get("channel", -1))


def _resolve_path(path: Path | str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _config_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_config(path: Path | str | None = None) -> QCConfig:
    root = project_root()
    cfg_path = _resolve_path(path or root / "qc_config.yaml", root)
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    slides = [
        SlideConfig(
            id=s["id"],
            batch=s.get("batch", "B01"),
            acquisition_date=s.get("acquisition_date", ""),
            celldive_zarr=s["celldive_zarr"],
            mask_zarr=s["mask_zarr"],
            adata_in=s["adata_in"],
            adata_out=s.get("adata_out"),
        )
        for s in raw.get("slides", [])
    ]

    cfg = QCConfig(
        schema_version=str(raw.get("schema_version", "1.0")),
        qc_dir=_resolve_path(raw.get("qc_dir", "qc"), root),
        tile_size=int(raw.get("tile_size", 16384)),
        pyramid_level=int(raw.get("pyramid_level", 4)),
        max_pixels=int(raw.get("max_pixels", 5_000_000)),
        markers=raw.get("markers", {}),
        dapi_rounds={k: int(v) for k, v in raw.get("dapi_rounds", {}).items()},
        fluorophore_groups={
            k: [int(x) for x in v] for k, v in raw.get("fluorophore_groups", {}).items()
        },
        thresholds=raw.get("thresholds", {}),
        slides=slides,
        config_hash=_config_hash(raw),
        raw=raw,
    )
    return cfg
