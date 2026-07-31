"""Write QC summaries, AnnData integration, and MultiQC custom content."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from scripts.qc.config import QCConfig, SlideConfig
from scripts.qc.thresholds import module_status, worst_status


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def flatten_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metric dicts for module summary."""
    flat: dict[str, Any] = {}
    for key, val in results.items():
        if isinstance(val, dict) and "value" in val and "status" in val:
            flat[key] = val
        elif isinstance(val, dict) and "per_marker" in val:
            flat["per_marker"] = val["per_marker"]
        elif isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, dict) and "value" in v2:
                    flat[k2] = v2
                elif k2 not in ("markers", "slide_ids", "batch_groups", "per_marker_cv", "per_cycle"):
                    flat[f"{key}.{k2}"] = v2
        elif isinstance(val, (list, str, int, float, bool)):
            flat[key] = val
        else:
            flat[key] = val
    return flat


def build_module_block(metric_results: dict[str, Any]) -> dict[str, Any]:
    metrics = flatten_metrics(metric_results)
    status_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, dict) and "status" in v
    }
    status = module_status(list(status_metrics.values()))
    if "per_marker" in metrics:
        marker_statuses = [
            m.get("status", "unknown")
            for m in metrics["per_marker"].values()
            if isinstance(m, dict)
        ]
        status = worst_status([status] + marker_statuses)
    return {"status": status, "metrics": metrics}


def build_summary_json(
    slide: SlideConfig,
    cfg: QCConfig,
    adata: ad.AnnData | None,
    module_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    modules = {name: build_module_block(res) for name, res in module_results.items()}
    n_cells = int(adata.n_obs) if adata is not None else 0
    n_pass = int(adata.obs["qc_pass"].sum()) if adata is not None and "qc_pass" in adata.obs else n_cells
    overall = worst_status([m["status"] for m in modules.values()])
    return {
        "schema_version": cfg.schema_version,
        "slide_id": slide.id,
        "batch_id": slide.batch,
        "acquisition_date": slide.acquisition_date,
        "config_hash": cfg.config_hash,
        "generated_at": now_iso(),
        "n_cells": n_cells,
        "n_cells_qc_pass": n_pass,
        "modules": modules,
        "overall_status": overall,
    }


def add_cell_flags(adata: ad.AnnData, flags: pd.DataFrame) -> None:
    if "cell_id" not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'cell_id'")
    cell_ids = adata.obs["cell_id"].values
    for col in flags.columns:
        adata.obs[col] = flags[col].reindex(cell_ids).fillna(False).astype(bool).values


FLAG_BITS = {
    "qc_area_outlier": 1,
    "qc_shape_outlier": 2,
    "qc_low_dapi": 4,
    "qc_spatial_outlier": 8,
    "qc_doublet": 16,
}


def finalize_qc_obs(adata: ad.AnnData) -> None:
    flag_cols = [c for c in adata.obs.columns if c.startswith("qc_") and c not in ("qc_pass", "qc_flags")]
    if not flag_cols:
        adata.obs["qc_pass"] = True
        adata.obs["qc_flags"] = 0
        return
    flags = adata.obs[flag_cols].astype(bool)
    adata.obs["qc_pass"] = ~flags.any(axis=1)
    bitmask = np.zeros(adata.n_obs, dtype=np.int32)
    for col, bit in FLAG_BITS.items():
        if col in adata.obs.columns:
            bitmask |= adata.obs[col].astype(bool).values.astype(np.int32) * bit
    adata.obs["qc_flags"] = bitmask


def write_marker_tsv(path: Path, per_marker: dict[str, Any]) -> None:
    rows = []
    for marker, stats in per_marker.items():
        row = {"marker": marker}
        if isinstance(stats, dict):
            row.update({k: v for k, v in stats.items() if k != "status"})
            row["status"] = stats.get("status", "unknown")
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def write_multiqc_customcontent(qc_slide_dir: Path, slide_id: str, summary: dict[str, Any]) -> None:
    mqc_dir = qc_slide_dir / "multiqc"
    mqc_dir.mkdir(parents=True, exist_ok=True)

    general = {
        "id": f"{slide_id}_general",
        "section_name": f"{slide_id} QC Summary",
        "description": "CellDIVE per-slide QC summary",
        "plot_type": "generalstats",
        "pconfig": {
            "id": f"{slide_id}_general",
            "title": f"{slide_id} General Stats",
            "format": "{value:.3f}",
            "ylab": "Value",
        },
        "data": {
            slide_id: {
                "n_cells": summary.get("n_cells"),
                "n_cells_qc_pass": summary.get("n_cells_qc_pass"),
                "overall_status": summary.get("overall_status"),
            }
        },
    }
    write_json(mqc_dir / f"{slide_id}_general_mqc.json", general)

    marker_block = summary.get("modules", {}).get("marker", {}).get("metrics", {}).get("per_marker", {})
    if marker_block:
        rows = []
        for marker, stats in marker_block.items():
            rows.append(
                {
                    "marker": marker,
                    "staining_index": stats.get("staining_index"),
                    "snr": stats.get("snr"),
                    "pct_positive": stats.get("pct_positive"),
                    "antibody_failed": stats.get("antibody_failed"),
                    "status": stats.get("status"),
                }
            )
        pd.DataFrame(rows).to_csv(mqc_dir / f"{slide_id}_marker_mqc.tsv", sep="\t", index=False)

    for module in ("image", "segmentation", "spatial"):
        mod = summary.get("modules", {}).get(module, {})
        metrics = mod.get("metrics", {})
        scalar = {
            k: v.get("value")
            for k, v in metrics.items()
            if isinstance(v, dict) and "value" in v
        }
        if scalar:
            payload = {
                "id": f"{slide_id}_{module}",
                "section_name": f"{slide_id} {module.title()} QC",
                "description": f"{module} QC metrics for {slide_id}",
                "plot_type": "generalstats",
                "data": {slide_id: scalar},
            }
            write_json(mqc_dir / f"{slide_id}_{module}_mqc.json", payload)


def finalize_qc(
    adata: ad.AnnData,
    slide: SlideConfig,
    cfg: QCConfig,
    module_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    finalize_qc_obs(adata)
    summary = build_summary_json(slide, cfg, adata, module_results)

    qc_slide_dir = cfg.qc_dir / slide.id
    qc_slide_dir.mkdir(parents=True, exist_ok=True)
    write_json(qc_slide_dir / "qc_summary.json", summary)

    for module, res in module_results.items():
        if module == "marker" and "per_marker" in flatten_metrics(res):
            write_marker_tsv(qc_slide_dir / "marker_qc.tsv", flatten_metrics(res)["per_marker"])
        else:
            write_json(qc_slide_dir / f"{module}_qc.json", build_module_block(res))

    adata.uns["qc"] = {
        "schema_version": cfg.schema_version,
        "config_hash": cfg.config_hash,
        "slide_id": slide.id,
        "generated_at": summary["generated_at"],
        "modules": summary["modules"],
        "overall_status": summary["overall_status"],
    }

    write_multiqc_customcontent(qc_slide_dir, slide.id, summary)
    return summary
