"""Batch/cohort QC metrics (activates with multiple slides)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.qc.config import QCConfig
from scripts.qc.registry import register_qc
from scripts.qc.thresholds import metric


@register_qc("batch", "cohort_stats")
def cohort_stats(ctx, summaries: list[dict] | None = None) -> dict:
    """
    Cohort-level stats across per-slide qc_summary.json files.
    When called from run_cohort, summaries is provided directly.
    When called from run_slide with one slide, returns skip status.
    """
    if summaries is None:
        return {"status": "skip", "reason": "single-slide mode; run run_cohort() for batch QC"}

    if len(summaries) < 2:
        return {"status": "skip", "reason": f"only {len(summaries)} slide(s); need >=2 for batch QC"}

    slide_ids = [s["slide_id"] for s in summaries]
    marker_medians: dict[str, list[float]] = {}

    for summary in summaries:
        per_marker = (
            summary.get("modules", {})
            .get("marker", {})
            .get("metrics", {})
            .get("per_marker", {})
        )
        for marker, stats in per_marker.items():
            # use staining_index as proxy for slide-level marker median behavior
            marker_medians.setdefault(marker, []).append(float(stats.get("staining_index", 0.0)))

    marker_cv = {}
    for marker, vals in marker_medians.items():
        if len(vals) >= 2:
            marker_cv[marker] = float(np.std(vals) / (np.mean(vals) + 1e-9))

    max_cv = max(marker_cv.values()) if marker_cv else 0.0

    # batch grouping
    batch_groups: dict[str, list[str]] = {}
    for s in summaries:
        batch_groups.setdefault(s.get("batch_id", "unknown"), []).append(s["slide_id"])

    return {
        "n_slides": metric(float(len(summaries))),
        "marker_median_cv_max": metric(max_cv),
        "slide_ids": slide_ids,
        "batch_groups": batch_groups,
        "per_marker_cv": marker_cv,
        "status": "pass" if max_cv < 0.6 else "warn",
    }


def run_cohort_qc(cfg: QCConfig) -> dict:
    """Load per-slide summaries and compute cohort metrics."""
    summaries = []
    for slide in cfg.slides:
        summary_path = cfg.qc_dir / slide.id / "qc_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summaries.append(json.load(f))

    if len(summaries) < 2:
        result = {
            "status": "skip",
            "reason": f"only {len(summaries)} slide summary available",
            "n_slides": len(summaries),
        }
        out = cfg.qc_dir / "aggregate" / "cohort_qc.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        return result

    result = cohort_stats(None, summaries=summaries)
    agg_dir = cfg.qc_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    with open(agg_dir / "cohort_qc.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # slide x marker staining index matrix
    rows = []
    for summary in summaries:
        slide_id = summary["slide_id"]
        per_marker = (
            summary.get("modules", {})
            .get("marker", {})
            .get("metrics", {})
            .get("per_marker", {})
        )
        for marker, stats in per_marker.items():
            rows.append(
                {
                    "slide_id": slide_id,
                    "marker": marker,
                    "staining_index": stats.get("staining_index"),
                    "snr": stats.get("snr"),
                    "status": stats.get("status"),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(agg_dir / "slide_marker_median.tsv", sep="\t", index=False)

    status_rows = []
    for summary in summaries:
        row = {"slide_id": summary["slide_id"], "overall_status": summary.get("overall_status")}
        for mod, block in summary.get("modules", {}).items():
            row[f"{mod}_status"] = block.get("status")
        status_rows.append(row)
    pd.DataFrame(status_rows).to_csv(agg_dir / "slide_status_matrix.tsv", sep="\t", index=False)

    return result
