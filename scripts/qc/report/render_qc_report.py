#!/usr/bin/env python3
"""Render a simple HTML QC report from per-slide JSON/TSV outputs.

Use when Quarto is unavailable:
  python scripts/qc/report/render_qc_report.py --slide SLIDE-045

With Quarto installed, prefer:
  quarto render scripts/qc/report/qc_report.qmd -P slide_id:SLIDE-045
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.qc.config import load_config


def _metric_table(metrics: dict) -> str:
    rows = []
    for k, v in metrics.items():
        if isinstance(v, dict) and "value" in v:
            rows.append(
                f"<tr><td>{k}</td><td>{v.get('value','')}</td><td>{v.get('status','')}</td></tr>"
            )
    if not rows:
        return "<p><em>No scalar metrics.</em></p>"
    return (
        "<table border='1' cellpadding='4'><tr><th>Metric</th><th>Value</th><th>Status</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def render_html(slide_id: str, cfg_path: Path | None = None) -> Path:
    cfg = load_config(cfg_path)
    slide_dir = cfg.qc_dir / slide_id
    with open(slide_dir / "qc_summary.json") as f:
        summary = json.load(f)

    sections = []
    sections.append(f"<h1>CellDIVE QC Report: {slide_id}</h1>")
    sections.append(
        f"<p><b>Overall status:</b> {summary.get('overall_status')} | "
        f"<b>Cells:</b> {summary.get('n_cells_qc_pass'):,} / {summary.get('n_cells'):,} pass QC</p>"
    )

    sections.append("<h2>Module status</h2><ul>")
    for mod, block in summary.get("modules", {}).items():
        sections.append(f"<li>{mod}: <b>{block.get('status')}</b></li>")
    sections.append("</ul>")

    for mod in ("image", "segmentation", "spatial"):
        p = slide_dir / f"{mod}_qc.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            sections.append(f"<h2>{mod.title()} QC</h2>")
            sections.append(_metric_table(data.get("metrics", {})))
            fig_dir = slide_dir / "figures"
            for fig in sorted(fig_dir.glob("*.png")):
                if mod in fig.name or (mod == "image" and any(x in fig.name for x in ("focus", "illumination", "drift"))):
                    rel = fig.relative_to(cfg.qc_dir.parent)
                    sections.append(f'<figure><img src="../../{rel}" width="700"/><figcaption>{fig.name}</figcaption></figure>')

    marker_tsv = slide_dir / "marker_qc.tsv"
    if marker_tsv.exists():
        df = pd.read_csv(marker_tsv, sep="\t")
        sections.append("<h2>Marker QC</h2>")
        sections.append(df.to_html(index=False))
        for fig in ["marker_staining_index.png", "dapi1_dapi2_scatter.png", "marker_correlation_heatmap.png"]:
            p = slide_dir / "figures" / fig
            if p.exists():
                rel = p.relative_to(cfg.qc_dir.parent)
                sections.append(f'<figure><img src="../../{rel}" width="700"/><figcaption>{fig}</figcaption></figure>')

    sections.append("<h2>Provenance</h2><ul>")
    sections.append(f"<li>Schema: {summary.get('schema_version')}</li>")
    sections.append(f"<li>Config hash: {summary.get('config_hash')}</li>")
    sections.append(f"<li>Generated: {summary.get('generated_at')}</li>")
    sections.append("</ul>")

    out_dir = cfg.qc_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slide_id}_qc.html"
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>CellDIVE QC</title><style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc}</style></head><body>"
        + "\n".join(sections)
        + "</body></html>"
    )
    out_path.write_text(html)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", default="SLIDE-045")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    out = render_html(args.slide, args.config)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
