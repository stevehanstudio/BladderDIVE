"""QC orchestrator."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.qc.config import QCConfig, load_config
from scripts.qc.io import QCContext, free_memory
from scripts.qc.registry import REGISTRY
from scripts.qc.thresholds import module_status
from scripts.qc.writers import finalize_qc

# Import modules to populate REGISTRY
from scripts.qc.modules import batch_qc  # noqa: F401
from scripts.qc.modules import image_qc  # noqa: F401
from scripts.qc.modules import marker_qc  # noqa: F401
from scripts.qc.modules import segmentation_qc  # noqa: F401
from scripts.qc.modules import spatial_qc  # noqa: F401
from scripts.qc.modules.batch_qc import run_cohort_qc


DEFAULT_MODULES = ["image", "segmentation", "marker", "spatial"]


def run_module(ctx: QCContext, module: str) -> dict[str, Any]:
    fns = REGISTRY.get(module, {})
    if not fns:
        return {"status": "skip", "reason": f"no metrics registered for {module}"}
    results: dict[str, Any] = {}
    for name, fn in fns.items():
        if module == "batch":
            continue  # batch handled by run_cohort
        t0 = time.perf_counter()
        print(f"  [{module}] {name}...", flush=True)
        results[name] = fn(ctx)
        ctx.clear_caches()
        free_memory()
        elapsed = time.perf_counter() - t0
        print(f"      done in {elapsed:.1f}s", flush=True)
    return results


def run_slide(
    slide_id: str,
    cfg: QCConfig | None = None,
    modules: list[str] | None = None,
    save_adata: bool = True,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    modules = modules or DEFAULT_MODULES
    slide = cfg.slide(slide_id)

    print(f"Running QC for slide {slide_id}", flush=True)
    ctx = QCContext.from_slide(cfg, slide_id)

    module_results: dict[str, dict[str, Any]] = {}
    for module in modules:
        print(f"Module: {module}", flush=True)
        module_results[module] = run_module(ctx, module)
        ctx.clear_caches()
        free_memory()

    ctx.apply_cell_flags_to_adata()
    summary = finalize_qc(ctx.adata, slide, cfg, module_results)

    if save_adata:
        out_path = slide.adata_out or slide.adata_in
        print(f"Writing AnnData with QC fields to {out_path}", flush=True)
        adata_out = ctx.materialize_adata_for_write()
        adata_out.write_h5ad(out_path)
        if adata_out is not ctx.adata:
            del adata_out
        free_memory()

    print(f"QC complete. Overall status: {summary['overall_status']}", flush=True)
    print(f"Summary: {cfg.qc_dir / slide_id / 'qc_summary.json'}", flush=True)
    return summary


def run_cohort(cfg: QCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    return run_cohort_qc(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CellDIVE QC pipeline")
    parser.add_argument("--slide", default=None, help="Slide ID (default: all slides in config)")
    parser.add_argument("--config", type=Path, default=None, help="Path to qc_config.yaml")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=DEFAULT_MODULES,
        help=f"QC modules to run (default: {DEFAULT_MODULES})",
    )
    parser.add_argument("--cohort", action="store_true", help="Also run cohort/batch QC aggregation")
    parser.add_argument("--no-save-adata", action="store_true", help="Do not write updated h5ad")
    args = parser.parse_args()

    cfg = load_config(args.config)
    slide_ids = [args.slide] if args.slide else [s.id for s in cfg.slides]

    for slide_id in slide_ids:
        run_slide(
            slide_id,
            cfg=cfg,
            modules=args.modules,
            save_adata=not args.no_save_adata,
        )

    if args.cohort or len(cfg.slides) > 1:
        result = run_cohort(cfg)
        print(f"Cohort QC: {result.get('status', result)}", flush=True)


if __name__ == "__main__":
    main()
