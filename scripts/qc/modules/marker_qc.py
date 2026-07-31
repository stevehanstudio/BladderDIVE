"""Marker-level QC metrics."""

from __future__ import annotations

import numpy as np

from scripts.qc.io import QCContext, free_memory
from scripts.qc.registry import register_qc
from scripts.qc.thresholds import metric, worst_status

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _staining_index(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / (2.0 * neg.std() + 1e-9))


def _bimodality_score(x: np.ndarray) -> float:
    if x.size < 100:
        return 0.0
    xs = np.log1p(x)
    xs = xs[np.isfinite(xs)]
    if xs.size < 100:
        return 0.0
    hist, _ = np.histogram(xs, bins=50)
    hist = hist.astype(float) + 1e-9
    hist /= hist.sum()
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 0.02:
            peaks += 1
    return float(peaks >= 2)


def _antibody_failed(si: float, snr: float, pct_pos: float, bimodal: float, thr) -> bool:
    si_fail = thr("marker", "staining_index")
    snr_fail = thr("marker", "snr")
    low = thr("marker", "pct_positive_low")
    high = thr("marker", "pct_positive_high")
    failed = False
    if si_fail and si < si_fail.fail:
        failed = True
    if snr_fail and snr < snr_fail.fail:
        failed = True
    if low and pct_pos < low.fail:
        failed = True
    if high and pct_pos > high.fail:
        failed = True
    if bimodal < 1.0 and snr < 1.5:
        failed = True
    return failed


@register_qc("marker", "signal_quality")
def signal_quality(ctx: QCContext) -> dict:
    per_marker: dict[str, dict] = {}
    markers = [m for m in ctx.markers() if m not in ("DAPI", "DAPI2")]

    for marker in markers:
        x = ctx.marker_vector(marker, subsample=True)
        bg = ctx.marker_background(marker)
        pos = x[x > bg]
        neg = x[x <= bg]
        si = _staining_index(pos, neg)
        snr = float(np.median(pos) / (bg + 1e-9)) if pos.size else 0.0
        pct_pos = float((x > bg).mean())
        p99 = float(np.percentile(x, 99)) if x.size else 0.0
        p1 = float(np.percentile(x, 1)) if x.size else 0.0
        dynamic_range = float(p99 / (p1 + 1e-9))
        bimodal = _bimodality_score(x)
        failed = _antibody_failed(si, snr, pct_pos, bimodal, ctx.thr)

        statuses = [
            ctx.thr("marker", "staining_index").status(si) if ctx.thr("marker", "staining_index") else "pass",
            ctx.thr("marker", "snr").status(snr) if ctx.thr("marker", "snr") else "pass",
        ]
        if failed:
            statuses.append("fail")
        status = worst_status(statuses)

        per_marker[marker] = {
            "staining_index": si,
            "snr": snr,
            "pct_positive": pct_pos,
            "dynamic_range": dynamic_range,
            "bimodal": bimodal,
            "background": bg,
            "antibody_failed": failed,
            "status": status,
        }
        del x, pos, neg
        free_memory()

    if plt is not None and per_marker:
        fig_path = ctx.figure_path("marker_staining_index.png")
        names = list(per_marker.keys())
        vals = [per_marker[m]["staining_index"] for m in names]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(names, vals, color="steelblue", alpha=0.85)
        ax.axhline(2.0, color="orange", ls="--", label="warn=2")
        ax.axhline(1.0, color="crimson", ls="--", label="fail=1")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Staining index")
        ax.set_title("Per-marker staining index")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {"per_marker": per_marker}


@register_qc("marker", "dapi_correlation")
def dapi_correlation(ctx: QCContext) -> dict:
    if "DAPI" not in ctx.adata.var_names or "DAPI2" not in ctx.adata.var_names:
        return {"dapi_corr": metric(float("nan"), ctx.thr("marker", "dapi_corr"))}
    d1 = ctx.marker_vector("DAPI", subsample=True)
    d2 = ctx.marker_vector("DAPI2", subsample=True)
    n = min(len(d1), len(d2))
    d1, d2 = d1[:n], d2[:n]
    corr = float(np.corrcoef(d1, d2)[0, 1]) if d1.size else float("nan")

    if plt is not None:
        fig_path = ctx.figure_path("dapi1_dapi2_scatter.png")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hexbin(d1, d2, gridsize=80, cmap="viridis", mincnt=1)
        ax.set_xlabel("DAPI (R01)")
        ax.set_ylabel("DAPI2 (R06)")
        ax.set_title(f"DAPI correlation r={corr:.3f}")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    del d1, d2
    free_memory()

    return {"dapi_corr": metric(corr, ctx.thr("marker", "dapi_corr"))}


@register_qc("marker", "cross_marker_correlation")
def cross_marker_correlation(ctx: QCContext) -> dict:
    markers = [m for m in ctx.markers() if m not in ("DAPI", "DAPI2")]
    if len(markers) < 2:
        return {"max_abs_correlation": metric(0.0)}

    n = min(ctx.adata.n_obs, int(ctx.cfg.raw.get("correlation_cells", 20_000)))
    X = ctx.marker_matrix(markers, n_samples=n)
    corr = np.corrcoef(X, rowvar=False)
    del X
    free_memory()
    np.fill_diagonal(corr, 0.0)
    max_abs = float(np.nanmax(np.abs(corr)))

    if plt is not None:
        fig_path = ctx.figure_path("marker_correlation_heatmap.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(markers)))
        ax.set_yticks(range(len(markers)))
        ax.set_xticklabels(markers, rotation=90)
        ax.set_yticklabels(markers)
        ax.set_title("Cross-marker correlation")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "max_abs_correlation": metric(max_abs),
        "markers": markers,
    }
