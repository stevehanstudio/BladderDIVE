"""
Scoring functions shared by the mask requantification and the ROI segmentation bake-off.

No ground-truth masks exist for this slide, so mask quality cannot be scored by IoU.
Instead the biological-impossibility gates already used in production act as the objective
function: a better mask assigns fewer cells a combination of markers that cannot co-occur.

The subtlety is thresholding. `gating_thresholds.csv` was derived on the DAPI-only mask.
Reusing those absolute cutoffs on a larger, membrane-informed mask is not neutral: bigger
masks average over more background pixels, so every marker's mean intensity falls and
positivity drops across the board. That alone lowers the double-positive rate without any
real improvement in segmentation. Three threshold policies are therefore provided:

    fixed    the production cutoffs, unchanged. Directly comparable to the published
             15.83 % / 11.52 % numbers, but confounded by mask-size-driven dilution.
    gmm      re-fit per mask with the same two-component log-space GMM used originally.
             Judges each mask on its own intensity distribution.
    matched  per marker, the cutoff that reproduces the DAPI-only positive fraction.
             Removes the dilution confound entirely, so any change in the double-positive
             rate reflects a change in marker CO-OCCURRENCE rather than in marker calling.

`matched` is the fairest comparison of segmentation quality; `fixed` is the one that
answers "what would the pipeline report today".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GATING_CSV = PROJECT_ROOT / "output" / "gating_thresholds.csv"
RULES_H5AD = PROJECT_ROOT / "output" / "celldive_protein_matrix_celltypes.h5ad"

# Verbatim from notebooks/analyze_cell_types_from_markers.ipynb. Order matters: gates are
# applied first-match-wins, so per-pair counts are conditional on the earlier gates.
SPILLOVER_GATES = [
    ("CD45", "PANCK", "Immune+Epithelial"),
    ("CD45", "CD31", "Immune+Endothelial"),
    ("PANCK", "CD31", "Epithelial+Endothelial"),
    ("CD3E", "CD20", "TCell+BCell"),
]

# PANCK+VIM is deliberately NOT a gate: those are EMT tumour cells, the most invasive
# population in the sample, and flagging them would silently delete them.


def load_thresholds(path: Path = GATING_CSV) -> dict[str, float]:
    return pd.read_csv(path).set_index("marker")["threshold"].to_dict()


def load_rules(path: Path = RULES_H5AD) -> list[tuple[str, list[str], list[str]]]:
    """Cell-type rules in their original priority order, read from the production h5ad."""
    import anndata as ad

    raw = json.loads(ad.read_h5ad(path, backed="r").uns["cell_type_rules"])
    return [(name, v["positive"], v["negative"]) for name, v in raw.items()]


# ---------------------------------------------------------------- threshold policies


def thresholds_fixed(X, names, base) -> dict[str, float]:
    return dict(base)


def thresholds_matched(X, names, base, reference_rates: dict[str, float]) -> dict[str, float]:
    """Per-marker cutoff reproducing a reference positive fraction on this mask."""
    col = {m: i for i, m in enumerate(names)}
    out = {}
    for m, rate in reference_rates.items():
        if m not in col:
            continue
        v = X[:, col[m]]
        out[m] = float(np.quantile(v, 1.0 - rate)) if 0 < rate < 1 else float(base.get(m, np.inf))
    return out


def thresholds_gmm(X, names, base, n_sigma: float = 2.0, max_cells: int = 200_000) -> dict[str, float]:
    """Two-component Gaussian mixture in log1p space; cutoff = background mu + n_sigma*sigma."""
    from sklearn.mixture import GaussianMixture

    col = {m: i for i, m in enumerate(names)}
    rng = np.random.default_rng(0)
    out = {}
    for m in base:
        if m not in col:
            continue
        v = X[:, col[m]]
        pos = v[v > 0]
        if pos.size < 1000:
            out[m] = float(base[m])
            continue
        if pos.size > max_cells:
            pos = rng.choice(pos, max_cells, replace=False)
        lv = np.log1p(pos.astype(np.float64)).reshape(-1, 1)
        try:
            g = GaussianMixture(n_components=2, random_state=0).fit(lv)
            mu = g.means_.flatten()
            sd = np.sqrt(g.covariances_.flatten())
            bg = int(np.argmin(mu))
            out[m] = float(np.expm1(mu[bg] + n_sigma * sd[bg]))
        except Exception:
            out[m] = float(base[m])
    return out


# ---------------------------------------------------------------- scoring


def binarize(X, names, thresholds) -> np.ndarray:
    B = np.zeros(X.shape, dtype=np.int8)
    for j, m in enumerate(names):
        if m in thresholds:
            B[:, j] = (X[:, j] > thresholds[m]).astype(np.int8)
    return B


def positive_rates(B, names) -> dict[str, float]:
    return {m: float(B[:, j].mean()) for j, m in enumerate(names)}


def apply_gates(B, names, rules) -> tuple[np.ndarray, dict]:
    """Reproduce the production QC + phenotyping cascade. Returns (labels, per-gate stats)."""
    col = {m: i for i, m in enumerate(names)}
    n = B.shape[0]
    labels = np.array(["Pass"] * n, dtype=object)

    per_pair = {}
    for m1, m2, reason in SPILLOVER_GATES:
        if m1 not in col or m2 not in col:
            continue
        sel = (B[:, col[m1]] == 1) & (B[:, col[m2]] == 1) & (labels == "Pass")
        k = int(sel.sum())
        labels[sel] = "Artifact_Spillover"
        per_pair[f"{m1}+{m2}"] = 100 * k / n

    for name, pos, neg in rules:
        if any(m not in col for m in pos + neg):
            continue
        ok = np.ones(n, dtype=bool)
        for m in pos:
            ok &= B[:, col[m]] == 1
        for m in neg:
            ok &= B[:, col[m]] == 0
        labels[ok & (labels == "Pass")] = name
    labels[labels == "Pass"] = "Unassigned"

    stats = {
        "spillover_pct": 100 * float((labels == "Artifact_Spillover").mean()),
        "unassigned_pct": 100 * float((labels == "Unassigned").mean()),
        "per_pair_pct": per_pair,
    }
    return labels, stats


def score_matrix(X, names, obs, thresholds, rules, label: str) -> dict:
    """Full scorecard for one (mask, threshold-policy) combination."""
    B = binarize(X, names, thresholds)
    labels, stats = apply_gates(B, names, rules)
    area = np.asarray(obs["area"], dtype=np.float64)
    row = {
        "label": label,
        "n_cells": int(X.shape[0]),
        "mean_area_px": float(area.mean()),
        "median_area_px": float(np.median(area)),
        "spillover_pct": stats["spillover_pct"],
        "unassigned_pct": stats["unassigned_pct"],
        "usable_pct": 100 - stats["spillover_pct"] - stats["unassigned_pct"],
    }
    for k, v in stats["per_pair_pct"].items():
        row[f"pair_{k}_pct"] = v
    return row, labels
