"""
Score the full-slide requantifications produced by scripts/requantify_masks.py.

This is the Step 0 decision gate: it measures what switching to an already-segmented,
membrane-informed mask buys, using zero new segmentation tools.

The `dapi_only` row doubles as a correctness check — it must reproduce the published
15.83 % Artifact_Spillover / 11.52 % Unassigned under the `fixed` policy, because it is
the same mask and the same thresholds as production.

Usage:
    python scripts/score_requant.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import spillover_score as ss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUANT_DIR = PROJECT_ROOT / "output" / "requant"
REFERENCE_H5AD = PROJECT_ROOT / "output" / "celldive_protein_matrix.h5ad"
OUT_CSV = PROJECT_ROOT / "output" / "requant_scores.csv"

PUBLISHED = {"spillover_pct": 15.83, "unassigned_pct": 11.52}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policies", nargs="+", default=["fixed", "matched", "gmm"])
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    base_thr = ss.load_thresholds()
    rules = ss.load_rules()

    ref = ad.read_h5ad(REFERENCE_H5AD)
    ref_names = list(ref.var_names)
    ref_B = np.asarray(ref.layers["binary"])
    ref_rates = {m: float(ref_B[:, j].mean()) for j, m in enumerate(ref_names)}
    del ref, ref_B

    files = sorted(REQUANT_DIR.glob("protein_matrix_*.h5ad"))
    if not files:
        raise SystemExit(f"no requant h5ad found in {REQUANT_DIR}; run requantify_masks.py first")

    rows = []
    for f in files:
        a = ad.read_h5ad(f)
        tag = str(a.uns.get("mask_tag", f.stem))
        X = np.asarray(a.X, dtype=np.float64)
        names = list(a.var_names)
        print(f"\n{tag}: {a.n_obs:,} cells, mean area {a.obs['area'].mean():.1f} px")
        for policy in args.policies:
            if policy == "fixed":
                thr = ss.thresholds_fixed(X, names, base_thr)
            elif policy == "matched":
                thr = ss.thresholds_matched(X, names, base_thr, ref_rates)
            elif policy == "gmm":
                thr = ss.thresholds_gmm(X, names, base_thr)
            else:
                raise SystemExit(f"unknown policy {policy}")
            row, _ = ss.score_matrix(X, names, a.obs, thr, rules, tag)
            row["policy"] = policy
            rows.append(row)
            print(
                f"  {policy:8s} spillover={row['spillover_pct']:5.2f}%  "
                f"unassigned={row['unassigned_pct']:5.2f}%  usable={row['usable_pct']:5.2f}%  "
                f"CD3E+CD20={row.get('pair_CD3E+CD20_pct', float('nan')):5.2f}%  "
                f"CD45+CD31={row.get('pair_CD45+CD31_pct', float('nan')):5.2f}%"
            )
        del a, X

    df = pd.DataFrame(rows)
    front = ["policy", "label", "n_cells", "mean_area_px", "median_area_px",
             "spillover_pct", "unassigned_pct", "usable_pct"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    chk = df[(df.policy == "fixed") & (df.label == "dapi_only")]
    if not chk.empty:
        r = chk.iloc[0]
        ok = (abs(r.spillover_pct - PUBLISHED["spillover_pct"]) < 0.05
              and abs(r.unassigned_pct - PUBLISHED["unassigned_pct"]) < 0.05)
        print(f"\nsanity check vs published (dapi_only, fixed): "
              f"spillover {r.spillover_pct:.2f}% vs 15.83%, "
              f"unassigned {r.unassigned_pct:.2f}% vs 11.52% -> {'MATCH' if ok else 'MISMATCH'}")

    for policy in df.policy.unique():
        sub = df[df.policy == policy].set_index("label")
        print(f"\n--- policy: {policy} ---")
        cols = ["n_cells", "mean_area_px", "spillover_pct", "unassigned_pct", "usable_pct",
                "pair_CD3E+CD20_pct", "pair_CD45+CD31_pct"]
        print(sub[[c for c in cols if c in sub.columns]].to_string(float_format=lambda v: f"{v:9.2f}"))


if __name__ == "__main__":
    main()
