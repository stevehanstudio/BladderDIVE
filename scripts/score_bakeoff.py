"""
Score every candidate mask on the validation ROIs using the production spillover gates.

For each (ROI, mask) pair this requantifies mean marker intensities directly from the ROI
OME-TIFF, then runs the same QC cascade the pipeline uses, under three threshold policies
(see scripts/spillover_score.py for why more than one is needed).

Border-touching cells are dropped: they are truncated by the ROI window, so their means
are not comparable.

Usage:
    python scripts/score_bakeoff.py
    python scripts/score_bakeoff.py --rois lymphoid --policies matched
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import spillover_score as ss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROI_DIR = PROJECT_ROOT / "output" / "validation_rois"
MASK_DIR = PROJECT_ROOT / "output" / "bakeoff_masks"
REFERENCE_H5AD = PROJECT_ROOT / "output" / "celldive_protein_matrix.h5ad"
OUT_CSV = PROJECT_ROOT / "output" / "bakeoff_scores.csv"


def reference_positive_rates(X, names, base_thr) -> dict[str, float]:
    """
    Per-marker positive fraction of the production mask WITHIN THIS ROI.

    The reference must be local. A lymphoid ROI is genuinely ~60 % T cells, so matching it
    to the slide-wide CD3E positive fraction would suppress real biology and make every
    mask look equally clean for the wrong reason. Anchoring to the production mask's own
    positivity in the same ROI asks the question that matters: at equal marker calling,
    does this mask produce fewer impossible co-expressions?
    """
    B = ss.binarize(X, names, base_thr)
    return {m: float(B[:, j].mean()) for j, m in enumerate(names)}


def quantify(mask: np.ndarray, stack: np.ndarray):
    """Mean intensity and area per label, excluding labels touching the ROI border."""
    maxlab = int(mask.max())
    flat = mask.ravel()
    cnt = np.bincount(flat, minlength=maxlab + 1).astype(np.float64)
    means = np.zeros((maxlab + 1, stack.shape[0]), np.float64)
    for c in range(stack.shape[0]):
        s = np.bincount(flat, weights=stack[c].ravel().astype(np.float64), minlength=maxlab + 1)
        np.divide(s, cnt, out=means[:, c], where=cnt > 0)

    border = np.unique(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]))
    keep = cnt > 0
    keep[0] = False
    keep[border[border > 0]] = False
    ids = np.flatnonzero(keep)
    return means[ids], pd.DataFrame({"cell_id": ids, "area": cnt[ids]})


def candidate_masks(roi: str) -> dict[str, Path]:
    out = {"production_dapi_only": ROI_DIR / f"{roi}_mask_dapi_only.tif"}
    for p in sorted(MASK_DIR.glob(f"{roi}__*.tif")):
        if p.name.endswith("_nuclei.tif"):
            continue
        out[p.stem.split("__", 1)[1]] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rois", nargs="+", default=["lymphoid", "vascular", "invasive"])
    ap.add_argument("--policies", nargs="+", default=["fixed", "matched", "gmm"])
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    manifest = json.loads((ROI_DIR / "rois.json").read_text())
    base_thr = ss.load_thresholds()
    rules = ss.load_rules()

    rows = []
    for roi in args.rois:
        names = manifest[roi]["channels"]
        stack = tifffile.imread(ROI_DIR / f"{roi}.ome.tif")
        print(f"\n{'='*100}\nROI {roi}  origin={manifest[roi]['origin_yx_level0']}\n{'='*100}")

        candidates = candidate_masks(roi)
        # Establish the local reference from the production mask before scoring anything,
        # so every candidate in this ROI is matched against the same positivity target.
        ref_mask = tifffile.imread(candidates["production_dapi_only"]).astype(np.int64)
        ref_X, _ = quantify(ref_mask, stack)
        ref_rates = reference_positive_rates(ref_X, names, base_thr)
        del ref_mask, ref_X

        for variant, path in candidates.items():
            if not path.exists():
                continue
            mask = tifffile.imread(path).astype(np.int64)
            X, obs = quantify(mask, stack)
            runtime = None
            jp = path.with_suffix(".json")
            if jp.exists():
                runtime = json.loads(jp.read_text()).get("runtime_s")

            for policy in args.policies:
                if policy == "fixed":
                    thr = ss.thresholds_fixed(X, names, base_thr)
                elif policy == "matched":
                    thr = ss.thresholds_matched(X, names, base_thr, ref_rates)
                elif policy == "gmm":
                    thr = ss.thresholds_gmm(X, names, base_thr)
                else:
                    raise SystemExit(f"unknown policy {policy}")
                row, _ = ss.score_matrix(X, names, obs, thr, rules, variant)
                row.update({"roi": roi, "policy": policy, "runtime_s": runtime})
                rows.append(row)
                print(
                    f"  {variant:22s} {policy:8s} "
                    f"n={row['n_cells']:>7,} area={row['mean_area_px']:6.1f} "
                    f"spill={row['spillover_pct']:5.2f}% unassigned={row['unassigned_pct']:5.2f}% "
                    f"usable={row['usable_pct']:5.2f}%"
                )
            del mask
        del stack

    df = pd.DataFrame(rows)
    front = ["roi", "policy", "label", "n_cells", "mean_area_px", "median_area_px",
             "spillover_pct", "unassigned_pct", "usable_pct", "runtime_s"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    for policy in args.policies:
        sub = df[df.policy == policy]
        if sub.empty:
            continue
        print(f"\n--- policy: {policy} — CD3E+CD20 and CD45+CD31 per ROI ---")
        piv = sub.pivot_table(index="label", columns="roi",
                              values=["pair_CD3E+CD20_pct", "pair_CD45+CD31_pct", "usable_pct"])
        print(piv.to_string(float_format=lambda v: f"{v:7.2f}"))


if __name__ == "__main__":
    main()
