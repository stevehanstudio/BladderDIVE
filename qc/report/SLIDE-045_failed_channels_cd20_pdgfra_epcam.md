# Did CD20, PDGFRA, and EPCAM work?

**Slide:** SLIDE-045 · **926,006 cells** · 23-channel CellDIVE

**Verdict**

| Channel | Round / fluor | Verdict | One-line reason |
| --- | --- | --- | --- |
| **PDGFRA** | R05 AF488 | **Failed** | Tight blob on background; tracks epithelium, not stroma |
| **EPCAM** | R06 AF488 | **Mostly failed** | Unimodal haze; only 18% of EPCAM+ cells are PANCK+ |
| **CD20** | R03 AF555 | **Contaminated** (antibody likely worked) | Real bright B-cell tail, but 74% of gated CD20+ cells are also CD3E+ |

A working lineage marker has a bright tail (high p99 / background), sits in the expected compartment, and co-occurs with a sister marker from a different fluorophore and round. Global round failure is ruled out when a same-round, different-fluorophore channel still looks healthy (CD14 on R05, CD45RO on R06).

**Sources:** `output/celldive_protein_matrix_celltypes.h5ad`, `output/gating_thresholds.csv`, `qc/SLIDE-045/`, `notebooks/determine_binary_threshold.ipynb`, `notebooks/investigate_spillover_mechanism.ipynb`. Compared against lineage-matched channels that clearly did work (CD3E, PANCK, VIM, COL1A1, CD14).

---

## Bright-tail dynamic range

p99 intensity divided by per-marker background (`uns.background_estimate`, p10 of positive cells). Working markers are tens to thousands; a failed stain sits near 1–15.

| Marker | p99 / background |
| --- | ---: |
| PDGFRA | 3.3× |
| EPCAM | 14.4× |
| CD20 | 46.1× |
| ACTA2 | 39.3× |
| COL1A1 | 50.2× |
| VIM | 82.9× |
| PANCK | 96.8× |
| CD45 | 430× |
| CD3E | 3,806× |

---

## AF488 signal by imaging round

PDGFRA and EPCAM are the last two AF488 stains. PANCK on the previous AF488 round still has a 97× bright tail, and CD14-AF647 on the same round as PDGFRA has a 1,709× tail. This is not wholesale tissue destruction after R04. It is those two antibodies (or AF488 on those cycles) producing almost no specific signal.

| Round | Marker | log10(p99 / background) | p99 / background |
| --- | --- | ---: | ---: |
| R01 | CD45 | 2.63 | 430× |
| R02 | VIM | 1.92 | 83× |
| R03 | ACTA2 | 1.59 | 39× |
| R04 | PANCK | 1.99 | 97× |
| R05 | **PDGFRA** | **0.51** | **3.3×** |
| R06 | **EPCAM** | **1.16** | **14×** |

CD14 (AF647, R05) remains 3.23 on the log10 scale (1,709×).

---

## Biological co-expression

If the stain is specific, EPCAM should recover PANCK+ epithelium, PDGFRA should recover VIM+ stroma, and CD20 should be almost mutually exclusive with CD3E. Observed rates go the wrong way for all three.

Binary layer from `gating_thresholds.csv`: CD20 gate 300, EPCAM p90 fallback 78.5, PDGFRA p90 fallback 132.1, PANCK GMM 673.4.

| Quantity | Observed | Expected if the stain worked |
| --- | ---: | --- |
| P(CD3E \| CD20) | 74.1% | Near 0% (T vs B) |
| P(PANCK \| EPCAM) | 18.4% | High (both epithelial) |
| P(EPCAM \| PANCK) | 49.4% | High |
| P(VIM \| PDGFRA) | 24.0% | High (fibroblasts) |
| P(PANCK \| PDGFRA) | 8.5% | Low (not epithelial) |

---

## PDGFRA — antibody almost certainly did not work

The distribution is a single peak sitting on background (median 111, background 75, p99 244, p99.9 330). QC staining index is 1.46 (warn; fail = 1.0) and the GMM separation is 0.77 (unimodal). The 10% “positive” rate is an artifact of the p90 fallback gate, not a fibroblast population.

| Test | PDGFRA | What a working stain would do |
| --- | --- | --- |
| p99 / background | 3.3× | VIM 83×, PANCK 97×, CD3E 3,806× |
| Top correlation | EPCAM r=0.51, PANCK r=0.51 | VIM / COL1A1 / ACTA2 (stroma), not epithelium |
| P(VIM+ \| PDGFRA+) | 24% | Most fibroblasts should be vimentin+ |
| Median in stroma vs all cells | 110 vs 111 | Enriched in VIM+ CD45− cells |
| Same-round sister channel | CD14-AF647 p99/bg = 1,709× | R05 tissue and imaging are intact |

**Most likely reason:** non-functional PDGFRA-AF488 staining (dead/wrong clone, too dilute, or epitope not accessible) leaving autofluorescence. Incomplete stripping of PANCK-AF488 is a weaker fit: PDGFRA vs PANCK is r=0.81 in PANCK-low cells and r=−0.35 in PANCK-high cells, which is shared haze, not leftover bright PanCK. R05 stripping did not destroy the section — COL1A1-AF555 and CD14-AF647 on the same cycle are bright and structured.

**Do not use PDGFRA for phenotyping.** Drop the Fibroblast = PDGFRA+ rule. Use VIM, COL1A1, and ACTA2 for stroma. The 48,012 “Fibroblast” labels are the top 10% of a failed channel.

---

## EPCAM — largely failed; PanCK is the epithelial channel

EPCAM is unimodal (GMM sep 0.85) with p99/background = 14×. QC marked it **pass** (staining index 4.2, SNR 2.9) because background is only 21, so dim haze still clears the SNR cutoff. That is a false pass.

| Test | EPCAM | PANCK (working sister) |
| --- | --- | --- |
| Gate source | p90 fallback (78.5) | GMM, clearly bimodal (sep 6.04) |
| % positive | 9.9% (by construction ~10%) | 3.7% |
| Jaccard with the other epithelial marker | 0.15 | 0.15 |
| Median in PANCK+ vs PANCK− | 78 vs 57 (almost no contrast) | PANCK is the compartment definition |

81.6% of EPCAM+ cells are PANCK−. EPCAM’s strongest correlation is PDGFRA (r=0.51) — the other failed AF488 channel — not PANCK (r=0.28). There is a faint high tail inside PANCK+ cells (p99 1,015 vs 277 in PANCK−), so a weak real epitope cannot be ruled out, but it is not usable for gating. DAPI R01 vs R06 correlation is 0.69 (warn) with 0 px XY drift, consistent with late-round intensity change, yet CD45RO-AF647 on R06 still has a 231× bright tail, so R06 imaging works for other fluorophores.

**Do not use EPCAM for phenotyping.** The 61,775 Epithelial_EPCAM labels are mostly the brightest 10% of haze. Keep PANCK as the epithelial call. CD56-AF555 on the same round is similarly weak (p99/bg = 13×) and should be treated as suspect too.

---

## CD20 — antibody likely worked; the gate is not trustworthy

Unlike PDGFRA/EPCAM, CD20 has a real bright tail (p99 5,245, p99.9 10,165, p99/p50 = 39). The brightest 1% of CD20 cells are 99.7% CD45+ and spatially the most concentrated marker on the slide (98% of top-5% cells fall in the hottest 10% of bins) — consistent with lymphoid aggregates / TLS. A plausible B-cell set exists: CD20+ CD45+ CD3E− = 12,702 cells (1.4%).

### Why the channel still looks broken

The gate at 300 still calls 14.6% of cells CD20+. Of those, 74% are CD3E+ and 69% are CD8a+. T-cell / B-cell double positives are 10.8% of the slide; 77.5% of gated CD20+ cells were labeled Artifact_Spillover. Automated thresholding first called 37% positive, which is why the gate was raised by hand.

CD20-AF555 is the same fluorophore as the extremely bright CD3E-AF555 (R01, p99/bg = 3,806×). The strongest AF555 pair on the slide is CD3E–CD20 (r=0.42). Incomplete stripping of CD3E into the CD20 channel puts T cells over the CD20 gate. Independently, the spillover notebook found CD3E+/CD20+ cells 4.8× enriched for a CD20-only neighbor, so segmentation bleed in crowded lymphoid regions is also real.

**Gated CD20+ cell types** (134,943 cells at gate 300; B_cell requires CD45+ CD20+):

| Cell type | % of CD20+ |
| --- | ---: |
| Artifact_Spillover | 77.5% |
| B_cell | 6.2% |
| Unassigned | 3.7% |
| Myofibroblast | 2.8% |
| Other | 9.8% |

| CD20 subset | n | Median CD20 | % CD3E+ | % CD45+ |
| --- | ---: | ---: | ---: | ---: |
| Top 1% intensity | 9,261 | 6,914 | 66.6% | 99.7% |
| All gated CD20+ | 134,943 | 697 | 74.1% | 75.8% |

Even the bright tail is 67% CD3E+, so stripping/spillover reaches the high end. True B cells are the CD45+ CD3E− fraction of that tail (33% of top 1%), not the gate.

**Keep CD20 only with a T-cell veto.** Do not interpret CD20+ as B cells. A salvage rule is CD20-high, CD45+, CD3E−, CD8a−, preferably restricted to lymphoid aggregates. The 8.41–10.8% CD3E+/CD20+ pool is artifact (incomplete AF555 stripping plus segmentation spillover), not double-positive lymphocytes.

---

## Causes ranked

| Cause | PDGFRA | EPCAM | CD20 |
| --- | --- | --- | --- |
| Antibody / epitope produced no specific stain | **Primary** | **Primary** (possible faint epithelial tail) | Unlikely — bright CD45+ TLS-like tail exists |
| Incomplete stripping of the previous same-fluorophore cycle | Unlikely (anti-correlates with bright PANCK) | Possible weak PANCK-AF488 residual | **Primary** for the T-cell contamination (CD3E-AF555 → CD20-AF555) |
| Segmentation spillover from neighbors | N/A (no real signal to spill) | N/A | Confirmed in lymphoid ROI (4.8× neighbor enrichment) |
| Late-round tissue destruction | Ruled out (CD14 and COL1A1 on R05 work) | Partial (DAPI r=0.69) but CD45RO on R06 works | No (R03; CD163-AF647 on R03 is excellent) |

---

## Practical next steps

1. Visually confirm in napari/QuPath on a known epithelial region (PANCK vs EPCAM) and a known lymphoid aggregate (CD20 vs CD3E).
2. Drop PDGFRA and EPCAM from phenotyping rules.
3. Stroma: VIM / COL1A1 / ACTA2. Epithelium: PANCK only.
4. B cells: CD20-high **and** CD45+ **and** CD3E− (and preferably CD8a−), not CD20+ alone.
5. Treat CD56 (R06 AF555, p99/bg = 13×) as suspect in the same class as EPCAM.
