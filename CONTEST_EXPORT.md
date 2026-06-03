# UCSF BioGraphic Image Contest 2026 — Export Guide

Export multi-channel RGB composites from CellDIVE multiplexed immunofluorescence data for the UCSF BioGraphic Image Contest. Each composite blends 4–5 fluorescence channels with custom colors, gamma, opacity, and blending modes.

---

## Quick Start

```bash
conda activate napari-env

# Quick preview (2000 px, JPEG)
python scripts/export_biographic_contest_composites.py --preview

# Hi-res (50% of full res, reads from pyramid level 1)
python scripts/export_biographic_contest_composites.py --max-size 36978 --format jpeg

# Full resolution (tiled BigTIFF, ~14 GB each)
python scripts/export_biographic_contest_composites.py

# With palette and blend options
python scripts/export_biographic_contest_composites.py --preview --palette original --blend max
```

---

## Composites (Hybrid Palette — Contest Default)

| Composite | Channels | Colors | Purpose |
|-----------|----------|--------|---------|
| **Invasive_Front** | PANCK, VIM, ACTA2, Ki67, DAPI | Yellow-green, blue, teal, gold, white | Tumor vs. stroma boundary |
| **Immune_Surveillance** | CD8a, CD68, CD163, DAPI | Bright yellow, cyan, green, neon blue | Myeloid vs. lymphoid immune cells |
| **Vascular_Architecture** | CD31, COL1A1, PDGFRA, DAPI | Light cyan, orange, red, blue | Vessels, collagen, perivascular stroma |
| **Proliferation_Zone** | Ki67, PANCK, CD44, DAPI | Chartreuse, neon blue, deep pink, white | Proliferating cells and stemness |
| **Tertiary_Lymphoid_Structures** | CD20, CD3E, CD38, PANCK, DAPI | Green, blue, yellow, yellow-green, blue | Organized adaptive immunity |

---

## Color Palettes

Two palettes are available via `--palette`:

- **`contest`** (default): Hybrid palette with curated colors optimized for complementary contrast (blue+yellow, orange+cyan, red accents).
- **`original`**: Spectral colors extracted from the CellDIVE Zarr metadata, preserving the instrument's native color assignments. Opacity and gamma settings are inherited from the contest palette.

---

## Blending Modes

Three modes available via `--blend`:

| Mode | Behavior | Best for |
|------|----------|----------|
| **`additive`** (default) | Sum channel intensities, clip to 1.0 | Standard fluorescence compositing |
| **`screen`** | `1 - (1-a)(1-b)` — preserves color in bright regions | Reducing washout in dense tissue |
| **`max`** | Per-pixel maximum across channels | Saturated colors, no blending artifacts |

---

## Tuning: Gamma, Opacity, and Colors

Composites are defined in `scripts/export_biographic_contest_composites.py` in `COMPOSITES_CONTEST`. Each channel entry: `(alias, hex_color, opacity, gamma)`.

### Gamma (contrast / brightness)

| Gamma | Effect |
|-------|--------|
| 0.4–0.5 | Bright, reveals dim structures |
| 0.55–0.6 | Balanced |
| 0.7–0.85 | Higher contrast, dimmer midtones |

### Opacity (channel weight)

| Opacity | Use case |
|---------|----------|
| 0.3–0.5 | Background / context (DAPI, structural) |
| 0.6–0.8 | Supporting channels |
| 0.9–1.0 | Primary markers |

### Hex colors

Format: `#RRGGBB`. Brighter colors contribute more to composites; darker colors reduce saturation overlap.

---

## CLI Options

| Option | Description |
|--------|-------------|
| `--preview [N]` | Quick preview at N px (default 2000), JPEG format |
| `--max-size N` | Cap longest dimension to N px |
| `--format {tiff,png,jpeg}` | Output format (default: tiff) |
| `--palette {contest,original}` | Color palette (default: contest) |
| `--blend {additive,screen,max}` | Blending mode (default: additive) |
| `--photoshop` | 32768 px max, PNG format |
| `--test` | 2000x2000 center crop only |
| `-o DIR` | Output directory (default: `output/contest_2026`) |

---

## Resolution Guide

| Setting | Dimensions | File size | Use |
|---------|-----------|-----------|-----|
| `--preview` | ~2000 px | ~2 MB | Quick review |
| `--max-size 36978` | 32312 x 36978 | ~400–600 MB JPEG | Hi-res editing |
| Full resolution | 64624 x 73957 | ~14 GB TIFF | Archival |

The script automatically selects the optimal Zarr pyramid level to minimize memory usage.

---

## Percentile Normalization

Each channel is normalized using global 1st–99.5th percentiles computed from the coarsest pyramid level. This ensures consistent contrast across the entire image without banding artifacts.

---

## Tissue Masking

A DAPI-based tissue mask (computed from pyramid level 3) removes background and preserves tile gaps. Missing acquisition tiles remain masked and are not filled.

---

## Channel Aliases

Short names for CLI and code: `DAPI`, `PANCK`, `ACTA2` (SMA), `VIM`, `CD8a`, `CD68`, `CD163`, `CD31`, `PDGFRA`, `COL1A1`, `CD45`, `Ki67`, `CD44`, `CD20`, `CD3E`, `CD38`, `CD11c`, `CD14`, `EPCAM`, `CD56`, `CD45RO`, `HLADR`. Full mapping in `CHANNEL_ALIAS` dict.

---

## File Locations

- **Input:** `data/CellDIVE_SLIDE-045.zarr`
- **Output:** `output/contest_2026/`
- **Script:** `scripts/export_biographic_contest_composites.py`
