# PIPEX Analysis Guide for CellDIVE Bladder Dataset

## Overview

[PIPEX](https://github.com/CellProfiling/pipex) is perfect for your 23-channel CellDIVE data as it provides:
- **Advanced cell segmentation** using StarDist + membrane refinement
- **Automated analysis** of multiplexed imaging data
- **Cell phenotyping** and clustering
- **Spatial analysis** and cell-cell interactions  
- **QuPath integration** for visualization

## Quick Start

### 1. Setup PIPEX Environment
```bash
cd /home/steve/Projects/HeLab/bladder
python setup_pipex_for_celldive.py
```

This will:
- Install PIPEX dependencies
- Create the required directory structure
- Generate configuration files for your 23 markers
- Create extraction and analysis scripts

### 2. Prepare Your Data

PIPEX needs individual channel files. The setup script creates an extraction tool:

```bash
cd /mnt/data/Projects/HeLab/bladder_proteomics_data/pipex_analysis
python extract_for_pipex.py
```

This extracts your 23 channels as individual TIFF files with PIPEX-compatible names:
- `DAPI.tif` (nuclear segmentation)
- `CD45.tif`, `CD3E.tif`, `CD8a.tif` (immune markers)
- `PANCK.tif`, `EPCAM.tif` (epithelial markers)
- `VIM.tif`, `PDGFRA.tif` (stromal markers)
- And 15 more...

### 3. Run PIPEX Analysis

```bash
cd /mnt/data/Projects/HeLab/bladder_proteomics_data/pipex_analysis
./run_pipex.sh
```

## Key PIPEX Features for Your Dataset

### Cell Segmentation Strategy
```
DAPI (nuclei) → StarDist → Cell expansion → PANCK refinement
```

PIPEX will:
1. **Detect nuclei** using DAPI with StarDist deep learning
2. **Expand cells** around nuclei (configurable radius)
3. **Refine boundaries** using PanCK (epithelial membrane marker)
4. **Generate masks** for accurate single-cell analysis

### Automated Analysis Pipeline

**Cell Phenotyping:**
- Epithelial cells: `PANCK+`, `EPCAM+` 
- T cells: `CD3E+`, `CD8a+` (cytotoxic), `CD45RO+` (memory)
- B cells: `CD20+`, `CD38+` (plasma cells)
- Macrophages: `CD68+`, `CD163+` (M2 type)
- Stromal cells: `VIM+`, `PDGFRA+`, `COL1A1+`

**Spatial Analysis:**
- Cell-cell interactions
- Neighborhood analysis  
- Spatial clustering
- Distance-based statistics

**Functional States:**
- Proliferation: `Ki67+` cells
- Activation: `HLADR+` cells
- Adhesion: `CD44+` cells

### Configuration Files Created

**markers.txt** - Your 23 channel names:
```
DAPI
CD45
CD3E
Ki67
CD8a
VIM
...
```

**cell_types.csv** - Bladder cancer-specific cell type definitions:
```
epithelial,tumor,panck_positive → PANCK+ EPCAM+ CD45-
immune,tcell,cd8_cytotoxic → CD8a+ CD3E+ CD45+
immune,macrophage,m2_type → CD163+ CD68+ CD45+
...
```

## Expected Results

PIPEX will generate:

### 📁 Segmentation Results
- `segmentation_mask.npy` - Cell segmentation masks
- `nuclei_mask.npy` - Nuclear masks
- Quality control images for validation

### 📊 Analysis Data
- `analysis.csv` - Single-cell measurements for all 23 markers
- `clusters_leiden.csv` - Unsupervised cell clustering
- `clusters_refined.csv` - Annotated cell types
- `interactions.csv` - Cell-cell interaction analysis

### 🎯 QuPath Integration
- `qupath_annotations.json` - Import cell segmentation to QuPath
- Direct compatibility with QuPath 0.6.0

### 📈 Visualizations
- Marker expression heatmaps
- UMAP/t-SNE plots colored by:
  - Cell types
  - Marker expression
  - Spatial location
- Interaction networks
- Spatial maps

## Customization Options

### Segmentation Parameters
Edit `run_pipex.sh` to adjust:
```bash
--nuclei_size 20        # Nucleus detection sensitivity
--expansion_size 10     # Cell boundary expansion (pixels)
--membrane_size 30      # Membrane refinement strength
--compactness 0.5       # Cell shape constraint
```

### Analysis Options
```bash
--clustering            # Enable unsupervised clustering
--interactions          # Analyze cell-cell interactions
--spatial_analysis      # Spatial statistics
--generate_qupath       # Create QuPath annotations
```

### Preprocessing (if needed)
For problematic images:
```bash
--preprocessing
--threshold_min 1       # Remove background
--threshold_max 99      # Remove artifacts
--balance_tiles         # Correct uneven illumination
```

## Bladder Cancer-Specific Insights

PIPEX will help identify:

**Tumor Microenvironment:**
- Epithelial tumor cells vs. normal urothelium
- Immune infiltration patterns
- Stromal composition

**Immune Landscape:**
- T cell subsets and activation states
- Macrophage polarization (M1 vs M2)
- B cell and plasma cell distribution

**Spatial Organization:**
- Tumor-immune interfaces
- Vascular architecture (`CD31+` endothelial cells)
- Fibroblast networks (`PDGFRA+`, `COL1A1+`)

**Functional States:**
- Proliferative activity (`Ki67+`)
- Immune activation (`HLADR+`)
- Cell adhesion changes (`CD44+`)

## Troubleshooting

**If channel extraction fails:**
- Use the individual channel extraction script from earlier
- Copy extracted channels to `/input/` folder manually

**If segmentation looks poor:**
- Adjust `--nuclei_size` (10-30 range)
- Try different `--membrane_channel` (VIM, EPCAM)
- Enable `--preprocessing` for image correction

**If cell typing seems wrong:**
- Edit `cell_types.csv` with your domain knowledge
- Adjust confidence thresholds
- Add marker combinations specific to your research

## Integration with Other Tools

**QuPath Workflow:**
1. PIPEX generates segmentation
2. Import annotations to QuPath
3. Visualize with your original 23-channel data
4. Manual validation and annotation refinement

**Downstream Analysis:**
- Export data to R/Python for custom analysis
- Use generated FCS files in FlowJo or similar
- Integrate with spatial analysis packages (SpatialData, squidpy)

This workflow combines PIPEX's automated analysis with your domain expertise for comprehensive bladder cancer tissue analysis.







