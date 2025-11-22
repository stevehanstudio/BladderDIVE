# Cell Type Identification Workflow
## No H&E Registration Required!

---

## 🎯 **Goal**
Identify cell types in CellDIVE data based on 23-channel marker expression.

---

## 📊 **What You Have**
- ✅ `DAPI_AF_R01.ome.tif` - Original 23-channel CellDIVE image
- ✅ `input/*.tif` - 23 extracted channel files (DAPI, CD45, CD3E, etc.)
- ✅ PIPEX pipeline - Cell segmentation & analysis tools

---

## 🚀 **Quick Start**

### Option 1: Run Complete Pipeline (Recommended)
```bash
cd /home/steve/Projects/HeLab/BladderDIVE
./run_cell_analysis.sh
```

### Option 2: Step-by-Step Execution

#### Step 1: Cell Segmentation
```bash
cd /mnt/data/Projects/HeLab/pipex
conda run -n pipex_seg python segmentation.py \
    -data=/home/steve/Projects/HeLab/BladderDIVE/input \
    -nuclei_marker=DAPI \
    -nuclei_diameter=20 \
    -nuclei_expansion=10 \
    -membrane_marker=PANCK \
    -membrane_diameter=30 \
    -measure_markers="DAPI,CD45,CD3E,CD8a,CD20,CD68,PANCK,EPCAM,VIM,Ki67"
```

**Output:**
- `input/analysis/segmentation_data.npy` - Cell masks
- `input/analysis/cell_data.csv` - Raw marker intensities

#### Step 2: Clustering & Analysis
```bash
conda run -n pipex_analysis python analysis.py \
    -data=/home/steve/Projects/HeLab/BladderDIVE/input \
    -log_norm=yes \
    -z_norm=yes \
    -dim_red=UMAP \
    -clustering=leiden \
    -interactions=yes
```

**Output:**
- `input/analysis/analysis_data.csv` - Normalized single-cell data
- `input/analysis/clusters_leiden.csv` - Cell cluster assignments
- `input/analysis/umap_coordinates.csv` - UMAP embedding
- `input/analysis/interactions.csv` - Cell-cell interactions

---

## 📋 **Output Files Explained**

### `cell_data.csv` (Raw Data)
```csv
CellID,X,Y,DAPI,CD45,CD3E,CD8a,CD20,CD68,PANCK,EPCAM,VIM,Ki67
1,1000,2000,5000,800,300,50,10,5,100,50,200,10
2,1020,2015,4800,50,20,10,5,10,3000,2500,100,5
...
```

### `analysis_data.csv` (Normalized & Analyzed)
```csv
CellID,X,Y,DAPI_norm,CD45_norm,...,Cluster,Neighbors,LocalDensity
1,1000,2000,2.3,3.5,...,0,15,0.82
2,1020,2015,2.1,0.5,...,3,20,0.91
...
```

### `clusters_leiden.csv` (Cell Clusters)
```csv
CellID,Cluster,ClusterSize
1,0,5420
2,3,1893
...
```

---

## 🔬 **Cell Type Identification Strategy**

### Based on Marker Expression Patterns

| Cell Type | Markers | Logic |
|-----------|---------|-------|
| **T Cells** | `CD45+, CD3E+` | Immune, T-cell specific |
| **Cytotoxic T Cells** | `CD8a+, CD3E+` | CD8+ T cells |
| **B Cells** | `CD45+, CD20+` | Immune, B-cell specific |
| **Macrophages** | `CD45+, CD68+` | Immune, myeloid |
| **M2 Macrophages** | `CD68+, CD163+` | Anti-inflammatory |
| **Epithelial Cells** | `PANCK+, EPCAM+, CD45-` | Epithelial markers |
| **Stromal Cells** | `VIM+, CD45-, PANCK-` | Fibroblasts/stroma |
| **Proliferating** | `Ki67+` | Any cell type, actively dividing |

### Using `cell_types.csv`

You already have `cell_types.csv` which defines these rules. PIPEX will apply them automatically!

---

## 📊 **Visualizing Results**

### In Python
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
cells = pd.read_csv('input/analysis/analysis_data.csv')
umap = pd.read_csv('input/analysis/umap_coordinates.csv')
clusters = pd.read_csv('input/analysis/clusters_leiden.csv')

# Merge
cells = cells.merge(umap, on='CellID').merge(clusters, on='CellID')

# Plot UMAP colored by cluster
plt.figure(figsize=(10, 8))
sns.scatterplot(data=cells, x='UMAP1', y='UMAP2', hue='Cluster', 
                palette='tab20', s=10, alpha=0.6)
plt.title('Cell Clusters (UMAP)')
plt.savefig('results/umap_clusters.png', dpi=300)

# Plot marker expression
markers = ['CD45', 'CD3E', 'CD8a', 'CD20', 'PANCK']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, marker in enumerate(markers):
    ax = axes.flatten()[i]
    sc = ax.scatter(cells['X'], cells['Y'], c=cells[f'{marker}_norm'], 
                    s=1, cmap='viridis', alpha=0.8)
    ax.set_title(f'{marker} Expression')
    plt.colorbar(sc, ax=ax)
plt.tight_layout()
plt.savefig('results/spatial_markers.png', dpi=300)
```

### In QuPath
1. Open `DAPI_AF_R01.ome.tif` in QuPath
2. Run → Groovy Script:
```groovy
// Import cell segmentation from PIPEX
def cellsFile = '/home/steve/Projects/HeLab/BladderDIVE/input/analysis/analysis_data.csv'
def cells = new File(cellsFile).readLines().drop(1).collect { line ->
    def parts = line.split(',')
    // Create cell annotations with measurements
    // (Full script in qupath_import/ directory)
}
```

---

## 🔄 **When Would You Need H&E Registration?**

### You DON'T need it for:
- ✅ Cell type identification
- ✅ Marker quantification  
- ✅ Clustering analysis
- ✅ Cell-cell interactions
- ✅ Global spatial patterns

### You WOULD need it for:
- ❌ Comparing to pathologist H&E annotations
- ❌ Correlating with H&E morphology
- ❌ "Inside vs outside tumor" analysis (if tumor defined on H&E)
- ❌ Validating cell types against H&E features

### If you need registration later:
The registration scripts (`corrected_registration.py`, `manual_registration_napari.py`) are ready to use. You can run them AFTER completing cell typing to add spatial context from H&E.

---

## 🎯 **Next Steps**

1. **Run the pipeline:**
   ```bash
   ./run_cell_analysis.sh
   ```

2. **Check outputs:**
   ```bash
   ls input/analysis/
   ```

3. **Analyze results:**
   - Load `analysis_data.csv` in Python/R
   - Visualize marker expression
   - Identify cell type proportions
   - Examine spatial distributions

4. **(Optional) Add H&E registration:**
   - Only if you need to correlate with H&E annotations
   - Run `python corrected_registration.py`
   - Then `python overlay_annotations.py`

---

## 📚 **Resources**

- **PIPEX Documentation**: `PIPEX_CellDIVE_Guide.md`
- **Cell Type Definitions**: `cell_types.csv`
- **QuPath Import**: `qupath_import_guide.md`
- **Pipeline Details**: `README.md`

---

## ⚠️ **Memory Considerations**

The analysis only measures **10 key markers** instead of all 23 to avoid out-of-memory errors on your mini-PC:
```
DAPI, CD45, CD3E, CD8a, CD20, CD68, PANCK, EPCAM, VIM, Ki67
```

This is sufficient for identifying main cell types. To measure all 23 markers, run on your server with more RAM.

---

## ✅ **Expected Runtime**
- Segmentation: ~30-60 minutes
- Analysis: ~30-45 minutes
- **Total: ~1-2 hours**

---

**🎉 You're ready to start! Run `./run_cell_analysis.sh` to begin!**


