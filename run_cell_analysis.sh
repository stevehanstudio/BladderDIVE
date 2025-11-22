#!/bin/bash
# Simple script to run PIPEX cell segmentation and analysis
# NO H&E or registration required - just CellDIVE channel analysis

set -e  # Exit on error

echo "=========================================="
echo " BladderDIVE Cell Type Identification"
echo "=========================================="
echo

# Paths
PIPEX_DIR="/mnt/data/Projects/HeLab/pipex"
WORK_DIR="/home/steve/Projects/HeLab/BladderDIVE"
INPUT_DIR="$WORK_DIR/input"
OUTPUT_DIR="$WORK_DIR/output"

cd "$PIPEX_DIR"

echo "[1/3] Cell Segmentation (StarDist + PanCK refinement)"
echo "   Using DAPI for nuclei detection..."
echo "   Measuring 10 key markers to avoid OOM..."
echo

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "pipex_seg"; then
    echo "Creating segmentation environment..."
    conda env create -f "$WORK_DIR/envs/segmentation.yaml" -n pipex_seg
fi

# Run segmentation
conda run -n pipex_seg python segmentation.py \
    -data="$INPUT_DIR" \
    -nuclei_marker=DAPI \
    -nuclei_diameter=20 \
    -nuclei_expansion=10 \
    -membrane_marker=PANCK \
    -membrane_diameter=30 \
    -membrane_compactness=0.5 \
    -measure_markers="DAPI,CD45,CD3E,CD8a,CD20,CD68,PANCK,EPCAM,VIM,Ki67"

echo
echo "✓ Segmentation complete!"
echo "   Output: $INPUT_DIR/analysis/segmentation_data.npy"
echo "   Output: $INPUT_DIR/analysis/cell_data.csv"
echo

echo "[2/3] Single-cell Analysis & Clustering"
echo "   Running UMAP, Leiden clustering, interaction analysis..."
echo

# Create analysis environment if needed
if ! conda env list | grep -q "pipex_analysis"; then
    echo "Creating analysis environment..."
    conda env create -f "$WORK_DIR/envs/analysis.yaml" -n pipex_analysis
fi

# Run analysis
conda run -n pipex_analysis python analysis.py \
    -data="$INPUT_DIR" \
    -log_norm=yes \
    -z_norm=yes \
    -dim_red=UMAP \
    -clustering=leiden \
    -interactions=yes \
    -spatial_analysis=yes

echo
echo "✓ Analysis complete!"
echo "   Output: $INPUT_DIR/analysis/analysis_data.csv"
echo "   Output: $INPUT_DIR/analysis/clusters_leiden.csv"
echo "   Output: $INPUT_DIR/analysis/umap_coordinates.csv"
echo

echo "[3/3] Cell Type Annotation"
echo "   Classifying cells based on marker expression..."
echo

# Apply cell type definitions
if [ -f "$WORK_DIR/cell_types.csv" ]; then
    echo "   Using cell type definitions from: cell_types.csv"
    # TODO: Add cell typing script call here
else
    echo "   ⚠ No cell_types.csv found - using unsupervised clusters only"
fi

echo
echo "=========================================="
echo " ✓ Cell Type Identification Complete!"
echo "=========================================="
echo
echo "Key outputs:"
echo "  • Cell segmentation: $INPUT_DIR/analysis/segmentation_data.npy"
echo "  • Cell measurements: $INPUT_DIR/analysis/cell_data.csv"
echo "  • Analysis matrix:   $INPUT_DIR/analysis/analysis_data.csv"
echo "  • Cell clusters:     $INPUT_DIR/analysis/clusters_leiden.csv"
echo "  • UMAP coords:       $INPUT_DIR/analysis/umap_coordinates.csv"
echo "  • Interactions:      $INPUT_DIR/analysis/interactions.csv"
echo
echo "Next steps:"
echo "  1. View results in QuPath: import analysis_data.csv"
echo "  2. Visualize in Python: load analysis_data.csv + umap_coordinates.csv"
echo "  3. (Optional) Add H&E registration later if needed for spatial context"
echo


