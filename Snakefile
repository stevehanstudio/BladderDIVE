# Snakemake workflow for CellDIVE PIPEX analysis
# Author: AI Assistant
# Description: Complete pipeline for 23-channel CellDIVE bladder proteomics analysis

import os
from pathlib import Path

# Configuration
configfile: "config.yaml"

# Define paths
# Use current working directory (where snakemake is run from)
WORK_DIR = str(Path.cwd())
PIPEX_DIR = "/mnt/data/Projects/HeLab/pipex"
INPUT_DIR = f"{WORK_DIR}/input"
OUTPUT_DIR = f"{WORK_DIR}/output"
RESULTS_DIR = f"{WORK_DIR}/results"

# Channel list for your CellDIVE data
CHANNELS = [
    "DAPI", "CD45", "CD3E", "Ki67", "CD8a", "VIM", "CD68", "HLADR",
    "CD31", "ACTA2", "CD20", "CD163", "CD44", "PANCK", "CD38", "CD11c",
    "PDGFRA", "COL1A1", "CD14", "EPCAM", "CD56", "CD45RO", "DAPI2"
]

# Print paths for debugging
print(f"WORK_DIR: {WORK_DIR}")
print(f"INPUT_DIR: {INPUT_DIR}")

# Create required directories if they don't exist
Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{INPUT_DIR}/analysis").mkdir(parents=True, exist_ok=True)

# Check if input files exist
missing_files = []
for channel in CHANNELS:
    tif_file = Path(f"{INPUT_DIR}/{channel}.tif")
    if not tif_file.exists():
        missing_files.append(str(tif_file))

if missing_files:
    print(f"\n⚠️  WARNING: Missing {len(missing_files)} input TIF files:")
    for f in missing_files[:5]:  # Show first 5
        print(f"   - {f}")
    if len(missing_files) > 5:
        print(f"   ... and {len(missing_files) - 5} more")
    print(f"\nPlease ensure all TIF files are in: {INPUT_DIR}")
    print(f"Expected files: {', '.join(CHANNELS)}.tif\n")

# Target rule - what we want to achieve
rule all:
    input:
        # Segmentation outputs
        f"{INPUT_DIR}/analysis/segmentation_data.npy",
        f"{INPUT_DIR}/analysis/cell_data.csv",
        
        # Analysis outputs
        f"{INPUT_DIR}/analysis/analysis_data.csv",
        f"{INPUT_DIR}/analysis/clusters_leiden.csv",
        f"{INPUT_DIR}/analysis/interactions.csv",
        
        # Visualization outputs
        f"{RESULTS_DIR}/umap_plot.png",
        f"{RESULTS_DIR}/markers_heatmap.png",
        
        # QuPath integration
        f"{INPUT_DIR}/analysis/qupath_annotations.geojson",
        
        # Final report
        f"{RESULTS_DIR}/analysis_report.html"

# Rule 0: Prepare input TIF files (copy/link from source directory if specified)
source_data_dir = config.get("source_data_dir", "")
if source_data_dir and Path(source_data_dir).exists():
    rule prepare_inputs:
        input:
            source_files = expand(f"{source_data_dir}/{{channel}}.tif", channel=CHANNELS)
        output:
            input_files = expand(f"{INPUT_DIR}/{{channel}}.tif", channel=CHANNELS)
        params:
            channels = " ".join(CHANNELS),
            source = source_data_dir,
            dest = INPUT_DIR
        shell:
            """
            # Create symlinks to avoid copying large files
            for channel in {params.channels}; do
                if [ ! -f {params.dest}/${{channel}}.tif ]; then
                    ln -s {params.source}/${{channel}}.tif {params.dest}/${{channel}}.tif || \
                    cp {params.source}/${{channel}}.tif {params.dest}/${{channel}}.tif
                fi
            done
            """
else:
    # If no source directory, create a dummy rule that does nothing
    # (files should already exist in input/)
    rule prepare_inputs:
        output:
            input_files = expand(f"{INPUT_DIR}/{{channel}}.tif", channel=CHANNELS)
        shell:
            "echo 'No source_data_dir configured. Ensure TIF files are in {INPUT_DIR}/'"

# Rule 1: Cell segmentation using StarDist + membrane refinement
# Note: Measuring all 23 channels (23 x 9GB TIFs) causes OOM, so we measure only key markers
rule segmentation:
    input:
        channels = expand(f"{INPUT_DIR}/{{channel}}.tif", channel=CHANNELS)
    output:
        mask = f"{INPUT_DIR}/analysis/segmentation_data.npy",
        measurements = f"{INPUT_DIR}/analysis/cell_data.csv",
        qc_dir = directory(f"{INPUT_DIR}/analysis/quality_control")
    params:
        data_dir = INPUT_DIR,
        nuclei_diameter = config.get("nuclei_diameter", 20),
        nuclei_expansion = config.get("nuclei_expansion", 10),
        membrane_diameter = config.get("membrane_diameter", 30),
        membrane_compactness = config.get("membrane_compactness", 0.5),
        # Measure only essential markers to avoid OOM (reduced from 23 to 10 markers)
        markers = "DAPI,CD45,CD3E,CD8a,CD20,CD68,PANCK,EPCAM,VIM,Ki67"
    conda:
        "envs/segmentation.yaml"
    shell:
        """
        cd {PIPEX_DIR}
        python segmentation.py \
            -data={params.data_dir} \
            -nuclei_marker=DAPI \
            -nuclei_diameter={params.nuclei_diameter} \
            -nuclei_expansion={params.nuclei_expansion} \
            -membrane_marker=PANCK \
            -membrane_diameter={params.membrane_diameter} \
            -membrane_compactness={params.membrane_compactness} \
            -measure_markers={params.markers}
        """

# Rule 2: Single-cell analysis and clustering
rule analysis:
    input:
        mask = f"{INPUT_DIR}/analysis/segmentation_data.npy",
        measurements = f"{INPUT_DIR}/analysis/cell_data.csv",
        channels = expand(f"{INPUT_DIR}/{{channel}}.tif", channel=CHANNELS)
    output:
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv",
        leiden = f"{INPUT_DIR}/analysis/clusters_leiden.csv",
        kmeans = f"{INPUT_DIR}/analysis/clusters_kmeans.csv",
        interactions = f"{INPUT_DIR}/analysis/interactions.csv",
        umap_coords = f"{INPUT_DIR}/analysis/umap_coordinates.csv"
    params:
        data_dir = INPUT_DIR,
        log_norm = config.get("log_norm", "yes"),
        z_norm = config.get("z_norm", "yes"),
        custom_filter = config.get("custom_filter", "no")
    conda:
        "envs/analysis.yaml"
    shell:
        """
        cd {PIPEX_DIR}
        python analysis.py \
            -data={params.data_dir} \
            -log_norm={params.log_norm} \
            -z_norm={params.z_norm} \
            -umap=yes \
            -leiden=yes \
            -kmeans=yes \
            -interactions=yes \
            -custom_filter={params.custom_filter}
        """

# Rule 3: Cell type refinement using predefined markers
rule cell_typing:
    input:
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv",
        leiden = f"{INPUT_DIR}/analysis/clusters_leiden.csv",
        cell_types_config = f"{WORK_DIR}/cell_types.csv"
    output:
        refined_clusters = f"{INPUT_DIR}/analysis/clusters_refined.csv",
        cell_type_report = f"{RESULTS_DIR}/cell_typing_report.json"
    params:
        data_dir = INPUT_DIR
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/refine_cell_types.py"

# Rule 4: Generate visualizations
rule visualizations:
    input:
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv",
        clusters = f"{INPUT_DIR}/analysis/clusters_refined.csv",
        umap_coords = f"{INPUT_DIR}/analysis/umap_coordinates.csv",
        interactions = f"{INPUT_DIR}/analysis/interactions.csv"
    output:
        umap_plot = f"{RESULTS_DIR}/umap_plot.png",
        markers_heatmap = f"{RESULTS_DIR}/markers_heatmap.png",
        interaction_network = f"{RESULTS_DIR}/interaction_network.png",
        spatial_plot = f"{RESULTS_DIR}/spatial_distribution.png"
    conda:
        "envs/visualization.yaml"
    script:
        "scripts/generate_visualizations.py"

# Rule 5: Generate QuPath-compatible outputs
rule qupath_export:
    input:
        mask = f"{INPUT_DIR}/analysis/segmentation_data.npy",
        clusters = f"{INPUT_DIR}/analysis/clusters_refined.csv",
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv"
    output:
        geojson = f"{INPUT_DIR}/analysis/qupath_annotations.geojson",
        measurements_qupath = f"{INPUT_DIR}/analysis/qupath_measurements.txt"
    params:
        data_dir = INPUT_DIR
    conda:
        "envs/qupath.yaml"
    shell:
        """
        cd {PIPEX_DIR}
        python generate_geojson.py -data={params.data_dir}
        """

# Rule 6: Spatial analysis
rule spatial_analysis:
    input:
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv",
        clusters = f"{INPUT_DIR}/analysis/clusters_refined.csv",
        mask = f"{INPUT_DIR}/analysis/segmentation_data.npy"
    output:
        spatial_stats = f"{RESULTS_DIR}/spatial_statistics.csv",
        neighborhood_analysis = f"{RESULTS_DIR}/neighborhood_analysis.csv",
        spatial_plots = directory(f"{RESULTS_DIR}/spatial_plots")
    conda:
        "envs/spatial.yaml"
    script:
        "scripts/spatial_analysis.py"

# Rule 7: Generate comprehensive report
rule generate_report:
    input:
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv",
        clusters = f"{INPUT_DIR}/analysis/clusters_refined.csv",
        interactions = f"{INPUT_DIR}/analysis/interactions.csv",
        spatial_stats = f"{RESULTS_DIR}/spatial_statistics.csv",
        umap_plot = f"{RESULTS_DIR}/umap_plot.png",
        markers_heatmap = f"{RESULTS_DIR}/markers_heatmap.png",
        interaction_network = f"{RESULTS_DIR}/interaction_network.png",
        spatial_plot = f"{RESULTS_DIR}/spatial_distribution.png"
    output:
        report = f"{RESULTS_DIR}/analysis_report.html",
        summary_stats = f"{RESULTS_DIR}/summary_statistics.json"
    params:
        project_name = config.get("project_name", "CellDIVE_Bladder_Analysis"),
        channels = CHANNELS
    conda:
        "envs/visualization.yaml"
    script:
        "scripts/generate_report.py"

# Rule 8: Quality control checks
rule quality_control:
    input:
        channels = expand(f"{INPUT_DIR}/{{channel}}.tif", channel=CHANNELS),
        mask = f"{INPUT_DIR}/analysis/segmentation_data.npy",
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv"
    output:
        qc_report = f"{RESULTS_DIR}/quality_control_report.html",
        qc_plots = directory(f"{RESULTS_DIR}/qc_plots")
    conda:
        "envs/visualization.yaml"
    script:
        "scripts/quality_control.py"

# Rule 9: Export for downstream analysis
rule export_data:
    input:
        analysis = f"{INPUT_DIR}/analysis/analysis_data.csv",
        clusters = f"{INPUT_DIR}/analysis/clusters_refined.csv",
        spatial_stats = f"{RESULTS_DIR}/spatial_statistics.csv",
        interactions = f"{INPUT_DIR}/analysis/interactions.csv"
    output:
        h5ad = f"{RESULTS_DIR}/celldive_data.h5ad",  # AnnData format for scanpy/squidpy
        fcs = f"{RESULTS_DIR}/celldive_data.fcs",    # Flow cytometry format
        csv_combined = f"{RESULTS_DIR}/combined_analysis.csv"
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/export_data.py"

# Utility rules
rule clean:
    shell:
        """
        rm -rf {INPUT_DIR}/analysis/quality_control
        rm -rf {RESULTS_DIR}/spatial_plots
        rm -rf {RESULTS_DIR}/qc_plots
        rm -f {RESULTS_DIR}/*.png
        rm -f {RESULTS_DIR}/*.jpg
        """

rule clean_all:
    shell:
        """
        rm -rf {OUTPUT_DIR}/*
        rm -rf {RESULTS_DIR}/*
        rm -rf {INPUT_DIR}/analysis/*
        """
