# Snakemake workflow for CellDIVE PIPEX analysis
# Author: AI Assistant
# Description: Complete pipeline for 23-channel CellDIVE bladder proteomics analysis

import os
from pathlib import Path

# Configuration
configfile: "config.yaml"

# Define paths
PIPEX_DIR = "/mnt/data/Projects/HeLab/pipex"
WORK_DIR = "/home/steve/Projects/HeLab/bladder"
INPUT_DIR = f"{WORK_DIR}/input"
OUTPUT_DIR = f"{WORK_DIR}/output"
RESULTS_DIR = f"{WORK_DIR}/results"

# Channel list for your CellDIVE data
CHANNELS = [
    "DAPI", "CD45", "CD3E", "Ki67", "CD8a", "VIM", "CD68", "HLADR",
    "CD31", "ACTA2", "CD20", "CD163", "CD44", "PANCK", "CD38", "CD11c",
    "PDGFRA", "COL1A1", "CD14", "EPCAM", "CD56", "CD45RO", "DAPI2"
]

# Target rule - what we want to achieve
rule all:
    input:
        # Segmentation outputs
        f"{INPUT_DIR}/segmentation_mask.npy",
        f"{INPUT_DIR}/nuclei_mask.npy",
        
        # Analysis outputs
        f"{INPUT_DIR}/analysis.csv",
        f"{INPUT_DIR}/clusters_leiden.csv",
        f"{INPUT_DIR}/interactions.csv",
        
        # Visualization outputs
        f"{INPUT_DIR}/umap_plot.png",
        f"{INPUT_DIR}/markers_heatmap.png",
        
        # QuPath integration
        f"{INPUT_DIR}/qupath_annotations.geojson",
        
        # Final report
        f"{RESULTS_DIR}/analysis_report.html"

# Rule 1: Cell segmentation using StarDist + membrane refinement
rule segmentation:
    input:
        channels = expand(f"{INPUT_DIR}/{{channel}}.tif", channel=CHANNELS),
        dapi = f"{INPUT_DIR}/DAPI.tif",
        membrane = f"{INPUT_DIR}/PANCK.tif"
    output:
        mask = f"{INPUT_DIR}/segmentation_mask.npy",
        nuclei = f"{INPUT_DIR}/nuclei_mask.npy",
        measurements = f"{INPUT_DIR}/measurements.csv",
        qc_images = directory(f"{INPUT_DIR}/segmentation_qc")
    params:
        data_dir = INPUT_DIR,
        nuclei_diameter = config.get("nuclei_diameter", 20),
        nuclei_expansion = config.get("nuclei_expansion", 10),
        membrane_diameter = config.get("membrane_diameter", 30),
        membrane_compactness = config.get("membrane_compactness", 0.5),
        markers = ",".join(CHANNELS)
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
        mask = f"{INPUT_DIR}/segmentation_mask.npy",
        measurements = f"{INPUT_DIR}/measurements.csv",
        channels = expand(f"{INPUT_DIR}/{{channel}}.tif", channel=CHANNELS)
    output:
        analysis = f"{INPUT_DIR}/analysis.csv",
        leiden = f"{INPUT_DIR}/clusters_leiden.csv",
        kmeans = f"{INPUT_DIR}/clusters_kmeans.csv",
        interactions = f"{INPUT_DIR}/interactions.csv",
        umap_coords = f"{INPUT_DIR}/umap_coordinates.csv"
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
        analysis = f"{INPUT_DIR}/analysis.csv",
        leiden = f"{INPUT_DIR}/clusters_leiden.csv",
        cell_types_config = f"{WORK_DIR}/cell_types.csv"
    output:
        refined_clusters = f"{INPUT_DIR}/clusters_refined.csv",
        cell_type_report = f"{INPUT_DIR}/cell_typing_report.json"
    params:
        data_dir = INPUT_DIR
    conda:
        "envs/analysis.yaml"
    script:
        "scripts/refine_cell_types.py"

# Rule 4: Generate visualizations
rule visualizations:
    input:
        analysis = f"{INPUT_DIR}/analysis.csv",
        clusters = f"{INPUT_DIR}/clusters_refined.csv",
        umap_coords = f"{INPUT_DIR}/umap_coordinates.csv",
        interactions = f"{INPUT_DIR}/interactions.csv"
    output:
        umap_plot = f"{INPUT_DIR}/umap_plot.png",
        markers_heatmap = f"{INPUT_DIR}/markers_heatmap.png",
        interaction_network = f"{INPUT_DIR}/interaction_network.png",
        spatial_plot = f"{INPUT_DIR}/spatial_distribution.png"
    conda:
        "envs/visualization.yaml"
    script:
        "scripts/generate_visualizations.py"

# Rule 5: Generate QuPath-compatible outputs
rule qupath_export:
    input:
        mask = f"{INPUT_DIR}/segmentation_mask.npy",
        clusters = f"{INPUT_DIR}/clusters_refined.csv",
        analysis = f"{INPUT_DIR}/analysis.csv"
    output:
        geojson = f"{INPUT_DIR}/qupath_annotations.geojson",
        measurements_qupath = f"{INPUT_DIR}/qupath_measurements.txt"
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
        analysis = f"{INPUT_DIR}/analysis.csv",
        clusters = f"{INPUT_DIR}/clusters_refined.csv",
        mask = f"{INPUT_DIR}/segmentation_mask.npy"
    output:
        spatial_stats = f"{INPUT_DIR}/spatial_statistics.csv",
        neighborhood_analysis = f"{INPUT_DIR}/neighborhood_analysis.csv",
        spatial_plots = directory(f"{INPUT_DIR}/spatial_plots")
    conda:
        "envs/spatial.yaml"
    script:
        "scripts/spatial_analysis.py"

# Rule 7: Generate comprehensive report
rule generate_report:
    input:
        analysis = f"{INPUT_DIR}/analysis.csv",
        clusters = f"{INPUT_DIR}/clusters_refined.csv",
        interactions = f"{INPUT_DIR}/interactions.csv",
        spatial_stats = f"{INPUT_DIR}/spatial_statistics.csv",
        umap_plot = f"{INPUT_DIR}/umap_plot.png",
        markers_heatmap = f"{INPUT_DIR}/markers_heatmap.png",
        interaction_network = f"{INPUT_DIR}/interaction_network.png",
        spatial_plot = f"{INPUT_DIR}/spatial_distribution.png"
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
        mask = f"{INPUT_DIR}/segmentation_mask.npy",
        analysis = f"{INPUT_DIR}/analysis.csv"
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
        analysis = f"{INPUT_DIR}/analysis.csv",
        clusters = f"{INPUT_DIR}/clusters_refined.csv",
        spatial_stats = f"{INPUT_DIR}/spatial_statistics.csv",
        interactions = f"{INPUT_DIR}/interactions.csv"
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
        rm -rf {INPUT_DIR}/segmentation_qc
        rm -rf {INPUT_DIR}/spatial_plots
        rm -rf {RESULTS_DIR}/qc_plots
        rm -f {INPUT_DIR}/*.png
        rm -f {INPUT_DIR}/*.jpg
        """

rule clean_all:
    shell:
        """
        rm -rf {OUTPUT_DIR}/*
        rm -rf {RESULTS_DIR}/*
        rm -f {INPUT_DIR}/*.npy
        rm -f {INPUT_DIR}/*.csv
        rm -f {INPUT_DIR}/*.json
        rm -f {INPUT_DIR}/*.geojson
        """
