#!/usr/bin/env python3
"""
Generate visualizations for CellDIVE analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load analysis results"""
    analysis = pd.read_csv(snakemake.input.analysis)
    clusters = pd.read_csv(snakemake.input.clusters)
    umap_coords = pd.read_csv(snakemake.input.umap_coords)
    interactions = pd.read_csv(snakemake.input.interactions)
    
    return analysis, clusters, umap_coords, interactions

def plot_umap(analysis, clusters, umap_coords, output_path):
    """Generate UMAP plot colored by cell types"""
    
    # Merge data
    plot_data = analysis.merge(clusters, left_index=True, right_index=True)
    plot_data = plot_data.merge(umap_coords, left_index=True, right_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: UMAP colored by refined clusters
    scatter = axes[0,0].scatter(plot_data['UMAP_1'], plot_data['UMAP_2'], 
                               c=plot_data['leiden_refined'], 
                               s=1, alpha=0.7, cmap='tab20')
    axes[0,0].set_title('UMAP - Cell Types')
    axes[0,0].set_xlabel('UMAP 1')
    axes[0,0].set_ylabel('UMAP 2')
    
    # Plot 2: UMAP colored by CD45 (immune marker)
    if 'CD45' in plot_data.columns:
        scatter2 = axes[0,1].scatter(plot_data['UMAP_1'], plot_data['UMAP_2'], 
                                    c=plot_data['CD45'], 
                                    s=1, alpha=0.7, cmap='viridis')
        axes[0,1].set_title('UMAP - CD45 Expression')
        plt.colorbar(scatter2, ax=axes[0,1])
    
    # Plot 3: UMAP colored by PANCK (epithelial marker)
    if 'PANCK' in plot_data.columns:
        scatter3 = axes[1,0].scatter(plot_data['UMAP_1'], plot_data['UMAP_2'], 
                                    c=plot_data['PANCK'], 
                                    s=1, alpha=0.7, cmap='plasma')
        axes[1,0].set_title('UMAP - PanCK Expression')
        plt.colorbar(scatter3, ax=axes[1,0])
    
    # Plot 4: UMAP colored by Ki67 (proliferation)
    if 'Ki67' in plot_data.columns:
        scatter4 = axes[1,1].scatter(plot_data['UMAP_1'], plot_data['UMAP_2'], 
                                    c=plot_data['Ki67'], 
                                    s=1, alpha=0.7, cmap='hot')
        axes[1,1].set_title('UMAP - Ki67 Expression')
        plt.colorbar(scatter4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_markers_heatmap(analysis, clusters, output_path):
    """Generate marker expression heatmap by cell type"""
    
    # Merge data
    plot_data = analysis.merge(clusters, left_index=True, right_index=True)
    
    # Get marker columns (exclude metadata)
    marker_cols = [col for col in analysis.columns if col not in 
                   ['cell_id', 'x_centroid', 'y_centroid', 'area', 'perimeter']]
    
    # Calculate mean expression by cell type
    heatmap_data = plot_data.groupby('leiden_refined')[marker_cols].mean()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, cbar_kws={'label': 'Mean Expression'})
    plt.title('Marker Expression by Cell Type')
    plt.xlabel('Cell Type')
    plt.ylabel('Markers')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_interaction_network(interactions, output_path):
    """Generate cell-cell interaction network plot"""
    
    # Create interaction matrix
    cell_types = list(set(interactions['cell_type_1'].tolist() + 
                         interactions['cell_type_2'].tolist()))
    
    interaction_matrix = pd.DataFrame(0, index=cell_types, columns=cell_types)
    
    for _, row in interactions.iterrows():
        interaction_matrix.loc[row['cell_type_1'], row['cell_type_2']] = row['interaction_score']
        interaction_matrix.loc[row['cell_type_2'], row['cell_type_1']] = row['interaction_score']
    
    # Plot network
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, annot=True, fmt='.2f', cmap='Reds',
                cbar_kws={'label': 'Interaction Score'})
    plt.title('Cell-Cell Interaction Network')
    plt.xlabel('Cell Type')
    plt.ylabel('Cell Type')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_distribution(analysis, clusters, output_path):
    """Generate spatial distribution plot"""
    
    # Merge data
    plot_data = analysis.merge(clusters, left_index=True, right_index=True)
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot of cell positions colored by type
    unique_types = plot_data['leiden_refined'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    
    for i, cell_type in enumerate(unique_types):
        mask = plot_data['leiden_refined'] == cell_type
        plt.scatter(plot_data.loc[mask, 'x_centroid'], 
                   plot_data.loc[mask, 'y_centroid'],
                   c=[colors[i]], label=cell_type, s=0.5, alpha=0.7)
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Spatial Distribution of Cell Types')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    
    # Load data
    analysis, clusters, umap_coords, interactions = load_data()
    
    # Generate plots
    plot_umap(analysis, clusters, umap_coords, snakemake.output.umap_plot)
    plot_markers_heatmap(analysis, clusters, snakemake.output.markers_heatmap)
    plot_interaction_network(interactions, snakemake.output.interaction_network)
    plot_spatial_distribution(analysis, clusters, snakemake.output.spatial_plot)
    
    print("Visualizations generated successfully!")

if __name__ == "__main__":
    main()
