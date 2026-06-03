#!/usr/bin/env python3
"""
Exploratory Data Analysis for Cellpose Channel Selection

Analyzes all channels in the zarr file to determine which channels
work best for Cellpose 4 segmentation (nuclei + cytoplasm).

This script:
1. Loads all channels from the zarr file
2. Calculates metrics for each channel (intensity, contrast, SNR, etc.)
3. Tests different channel combinations
4. Provides recommendations for nuclei and cytoplasm channels
"""

import numpy as np
import zarr
import dask.array as da
import json
import os
from pathlib import Path
import pandas as pd
from scipy import ndimage
from scipy.stats import entropy
from skimage import filters, feature, measure
from skimage.morphology import disk
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_channel_metadata(zarr_path):
    """Load channel names and metadata from zarr file."""
    zattrs_path = os.path.join(zarr_path, ".zattrs")
    names = []
    
    if os.path.exists(zattrs_path):
        with open(zattrs_path) as f:
            meta = json.load(f)
            channels_meta = meta.get("omero", {}).get("channels", [])
            if not channels_meta and "multiscales" in meta:
                channels_meta = meta["multiscales"][0].get("omero", {}).get("channels", [])
        
        for i, ch in enumerate(channels_meta):
            names.append(ch.get("label", f"Ch{i}"))
    else:
        # Fallback: try to get number of channels from zarr
        group = zarr.open(zarr_path, mode='r')
        n_channels = group['0'].shape[0]
        names = [f"Ch{i}" for i in range(n_channels)]
    
    return names


def sample_image_region(zarr_path, channel_idx, sample_size=2048, n_samples=4):
    """
    Sample multiple regions from a channel to analyze.
    
    Parameters:
    -----------
    zarr_path : str
        Path to zarr file
    channel_idx : int
        Channel index to sample
    sample_size : int
        Size of each sample region (square)
    n_samples : int
        Number of sample regions to extract
    
    Returns:
    --------
    list of np.ndarray: Sample regions
    """
    group = zarr.open(zarr_path, mode='r')
    # Use level 0 (full resolution)
    zarr_array = group['0']
    
    # Get image dimensions
    n_channels, height, width = zarr_array.shape
    
    # Calculate sample regions (avoid edges)
    margin = sample_size // 2
    samples = []
    
    for _ in range(n_samples):
        # Random position within valid range
        y = np.random.randint(margin, height - margin)
        x = np.random.randint(margin, width - margin)
        
        y_start = max(0, y - sample_size // 2)
        y_end = min(height, y + sample_size // 2)
        x_start = max(0, x - sample_size // 2)
        x_end = min(width, x + sample_size // 2)
        
        # Load sample region
        sample = zarr_array[channel_idx, y_start:y_end, x_start:x_end]
        # Convert to numpy if dask
        if isinstance(sample, da.Array):
            sample = sample.compute()
        samples.append(sample)
    
    return samples


def calculate_channel_metrics(samples):
    """
    Calculate metrics for a channel based on sample regions.
    
    Parameters:
    -----------
    samples : list of np.ndarray
        Sample image regions
    
    Returns:
    --------
    dict: Metrics dictionary
    """
    # Combine all samples for overall statistics
    combined = np.concatenate([s.flatten() for s in samples])
    
    # Basic intensity statistics
    mean_intensity = np.mean(combined)
    median_intensity = np.median(combined)
    std_intensity = np.std(combined)
    cv = std_intensity / (mean_intensity + 1e-10)  # Coefficient of variation
    
    # Dynamic range
    min_val = np.min(combined)
    max_val = np.max(combined)
    dynamic_range = max_val - min_val
    percentile_range = np.percentile(combined, 99) - np.percentile(combined, 1)
    
    # Signal-to-noise ratio (using median as signal, std as noise)
    snr = median_intensity / (std_intensity + 1e-10)
    
    # Contrast metrics (per sample, then average)
    contrast_scores = []
    edge_scores = []
    texture_scores = []
    
    for sample in samples:
        # Normalize sample for analysis
        sample_norm = (sample - sample.min()) / (sample.max() - sample.min() + 1e-10)
        
        # Contrast: standard deviation of normalized image
        contrast = np.std(sample_norm)
        contrast_scores.append(contrast)
        
        # Edge content: Sobel edge detection
        edges = filters.sobel(sample_norm)
        edge_score = np.mean(edges)
        edge_scores.append(edge_score)
        
        # Texture: Local entropy (measure of local variation)
        # Use a small window to compute local entropy
        try:
            from skimage.filters.rank import entropy as rank_entropy
            kernel_size = 5
            local_entropy = rank_entropy(
                (sample_norm * 255).astype(np.uint8),
                disk(kernel_size)
            )
            texture_score = np.mean(local_entropy) / 8.0  # Normalize by max entropy
        except (ImportError, AttributeError):
            # Fallback: use variance as texture measure
            from scipy.ndimage import uniform_filter
            kernel_size = 5
            local_var = uniform_filter(sample_norm**2, size=kernel_size) - \
                       uniform_filter(sample_norm, size=kernel_size)**2
            texture_score = np.mean(local_var)
        texture_scores.append(texture_score)
    
    avg_contrast = np.mean(contrast_scores)
    avg_edge = np.mean(edge_scores)
    avg_texture = np.mean(texture_scores)
    
    # Nuclei-like characteristics (bright, compact objects)
    # Look for bright spots that could be nuclei
    bright_threshold = np.percentile(combined, 95)
    bright_pixels = np.sum([np.sum(s > bright_threshold) for s in samples])
    total_pixels = np.sum([s.size for s in samples])
    bright_fraction = bright_pixels / total_pixels
    
    # Cytoplasm-like characteristics (moderate intensity, more diffuse)
    # Look for moderate intensity regions
    moderate_low = np.percentile(combined, 30)
    moderate_high = np.percentile(combined, 70)
    moderate_pixels = np.sum([
        np.sum((s > moderate_low) & (s < moderate_high)) for s in samples
    ])
    moderate_fraction = moderate_pixels / total_pixels
    
    return {
        'mean_intensity': float(mean_intensity),
        'median_intensity': float(median_intensity),
        'std_intensity': float(std_intensity),
        'cv': float(cv),
        'dynamic_range': float(dynamic_range),
        'percentile_range': float(percentile_range),
        'snr': float(snr),
        'contrast': float(avg_contrast),
        'edge_content': float(avg_edge),
        'texture': float(avg_texture),
        'bright_fraction': float(bright_fraction),
        'moderate_fraction': float(moderate_fraction),
        'min_val': float(min_val),
        'max_val': float(max_val),
    }


def calculate_nuclei_score(metrics):
    """
    Score how well a channel might work for nuclei detection.
    
    Higher scores indicate better nuclei channels.
    """
    # Nuclei should have:
    # - High contrast (distinct objects)
    # - Moderate to high intensity
    # - Good edge content (clear boundaries)
    # - Some bright spots (nuclei are typically bright)
    
    score = (
        metrics['contrast'] * 0.3 +
        (metrics['mean_intensity'] / 1000.0) * 0.2 +  # Normalize intensity
        metrics['edge_content'] * 0.3 +
        metrics['bright_fraction'] * 0.2
    )
    
    # Penalize very low or very high intensity (too dim or saturated)
    if metrics['mean_intensity'] < 100:
        score *= 0.5
    if metrics['mean_intensity'] > 50000:
        score *= 0.7
    
    return score


def calculate_cytoplasm_score(metrics):
    """
    Score how well a channel might work for cytoplasm detection.
    
    Higher scores indicate better cytoplasm channels.
    """
    # Cytoplasm should have:
    # - Moderate intensity (not too bright, not too dim)
    # - Good texture (cytoplasm has structure)
    # - Moderate contrast (more diffuse than nuclei)
    # - Moderate fraction (should cover substantial area)
    
    # Ideal intensity range (normalized)
    intensity_norm = metrics['mean_intensity'] / 1000.0
    intensity_score = 1.0 - abs(intensity_norm - 5.0) / 5.0  # Peak around 5000
    intensity_score = max(0, intensity_score)
    
    score = (
        metrics['texture'] * 0.3 +
        intensity_score * 0.3 +
        metrics['moderate_fraction'] * 0.2 +
        metrics['contrast'] * 0.2
    )
    
    # Penalize very low intensity
    if metrics['mean_intensity'] < 200:
        score *= 0.5
    
    return score


def analyze_channel_combinations(channel_metrics, channel_names):
    """
    Analyze pairwise combinations of channels for segmentation.
    
    Parameters:
    -----------
    channel_metrics : dict
        Dictionary mapping channel index to metrics
    channel_names : list
        List of channel names
    
    Returns:
    --------
    pd.DataFrame: DataFrame with combination scores
    """
    combinations = []
    
    n_channels = len(channel_names)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            nuc_metrics = channel_metrics[i]
            cyto_metrics = channel_metrics[j]
            
            # Score this combination
            nuc_score = calculate_nuclei_score(nuc_metrics)
            cyto_score = calculate_cytoplasm_score(cyto_metrics)
            combined_score = nuc_score * 0.6 + cyto_score * 0.4
            
            combinations.append({
                'nuclei_channel': channel_names[i],
                'nuclei_idx': i,
                'cytoplasm_channel': channel_names[j],
                'cytoplasm_idx': j,
                'nuclei_score': nuc_score,
                'cytoplasm_score': cyto_score,
                'combined_score': combined_score,
                'nuclei_mean_intensity': nuc_metrics['mean_intensity'],
                'cytoplasm_mean_intensity': cyto_metrics['mean_intensity'],
                'nuclei_contrast': nuc_metrics['contrast'],
                'cytoplasm_contrast': cyto_metrics['contrast'],
            })
            
            # Also try reverse (j as nuclei, i as cytoplasm)
            nuc_score_rev = calculate_nuclei_score(cyto_metrics)
            cyto_score_rev = calculate_cytoplasm_score(nuc_metrics)
            combined_score_rev = nuc_score_rev * 0.6 + cyto_score_rev * 0.4
            
            combinations.append({
                'nuclei_channel': channel_names[j],
                'nuclei_idx': j,
                'cytoplasm_channel': channel_names[i],
                'cytoplasm_idx': i,
                'nuclei_score': nuc_score_rev,
                'cytoplasm_score': cyto_score_rev,
                'combined_score': combined_score_rev,
                'nuclei_mean_intensity': nuc_metrics['mean_intensity'],
                'cytoplasm_mean_intensity': cyto_metrics['mean_intensity'],
                'nuclei_contrast': nuc_metrics['contrast'],
                'cytoplasm_contrast': cyto_metrics['contrast'],
            })
    
    df = pd.DataFrame(combinations)
    df = df.sort_values('combined_score', ascending=False)
    
    return df


def create_visualizations(channel_metrics, channel_names, combinations_df, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Channel metrics summary
    metrics_df = pd.DataFrame([
        {**metrics, 'channel': name, 'channel_idx': idx}
        for idx, (name, metrics) in enumerate(zip(channel_names, channel_metrics))
    ])
    
    # Add scores
    metrics_df['nuclei_score'] = metrics_df.apply(
        lambda row: calculate_nuclei_score(row.to_dict()), axis=1
    )
    metrics_df['cytoplasm_score'] = metrics_df.apply(
        lambda row: calculate_cytoplasm_score(row.to_dict()), axis=1
    )
    
    # Sort by nuclei score
    metrics_df = metrics_df.sort_values('nuclei_score', ascending=False)
    
    # Plot 1: Channel scores comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Nuclei scores
    ax = axes[0, 0]
    top_n = min(15, len(metrics_df))
    top_df = metrics_df.head(top_n)
    ax.barh(range(len(top_df)), top_df['nuclei_score'], color='steelblue')
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['channel'], fontsize=8)
    ax.set_xlabel('Nuclei Score', fontsize=10)
    ax.set_title(f'Top {top_n} Channels for Nuclei Detection', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Cytoplasm scores
    ax = axes[0, 1]
    metrics_df_cyto = metrics_df.sort_values('cytoplasm_score', ascending=False)
    top_df_cyto = metrics_df_cyto.head(top_n)
    ax.barh(range(len(top_df_cyto)), top_df_cyto['cytoplasm_score'], color='coral')
    ax.set_yticks(range(len(top_df_cyto)))
    ax.set_yticklabels(top_df_cyto['channel'], fontsize=8)
    ax.set_xlabel('Cytoplasm Score', fontsize=10)
    ax.set_title(f'Top {top_n} Channels for Cytoplasm Detection', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Intensity vs Contrast scatter
    ax = axes[1, 0]
    scatter = ax.scatter(
        metrics_df['mean_intensity'] / 1000.0,
        metrics_df['contrast'],
        c=metrics_df['nuclei_score'],
        s=100,
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidths=0.5
    )
    ax.set_xlabel('Mean Intensity (×1000)', fontsize=10)
    ax.set_ylabel('Contrast (std dev)', fontsize=10)
    ax.set_title('Channel Characteristics (colored by Nuclei Score)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Nuclei Score')
    ax.grid(alpha=0.3)
    
    # Top combinations
    ax = axes[1, 1]
    top_combos = combinations_df.head(10)
    combo_labels = [
        f"{row['nuclei_channel']}\n+ {row['cytoplasm_channel']}"
        for _, row in top_combos.iterrows()
    ]
    ax.barh(range(len(top_combos)), top_combos['combined_score'], color='mediumseagreen')
    ax.set_yticks(range(len(top_combos)))
    ax.set_yticklabels(combo_labels, fontsize=8)
    ax.set_xlabel('Combined Score', fontsize=10)
    ax.set_title('Top 10 Channel Combinations', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_analysis_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Detailed metrics heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(metrics_df) * 0.4)))
    
    # Select metrics to show
    metric_cols = ['mean_intensity', 'contrast', 'edge_content', 'texture', 
                   'snr', 'nuclei_score', 'cytoplasm_score']
    
    # Normalize for heatmap (except scores which are already normalized)
    heatmap_data = metrics_df[['channel'] + metric_cols].copy()
    for col in ['mean_intensity', 'snr']:
        heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (
            heatmap_data[col].max() - heatmap_data[col].min() + 1e-10
        )
    
    heatmap_data = heatmap_data.set_index('channel')[metric_cols]
    
    sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0.5,
        cbar_kws={'label': 'Normalized Value'},
        ax=ax,
        linewidths=0.5
    )
    ax.set_title('Channel Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_metrics_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze channels for Cellpose segmentation"
    )
    parser.add_argument(
        '--zarr',
        type=str,
        default='data/CellDIVE_SLIDE-045.zarr',
        help='Path to zarr file (default: CellDIVE_SLIDE-045.zarr)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=2048,
        help='Size of sample regions to analyze (default: 2048)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=4,
        help='Number of sample regions per channel (default: 4)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/channel_analysis_output',
        help='Output directory for results (default: channel_analysis_output)'
    )
    parser.add_argument(
        '--max-channels',
        type=int,
        default=None,
        help='Maximum number of channels to analyze (for testing, default: all)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    workspace_dir = script_dir.parent
    zarr_path = workspace_dir / args.zarr
    output_dir = workspace_dir / args.output_dir
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
    
    print(f"📊 Channel Analysis for Cellpose Segmentation")
    print(f"=" * 60)
    print(f"Zarr file: {zarr_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Load channel metadata
    print("Loading channel metadata...")
    channel_names = load_channel_metadata(str(zarr_path))
    n_channels = len(channel_names)
    
    if args.max_channels:
        n_channels = min(n_channels, args.max_channels)
        channel_names = channel_names[:n_channels]
    
    print(f"Found {n_channels} channels:")
    for i, name in enumerate(channel_names):
        print(f"  {i:2d}: {name}")
    print()
    
    # Analyze each channel
    print(f"Analyzing {n_channels} channels...")
    print(f"  Sample size: {args.sample_size}×{args.sample_size}")
    print(f"  Samples per channel: {args.n_samples}\n")
    
    channel_metrics = []
    
    for i, channel_name in enumerate(tqdm(channel_names, desc="Channels")):
        # Sample regions
        samples = sample_image_region(
            str(zarr_path),
            i,
            sample_size=args.sample_size,
            n_samples=args.n_samples
        )
        
        # Calculate metrics
        metrics = calculate_channel_metrics(samples)
        channel_metrics.append(metrics)
    
    print("\n✅ Channel analysis complete!\n")
    
    # Analyze combinations
    print("Analyzing channel combinations...")
    combinations_df = analyze_channel_combinations(channel_metrics, channel_names)
    print(f"  Evaluated {len(combinations_df)} combinations\n")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save channel metrics
    metrics_df = pd.DataFrame([
        {**metrics, 'channel': name, 'channel_idx': idx}
        for idx, (name, metrics) in enumerate(zip(channel_names, channel_metrics))
    ])
    metrics_df['nuclei_score'] = metrics_df.apply(
        lambda row: calculate_nuclei_score(row.to_dict()), axis=1
    )
    metrics_df['cytoplasm_score'] = metrics_df.apply(
        lambda row: calculate_cytoplasm_score(row.to_dict()), axis=1
    )
    
    metrics_df = metrics_df.sort_values('nuclei_score', ascending=False)
    metrics_csv = output_dir / 'channel_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✅ Saved channel metrics to {metrics_csv}")
    
    # Save combinations
    combinations_csv = output_dir / 'channel_combinations.csv'
    combinations_df.to_csv(combinations_csv, index=False)
    print(f"✅ Saved combinations to {combinations_csv}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(channel_metrics, channel_names, combinations_df, output_dir)
    
    # Print recommendations
    print("\n" + "=" * 60)
    print("📋 RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n🏆 Top 5 Channels for NUCLEI Detection:")
    top_nuclei = metrics_df.nlargest(5, 'nuclei_score')
    for idx, row in top_nuclei.iterrows():
        print(f"  {row['channel']:20s} (idx {row['channel_idx']:2d}): "
              f"score={row['nuclei_score']:.3f}, "
              f"intensity={row['mean_intensity']:.0f}, "
              f"contrast={row['contrast']:.3f}")
    
    print("\n🏆 Top 5 Channels for CYTOPLASM Detection:")
    top_cyto = metrics_df.nlargest(5, 'cytoplasm_score')
    for idx, row in top_cyto.iterrows():
        print(f"  {row['channel']:20s} (idx {row['channel_idx']:2d}): "
              f"score={row['cytoplasm_score']:.3f}, "
              f"intensity={row['mean_intensity']:.0f}, "
              f"texture={row['texture']:.3f}")
    
    print("\n🏆 Top 5 CHANNEL COMBINATIONS:")
    top_combos = combinations_df.head(5)
    for idx, row in top_combos.iterrows():
        print(f"  Nuclei: {row['nuclei_channel']:20s} (idx {row['nuclei_idx']:2d})")
        print(f"  Cytoplasm: {row['cytoplasm_channel']:20s} (idx {row['cytoplasm_idx']:2d})")
        print(f"  Combined Score: {row['combined_score']:.3f}")
        print()
    
    print("\n💡 Usage in Cellpose:")
    best_combo = combinations_df.iloc[0]
    print(f"  Recommended channels: [{best_combo['nuclei_idx']}, {best_combo['cytoplasm_idx']}]")
    print(f"  Command example:")
    print(f"    python scripts/cellpose_segmentation.py \\")
    print(f"      --nuclear data/{best_combo['nuclei_channel']}.tif \\")
    print(f"      --cyto data/{best_combo['cytoplasm_channel']}.tif")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete! Check output directory for detailed results.")
    print("=" * 60)


if __name__ == '__main__':
    main()
