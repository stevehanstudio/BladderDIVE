#!/usr/bin/env python3
"""
Measure per-cell marker intensities for CD45, CD3E, and CD8a.
For each cell mask, calculates:
  - Mean intensity inside cell
  - Composite intensity (sum of markers)
  - Positive/negative classification for each marker
"""

import tifffile
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from tqdm import tqdm
from skimage.measure import regionprops
import argparse

def measure_intensities_per_cell(mask_path, marker_paths, marker_names, 
                                 output_path, positive_threshold=None,
                                 batch_size=10000):
    """
    Measure marker intensities for each cell.
    
    Parameters
    ----------
    mask_path : Path or str
        Path to cell segmentation mask TIFF
    marker_paths : list
        List of paths to marker TIFF files
    marker_names : list
        List of marker names (same order as marker_paths)
    output_path : Path or str
        Output CSV path
    positive_threshold : dict or None
        Threshold for positive classification per marker
        If None, uses median of non-zero intensities
    batch_size : int
        Number of regions to process at a time (for memory management)
    """
    
    print(f"Loading segmentation mask: {mask_path}")
    mask = tifffile.imread(mask_path)
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    
    # Get region properties
    print("Computing region properties...")
    regions = regionprops(mask)
    n_cells = len(regions)
    print(f"Found {n_cells:,} cells")
    
    # Load marker images (memory-mapped to avoid loading everything)
    marker_images = {}
    print(f"\nLoading {len(marker_paths)} marker images...")
    for marker_path, marker_name in zip(marker_paths, marker_names):
        print(f"  Loading {marker_name}...")
        # Use memmap for large files
        marker_images[marker_name] = tifffile.memmap(marker_path)
        if marker_images[marker_name].shape != mask.shape:
            raise ValueError(
                f"{marker_name} shape {marker_images[marker_name].shape} "
                f"doesn't match mask shape {mask.shape}"
            )
    
    # Initialize results storage
    results = {
        'cell_id': [],
    }
    for name in marker_names:
        results[f'{name}_mean'] = []
        results[f'{name}_sum'] = []
        results[f'{name}_max'] = []
        results[f'{name}_positive'] = []
    
    results['composite_intensity'] = []  # Sum of all markers
    results['cell_area'] = []
    
    # Determine thresholds if not provided
    if positive_threshold is None:
        print("\nCalculating positive thresholds (median of non-zero intensities)...")
        positive_threshold = {}
        for marker_name in marker_names:
            img = marker_images[marker_name]
            # Sample non-zero values (don't need full image)
            nonzero = img[img > 0]
            if len(nonzero) > 0:
                # Use 10% sample for speed if image is huge
                if len(nonzero) > 1000000:
                    sample = np.random.choice(nonzero, size=1000000, replace=False)
                    threshold = np.median(sample)
                else:
                    threshold = np.median(nonzero)
                positive_threshold[marker_name] = float(threshold)
                print(f"  {marker_name}: threshold = {threshold:.2f}")
            else:
                positive_threshold[marker_name] = 0.0
                print(f"  {marker_name}: no positive pixels found")
    
    print(f"\nProcessing {n_cells:,} cells in batches of {batch_size}...")
    
    # Process cells in batches
    for batch_start in tqdm(range(0, n_cells, batch_size), desc="Batch"):
        batch_end = min(batch_start + batch_size, n_cells)
        batch_regions = regions[batch_start:batch_end]
        
        for region in batch_regions:
            cell_id = region.label
            min_row, min_col, max_row, max_col = region.bbox
            
            # Extract cell mask region
            mask_region = mask[min_row:max_row, min_col:max_col]
            cell_mask = (mask_region == cell_id)
            
            if not np.any(cell_mask):
                continue
            
            cell_area = region.area
            results['cell_id'].append(cell_id)
            results['cell_area'].append(cell_area)
            
            # Measure each marker
            marker_sums = []
            for marker_name in marker_names:
                marker_img = marker_images[marker_name]
                marker_region = marker_img[min_row:max_row, min_col:max_col]
                
                # Extract intensities inside cell
                intensities = marker_region[cell_mask]
                
                if len(intensities) > 0:
                    mean_int = float(np.mean(intensities))
                    sum_int = float(np.sum(intensities))
                    max_int = float(np.max(intensities))
                    
                    # Classify as positive if mean intensity > threshold
                    is_positive = mean_int > positive_threshold[marker_name]
                    
                    results[f'{marker_name}_mean'].append(mean_int)
                    results[f'{marker_name}_sum'].append(sum_int)
                    results[f'{marker_name}_max'].append(max_int)
                    results[f'{marker_name}_positive'].append(is_positive)
                    
                    marker_sums.append(sum_int)
                else:
                    results[f'{marker_name}_mean'].append(0.0)
                    results[f'{marker_name}_sum'].append(0.0)
                    results[f'{marker_name}_max'].append(0.0)
                    results[f'{marker_name}_positive'].append(False)
                    marker_sums.append(0.0)
            
            # Composite intensity = sum of all marker sums
            results['composite_intensity'].append(sum(marker_sums))
        
        # Periodic garbage collection
        if batch_start % (batch_size * 5) == 0:
            gc.collect()
    
    # Create DataFrame
    print("\nCreating results DataFrame...")
    df = pd.DataFrame(results)
    
    # Add summary statistics
    print("\nSummary Statistics:")
    print(f"  Total cells: {len(df):,}")
    for marker_name in marker_names:
        n_positive = df[f'{marker_name}_positive'].sum()
        pct_positive = 100 * n_positive / len(df)
        print(f"  {marker_name}+ cells: {n_positive:,} ({pct_positive:.2f}%)")
    
    # Save to CSV
    print(f"\nSaving results to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df):,} cells × {len(df.columns)} measurements")
    
    # Cleanup
    del mask, regions, marker_images
    gc.collect()
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Measure per-cell marker intensities for CD45, CD3E, and CD8a"
    )
    parser.add_argument(
        '--mask',
        type=str,
        default='output/cellpose_output/cellpose_masks.tif',
        help='Path to cell segmentation mask TIFF'
    )
    parser.add_argument(
        '--cd45',
        type=str,
        default='data/CD45.tif',
        help='Path to CD45 marker TIFF'
    )
    parser.add_argument(
        '--cd3e',
        type=str,
        default='data/CD3E.tif',
        help='Path to CD3E marker TIFF'
    )
    parser.add_argument(
        '--cd8a',
        type=str,
        default='data/CD8a.tif',
        help='Path to CD8a marker TIFF'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output/cellpose_output/cell_intensities.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--threshold-cd45',
        type=float,
        default=None,
        help='Threshold for CD45 positive classification (default: median)'
    )
    parser.add_argument(
        '--threshold-cd3e',
        type=float,
        default=None,
        help='Threshold for CD3E positive classification (default: median)'
    )
    parser.add_argument(
        '--threshold-cd8a',
        type=float,
        default=None,
        help='Threshold for CD8a positive classification (default: median)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Number of cells to process at a time (default: 10000)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    workspace_dir = script_dir.parent
    
    mask_path = workspace_dir / args.mask
    marker_paths = [
        workspace_dir / args.cd45,
        workspace_dir / args.cd3e,
        workspace_dir / args.cd8a,
    ]
    marker_names = ['CD45', 'CD3E', 'CD8a']
    output_path = workspace_dir / args.output
    
    # Check files exist
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    for path in marker_paths:
        if not path.exists():
            raise FileNotFoundError(f"Marker file not found: {path}")
    
    # Setup thresholds
    positive_threshold = None
    if any([args.threshold_cd45, args.threshold_cd3e, args.threshold_cd8a]):
        positive_threshold = {
            'CD45': args.threshold_cd45,
            'CD3E': args.threshold_cd3e,
            'CD8a': args.threshold_cd8a,
        }
        # Remove None values to use median instead
        positive_threshold = {k: v for k, v in positive_threshold.items() if v is not None}
        if not positive_threshold:
            positive_threshold = None
    
    # Run measurement
    df = measure_intensities_per_cell(
        mask_path=mask_path,
        marker_paths=marker_paths,
        marker_names=marker_names,
        output_path=output_path,
        positive_threshold=positive_threshold,
        batch_size=args.batch_size
    )
    
    print(f"\n✅ Complete! Results saved to {output_path}")


if __name__ == '__main__':
    main()
