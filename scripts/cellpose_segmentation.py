#!/usr/bin/env python3
"""
Cellpose Segmentation Script for CellDIVE Analysis
Run Cellpose segmentation on DAPI and VIM channels from command line.
"""

import numpy as np
import os
import sys
import argparse
import gc
from pathlib import Path
from tifffile import imread, imwrite

try:
    from cellpose import models
    import psutil
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Please install: pip install cellpose psutil tifffile")
    sys.exit(1)


def check_memory():
    """Check available system memory."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)
    return available_gb, total_gb


def run_segmentation(nuclear_file, cyto_file, output_dir='output', use_gpu=False, 
                    model_type='cyto3', diameter=None):
    """
    Run Cellpose segmentation on nuclear and cytoplasmic channels.
    
    Parameters:
    -----------
    nuclear_file : str
        Path to nuclear channel (DAPI) TIF file
    cyto_file : str
        Path to cytoplasmic channel (VIM) TIF file
    output_dir : str
        Directory to save output masks
    use_gpu : bool
        Whether to use GPU (default: False)
    model_type : str
        Cellpose model type (default: 'cyto3')
    diameter : float or None
        Cell diameter in pixels (None = auto-detect)
    """
    # Check memory
    available_gb, total_gb = check_memory()
    print(f"System memory: {total_gb:.2f} GB total, {available_gb:.2f} GB available")
    
    # Check file sizes
    if os.path.exists(nuclear_file):
        file_size_gb = os.path.getsize(nuclear_file) / (1024**3)
        print(f"Nuclear channel file size: {file_size_gb:.2f} GB")
    else:
        raise FileNotFoundError(f"Nuclear channel file not found: {nuclear_file}")
        
    if os.path.exists(cyto_file):
        file_size_gb = os.path.getsize(cyto_file) / (1024**3)
        print(f"Cytoplasmic channel file size: {file_size_gb:.2f} GB")
    else:
        raise FileNotFoundError(f"Cytoplasmic channel file not found: {cyto_file}")
    
    if available_gb < 30:
        print(f"⚠️  WARNING: Only {available_gb:.2f} GB available. Large images may cause issues.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load images
    print(f"\nLoading nuclear channel from: {nuclear_file}")
    try:
        img_nucleus = imread(nuclear_file)
        print(f"Nuclear image shape: {img_nucleus.shape}, dtype: {img_nucleus.dtype}")
        gc.collect()
    except MemoryError:
        print("❌ ERROR: Insufficient memory to load nuclear channel.")
        raise
    
    print(f"\nLoading cytoplasmic channel from: {cyto_file}")
    try:
        img_cyto = imread(cyto_file)
        print(f"Cytoplasmic image shape: {img_cyto.shape}, dtype: {img_cyto.dtype}")
        gc.collect()
    except MemoryError:
        print("❌ ERROR: Insufficient memory to load cytoplasmic channel.")
        raise
    
    # Combine channels (Cellpose expects shape: N_channels x Y x X)
    print("\nStacking channels...")
    imgs = np.stack([img_nucleus, img_cyto], axis=0)
    print(f"Combined image shape: {imgs.shape}")
    
    # Free memory
    del img_nucleus, img_cyto
    gc.collect()
    
    # Initialize model
    print(f"\nInitializing Cellpose model (type: {model_type}, GPU: {use_gpu})...")
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    
    # Run segmentation
    print("\nRunning segmentation (this may take a while for large images)...")
    try:
        masks, flows, styles, diams = model.eval(
            imgs, 
            diameter=diameter,
            channels=[0, 1]  # [nuclear_channel_index, cytoplasmic_channel_index]
        )
        n_cells = len(np.unique(masks)) - 1  # Subtract 1 for background (0)
        print(f"\n✅ Segmentation complete. Found {n_cells} cells.")
        
        # Save outputs
        masks_file = output_path / "cellpose_masks.tif"
        print(f"\nSaving masks to: {masks_file}")
        imwrite(masks_file, masks)
        
        # Save summary
        summary_file = output_path / "segmentation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Cellpose Segmentation Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model type: {model_type}\n")
            f.write(f"GPU used: {use_gpu}\n")
            f.write(f"Number of cells detected: {n_cells}\n")
            f.write(f"Image shape: {imgs.shape}\n")
            f.write(f"Average cell diameter: {np.mean(diams):.2f} pixels\n")
            f.write(f"Masks saved to: {masks_file}\n")
        
        print(f"Summary saved to: {summary_file}")
        print(f"\n✅ All outputs saved to: {output_path}")
        
        return masks, flows, styles, diams
        
    except MemoryError:
        print("❌ ERROR: Insufficient memory during segmentation. Consider processing in chunks.")
        raise
    finally:
        del imgs
        gc.collect()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Run Cellpose segmentation on DAPI and VIM channels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif -o results --gpu
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif -m cyto2 -d 30
        """
    )
    
    parser.add_argument('nuclear_file', type=str,
                       help='Path to nuclear channel (DAPI) TIF file')
    parser.add_argument('cyto_file', type=str,
                       help='Path to cytoplasmic channel (VIM) TIF file')
    parser.add_argument('-o', '--output', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('-m', '--model', type=str, default='cyto3',
                       choices=['cyto', 'cyto2', 'cyto3', 'nuclei'],
                       help='Cellpose model type (default: cyto3)')
    parser.add_argument('-d', '--diameter', type=float, default=None,
                       help='Cell diameter in pixels (None = auto-detect)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration (if available)')
    
    args = parser.parse_args()
    
    try:
        run_segmentation(
            nuclear_file=args.nuclear_file,
            cyto_file=args.cyto_file,
            output_dir=args.output,
            use_gpu=args.gpu,
            model_type=args.model,
            diameter=args.diameter
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()