#!/usr/bin/env python3
"""
Combine two channels into a single channel TIFF file for Cellpose cytoplasm input.

IMPORTANT: This script combines PanCK and CD45RO into a SINGLE channel (not 2 channels)
by averaging them. Cellpose 4 expects:
- Nuclear input: single channel (e.g., DAPI) - shape (H, W)
- Cytoplasmic input: single channel (e.g., combined PanCK+CD45RO) - shape (H, W)
- Final stack: (2, H, W) where [0] = nuclear, [1] = cytoplasmic

If you want to use PanCK and CD45RO as separate channels, you would need to modify
the Cellpose script to handle 3-channel input (DAPI + PanCK + CD45RO), but Cellpose
models are typically trained on 2-channel data (nuclear + cytoplasmic).

This script averages the two channels to create a single combined cytoplasmic channel.
"""

import tifffile
import numpy as np
import argparse
from pathlib import Path


def combine_channels(channel1_path, channel2_path, output_path, use_memmap=True, method='average'):
    """
    Combine two single-channel TIFF files into a single channel TIFF.

    This function averages (or sums/maxes) two channels to create a single
    combined channel for use as Cellpose's cytoplasmic input.

    Parameters
    ----------
    channel1_path : str or Path
        Path to first channel TIFF file (e.g., PanCK)
    channel2_path : str or Path
        Path to second channel TIFF file (e.g., CD45RO)
    output_path : str or Path
        Output path for combined single-channel TIFF
    use_memmap : bool
        If True, use memory-mapping for large files to save RAM
    method : str
        Combination method: 'average' (default), 'sum', or 'max'
    """
    channel1_path = Path(channel1_path)
    channel2_path = Path(channel2_path)
    output_path = Path(output_path)

    # Check input files exist
    if not channel1_path.exists():
        raise FileNotFoundError(f"Channel 1 file not found: {channel1_path}")
    if not channel2_path.exists():
        raise FileNotFoundError(f"Channel 2 file not found: {channel2_path}")

    print(f"Reading {channel1_path.name}...")
    if use_memmap:
        img1 = tifffile.memmap(channel1_path)
    else:
        img1 = tifffile.imread(channel1_path)

    print(f"Reading {channel2_path.name}...")
    if use_memmap:
        img2 = tifffile.memmap(channel2_path)
    else:
        img2 = tifffile.imread(channel2_path)

    # Check shapes match
    if img1.shape != img2.shape:
        raise ValueError(
            f"Channel shapes don't match:\n"
            f"  {channel1_path.name}: {img1.shape}\n"
            f"  {channel2_path.name}: {img2.shape}"
        )

    print(f"  Shape: {img1.shape}, dtype: {img1.dtype}")

    # Combine channels into a single channel (not stack them)
    # This creates a single cytoplasmic channel for Cellpose
    print(f"Combining channels using '{method}' method...")
    
    # Ensure arrays are loaded into memory for computation
    if use_memmap:
        img1 = np.array(img1, copy=True)
        img2 = np.array(img2, copy=True)
    
    # Combine based on method
    if method == 'average':
        combined = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0
    elif method == 'sum':
        combined = img1.astype(np.float32) + img2.astype(np.float32)
    elif method == 'max':
        combined = np.maximum(img1.astype(np.float32), img2.astype(np.float32))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'average', 'sum', or 'max'")
    
    # Convert back to uint16 for output
    combined = np.clip(combined, 0, 65535).astype(np.uint16)
    print(f"  Combined shape: {combined.shape} (single channel)")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write combined image
    # Use BigTIFF for large images to avoid uint32 overflow errors
    print(f"Writing to {output_path}...")
    
    # Calculate image size to determine if BigTIFF is needed
    # BigTIFF is needed for images > 4GB uncompressed or very large dimensions
    total_pixels = combined.size
    uncompressed_size_gb = total_pixels * combined.dtype.itemsize / (1024**3)
    use_bigtiff = uncompressed_size_gb > 4.0 or max(combined.shape) > 65535
    
    if use_bigtiff:
        print(f"  Using BigTIFF format (uncompressed size: {uncompressed_size_gb:.2f} GB)")
        print(f"  Note: Some viewers may have limited BigTIFF support")
    
    # Try multiple write strategies for maximum compatibility
    write_success = False
    
    # Strategy 1: Try uncompressed BigTIFF (most compatible with viewers)
    if use_bigtiff:
        try:
            print("  Attempting uncompressed BigTIFF (best compatibility)...")
            tifffile.imwrite(
                output_path,
                combined,
                bigtiff=True,
                compression=None,  # No compression for better compatibility
                ome=False
            )
            write_success = True
            print("  ✅ Written as uncompressed BigTIFF")
        except Exception as e:
            print(f"  ⚠️  Uncompressed BigTIFF failed: {e}")
    
    # Strategy 2: Try compressed BigTIFF (smaller file size)
    if not write_success:
        try:
            print("  Attempting compressed BigTIFF...")
            tifffile.imwrite(
                output_path,
                combined,
                compression='zlib',
                compressionargs={'level': 1},  # Lower compression for better compatibility
                bigtiff=use_bigtiff,
                ome=False
            )
            write_success = True
            print("  ✅ Written as compressed BigTIFF")
        except Exception as e:
            print(f"  ⚠️  Compressed BigTIFF failed: {e}")
    
    # Strategy 3: Try standard TIFF with compression (if file is small enough)
    if not write_success and not use_bigtiff:
        try:
            print("  Attempting standard TIFF with compression...")
            tifffile.imwrite(
                output_path,
                combined,
                compression='zlib',
                compressionargs={'level': 1},
                ome=False
            )
            write_success = True
            print("  ✅ Written as standard compressed TIFF")
        except Exception as e:
            print(f"  ⚠️  Standard TIFF failed: {e}")
    
    if not write_success:
        raise RuntimeError("Failed to write file with any method. File may be too large or corrupted.")

    # Calculate file sizes
    size1_mb = channel1_path.stat().st_size / (1024**2)
    size2_mb = channel2_path.stat().st_size / (1024**2)
    size_out_mb = output_path.stat().st_size / (1024**2)

    print(f"\n✅ Successfully combined channels!")
    print(f"  Input 1: {size1_mb:.1f} MB")
    print(f"  Input 2: {size2_mb:.1f} MB")
    print(f"  Output:  {size_out_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Combine two channels into a single channel TIFF for Cellpose cytoplasm input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine PanCK and CD45RO (averaged by default)
  python combine_channels.py data/PANCK.tif data/CD45RO.tif data/PANCK+CD45RO.tif
  
  # Use sum instead of average
  python combine_channels.py data/PANCK.tif data/CD45RO.tif data/PANCK+CD45RO.tif --method sum
  
  # Without memory-mapping (loads full images into RAM)
  python combine_channels.py data/PANCK.tif data/CD45RO.tif data/PANCK+CD45RO.tif --no-memmap
        """
    )

    parser.add_argument(
        'channel1',
        type=str,
        help='Path to first channel TIFF file (e.g., PanCK)'
    )
    parser.add_argument(
        'channel2',
        type=str,
        help='Path to second channel TIFF file (e.g., CD45RO)'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Output path for combined single-channel TIFF'
    )
    parser.add_argument(
        '--no-memmap',
        action='store_true',
        help='Disable memory-mapping (loads full images into RAM)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='average',
        choices=['average', 'sum', 'max'],
        help='Method to combine channels: average (default), sum, or max'
    )

    args = parser.parse_args()

    try:
        combine_channels(
            args.channel1,
            args.channel2,
            args.output,
            use_memmap=not args.no_memmap,
            method=args.method
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
