#!/usr/bin/env python3
"""
Convert Cellpose segmentation masks from TIFF to Zarr format.

This script converts large segmentation mask TIFF files to Zarr format with
multiscale pyramids for efficient viewing in Napari and other tools. Key features:

- Preserves cell IDs (uses nearest-neighbor scaling, not interpolation)
- Creates multiscale pyramids for efficient zooming
- Uses memory-efficient dask arrays for large files
- Automatically derives output filename from input filename

The output Zarr can be loaded in Napari for visualization and comparison
of different segmentation results.
"""

import argparse
import dask
from aicsimageio import AICSImage
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler
import zarr
import numpy as np
from pathlib import Path

# Default input when no argument is given
# DEFAULT_INPUT = "/home/steve/Projects/HeLab/BladderDIVE/cellpose_output/cellpose_masks_dapi_only_9tiles.tif"
DEFAULT_INPUT = "output/cellpose_output/cellpose_masks_dapi+panck+cd45_9tiles.tif"


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cellpose segmentation mask TIFF to Zarr (multiscale pyramids).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_mask_to_zarr.py
  python convert_mask_to_zarr.py output/cellpose_output/cellpose_masks_dapi+vim_9tiles.tif
  python convert_mask_to_zarr.py /path/to/masks.tif
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Input mask TIFF path (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = input_path.with_suffix(".zarr")
    output_file = str(output_path)

    dask.config.set(scheduler="threads", num_workers=4)

    print(f"Loading Mask: {input_path}...")
    img = AICSImage(input_path)
    dask_data = img.dask_data.squeeze()

    if not np.issubdtype(dask_data.dtype, np.integer):
        print(f"Warning: Data is {dask_data.dtype}. Casting to uint32 for safety.")
        dask_data = dask_data.astype(np.uint32)

    dask_data = dask_data.rechunk((1024, 1024))
    print(f"Converting shape {dask_data.shape} with chunks {dask_data.chunksize}")

    store = parse_url(output_file, mode="w").store
    root = zarr.group(store=store)

    print("Writing pyramids (this may take a while)...")
    write_image(
        image=dask_data,
        group=root,
        axes="yx",
        storage_options=dict(chunks=(1024, 1024)),
        scaler=Scaler(method="nearest"),
    )

    print(f"Done! Saved to {output_file}")


if __name__ == "__main__":
    main()