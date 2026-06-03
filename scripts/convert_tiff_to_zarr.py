#!/usr/bin/env python3
"""
Convert multi-channel TIFF images to Zarr format with metadata preservation.

This script converts large multi-channel TIFF files (e.g., CellDIVE images) to
Zarr format with multiscale pyramids. It:

- Preserves OME-XML metadata (channel names, colors, etc.)
- Creates multiscale pyramids for efficient viewing
- Uses memory-efficient dask arrays for large files
- Maintains channel information for proper visualization in Napari

The output Zarr can be loaded in Napari with proper channel names and colors
from the original OME metadata.
"""

import numpy as np
import dask
from aicsimageio import AICSImage
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler
import zarr

# --- CONFIGURATION ---
dask.config.set(scheduler='threads', num_workers=4)

INPUT_FILE = "raw/CellDIVE_SLIDE-045_R0.aivia.tif"
OUTPUT_FILE = "data/CellDIVE_SLIDE-045.zarr"

def int_to_hex(ome_int):
    """
    Convert OME integer color to hex string format.
    
    Parameters
    ----------
    ome_int : int or None
        OME color integer value
        
    Returns
    -------
    str
        Hex color string (e.g., "FFFFFF" for white)
    """
    if ome_int is None:
        return "FFFFFFFF"
    
    # Safety cast for strict types
    val = int(ome_int)
    
    unsigned_int = val & 0xFFFFFFFF
    hex_str = f"{unsigned_int:08X}"
    return hex_str[2:]

# --- EXECUTION ---
print(f"Loading {INPUT_FILE}...")
img = AICSImage(INPUT_FILE)
dask_data = img.dask_data.squeeze()

# 1. Extract Metadata
print("Parsing OME-XML metadata...")
channels_metadata = []

if hasattr(img, "ome_metadata") and img.ome_metadata:
    # Look for channels in the first image
    # Note: Structure can vary, sometimes it's img.ome_metadata.images[0].pixels.channels
    # aicsimageio usually maps this conveniently, but we access raw OME here.
    try:
        channels = img.ome_metadata.images[0].pixels.channels
        for i, channel in enumerate(channels):
            label = channel.name if channel.name else f"Channel {i}"
            color_hex = int_to_hex(channel.color)
            
            channels_metadata.append({
                "label": label,
                "color": color_hex,
                "active": True,
                "window": {"min": 0, "max": 65535} 
            })
    except Exception as e:
        print(f"Metadata warning: {e}. Using defaults.")
        channels_metadata = [{"label": f"Ch{i}", "color": "FFFFFF"} for i in range(dask_data.shape[0])]
else:
    channels_metadata = [{"label": f"Ch{i}", "color": "FFFFFF"} for i in range(dask_data.shape[0])]

print(f"Found {len(channels_metadata)} channels. First one: {channels_metadata[0]['label']}")

# 2. Re-chunking
dask_data = dask_data.rechunk((1, 1024, 1024))
print(f"Converting shape {dask_data.shape} with chunks {dask_data.chunksize}")

# 3. Write
store = parse_url(OUTPUT_FILE, mode="w").store
root = zarr.group(store=store)

write_image(
    image=dask_data,
    group=root,
    axes="cyx",
    storage_options=dict(chunks=(1, 1024, 1024)),
    scaler=Scaler(method="nearest"),
    omero={"channels": channels_metadata}
)

print(f"Done! Saved to {OUTPUT_FILE}")