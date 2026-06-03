#!/usr/bin/env python3
"""
Convert Cellpose segmentation masks to GeoJSON format.

This script processes large segmentation mask TIFF files and converts them
to GeoJSON format suitable for visualization and analysis tools. It:

1. Loads segmentation masks from TIFF
2. Extracts cell boundaries using contour detection
3. Validates and cleans polygon geometries
4. Simplifies polygons to reduce file size
5. Exports to GeoJSON FeatureCollection format

The output GeoJSON can be used in tools like QuPath, ImageJ, or web viewers.
"""

import tifffile
import numpy as np
import cv2
import geojson
import gc
from pathlib import Path
from skimage.measure import regionprops
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid

# --- 1. SETUP ---
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR.parent / "output" / "cellpose_output" / "cellpose_masks.tif"
OUTPUT_PATH = INPUT_PATH.with_suffix(".geojson")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Could not find: {INPUT_PATH}")

# --- 2. LOAD DATA ---
print(f"Loading image (19GB)...")
mask = tifffile.imread(INPUT_PATH)
print(f"Image loaded. Shape: {mask.shape}")

print("Mapping regions...")
regions = regionprops(mask)
print(f"Found {len(regions)} raw regions.")

# --- 3. PROCESSING ---
features = []
skipped_count = 0

print("Extracting & Cleaning Polygons...")
for region in tqdm(regions, desc="Processing"):
    
    # Extract contours
    min_y, min_x, max_y, max_x = region.bbox
    h, w = region.image.shape
    padded_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = region.image.astype(np.uint8)
    
    contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if len(contour) < 3: continue 
        
        contour[:, 0, 0] += (min_x - 1)
        contour[:, 0, 1] += (min_y - 1)
        coords = contour.squeeze().tolist()
        
        if len(np.shape(coords)) < 2 or len(coords) < 3: continue
        coords.append(coords[0]) 
        
        try:
            poly = Polygon(coords)
            
            # Fix invalid shapes
            if not poly.is_valid:
                clean_poly = make_valid(poly)
                
                # --- CRITICAL FIX: Handle Mixed GeometryCollections ---
                if isinstance(clean_poly, GeometryCollection):
                    # Keep only Polygons or MultiPolygons, discard Lines/Points
                    kept_geoms = [g for g in clean_poly.geoms if isinstance(g, (Polygon, MultiPolygon))]
                    if not kept_geoms:
                        skipped_count += 1
                        continue
                    # Merge them into a single MultiPolygon
                    from shapely.ops import unary_union
                    poly = unary_union(kept_geoms)
                else:
                    poly = clean_poly

            # Simplify
            poly = poly.simplify(0.25, preserve_topology=True)
            
            # Final check (ensure it is not a LineString or empty)
            if poly.is_empty or not isinstance(poly, (Polygon, MultiPolygon)):
                skipped_count += 1
                continue
                
            if poly.area < 10:
                skipped_count += 1
                continue

            feature = geojson.Feature(geometry=poly, properties={
                "objectType": "detection",
                "classification": {"name": "Cell", "color": [255, 0, 0]},
                "id": int(region.label)
            })
            features.append(feature)

        except Exception:
            skipped_count += 1
            continue

# --- FREE MEMORY ---
print("Processing complete. Releasing image memory...")
del mask
del regions
gc.collect()

# --- 4. STREAMING SAVE ---
print(f"Streaming {len(features)} cells to disk (Skipped {skipped_count} bad shapes)...")

try:
    with open(OUTPUT_PATH, 'w') as f:
        f.write('{"type": "FeatureCollection", "features": [\n')
        count = len(features)
        for i, feature in enumerate(tqdm(features, desc="Writing")):
            f.write(geojson.dumps(feature))
            if i < count - 1:
                f.write(',\n')
        f.write('\n]}')

    print(f"Success! Saved to {OUTPUT_PATH}")

except Exception as e:
    print(f"Error during save: {e}")