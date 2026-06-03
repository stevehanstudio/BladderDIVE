#!/usr/bin/env python3
"""
Check pixel size from CellDIVE image metadata and update napari_print.py
"""

import sys
import re
from pathlib import Path

# Try different methods to read pixel size
try:
    from tifffile import TiffFile
    tifffile_available = True
except ImportError:
    tifffile_available = False

try:
    from aicsimageio import AICSImage
    aics_available = True
except ImportError:
    aics_available = False

def check_with_tifffile(tiff_path):
    """Check pixel size using tifffile."""
    with TiffFile(tiff_path) as tif:
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
            # Parse XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(tif.ome_metadata)
            
            # Find PhysicalSizeX and PhysicalSizeY
            # Try different namespace variations
            namespaces = [
                {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'},
                {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'},
                {}
            ]
            
            for ns in namespaces:
                if ns:
                    pixels = root.find('.//ome:Pixels', ns)
                else:
                    pixels = root.find('.//Pixels')
                
                if pixels is not None:
                    phys_size_x = pixels.get('PhysicalSizeX')
                    phys_size_y = pixels.get('PhysicalSizeY')
                    if phys_size_x:
                        return {
                            'x': float(phys_size_x),
                            'y': float(phys_size_y) if phys_size_y else float(phys_size_x),
                            'x_unit': pixels.get('PhysicalSizeXUnit', 'µm'),
                            'y_unit': pixels.get('PhysicalSizeYUnit', 'µm')
                        }
    return None

def check_with_aics(tiff_path):
    """Check pixel size using aicsimageio."""
    img = AICSImage(tiff_path)
    
    # Try physical_pixel_sizes property
    if hasattr(img, 'physical_pixel_sizes'):
        sizes = img.physical_pixel_sizes
        if sizes:
            # Handle different return types
            if hasattr(sizes, 'X'):
                return {
                    'x': float(sizes.X),
                    'y': float(sizes.Y) if hasattr(sizes, 'Y') else float(sizes.X),
                    'x_unit': 'µm',
                    'y_unit': 'µm'
                }
            elif isinstance(sizes, (list, tuple)) and len(sizes) >= 2:
                return {
                    'x': float(sizes[1]),  # Usually Z, Y, X order
                    'y': float(sizes[2]) if len(sizes) > 2 else float(sizes[1]),
                    'x_unit': 'µm',
                    'y_unit': 'µm'
                }
    
    # Try OME metadata
    if hasattr(img, 'ome_metadata') and img.ome_metadata:
        try:
            pixels = img.ome_metadata.images[0].pixels
            if hasattr(pixels, 'PhysicalSizeX'):
                return {
                    'x': float(pixels.PhysicalSizeX),
                    'y': float(pixels.PhysicalSizeY) if hasattr(pixels, 'PhysicalSizeY') else float(pixels.PhysicalSizeX),
                    'x_unit': 'µm',
                    'y_unit': 'µm'
                }
        except Exception as e:
            print(f"  OME metadata access failed: {e}")
    
    return None

def update_napari_print(pixel_size):
    """Update napari_print.py with the actual pixel size."""
    script_dir = Path(__file__).parent.resolve()
    napari_print_path = script_dir / "napari_print.py"
    
    if not napari_print_path.exists():
        print(f"⚠️  napari_print.py not found at {napari_print_path}")
        return False
    
    # Read current content
    with open(napari_print_path, 'r') as f:
        content = f.read()
    
    # Replace the pixel size value
    # Pattern: scale = [0.325, 0.325] or similar
    pattern = r'scale\s*=\s*\[[\d.]+\s*,\s*[\d.]+\]'
    replacement = f'scale = [{pixel_size}, {pixel_size}]'
    
    new_content = re.sub(pattern, replacement, content)
    
    # Also update the comment if it has an example value
    comment_pattern = r'# .*?0\.\d+.*?microns per pixel'
    new_content = re.sub(comment_pattern, f'# Pixel size: {pixel_size} microns per pixel', new_content)
    
    if new_content != content:
        with open(napari_print_path, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    script_dir = Path(__file__).parent.resolve()
    workspace_dir = script_dir.parent
    
    # Try to find the original TIFF
    tiff_path = workspace_dir / "raw" / "CellDIVE_SLIDE-045_R0.aivia.tif"
    
    if not tiff_path.exists():
        print(f"❌ TIFF file not found: {tiff_path}")
        print("\nNote: The napari_print.py script mentions 0.325 microns per pixel as an example.")
        return 1
    
    print(f"📏 Checking pixel size from: {tiff_path}")
    print()
    
    pixel_size = None
    
    # Try aicsimageio first (more reliable)
    if aics_available:
        try:
            result = check_with_aics(str(tiff_path))
            if result:
                pixel_size = result['x']
                print("✅ Found pixel size using aicsimageio:")
                print(f"   X: {result['x']} {result['x_unit']}")
                print(f"   Y: {result['y']} {result['y_unit']}")
        except Exception as e:
            print(f"⚠️  aicsimageio method failed: {e}")
    
    # Try tifffile if aics didn't work
    if pixel_size is None and tifffile_available:
        try:
            result = check_with_tifffile(str(tiff_path))
            if result:
                pixel_size = result['x']
                print("✅ Found pixel size using tifffile:")
                print(f"   X: {result['x']} {result['x_unit']}")
                print(f"   Y: {result['y']} {result['y_unit']}")
        except Exception as e:
            print(f"⚠️  tifffile method failed: {e}")
    
    if pixel_size is None:
        print("❌ Could not read pixel size from metadata")
        print("\nNote: The napari_print.py script mentions 0.325 microns per pixel.")
        print("This may be the actual pixel size for your CellDIVE image.")
        return 1
    
    # Update napari_print.py
    print(f"\n📝 Updating napari_print.py with pixel size: {pixel_size} µm/pixel")
    if update_napari_print(pixel_size):
        print("✅ Successfully updated napari_print.py")
    else:
        print("⚠️  Could not update napari_print.py (file may already have correct value)")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
