#!/usr/bin/env python3
"""
Extract pixel size from CellDIVE TIFF and update napari_print.py
Tries multiple methods to read the pixel size.
"""

import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path

def extract_from_ome_xml(xml_string):
    """Extract pixel size from OME-XML string."""
    try:
        root = ET.fromstring(xml_string)
        
        # Try different namespace patterns
        namespaces = [
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'},
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2015-01'},
            {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2013-06'},
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
                    x_val = float(phys_size_x)
                    y_val = float(phys_size_y) if phys_size_y else x_val
                    unit_x = pixels.get('PhysicalSizeXUnit', 'µm')
                    unit_y = pixels.get('PhysicalSizeYUnit', 'µm')
                    return x_val, y_val, unit_x, unit_y
    except Exception as e:
        pass
    return None

def try_method_1_tifffile(tiff_path):
    """Method 1: Try tifffile library."""
    try:
        from tifffile import TiffFile
        with TiffFile(tiff_path) as tif:
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                result = extract_from_ome_xml(tif.ome_metadata)
                if result:
                    return result
    except ImportError:
        pass
    except Exception as e:
        pass
    return None

def try_method_2_aicsimageio(tiff_path):
    """Method 2: Try aicsimageio library."""
    try:
        from aicsimageio import AICSImage
        img = AICSImage(tiff_path)
        
        # Try physical_pixel_sizes
        if hasattr(img, 'physical_pixel_sizes'):
            sizes = img.physical_pixel_sizes
            if sizes:
                if hasattr(sizes, 'X'):
                    return (float(sizes.X), float(sizes.Y) if hasattr(sizes, 'Y') else float(sizes.X), 'µm', 'µm')
                elif isinstance(sizes, (list, tuple)) and len(sizes) >= 2:
                    # Usually Z, Y, X order
                    return (float(sizes[2]) if len(sizes) > 2 else float(sizes[1]), 
                           float(sizes[1]), 'µm', 'µm')
        
        # Try OME metadata
        if hasattr(img, 'ome_metadata') and img.ome_metadata:
            try:
                pixels = img.ome_metadata.images[0].pixels
                if hasattr(pixels, 'PhysicalSizeX'):
                    x = float(pixels.PhysicalSizeX)
                    y = float(pixels.PhysicalSizeY) if hasattr(pixels, 'PhysicalSizeY') else x
                    return (x, y, 'µm', 'µm')
            except:
                pass
    except ImportError:
        pass
    except Exception as e:
        pass
    return None

def try_method_3_pil(tiff_path):
    """Method 3: Try PIL/Pillow (limited OME support)."""
    try:
        from PIL import Image
        from PIL.TiffTags import TAGS
        
        with Image.open(tiff_path) as img:
            # Check for OME-XML in tags
            if hasattr(img, 'tag_v2'):
                # Look for OME-XML tag (usually 270 or 34665)
                for tag_id in [270, 34665, 50839]:
                    if tag_id in img.tag_v2:
                        tag_value = img.tag_v2[tag_id]
                        if isinstance(tag_value, (list, tuple)) and len(tag_value) > 0:
                            xml_str = tag_value[0] if isinstance(tag_value[0], str) else str(tag_value[0])
                            result = extract_from_ome_xml(xml_str)
                            if result:
                                return result
    except ImportError:
        pass
    except Exception as e:
        pass
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
    
    original_content = content
    
    # Replace the scale value
    pattern = r'scale\s*=\s*\[[\d.]+\s*,\s*[\d.]+\]'
    replacement = f'scale = [{pixel_size}, {pixel_size}]'
    content = re.sub(pattern, replacement, content)
    
    # Update the comment
    comment_pattern = r'# .*?0\.\d+.*?microns per pixel.*'
    new_comment = f'# Pixel size: {pixel_size} microns per pixel (extracted from TIFF metadata)'
    content = re.sub(comment_pattern, new_comment, content)
    
    # If no comment pattern matched, add a new comment
    if content == original_content:
        # Add comment before the scale line
        content = re.sub(
            r'(viewer\.scale_bar\.unit = "µm")',
            f'# Pixel size: {pixel_size} microns per pixel (extracted from TIFF metadata)\n\\1',
            content
        )
    
    if content != original_content:
        with open(napari_print_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    script_dir = Path(__file__).parent.resolve()
    workspace_dir = script_dir.parent
    tiff_path = workspace_dir / "raw" / "CellDIVE_SLIDE-045_R0.aivia.tif"
    
    if not tiff_path.exists():
        print(f"❌ TIFF file not found: {tiff_path}")
        return 1
    
    print(f"📏 Extracting pixel size from: {tiff_path.name}")
    print()
    
    # Try different methods
    methods = [
        ("tifffile", try_method_1_tifffile),
        ("aicsimageio", try_method_2_aicsimageio),
        ("PIL/Pillow", try_method_3_pil),
    ]
    
    pixel_size = None
    
    for method_name, method_func in methods:
        print(f"  Trying {method_name}...", end=" ")
        try:
            result = method_func(str(tiff_path))
            if result:
                x, y, unit_x, unit_y = result
                pixel_size = x
                print(f"✅ Found!")
                print(f"     X: {x} {unit_x}")
                print(f"     Y: {y} {unit_y}")
                break
            else:
                print("❌ Not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if pixel_size is None:
        print("\n❌ Could not extract pixel size from TIFF metadata")
        print("\n💡 Options:")
        print("   1. Install required packages:")
        print("      pip install tifffile aicsimageio")
        print("   2. Or manually check the TIFF metadata using:")
        print("      - ImageJ (Image > Show Info)")
        print("      - FIJI (Image > Properties)")
        print("      - Or check the acquisition settings")
        return 1
    
    # Update napari_print.py
    print(f"\n📝 Updating napari_print.py with pixel size: {pixel_size} µm/pixel")
    if update_napari_print(pixel_size):
        print("✅ Successfully updated napari_print.py")
        print(f"\n📄 Updated file: {script_dir / 'napari_print.py'}")
    else:
        print("⚠️  Could not update napari_print.py")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
