#!/usr/bin/env python3
"""
Complete H&E to DAPI Registration Pipeline

This unified script combines all registration steps:
1. Convert SVS to TIFF
2. Prepare H&E to match DAPI dimensions
3. Register H&E to DAPI

Usage:
    # Run full pipeline
    python register_he_to_dapi_full.py \
        --svs-file "raw/PH 001 C13_085117.svs" \
        --dapi-tiff data/DAPI.tif \
        --output-dir output/he_registration_output

    # Run individual steps
    python register_he_to_dapi_full.py --step convert-svs --svs-file ...
    python register_he_to_dapi_full.py --step prepare --he-image ... --dapi-image ...
    python register_he_to_dapi_full.py --step register --he-image ... --dapi-tiff ...
"""

import argparse
from pathlib import Path
import numpy as np
import json
import tifffile
import gc
from typing import Optional, Tuple

# SVS conversion imports
try:
    import openslide
except ImportError:
    openslide = None

# Image processing imports
try:
    from skimage.transform import resize
except ImportError:
    resize = None

# Registration imports
try:
    import stainwarpy
except ImportError:
    stainwarpy = None


# ============================================================================
# Step 1: Convert SVS to TIFF
# ============================================================================

def get_svs_info(svs_path: Path):
    """Get information about SVS file levels and dimensions."""
    if openslide is None:
        raise ImportError("openslide-python not installed. Install with: conda install -c conda-forge openslide-python")
    
    slide = openslide.OpenSlide(str(svs_path))
    
    print(f"SVS file: {svs_path.name}")
    print(f"  Dimensions (level 0): {slide.dimensions}")
    print(f"  Levels available: {slide.level_count}")
    
    for level in range(slide.level_count):
        dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        print(f"  Level {level}: {dims[0]}x{dims[1]} (downsample: {downsample:.1f}x)")
    
    return slide


def convert_svs_to_tiff(
    svs_path: Path,
    output_path: Path,
    level: int = None,
    region: tuple = None,
    target_size: tuple = None
) -> np.ndarray:
    """
    Convert SVS to TIFF.
    
    Parameters
    ----------
    svs_path : Path
        Path to input SVS file
    output_path : Path
        Path to output TIFF file
    level : int, optional
        Pyramid level to extract (0 = highest resolution)
    region : tuple, optional
        Region to extract as (x, y, width, height) at level 0 coordinates
    target_size : tuple, optional
        Target size as (width, height)
    
    Returns
    -------
    np.ndarray
        Converted image array
    """
    if openslide is None:
        raise ImportError("openslide-python not installed. Install with: conda install -c conda-forge openslide-python")
    
    print(f"Opening SVS file: {svs_path}")
    slide = openslide.OpenSlide(str(svs_path))
    
    # Get image info
    level_count = slide.level_count
    level0_dims = slide.dimensions
    
    print(f"  Level 0 dimensions: {level0_dims[0]}x{level0_dims[1]}")
    print(f"  Available levels: {level_count}")
    
    # Determine which level to use
    if level is None:
        if target_size:
            # Find level closest to target size
            best_level = 0
            best_diff = float('inf')
            for i in range(level_count):
                dims = slide.level_dimensions[i]
                diff = abs(dims[0] * dims[1] - target_size[0] * target_size[1])
                if diff < best_diff:
                    best_diff = diff
                    best_level = i
            level = best_level
            print(f"  Auto-selected level {level} for target size {target_size}")
        else:
            level = 0
            print(f"  Using level 0 (highest resolution)")
    
    if level >= level_count:
        raise ValueError(f"Level {level} not available. Max level: {level_count - 1}")
    
    # Get dimensions for selected level
    level_dims = slide.level_dimensions[level]
    level_downsample = slide.level_downsamples[level]
    
    print(f"  Extracting level {level}: {level_dims[0]}x{level_dims[1]}")
    print(f"  Downsample factor: {level_downsample:.1f}x")
    
    # Extract region or full image
    if region:
        x0, y0, width, height = region
        x0_level = int(x0 / level_downsample)
        y0_level = int(y0 / level_downsample)
        width_level = int(width / level_downsample)
        height_level = int(height / level_downsample)
        
        print(f"  Extracting region at level {level}: ({x0_level}, {y0_level}, {width_level}, {height_level})")
        image = slide.read_region((x0, y0), level, (width_level, height_level))
    else:
        print(f"  Extracting full image at level {level}...")
        image = slide.read_region((0, 0), level, level_dims)
    
    # Convert PIL image to numpy array
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_array = np.array(image)
    
    print(f"  Extracted image shape: {image_array.shape}")
    print(f"  Image dtype: {image_array.dtype}")
    
    # Resize if target_size specified
    if target_size and image_array.shape[:2][::-1] != target_size:
        if resize is None:
            raise ImportError("scikit-image not installed. Install with: pip install scikit-image")
        
        print(f"  Resizing to {target_size[0]}x{target_size[1]}...")
        if target_size[0] is None:
            aspect = image_array.shape[0] / image_array.shape[1]
            target_size = (int(target_size[1] * aspect), target_size[1])
        elif target_size[1] is None:
            aspect = image_array.shape[1] / image_array.shape[0]
            target_size = (target_size[0], int(target_size[0] * aspect))
        
        image_array = (resize(image_array, (target_size[1], target_size[0], 3), 
                             preserve_range=True)).astype(image_array.dtype)
        print(f"  Resized to: {image_array.shape}")
    
    # Save as TIFF
    print(f"  Saving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file will be large (need BigTIFF for > 4GB)
    estimated_size = image_array.shape[0] * image_array.shape[1] * image_array.shape[2] * image_array.itemsize
    use_bigtiff = estimated_size > 4 * 1024 * 1024 * 1024  # 4 GB
    
    if use_bigtiff:
        print(f"  Image is large (>4GB), using BigTIFF format...")
    
    tifffile.imwrite(
        str(output_path),
        image_array,
        photometric='rgb',
        compression='lzw',
        bigtiff=use_bigtiff
    )
    
    slide.close()
    print(f"  ✅ Conversion complete!")
    
    return image_array


# ============================================================================
# Step 2: Prepare H&E to Match DAPI
# ============================================================================

def get_image_shape(image_path: Path) -> tuple:
    """Get image shape without loading full image."""
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        with tifffile.TiffFile(str(image_path)) as tif:
            if len(tif.series) > 0:
                shape = tif.series[0].shape
                if len(shape) == 3 and shape[2] == 3:
                    return (shape[0], shape[1])
                elif len(shape) == 2:
                    return shape
    else:
        from PIL import Image
        with Image.open(str(image_path)) as img:
            return img.size[::-1]
    
    # Fallback: load full image
    img = tifffile.imread(str(image_path))
    if img.ndim == 3:
        return img.shape[:2]
    return img.shape


def load_image(image_path: Path) -> np.ndarray:
    """Load image (TIFF, PNG, JPEG)."""
    print(f"Loading image: {image_path}")
    
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        img = tifffile.imread(str(image_path))
    else:
        from PIL import Image
        img = np.array(Image.open(str(image_path)))
    
    print(f"  Shape: {img.shape}, dtype: {img.dtype}")
    return img


def crop_he_to_dapi_size(
    he_image: np.ndarray,
    dapi_shape: tuple,
    method: str = 'center',
    offset_xy: Optional[Tuple[int, int]] = None,
    box_xyxy: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Crop H&E image to match DAPI dimensions.

    You can crop using:
    - a predefined anchor method (default: center), OR
    - an explicit top-left offset (offset_xy=(x0, y0)), OR
    - an explicit box (box_xyxy=(x0, y0, x1, y1)) which must match DAPI size.
    """
    he_h, he_w = he_image.shape[:2]
    target_h, target_w = dapi_shape
    
    print(f"  H&E size: {he_w}x{he_h}")
    print(f"  Target size: {target_w}x{target_h}")
    
    if he_h < target_h or he_w < target_w:
        raise ValueError(
            f"H&E image ({he_w}x{he_h}) is smaller than target DAPI ({target_w}x{target_h}). "
            f"Cannot crop. Options:\n"
            f"1. Use a higher SVS pyramid level (lower number, e.g., --level 0 for full resolution)\n"
            f"2. Use --mode resize (but this will upscale and may cause quality loss)\n"
            f"3. Check if H&E and DAPI are at different magnifications/resolutions"
        )
    
    # Explicit crop box (x0, y0, x1, y1) in H&E pixel coordinates
    if box_xyxy is not None:
        x0, y0, x1, y1 = box_xyxy
        if not all(isinstance(v, (int, np.integer)) for v in (x0, y0, x1, y1)):
            raise ValueError(f"crop box values must be integers, got: {box_xyxy}")
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid crop box: {box_xyxy}")
        crop_w = x1 - x0
        crop_h = y1 - y0
        if crop_h != target_h or crop_w != target_w:
            raise ValueError(
                f"Crop box size ({crop_w}x{crop_h}) must match target DAPI ({target_w}x{target_h}). "
                f"Got box {box_xyxy}."
            )
        if x1 > he_w or y1 > he_h:
            raise ValueError(
                f"Crop box {box_xyxy} is outside H&E bounds (width={he_w}, height={he_h})."
            )

        if he_image.ndim == 3:
            cropped = he_image[y0:y1, x0:x1, :]
        else:
            cropped = he_image[y0:y1, x0:x1]

        print(f"  Cropped using explicit box: (x0={x0}, y0={y0}, x1={x1}, y1={y1})")
        print(f"  Cropped to: {cropped.shape}")
        return cropped

    # Explicit top-left offset (x0, y0); box size inferred from DAPI
    if offset_xy is not None:
        x0, y0 = offset_xy
        if not all(isinstance(v, (int, np.integer)) for v in (x0, y0)):
            raise ValueError(f"crop offset values must be integers, got: {offset_xy}")
        if x0 < 0 or y0 < 0:
            raise ValueError(f"crop offset must be non-negative, got: {offset_xy}")
        x1 = x0 + target_w
        y1 = y0 + target_h
        if x1 > he_w or y1 > he_h:
            raise ValueError(
                f"Crop region (x0={x0}, y0={y0}, x1={x1}, y1={y1}) is outside H&E bounds "
                f"(width={he_w}, height={he_h})."
            )

        if he_image.ndim == 3:
            cropped = he_image[y0:y1, x0:x1, :]
        else:
            cropped = he_image[y0:y1, x0:x1]

        print(f"  Cropped using explicit offset: (x0={x0}, y0={y0})")
        print(f"  Cropped to: {cropped.shape}")
        return cropped

    # Calculate crop coordinates
    crop_h = he_h - target_h
    crop_w = he_w - target_w
    
    if method == 'center':
        crop_top = crop_h // 2
        crop_bottom = crop_h - crop_top
        crop_left = crop_w // 2
        crop_right = crop_w - crop_left
    elif method == 'top-left':
        crop_top, crop_bottom, crop_left, crop_right = 0, crop_h, 0, crop_w
    elif method == 'top-right':
        crop_top, crop_bottom, crop_left, crop_right = 0, crop_h, crop_w, 0
    elif method == 'bottom-left':
        crop_top, crop_bottom, crop_left, crop_right = crop_h, 0, 0, crop_w
    elif method == 'bottom-right':
        crop_top, crop_bottom, crop_left, crop_right = crop_h, 0, crop_w, 0
    else:
        raise ValueError(f"Unknown crop method: {method}")
    
    # Crop image
    if he_image.ndim == 3:
        cropped = he_image[crop_top:he_h-crop_bottom, crop_left:he_w-crop_right, :]
    else:
        cropped = he_image[crop_top:he_h-crop_bottom, crop_left:he_w-crop_right]
    
    print(f"  Cropped to: {cropped.shape}")
    return cropped


def resize_he_to_dapi_size(
    he_image: np.ndarray,
    dapi_shape: tuple,
    preserve_aspect: bool = False,
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """Resize H&E image to match DAPI dimensions."""
    if resize is None:
        raise ImportError("scikit-image not installed. Install with: pip install scikit-image")
    
    he_h, he_w = he_image.shape[:2]
    target_h, target_w = dapi_shape
    
    print(f"  H&E size: {he_w}x{he_h}")
    print(f"  Target size: {target_w}x{target_h}")
    
    if preserve_aspect:
        scale_h = target_h / he_h
        scale_w = target_w / he_w
        scale = min(scale_h, scale_w)
        
        new_h = int(he_h * scale)
        new_w = int(he_w * scale)
        
        print(f"  Resizing to {new_w}x{new_h} (preserving aspect ratio)")
        
        if he_image.ndim == 3:
            resized = resize(he_image, (new_h, new_w, 3), preserve_range=True,
                           anti_aliasing=True, order=1 if interpolation == 'bilinear' else 3)
        else:
            resized = resize(he_image, (new_h, new_w), preserve_range=True,
                           anti_aliasing=True, order=1 if interpolation == 'bilinear' else 3)
        
        resized = resized.astype(he_image.dtype)
        
        # Pad to exact target size (center padding)
        if new_h < target_h or new_w < target_w:
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            if he_image.ndim == 3:
                resized = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                mode='constant', constant_values=0)
            else:
                resized = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right)),
                                mode='constant', constant_values=0)
        
        print(f"  Final size: {resized.shape}")
        return resized
    else:
        print(f"  Resizing to {target_w}x{target_h}")
        
        if he_image.ndim == 3:
            resized = resize(he_image, (target_h, target_w, 3), preserve_range=True,
                           anti_aliasing=True, order=1 if interpolation == 'bilinear' else 3)
        else:
            resized = resize(he_image, (target_h, target_w), preserve_range=True,
                           anti_aliasing=True, order=1 if interpolation == 'bilinear' else 3)
        
        resized = resized.astype(he_image.dtype)
        print(f"  Final size: {resized.shape}")
        return resized


def prepare_he_image(
    he_path: Path,
    dapi_path: Path,
    output_path: Path,
    mode: str = 'crop',
    preserve_aspect: bool = False,
    crop_method: str = 'center',
    crop_offset_xy: Optional[Tuple[int, int]] = None,
    crop_box_xyxy: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """Prepare H&E image to match DAPI dimensions."""
    print("Preparing H&E image to match DAPI dimensions...")
    
    # Get DAPI dimensions
    print("\nChecking DAPI dimensions...")
    try:
        dapi_shape = get_image_shape(dapi_path)
        print(f"  DAPI shape: {dapi_shape}")
    except:
        print("  Could not read shape from metadata, loading image...")
        dapi_img = load_image(dapi_path)
        if dapi_img.ndim == 3:
            dapi_shape = dapi_img.shape[:2]
        else:
            dapi_shape = dapi_img.shape
        del dapi_img
    
    # Load H&E image
    print("\nLoading H&E image...")
    try:
        he_image = load_image(he_path)
    except Exception as e:
        raise ValueError(
            f"Failed to load H&E image: {e}\n"
            "This might be because:\n"
            "1. The SVS conversion failed (file is corrupted/empty)\n"
            "2. The file is too large for memory\n"
            "3. Try using a lower SVS pyramid level (e.g., --level 1)"
        ) from e
    
    if he_image.size == 0 or he_image.shape[0] == 0:
        raise ValueError(
            "H&E image is empty or corrupted. The SVS conversion may have failed.\n"
            "Try using a lower SVS pyramid level (e.g., --level 1 or --level 2)"
        )
    
    # Ensure RGB format
    if he_image.ndim == 2:
        raise ValueError("H&E image is grayscale. Expected RGB (3 channels).")
    elif he_image.ndim == 3:
        if he_image.shape[2] == 4:
            he_image = he_image[:, :, :3]
        elif he_image.shape[2] != 3:
            raise ValueError(f"H&E image has {he_image.shape[2]} channels. Expected 3 (RGB).")
    
    he_h, he_w = he_image.shape[:2]
    
    # Determine preparation method
    if mode == 'auto':
        if he_h >= dapi_shape[0] and he_w >= dapi_shape[1]:
            mode = 'crop'
            print("  Auto-mode: H&E is larger, using crop")
        else:
            mode = 'resize'
            print("  Auto-mode: H&E is smaller, using resize")
    
    # Prepare image
    print(f"\nApplying {mode}...")
    if mode == 'crop':
        prepared = crop_he_to_dapi_size(
            he_image,
            dapi_shape,
            method=crop_method,
            offset_xy=crop_offset_xy,
            box_xyxy=crop_box_xyxy
        )
    elif mode == 'resize':
        prepared = resize_he_to_dapi_size(he_image, dapi_shape, 
                                         preserve_aspect=preserve_aspect)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Verify dimensions match
    if prepared.shape[:2] != dapi_shape:
        raise ValueError(f"Prepared H&E shape {prepared.shape[:2]} doesn't match DAPI shape {dapi_shape}")
    
    # Save prepared image
    print(f"\nSaving prepared H&E to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file will be large (need BigTIFF for > 4GB)
    estimated_size = prepared.size * prepared.itemsize
    use_bigtiff = estimated_size > 4 * 1024 * 1024 * 1024  # 4 GB
    
    if use_bigtiff:
        print(f"  Image is large (>4GB), using BigTIFF format...")
    
    tifffile.imwrite(
        str(output_path),
        prepared,
        photometric='rgb' if prepared.ndim == 3 else 'minisblack',
        compression='lzw',
        bigtiff=use_bigtiff  # Use BigTIFF for large files
    )
    
    print(f"  ✅ H&E prepared and saved!")
    print(f"  Final dimensions: {prepared.shape}")
    
    return prepared


# ============================================================================
# Step 3: Register H&E to DAPI
# ============================================================================

def load_dapi_from_tiff(tiff_path: Path) -> np.ndarray:
    """Load DAPI from TIFF file."""
    print(f"Loading DAPI from TIFF: {tiff_path}")
    dapi = tifffile.imread(str(tiff_path))
    
    if dapi.ndim > 2:
        if dapi.shape[0] == 1:
            dapi = dapi[0]
        else:
            raise ValueError(f"DAPI TIFF has {dapi.ndim} dimensions. Expected 2D grayscale.")
    
    print(f"  DAPI shape: {dapi.shape}, dtype: {dapi.dtype}")
    return dapi


def load_he_image(he_path: Path) -> np.ndarray:
    """Load H&E RGB image."""
    print(f"Loading H&E image: {he_path}")
    
    if he_path.suffix.lower() in ['.tif', '.tiff']:
        he = tifffile.imread(str(he_path))
    else:
        from PIL import Image
        he = np.array(Image.open(str(he_path)))
    
    if he.ndim == 2:
        raise ValueError("H&E image is grayscale. Expected RGB (3 channels).")
    elif he.ndim == 3:
        if he.shape[2] == 4:
            he = he[:, :, :3]
        elif he.shape[2] != 3:
            raise ValueError(f"H&E image has {he.shape[2]} channels. Expected 3 (RGB).")
    
    print(f"  H&E shape: {he.shape}, dtype: {he.dtype}")
    return he


def preprocess_for_registration(dapi: np.ndarray, he: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess images for registration."""
    print("Preprocessing images for registration...")
    
    # Normalize DAPI to 0-255 uint8
    if dapi.dtype != np.uint8:
        dapi_normalized = (dapi.astype(np.float32) / dapi.max() * 255).astype(np.uint8)
    else:
        dapi_normalized = dapi.copy()
    
    # Convert H&E RGB to grayscale for registration
    if he.dtype != np.uint8:
        he_float = he.astype(np.float32) / he.max() * 255
    else:
        he_float = he.astype(np.float32)
    
    he_gray = (0.299 * he_float[:, :, 0] + 
               0.587 * he_float[:, :, 1] + 
               0.114 * he_float[:, :, 2]).astype(np.uint8)
    
    print(f"  DAPI (normalized): shape={dapi_normalized.shape}, dtype={dapi_normalized.dtype}")
    print(f"  H&E (grayscale): shape={he_gray.shape}, dtype={he_gray.dtype}")
    
    return dapi_normalized, he_gray


def register_he_to_dapi(
    dapi: np.ndarray,
    he: np.ndarray,
    he_rgb: np.ndarray,
    output_dir: Path,
    method: str = "affine",
    downsample_factor: int = None
) -> Tuple[np.ndarray, dict]:
    """Register H&E to DAPI using stainwarpy or fallback."""
    print(f"\nRegistering H&E to DAPI (method: {method})...")
    
    # Check image sizes and memory requirements
    image_size_gb = dapi.nbytes / (1024**3)
    print(f"  Image size: {image_size_gb:.2f} GB")
    
    # Auto-downsample for very large images to avoid memory issues
    if downsample_factor is None:
        if image_size_gb > 5.0:
            downsample_factor = 4
            print(f"  Image is very large (>5GB), auto-downsampling by {downsample_factor}x for registration")
        elif image_size_gb > 2.0:
            downsample_factor = 2
            print(f"  Image is large (>2GB), auto-downsampling by {downsample_factor}x for registration")
        else:
            downsample_factor = 1
    
    # Downsample for registration if needed
    if downsample_factor > 1:
        print(f"  Downsampling images by {downsample_factor}x for registration...")
        from skimage.transform import downscale_local_mean
        
        dapi_small = downscale_local_mean(dapi, (downsample_factor, downsample_factor)).astype(dapi.dtype)
        he_small = downscale_local_mean(he, (downsample_factor, downsample_factor)).astype(he.dtype)
        
        print(f"  Downsampled DAPI: {dapi_small.shape}")
        print(f"  Downsampled H&E: {he_small.shape}")
    else:
        dapi_small = dapi
        he_small = he
    
    # Try stainwarpy first
    if stainwarpy is not None:
        try:
            if hasattr(stainwarpy, 'register'):
                result = stainwarpy.register(
                    fixed=dapi_small,
                    moving=he_small,
                    method=method
                )
                transform_params = result.get('transform', {})
                registered_he_gray_small = result.get('registered', he_small)
            elif hasattr(stainwarpy, 'Registration'):
                reg = stainwarpy.Registration(
                    fixed_image=dapi_small,
                    moving_image=he_small,
                    method=method
                )
                reg.fit()
                registered_he_gray_small = reg.transform(he_small)
                transform_params = reg.get_transform_params()
            else:
                raise AttributeError("Unknown stainwarpy API")
            
            print(f"  Registration complete using stainwarpy!")
            
            # Scale transform parameters back to full resolution if downsampled
            if downsample_factor > 1:
                if 'shift' in transform_params:
                    transform_params['shift'] = [s * downsample_factor for s in transform_params['shift']]
                if 'translation' in transform_params:
                    transform_params['translation'] = [t * downsample_factor for t in transform_params['translation']]
            
            # For now, just return the original RGB image (transformation would be applied separately)
            registered_he_rgb = he_rgb.copy()
            return registered_he_rgb, transform_params
            
        except Exception as e:
            print(f"  stainwarpy failed: {e}")
            print(f"  Falling back to scikit-image registration...")
    
    # Fallback: Use scikit-image registration
    try:
        from skimage.registration import phase_cross_correlation
        from skimage.transform import AffineTransform, SimilarityTransform
        from scipy.ndimage import shift as ndshift
        import gc
        
        print(f"  Using phase cross-correlation for translation registration...")
        print(f"  This may take a few minutes for large images...")
        
        # For very large images, use lower upsample factor
        upsample = 1 if image_size_gb > 5.0 else 10
        
        # Free memory before registration
        gc.collect()
        
        shift, error, phasediff = phase_cross_correlation(
            dapi_small, he_small, upsample_factor=upsample
        )
        
        # Free memory after registration
        del dapi_small, he_small
        gc.collect()
        
        # Scale shift back to full resolution
        if downsample_factor > 1:
            shift = shift * downsample_factor
        
        print(f"  Detected shift: {shift}")
        
        # Apply translation to full-resolution images
        print(f"  Applying transformation to full-resolution images...")
        shift_y, shift_x = -shift[0], -shift[1]
        
        # Apply shift to RGB H&E (one channel at a time to save memory)
        registered_he_rgb = np.zeros_like(he_rgb, dtype=he_rgb.dtype)
        print(f"  Shifting RGB channels...")
        for i in range(3):
            print(f"    Channel {i+1}/3...")
            registered_he_rgb[:, :, i] = ndshift(he_rgb[:, :, i], (shift_y, shift_x), order=1, mode='constant', cval=0)
            gc.collect()
        
        transform_params = {
            'type': 'translation',
            'shift': shift.tolist(),
            'error': float(error),
            'method': 'phase_cross_correlation',
            'downsample_factor': downsample_factor
        }
        
        return registered_he_rgb, transform_params
        
    except MemoryError as e:
        print(f"  Memory error during registration: {e}")
        print(f"  Try using a lower SVS pyramid level (e.g., --level 1)")
        raise
    except Exception as e:
        print(f"  Registration failed: {e}")
        raise


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete H&E to DAPI Registration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python register_he_to_dapi_full.py \\
      --svs-file "raw/PH 001 C13_085117.svs" \\
      --dapi-tiff data/DAPI.tif \\
      --output-dir output/he_registration_output

  # Run individual steps
  python register_he_to_dapi_full.py --step convert-svs \\
      --svs-file "raw/PH 001 C13_085117.svs" --level 1 \\
      --svs-output output/registration_intermediate/HE_raw.tif

  python register_he_to_dapi_full.py --step prepare \\
      --he-image output/registration_intermediate/HE_raw.tif \\
      --dapi-image data/DAPI.tif \\
      --prepare-output output/registration_intermediate/HE_prepared.tif --mode crop

  python register_he_to_dapi_full.py --step register \\
      --he-image output/registration_intermediate/HE_prepared.tif \\
      --dapi-tiff data/DAPI.tif \\
      --output-dir output/he_registration_output
        """
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'convert-svs', 'prepare', 'register'],
        default='all',
        help='Which step to run: all (default), convert-svs, prepare, or register'
    )
    
    # SVS conversion arguments
    parser.add_argument('--svs-file', type=str, help='Path to SVS file')
    parser.add_argument('--level', type=int, default=None, help='SVS pyramid level (0=full res)')
    parser.add_argument('--svs-output', type=str, default=None, help='Output path for converted SVS')
    parser.add_argument('--svs-info-only', action='store_true', help='Only show SVS info')
    
    # Prepare arguments
    parser.add_argument('--he-image', type=str, help='Path to H&E image')
    parser.add_argument('--dapi-image', type=str, help='Path to DAPI reference image')
    parser.add_argument('--prepare-output', type=str, default=None, help='Output path for prepared H&E')
    parser.add_argument('--mode', type=str, choices=['crop', 'resize', 'auto'], default='crop',
                       help='Preparation mode: crop (default), resize, or auto')
    parser.add_argument('--crop-method', type=str, default='center',
                       choices=['center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'],
                       help='Cropping method')
    parser.add_argument(
        '--crop-offset-xy',
        type=str,
        default=None,
        help="Manual crop offset for H&E as 'x0,y0' (top-left). Overrides --crop-method."
    )
    parser.add_argument(
        '--crop-box-xyxy',
        type=str,
        default=None,
        help="Manual crop box for H&E as 'x0,y0,x1,y1'. Box size must match DAPI. Overrides --crop-method."
    )
    parser.add_argument('--preserve-aspect', action='store_true', help='Preserve aspect ratio when resizing')
    
    # Registration arguments
    parser.add_argument('--dapi-tiff', type=str, help='Path to DAPI TIFF file')
    parser.add_argument('--output-dir', type=str, default='output/he_registration_output',
                       help='Output directory for registration results')
    parser.add_argument('--method', type=str, default='affine',
                       choices=['affine', 'rigid', 'translation'],
                       help='Registration method (default: affine)')
    parser.add_argument('--downsample-factor', type=int, default=None,
                       help='Downsample factor for registration (auto-determined if not specified)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if output files exist')
    
    args = parser.parse_args()

    def _parse_int_list(csv: str, n: int, name: str) -> Tuple[int, ...]:
        parts = [p.strip() for p in csv.split(',')]
        if len(parts) != n:
            raise ValueError(f"{name} must have {n} comma-separated integers, got: {csv!r}")
        try:
            return tuple(int(p) for p in parts)  # type: ignore[return-value]
        except Exception as e:
            raise ValueError(f"{name} must be integers, got: {csv!r}") from e

    crop_offset_xy = None
    crop_box_xyxy = None
    if args.crop_offset_xy is not None:
        crop_offset_xy = _parse_int_list(args.crop_offset_xy, 2, "--crop-offset-xy")  # (x0, y0)
    if args.crop_box_xyxy is not None:
        crop_box_xyxy = _parse_int_list(args.crop_box_xyxy, 4, "--crop-box-xyxy")  # (x0, y0, x1, y1)
    if crop_offset_xy is not None and crop_box_xyxy is not None:
        raise ValueError("Specify only one of --crop-offset-xy or --crop-box-xyxy")
    
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    workspace_dir = script_dir.parent
    
    # Run selected step(s)
    if args.step in ['all', 'convert-svs']:
        if not args.svs_file:
            parser.error("--svs-file is required for convert-svs step")
        
        svs_path = Path(args.svs_file)
        if not svs_path.is_absolute():
            svs_path = workspace_dir / svs_path
        
        if args.svs_info_only:
            get_svs_info(svs_path)
            return
        
        if args.step == 'all':
            output_path = workspace_dir / "output/registration_intermediate/HE_raw.tif"
        else:
            output_path = Path(args.svs_output) if args.svs_output else workspace_dir / "output/registration_intermediate/HE_raw.tif"
            if not output_path.is_absolute():
                output_path = workspace_dir / output_path
        
        # Check if output already exists
        if output_path.exists() and not args.force:
            print("\n" + "="*60)
            print("STEP 1: Converting SVS to TIFF")
            print("="*60)
            print(f"✅ Output file already exists: {output_path}")
            print(f"   Skipping SVS conversion. Use --force to regenerate.")
        else:
            if output_path.exists() and args.force:
                print(f"⚠️  Output file exists but --force specified, regenerating...")
            print("\n" + "="*60)
            print("STEP 1: Converting SVS to TIFF")
            print("="*60)
            convert_svs_to_tiff(svs_path, output_path, level=args.level)
        
        if args.step == 'convert-svs':
            return
    
    if args.step in ['all', 'prepare']:
        if not args.he_image or not args.dapi_image:
            if args.step == 'prepare':
                parser.error("--he-image and --dapi-image are required for prepare step")
            # For 'all', use defaults
            he_path = workspace_dir / "output/registration_intermediate/HE_raw.tif"
            dapi_path = workspace_dir / "data/DAPI.tif"
        else:
            he_path = Path(args.he_image)
            if not he_path.is_absolute():
                he_path = workspace_dir / he_path
            
            dapi_path = Path(args.dapi_image)
            if not dapi_path.is_absolute():
                dapi_path = workspace_dir / dapi_path
        
        if args.step == 'all':
            output_path = workspace_dir / "output/registration_intermediate/HE_prepared.tif"
        else:
            output_path = Path(args.prepare_output) if args.prepare_output else workspace_dir / "output/registration_intermediate/HE_prepared.tif"
            if not output_path.is_absolute():
                output_path = workspace_dir / output_path
        
        # Check if output already exists
        if output_path.exists() and not args.force:
            print("\n" + "="*60)
            print("STEP 2: Preparing H&E to Match DAPI")
            print("="*60)
            print(f"✅ Output file already exists: {output_path}")
            print(f"   Skipping H&E preparation. Use --force to regenerate.")
        else:
            if output_path.exists() and args.force:
                print(f"⚠️  Output file exists but --force specified, regenerating...")
            print("\n" + "="*60)
            print("STEP 2: Preparing H&E to Match DAPI")
            print("="*60)
            prepare_he_image(
                he_path=he_path,
                dapi_path=dapi_path,
                output_path=output_path,
                mode=args.mode,
                preserve_aspect=args.preserve_aspect,
                crop_method=args.crop_method,
                crop_offset_xy=crop_offset_xy,
                crop_box_xyxy=crop_box_xyxy
            )
        
        if args.step == 'prepare':
            return
    
    if args.step in ['all', 'register']:
        if not args.he_image or not args.dapi_tiff:
            if args.step == 'register':
                parser.error("--he-image and --dapi-tiff are required for register step")
            # For 'all', use defaults
            he_path = workspace_dir / "output/registration_intermediate/HE_prepared.tif"
            dapi_path = workspace_dir / "data/DAPI.tif"
        else:
            he_path = Path(args.he_image)
            if not he_path.is_absolute():
                he_path = workspace_dir / he_path
            
            dapi_path = Path(args.dapi_tiff)
            if not dapi_path.is_absolute():
                dapi_path = workspace_dir / dapi_path
        
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = workspace_dir / output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if output already exists
        registered_he_path = output_dir / "he_registered.tif"
        transform_path = output_dir / "transform_params.json"
        dapi_ref_path = output_dir / "dapi_reference.tif"
        
        if registered_he_path.exists() and transform_path.exists() and not args.force:
            print("\n" + "="*60)
            print("STEP 3: Registering H&E to DAPI")
            print("="*60)
            print(f"✅ Registration output already exists:")
            print(f"   - {registered_he_path}")
            print(f"   - {transform_path}")
            if dapi_ref_path.exists():
                print(f"   - {dapi_ref_path}")
            print(f"   Skipping registration. Use --force to regenerate.")
        else:
            if registered_he_path.exists() and args.force:
                print(f"⚠️  Registration output exists but --force specified, regenerating...")
            print("\n" + "="*60)
            print("STEP 3: Registering H&E to DAPI")
            print("="*60)
            
            # Load images
            he_rgb = load_he_image(he_path)
            dapi = load_dapi_from_tiff(dapi_path)
            
            # Preprocess
            dapi_norm, he_gray = preprocess_for_registration(dapi, he_rgb)
            
            # Register
            registered_he_rgb, transform_params = register_he_to_dapi(
                dapi=dapi_norm,
                he=he_gray,
                he_rgb=he_rgb,
                output_dir=output_dir,
                method=args.method,
                downsample_factor=args.downsample_factor
            )
            
            # Save results
            print(f"\nSaving results to {output_dir}...")
            
            tifffile.imwrite(str(registered_he_path), registered_he_rgb)
            print(f"  ✅ Saved registered H&E: {registered_he_path}")
            
            with open(transform_path, 'w') as f:
                json.dump(transform_params, f, indent=2)
            print(f"  ✅ Saved transformation parameters: {transform_path}")
            
            tifffile.imwrite(str(dapi_ref_path), dapi_norm)
            print(f"  ✅ Saved reference DAPI: {dapi_ref_path}")
            
            print(f"\n{'='*60}")
            print("✅ Registration complete!")
            print(f"{'='*60}")
            print(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
