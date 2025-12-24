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
import time
import subprocess
from pathlib import Path
from tifffile import imread, imwrite, TiffFile, memmap
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

try:
    from cellpose import models
    import psutil
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Please install: pip install cellpose psutil tifffile")
    sys.exit(1)

# Try to import GPU detection libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def check_memory():
    """Check available system memory."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)
    return available_gb, total_gb


def check_gpu_availability(min_free_memory_gb=2.0):
    """
    Check GPU availability and free memory.
    
    Parameters:
    -----------
    min_free_memory_gb : float
        Minimum free GPU memory required (GB)
    
    Returns:
    --------
    tuple: (available, free_memory_gb, total_memory_gb, gpu_id)
        available: bool - Whether GPU can be used
        free_memory_gb: float - Free GPU memory in GB
        total_memory_gb: float - Total GPU memory in GB
        gpu_id: int or None - GPU device ID
    """
    # Try PyTorch first (most reliable)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_id = 0  # Use first GPU
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            free_memory = total_memory - reserved
            
            available = free_memory >= min_free_memory_gb
            return available, free_memory, total_memory, gpu_id
        except Exception as e:
            print(f"Warning: Could not query GPU via PyTorch: {e}")
    
    # Try pynvml (nvidia-ml-py)
    if PYNVML_AVAILABLE:
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_id = 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = info.total / (1024**3)
            free_memory = info.free / (1024**3)
            used_memory = info.used / (1024**3)
            
            available = free_memory >= min_free_memory_gb
            return available, free_memory, total_memory, gpu_id
        except Exception as e:
            pass  # Silently fail and try next method
    
    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                free_str, total_str = lines[0].split(', ')
                free_memory = float(free_str) / 1024  # Convert MB to GB
                total_memory = float(total_str) / 1024
                available = free_memory >= min_free_memory_gb
                return available, free_memory, total_memory, 0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        pass
    
    # No GPU detected
    return False, 0.0, 0.0, None


def check_cpu_resources():
    """
    Check CPU cores and load.
    
    Returns:
    --------
    tuple: (available_cores, cpu_load_percent)
        available_cores: int - Number of CPU cores
        cpu_load_percent: float - Current CPU load percentage
    """
    available_cores = cpu_count()
    cpu_load_percent = psutil.cpu_percent(interval=0.1)
    return available_cores, cpu_load_percent


def get_current_gpu_memory():
    """
    Get current free GPU memory in GB.
    
    Returns:
    --------
    float or None: Free GPU memory in GB, or None if not available
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_id = 0
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            free_memory = total_memory - reserved
            return free_memory
        except Exception:
            pass
    
    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            free_mb = float(result.stdout.strip().split('\n')[0])
            return free_mb / 1024  # Convert MB to GB
    except Exception:
        pass
    
    return None


def get_gpu_memory_status():
    """
    Get detailed GPU memory status including usage by other processes.
    
    Returns:
    --------
    dict or None: Dictionary with GPU memory info, or None if not available
        Keys: 'total_gb', 'free_gb', 'used_gb', 'processes' (list of dicts with pid, name, memory_gb)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', 
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            total_mb, free_mb, used_mb = map(float, line.split(', '))
            
            # Get process info
            processes = []
            try:
                proc_result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                     '--format=csv,nounits,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if proc_result.returncode == 0:
                    for proc_line in proc_result.stdout.strip().split('\n'):
                        if proc_line.strip():
                            parts = proc_line.split(', ')
                            if len(parts) >= 3:
                                pid, name, mem_mb = parts[0], parts[1], parts[2]
                                try:
                                    processes.append({
                                        'pid': pid,
                                        'name': name,
                                        'memory_gb': float(mem_mb) / 1024
                                    })
                                except ValueError:
                                    pass
            except Exception:
                pass
            
            return {
                'total_gb': total_mb / 1024,
                'free_gb': free_mb / 1024,
                'used_gb': used_mb / 1024,
                'processes': processes
            }
    except Exception:
        pass
    
    return None


def print_gpu_status():
    """Print current GPU memory status."""
    status = get_gpu_memory_status()
    if status:
        print(f"\n  📊 GPU Memory Status:")
        print(f"     Total: {status['total_gb']:.2f} GB")
        print(f"     Free:  {status['free_gb']:.2f} GB")
        print(f"     Used:  {status['used_gb']:.2f} GB ({status['used_gb']/status['total_gb']*100:.1f}%)")
        
        if status['processes']:
            print(f"     Processes using GPU:")
            for proc in status['processes']:
                print(f"       PID {proc['pid']}: {proc['name']} ({proc['memory_gb']:.2f} GB)")
        else:
            print(f"     No other processes detected")
    else:
        # Fallback to simple check
        gpu_free = get_current_gpu_memory()
        if gpu_free is not None:
            print(f"\n  📊 GPU Memory: {gpu_free:.2f} GB free")


def clear_gpu_cache():
    """Clear PyTorch GPU cache if available."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # Also synchronize to ensure cache clearing completes
            torch.cuda.synchronize()
        except Exception:
            pass


def check_gpu_allocation_possible(size_mb=100):
    """
    Check if we can actually allocate a test tensor on GPU.
    This helps detect memory fragmentation issues.
    
    Parameters:
    -----------
    size_mb : float
        Size of test allocation in MB
    
    Returns:
    --------
    bool: True if allocation succeeded, False otherwise
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    
    try:
        # Try to allocate a small test tensor
        test_size = int(size_mb * 1024 * 1024 / 4)  # Convert MB to float32 elements
        test_tensor = torch.zeros(test_size, device='cuda:0', dtype=torch.float32)
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False
        # Other errors, assume it's possible
        return True
    except Exception:
        # Unknown error, assume it's possible
        return True


def aggressive_gpu_cleanup():
    """
    Perform aggressive GPU memory cleanup to handle fragmentation.
    This includes multiple cache clears and synchronization.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return
    
    try:
        # Multiple cache clears to help with fragmentation
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force Python garbage collection
        gc.collect()
        
        # One more cache clear after GC
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def calculate_tile_size(image_shape, available_memory_gb, channels=2, dtype_size=2, 
                        model_memory_gb=2.0, safety_margin=0.25):
    """
    Calculate optimal tile size based on available memory.
    
    Parameters:
    -----------
    image_shape : tuple
        Full image shape (height, width)
    available_memory_gb : float
        Available memory in GB
    channels : int
        Number of channels
    dtype_size : int
        Size of data type in bytes (2 for uint16)
    model_memory_gb : float
        Estimated model memory usage in GB
    safety_margin : float
        Safety margin as fraction (0.25 = 25% buffer)
    
    Returns:
    --------
    tuple: (tile_height, tile_width)
    """
    height, width = image_shape
    
    # Calculate usable memory (account for model and safety margin)
    usable_memory_gb = available_memory_gb - model_memory_gb
    usable_memory_gb *= (1 - safety_margin)
    usable_memory_bytes = usable_memory_gb * (1024**3)
    
    # Memory needed per pixel: channels * dtype_size + intermediate arrays
    # Use 10x multiplier to be very conservative (Cellpose can use significant memory)
    bytes_per_pixel = channels * dtype_size * 10  # 10x for all intermediate arrays and safety
    
    # Calculate max pixels per tile
    max_pixels = usable_memory_bytes / bytes_per_pixel
    
    # HARD LIMIT: Never create tiles larger than 1 GB (very conservative)
    # This prevents issues even if calculation is off
    max_tile_size_bytes = 1.0 * (1024**3)  # 1 GB max per tile
    max_pixels_hard_limit = max_tile_size_bytes / (channels * dtype_size)
    max_pixels = min(max_pixels, max_pixels_hard_limit)
    
    # Calculate tile dimensions (try to keep aspect ratio similar to image)
    aspect_ratio = width / height
    tile_height = int(np.sqrt(max_pixels / aspect_ratio))
    tile_width = int(tile_height * aspect_ratio)
    
    # Round to reasonable sizes (multiples of 256 for efficiency)
    tile_height = max(256, (tile_height // 256) * 256)
    tile_width = max(256, (tile_width // 256) * 256)
    
    # Ensure tiles aren't larger than the image
    tile_height = min(tile_height, height)
    tile_width = min(tile_width, width)
    
    return tile_height, tile_width


def calculate_overlap(tile_size, cell_diameter=None, overlap_percent=None, min_overlap_pixels=100):
    """
    Calculate tile overlap in pixels.
    
    Parameters:
    -----------
    tile_size : int
        Tile size (height or width)
    cell_diameter : float or None
        Expected cell diameter in pixels
    overlap_percent : float or None
        Overlap as percentage of tile size (0.1 = 10%)
    min_overlap_pixels : int
        Minimum overlap in pixels
    
    Returns:
    --------
    int: Overlap in pixels
    """
    if overlap_percent is not None:
        overlap = int(tile_size * overlap_percent)
    elif cell_diameter is not None:
        # Use 2x cell diameter as overlap
        overlap = int(2 * cell_diameter)
    else:
        # Default: 10% of tile size
        overlap = int(tile_size * 0.1)
    
    # Ensure minimum overlap
    overlap = max(overlap, min_overlap_pixels)
    
    # Don't exceed 50% of tile size
    overlap = min(overlap, tile_size // 2)
    
    return overlap


def generate_tiles(image_shape, tile_height, tile_width, overlap_pixels):
    """
    Generate tile coordinates with overlap.
    
    Parameters:
    -----------
    image_shape : tuple
        Full image shape (height, width)
    tile_height : int
        Tile height in pixels
    tile_width : int
        Tile width in pixels
    overlap_pixels : int
        Overlap in pixels
    
    Returns:
    --------
    list: List of (y_start, y_end, x_start, x_end) tuples
    """
    height, width = image_shape
    tiles = []
    
    # Calculate step size (tile size minus overlap)
    step_y = tile_height - overlap_pixels
    step_x = tile_width - overlap_pixels
    
    y = 0
    while y < height:
        y_end = min(y + tile_height, height)
        y_start = max(0, y_end - tile_height)  # Adjust if near edge
        
        x = 0
        while x < width:
            x_end = min(x + tile_width, width)
            x_start = max(0, x_end - tile_width)  # Adjust if near edge
            
            tiles.append((y_start, y_end, x_start, x_end))
            
            if x_end >= width:
                break
            x += step_x
        
        if y_end >= height:
            break
        y += step_y
    
    return tiles


def process_tile(model, img_tile, tile_coords, diameter=None, channels=[0, 1], use_gpu=False):
    """
    Process a single tile with Cellpose.
    
    Parameters:
    -----------
    model : CellposeModel
        Initialized Cellpose model
    img_tile : np.ndarray
        Image tile (channels, height, width)
    tile_coords : tuple
        (y_start, y_end, x_start, x_end) coordinates
    diameter : float or None
        Cell diameter in pixels
    channels : list
        Channel indices [nuclear, cytoplasmic]
    use_gpu : bool
        Whether using GPU
    
    Returns:
    --------
    tuple: (masks, flows, styles, diams, tile_coords)
    """

    def _run_eval(img):
        """
        Wrapper around model.eval that is robust to the number of
        returned values (3 or 4), which can vary across Cellpose
        versions and GPU/CPU paths.
        """
        result = model.eval(img, diameter=diameter)
        # Cellpose typically returns (masks, flows, styles, diams),
        # but in some cases (e.g. very large images / disabled QC)
        # it may return only 3 values.
        if isinstance(result, tuple):
            if len(result) == 4:
                masks, flows, styles, diams = result
            elif len(result) == 3:
                masks, flows, styles = result
                diams = None
            else:
                # Fallback: only masks returned, or unexpected structure
                masks = result[0]
                flows = styles = diams = None
        else:
            # Unexpected non-tuple return; treat as masks only
            masks = result
            flows = styles = diams = None
        return masks, flows, styles, diams

    try:
        # Cellpose v4+ handles channels differently
        # If image has 2 channels, it automatically uses them as [nuclear, cytoplasmic]
        # The channels parameter is deprecated in v4.0.1+
        if img_tile.shape[0] == 2:
            # Two-channel image - Cellpose will auto-detect
            masks, flows, styles, diams = _run_eval(img_tile)
        else:
            # Single channel or more than 2 - specify channels
            result = model.eval(
                img_tile,
                diameter=diameter,
                channels=channels if len(channels) == 2 else None
            )
            # Normalize return shape
            if isinstance(result, tuple):
                if len(result) == 4:
                    masks, flows, styles, diams = result
                elif len(result) == 3:
                    masks, flows, styles = result
                    diams = None
                else:
                    masks = result[0]
                    flows = styles = diams = None
            else:
                masks = result
                flows = styles = diams = None
        return masks, flows, styles, diams, tile_coords
    except RuntimeError as e:
        if "expanded size" in str(e) or "tensor" in str(e).lower():
            # Tensor size mismatch - try without channels parameter
            try:
                masks, flows, styles, diams = _run_eval(img_tile)
                return masks, flows, styles, diams, tile_coords
            except Exception:
                raise RuntimeError(f"Tile processing failed: {e}")
        else:
            raise
    except Exception as e:
        print(f"\nError processing tile {tile_coords}: {e}")
        raise


def stitch_tiles(tile_results, tile_coords_list, image_shape, overlap_pixels):
    """
    Stitch tile results into full image mask.
    
    Parameters:
    -----------
    tile_results : list
        List of (masks, flows, styles, diams, coords) tuples
    tile_coords_list : list
        List of tile coordinates
    image_shape : tuple
        Full image shape (height, width)
    overlap_pixels : int
        Overlap in pixels
    
    Returns:
    --------
    np.ndarray: Full stitched mask
    """
    height, width = image_shape
    full_mask = np.zeros((height, width), dtype=np.uint32)
    
    # Track next available cell ID
    next_cell_id = 1
    
    for masks, flows, styles, diams, (y_start, y_end, x_start, x_end) in tile_results:
        tile_height = y_end - y_start
        tile_width = x_end - x_start
        
        # Calculate overlap regions
        overlap_half = overlap_pixels // 2
        
        # Determine which region to use (center region, avoiding overlaps)
        if len(tile_coords_list) > 1:  # Multiple tiles, use center region
            use_y_start = overlap_half if y_start > 0 else 0
            use_y_end = tile_height - overlap_half if y_end < height else tile_height
            use_x_start = overlap_half if x_start > 0 else 0
            use_x_end = tile_width - overlap_half if x_end < width else tile_width
        else:  # Single tile, use all
            use_y_start, use_y_end = 0, tile_height
            use_x_start, use_x_end = 0, tile_width
        
        # Extract the region to use
        mask_region = masks[use_y_start:use_y_end, use_x_start:use_x_end]
        
        # Remap cell IDs to be unique
        unique_ids = np.unique(mask_region)
        unique_ids = unique_ids[unique_ids > 0]  # Exclude background (0)
        
        remapped_region = mask_region.copy()
        for old_id in unique_ids:
            remapped_region[mask_region == old_id] = next_cell_id
            next_cell_id += 1
        
        # Place in full mask
        full_y_start = y_start + use_y_start
        full_y_end = y_start + use_y_end
        full_x_start = x_start + use_x_start
        full_x_end = x_start + use_x_end
        
        full_mask[full_y_start:full_y_end, full_x_start:full_x_end] = remapped_region
    
    return full_mask


def format_time(seconds):
    """
    Format seconds as human-readable time.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
    
    Returns:
    --------
    str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} hours {minutes} minutes"


def estimate_initial_time(num_tiles, tile_height, tile_width, use_gpu=False, 
                            image_shape=None, channels=2):
    """
    Estimate initial processing time before any tiles are processed.
    Uses heuristics based on typical Cellpose performance.
    
    Parameters:
    -----------
    num_tiles : int
        Number of tiles to process
    tile_height : int
        Tile height in pixels
    tile_width : int
        Tile width in pixels
    use_gpu : bool
        Whether using GPU acceleration
    image_shape : tuple or None
        Full image shape (height, width) for full image processing
    channels : int
        Number of channels
    
    Returns:
    --------
    tuple: (estimated_seconds_min, estimated_seconds_max)
        Range of estimated time in seconds
    """
    # Calculate pixels per tile
    if image_shape is not None:
        # Full image processing
        height, width = image_shape
        total_pixels = height * width * channels
    else:
        # Tiled processing
        pixels_per_tile = tile_height * tile_width * channels
        total_pixels = pixels_per_tile * num_tiles
    
    # Heuristic: Cellpose processing speed (pixels per second)
    # Based on typical performance:
    # - CPU: ~0.5-2 MPixels/sec (varies by CPU)
    # - GPU: ~5-20 MPixels/sec (varies by GPU)
    # Using conservative estimates
    
    if use_gpu:
        # GPU: 5-15 MPixels/sec (conservative range)
        pixels_per_sec_min = 5_000_000
        pixels_per_sec_max = 15_000_000
    else:
        # CPU: 0.5-2 MPixels/sec (conservative range)
        pixels_per_sec_min = 500_000
        pixels_per_sec_max = 2_000_000
    
    # Calculate time estimates
    time_min = total_pixels / pixels_per_sec_max
    time_max = total_pixels / pixels_per_sec_min
    
    # Add overhead for tiling (10-20% for tile I/O and stitching)
    if image_shape is None:  # Tiled processing
        time_min *= 1.1
        time_max *= 1.2
    
    return time_min, time_max


def estimate_time(num_tiles, sample_time_per_tile, processing_mode='sequential', num_workers=1):
    """
    Estimate total processing time.
    
    Parameters:
    -----------
    num_tiles : int
        Total number of tiles
    sample_time_per_tile : float
        Time to process one tile (seconds)
    processing_mode : str
        'sequential' or 'parallel'
    num_workers : int
        Number of parallel workers (for parallel mode)
    
    Returns:
    --------
    float: Estimated total time in seconds
    """
    if processing_mode == 'parallel' and num_workers > 1:
        # Account for parallelization efficiency (assume 80% efficiency)
        effective_workers = num_workers * 0.8
        estimated = (num_tiles * sample_time_per_tile) / effective_workers
    else:
        estimated = num_tiles * sample_time_per_tile
    
    return estimated


def process_tile_wrapper(args):
    """
    Wrapper function for parallel tile processing.
    This needs to be a top-level function for multiprocessing.
    """
    (model_path, img_tile, tile_coords, diameter, channels, use_gpu) = args
    
    # Re-initialize model in worker process
    # Note: Cellpose models can't be pickled, so we'll need to handle this differently
    # For now, we'll process sequentially but with progress tracking
    # This is a limitation - we'll need to use threading or a different approach
    pass  # Will be implemented in the main function


class ProgressTracker:
    """Track and report progress during tile processing."""
    
    def __init__(self, total_tiles):
        self.total_tiles = total_tiles
        self.completed = 0
        self.start_time = time.time()
        self.tile_times = []
    
    def update(self, tile_num, tile_time=None):
        """Update progress after completing a tile."""
        self.completed = tile_num
        if tile_time is not None:
            self.tile_times.append(tile_time)
        
        # Calculate statistics
        elapsed = time.time() - self.start_time
        percent = (self.completed / self.total_tiles) * 100
        
        if self.completed > 0:
            avg_time_per_tile = np.mean(self.tile_times) if self.tile_times else elapsed / self.completed
            remaining_tiles = self.total_tiles - self.completed
            eta_seconds = remaining_tiles * avg_time_per_tile
            eta_str = format_time(eta_seconds)
            tiles_per_min = 60 / avg_time_per_tile if avg_time_per_tile > 0 else 0
        else:
            eta_str = "calculating..."
            tiles_per_min = 0
        
        # Print progress
        print(f"  Progress: {self.completed}/{self.total_tiles} tiles ({percent:.1f}%) | "
              f"ETA: {eta_str} | Speed: {tiles_per_min:.1f} tiles/min", flush=True)
    
    def finish(self):
        """Finish progress tracking and print summary."""
        total_time = time.time() - self.start_time
        print(f"\n  ✅ Completed {self.total_tiles} tiles in {format_time(total_time)}")
        if self.tile_times:
            print(f"  Average time per tile: {format_time(np.mean(self.tile_times))}")


def read_tile_from_files(nuclear_file, cyto_file, y_start, y_end, x_start, x_end):
    """
    Read a tile directly from files without loading full images.
    Uses memory-mapped reading for efficient region access.
    
    Note: For compressed TIFF files (e.g., JPEG compression), memory-mapped
    reading may still require decompressing large portions. If you encounter
    memory issues, try using smaller tile sizes or decompressing the TIFF files.
    """
    try:
        # Use memory-mapped arrays to read only the needed region
        nuc_memmap = memmap(nuclear_file)
        cyto_memmap = memmap(cyto_file)
        
        # Extract the tile region (this only reads that region from disk)
        # Note: For some compression types, this may still read more than needed
        img_nuc_tile = np.array(nuc_memmap[y_start:y_end, x_start:x_end], copy=True)
        img_cyto_tile = np.array(cyto_memmap[y_start:y_end, x_start:x_end], copy=True)
        
        # Close memory maps
        del nuc_memmap, cyto_memmap
        
        return np.stack([img_nuc_tile, img_cyto_tile], axis=0)
    except Exception as e:
        # Fallback: if memmap fails, try TiffFile (may be less efficient)
        print(f"⚠️  Warning: Memory-mapped reading failed, using fallback method: {e}")
        with TiffFile(nuclear_file) as tif:
            img_nuc_tile = tif.asarray(key=0)[y_start:y_end, x_start:x_end]
        with TiffFile(cyto_file) as tif:
            img_cyto_tile = tif.asarray(key=0)[y_start:y_end, x_start:x_end]
        return np.stack([img_nuc_tile, img_cyto_tile], axis=0)


def process_tiles_with_progress(model, imgs, tiles, diameter, channels, use_gpu, 
                                progress_tracker=None, nuclear_file=None, cyto_file=None):
    """
    Process tiles with progress tracking.
    Supports both in-memory array and file-based reading.
    
    Parameters:
    -----------
    model : CellposeModel
        Initialized model
    imgs : np.ndarray or None
        Full image array (None if reading from files)
    tiles : list
        List of tile coordinates
    diameter : float or None
        Cell diameter
    channels : list
        Channel indices
    use_gpu : bool
        Whether using GPU
    progress_tracker : ProgressTracker or None
        Progress tracker instance
    nuclear_file : str or None
        Path to nuclear channel file (if reading from files)
    cyto_file : str or None
        Path to cytoplasmic channel file (if reading from files)
    
    Returns:
    --------
    list: List of tile results
    """
    results = []
    num_tiles = len(tiles)
    read_from_files = (nuclear_file is not None and cyto_file is not None)
    
    # Debug: Track memory usage
    debug_mode = os.environ.get('CELLPOSE_DEBUG', '0') == '1'
    
    # GPU status monitoring
    last_gpu_status_time = time.time()
    gpu_status_interval = 30  # Print GPU status every 30 seconds
    gpu_status_tile_interval = 5  # Also print every 5 tiles
    
    for i, (y_start, y_end, x_start, x_end) in enumerate(tiles):
        tile_start = time.time()
        
        # Check memory before reading tile (always check, warn if low)
        mem_before = psutil.virtual_memory()
        available_gb = mem_before.available / (1024**3)
        
        # Warn if memory is getting low (< 2 GB available)
        if available_gb < 2.0:
            print(f"\n⚠️  WARNING: Low memory before tile {i+1}/{num_tiles} - "
                  f"Only {available_gb:.2f} GB available!")
        
        # Debug: Detailed memory info
        if debug_mode:
            print(f"\n  [DEBUG] Tile {i+1}/{num_tiles}: Before reading - "
                  f"Available: {available_gb:.2f} GB, "
                  f"Used: {mem_before.used / (1024**3):.2f} GB, "
                  f"Percent: {mem_before.percent:.1f}%")
        
        # Check GPU memory before reading tile (if using GPU)
        if use_gpu:
            gpu_free = get_current_gpu_memory()
            if gpu_free is not None:
                if gpu_free < 0.5:  # Less than 500 MB free
                    print(f"\n⚠️  Low GPU memory before tile {i+1}/{num_tiles}: {gpu_free:.2f} GB free")
                    print(f"  Clearing GPU cache...")
                    clear_gpu_cache()
                    # Re-check after clearing
                    gpu_free_after = get_current_gpu_memory()
                    if gpu_free_after is not None and gpu_free_after < 0.3:
                        print(f"  ⚠️  Still low GPU memory: {gpu_free_after:.2f} GB free")
        
        # Extract tile from array or read from files
        if read_from_files:
            img_tile = read_tile_from_files(nuclear_file, cyto_file, y_start, y_end, x_start, x_end)
        else:
            img_tile = imgs[:, y_start:y_end, x_start:x_end]
        
        # Debug: Check memory after reading tile
        if debug_mode:
            mem_after_read = psutil.virtual_memory()
            tile_size_mb = img_tile.nbytes / (1024**2)
            print(f"  [DEBUG] Tile {i+1}/{num_tiles}: After reading ({tile_size_mb:.1f} MB) - "
                  f"Available: {mem_after_read.available / (1024**3):.2f} GB, "
                  f"Used: {mem_after_read.used / (1024**3):.2f} GB")
        
        # Check GPU memory fragmentation after reading tile (if using GPU)
        if use_gpu:
            # Estimate tile memory needs (rough: tile size * channels * dtype * overhead)
            tile_size_mb = img_tile.nbytes / (1024**2)
            estimated_needed_mb = tile_size_mb * 5  # Conservative estimate with overhead
            
            if not check_gpu_allocation_possible(estimated_needed_mb):
                print(f"\n⚠️  GPU memory fragmentation detected - cannot allocate {estimated_needed_mb:.0f} MB")
                print(f"  Performing aggressive cleanup...")
                aggressive_gpu_cleanup()
                
                # Re-check allocation
                if not check_gpu_allocation_possible(estimated_needed_mb):
                    print(f"  ⚠️  Fragmentation persists - may fail during processing")
                    print(f"  💡 Tip: Set PYTORCH_ALLOC_CONF=expandable_segments:True to reduce fragmentation")
        
        # Process tile
        # Print status for first tile (which can take longer due to model warmup)
        if i == 0:
            print(f"\n  🚀 Starting tile 1/{num_tiles} (first tile may take longer due to model initialization)...")
        
        try:
            result = process_tile(model, img_tile, (y_start, y_end, x_start, x_end), 
                                diameter, channels, use_gpu)
            results.append(result)
            
            tile_time = time.time() - tile_start
            
            # Debug: Check memory after processing
            if debug_mode:
                mem_after_process = psutil.virtual_memory()
                print(f"  [DEBUG] Tile {i+1}/{num_tiles}: After processing - "
                      f"Available: {mem_after_process.available / (1024**3):.2f} GB, "
                      f"Used: {mem_after_process.used / (1024**3):.2f} GB")
            
            if progress_tracker:
                progress_tracker.update(i + 1, tile_time)
            
            # Free tile memory immediately
            del img_tile
            del result  # Also free result immediately
            gc.collect()
            
            # Clear GPU cache after each tile (if using GPU)
            if use_gpu:
                clear_gpu_cache()
                
                # Print GPU status periodically
                current_time = time.time()
                should_print_status = (
                    (i + 1) % gpu_status_tile_interval == 0 or  # Every N tiles
                    (current_time - last_gpu_status_time) >= gpu_status_interval  # Every N seconds
                )
                
                if should_print_status:
                    print_gpu_status()
                    last_gpu_status_time = current_time
            
            # Debug: Check memory after cleanup
            if debug_mode:
                mem_after_cleanup = psutil.virtual_memory()
                print(f"  [DEBUG] Tile {i+1}/{num_tiles}: After cleanup - "
                      f"Available: {mem_after_cleanup.available / (1024**3):.2f} GB")
            
        except RuntimeError as e:
            error_str = str(e)
            if "CUDA out of memory" in error_str or "out of memory" in error_str.lower():
                print(f"\n❌ GPU out of memory error processing tile {i+1}/{num_tiles}")
                print(f"  Error: {error_str}")
                
                # Try multiple retry strategies for GPU OOM
                if use_gpu:
                    max_retries = 3
                    retry_successful = False
                    
                    for retry_attempt in range(1, max_retries + 1):
                        print(f"  Retry attempt {retry_attempt}/{max_retries}...")
                        
                        if retry_attempt == 1:
                            # First retry: simple cache clear
                            print(f"    Clearing GPU cache...")
                            clear_gpu_cache()
                            gc.collect()
                        elif retry_attempt == 2:
                            # Second retry: aggressive cleanup
                            print(f"    Performing aggressive GPU cleanup...")
                            aggressive_gpu_cleanup()
                        else:
                            # Third retry: aggressive cleanup + wait a bit
                            print(f"    Final aggressive cleanup with delay...")
                            aggressive_gpu_cleanup()
                            time.sleep(1)  # Brief pause to let GPU settle
                        
                        # Check memory status
                        gpu_free = get_current_gpu_memory()
                        if gpu_free is not None:
                            print(f"    GPU memory: {gpu_free:.2f} GB free")
                        
                        # Try processing again
                        try:
                            result = process_tile(model, img_tile, (y_start, y_end, x_start, x_end), 
                                                diameter, channels, use_gpu)
                            results.append(result)
                            tile_time = time.time() - tile_start
                            if progress_tracker:
                                progress_tracker.update(i + 1, tile_time)
                            del img_tile
                            del result
                            gc.collect()
                            if use_gpu:
                                clear_gpu_cache()
                            print(f"  ✅ Retry {retry_attempt} successful!")
                            retry_successful = True
                            break
                        except RuntimeError as retry_e:
                            retry_error_str = str(retry_e)
                            if "out of memory" in retry_error_str.lower():
                                print(f"    Retry {retry_attempt} failed: still OOM")
                                if retry_attempt < max_retries:
                                    continue
                            else:
                                # Different error, re-raise
                                raise
                        except Exception as retry_e:
                            print(f"    Retry {retry_attempt} failed: {retry_e}")
                            if retry_attempt < max_retries:
                                continue
                            else:
                                raise
                    
                    if not retry_successful:
                        # All retries failed
                        # Calculate tile size from coordinates for error message
                        current_tile_height = y_end - y_start
                        current_tile_width = x_end - x_start
                        print(f"\n  ❌ All {max_retries} retry attempts failed")
                        print(f"  💡 Suggestions:")
                        print(f"     - Use smaller tile size (current tile: {current_tile_height} × {current_tile_width})")
                        print(f"     - Set PYTORCH_ALLOC_CONF=expandable_segments:True")
                        print(f"     - Process on CPU instead (remove --gpu flag)")
                        print(f"     - Free GPU memory from other processes")
                        raise RuntimeError(f"GPU OOM error persisted after {max_retries} retry attempts. "
                                         f"Original error: {error_str}")
                else:
                    raise
            else:
                raise
        except Exception as e:
            print(f"\n❌ Error processing tile {i+1}/{num_tiles}: {e}")
            if debug_mode:
                mem_error = psutil.virtual_memory()
                print(f"  [DEBUG] Memory at error: Available: {mem_error.available / (1024**3):.2f} GB")
            raise
    
    if progress_tracker:
        progress_tracker.finish()
    
    return results


def run_segmentation(nuclear_file, cyto_file, output_dir='output', use_gpu=False, 
                    model_type='cyto3', diameter=None, tile_size=None, overlap=None,
                    max_workers=None, force_cpu=False, gpu_memory_limit=2.0):
    """
    Run Cellpose segmentation on nuclear and cytoplasmic channels with resource-aware tiling.
    
    Parameters:
    -----------
    nuclear_file : str
        Path to nuclear channel (DAPI) TIF file
    cyto_file : str
        Path to cytoplasmic channel (VIM) TIF file
    output_dir : str
        Directory to save output masks
    use_gpu : bool
        Whether to request GPU (will check availability)
    model_type : str
        Cellpose model type (default: 'cyto3')
    diameter : float or None
        Cell diameter in pixels (None = auto-detect)
    tile_size : tuple or None
        Manual tile size (height, width) in pixels, or None for auto
    overlap : int or float or None
        Tile overlap in pixels, percentage (0.1 = 10%), or None for auto
    max_workers : int or None
        Maximum parallel workers (None = auto-detect)
    force_cpu : bool
        Force CPU even if GPU is available
    gpu_memory_limit : float
        Minimum free GPU memory required to use GPU (GB)
    """
    print("=" * 70)
    print("Cellpose Segmentation - Resource-Aware Processing")
    print("=" * 70)
    
    # Resource detection
    available_gb, total_gb = check_memory()
    print(f"\n📊 System Resources:")
    print(f"  RAM: {total_gb:.2f} GB total, {available_gb:.2f} GB available")
    
    # Check GPU availability
    gpu_available, gpu_free_gb, gpu_total_gb, gpu_id = check_gpu_availability(gpu_memory_limit)
    if gpu_available and not force_cpu:
        print(f"  GPU: Available ({gpu_free_gb:.2f} GB free / {gpu_total_gb:.2f} GB total)")
        if use_gpu:
            actual_use_gpu = True
            print(f"  ✅ Will use GPU (device {gpu_id})")
        else:
            actual_use_gpu = False
            print(f"  ℹ️  GPU available but not requested (use --gpu to enable)")
    else:
        if force_cpu:
            print(f"  ℹ️  CPU forced (--force-cpu)")
        elif not gpu_available:
            print(f"  ⚠️  GPU not available or insufficient memory (< {gpu_memory_limit} GB free)")
        actual_use_gpu = False
        gpu_free_gb = 0.0  # Set to 0 if GPU not available
    
    # Check CPU resources
    cpu_cores, cpu_load = check_cpu_resources()
    print(f"  CPU: {cpu_cores} cores, {cpu_load:.1f}% load")
    
    # Check file sizes
    if not os.path.exists(nuclear_file):
        raise FileNotFoundError(f"Nuclear channel file not found: {nuclear_file}")
    if not os.path.exists(cyto_file):
        raise FileNotFoundError(f"Cytoplasmic channel file not found: {cyto_file}")
    
    nuc_size_gb = os.path.getsize(nuclear_file) / (1024**3)
    cyto_size_gb = os.path.getsize(cyto_file) / (1024**3)
    print(f"\n📁 Input Files:")
    print(f"  Nuclear channel: {nuc_size_gb:.2f} GB")
    print(f"  Cytoplasmic channel: {cyto_size_gb:.2f} GB")
    
    # Get image shape from metadata WITHOUT loading full images
    print(f"\n📐 Reading image metadata...")
    try:
        with TiffFile(nuclear_file) as tif:
            image_shape = tif.pages[0].shape
            dtype_size = tif.pages[0].dtype.itemsize
        print(f"  Image shape: {image_shape}, dtype size: {dtype_size} bytes")
    except Exception as e:
        print(f"  ⚠️  Could not read metadata, will try loading sample: {e}")
        # Fallback: read small sample
        try:
            sample = imread(nuclear_file)[:100, :100]
            image_shape = sample.shape
            dtype_size = sample.itemsize
            del sample
            gc.collect()
        except Exception as e2:
            print(f"  ❌ Could not read image: {e2}")
            raise
    
    # Estimate memory needed BEFORE loading
    # Two images + stacked array + processing overhead
    single_image_memory_gb = (image_shape[0] * image_shape[1] * dtype_size) / (1024**3)
    estimated_total_memory_gb = single_image_memory_gb * 6  # Conservative: 2 images + stack + 3x overhead
    model_memory_gb = 2.0 if actual_use_gpu else 1.0
    total_needed_gb = estimated_total_memory_gb + model_memory_gb
    
    # Determine if tiling is needed BEFORE loading
    # IMPORTANT: when using GPU, we should compare against GPU memory,
    # not just system RAM, otherwise we may try to process a full image
    # that fits in RAM but does NOT fit in GPU memory.
    use_tiling = False
    tile_height = None
    tile_width = None
    
    # Effective memory budget for deciding tiling
    # - On CPU: use available system RAM
    # - On GPU: use the minimum of system RAM and free GPU memory
    if actual_use_gpu and gpu_free_gb > 0:
        memory_limit_gb = min(available_gb, gpu_free_gb)
    else:
        memory_limit_gb = available_gb
    
    if tile_size is not None:
        use_tiling = True
        tile_height, tile_width = tile_size
        print(f"\n🔲 Using manual tile size: {tile_height} × {tile_width} pixels")
    elif estimated_total_memory_gb > memory_limit_gb * 0.5:  # Use 50% threshold (conservative)
        use_tiling = True
        if actual_use_gpu and gpu_free_gb > 0:
            print(f"\n⚠️  Images too large for available GPU memory "
                  f"(estimated {estimated_total_memory_gb:.2f} GB needed, {gpu_free_gb:.2f} GB GPU free)")
        else:
            print(f"\n⚠️  Images too large for available memory "
                  f"(estimated {estimated_total_memory_gb:.2f} GB needed, {available_gb:.2f} GB available)")
        print(f"  🔲 Will use automatic tiling (reading tiles on-demand from files)")
        
        # Calculate optimal tile size
        memory_for_tiling = memory_limit_gb
        tile_height, tile_width = calculate_tile_size(
            image_shape, memory_for_tiling, channels=2, dtype_size=dtype_size, 
            model_memory_gb=model_memory_gb
        )
        print(f"  Calculated tile size: {tile_height} × {tile_width} pixels")
    else:
        print(f"\n✅ Sufficient memory available (estimated {estimated_total_memory_gb:.2f} GB needed)")
        print(f"  Will process full image without tiling")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load images only if NOT using tiling
    imgs = None
    if not use_tiling:
        print(f"\n📥 Loading images...")
        try:
            img_nucleus = imread(nuclear_file)
            print(f"  Nuclear: {img_nucleus.shape}, dtype: {img_nucleus.dtype}")
            gc.collect()
            
            img_cyto = imread(cyto_file)
            print(f"  Cytoplasmic: {img_cyto.shape}, dtype: {img_cyto.dtype}")
            gc.collect()
            
            # Combine channels
            print(f"\n🔗 Stacking channels...")
            imgs = np.stack([img_nucleus, img_cyto], axis=0)
            print(f"  Combined shape: {imgs.shape}")
            
            # Free individual arrays
            del img_nucleus, img_cyto
            gc.collect()
        except MemoryError as e:
            print(f"❌ ERROR: Insufficient memory to load images: {e}")
            print(f"  Falling back to tiling mode...")
            use_tiling = True
            memory_for_tiling = available_gb if not actual_use_gpu else min(available_gb, gpu_free_gb if gpu_free_gb > 0 else available_gb)
            tile_height, tile_width = calculate_tile_size(
                image_shape, memory_for_tiling, channels=2, dtype_size=dtype_size, 
                model_memory_gb=model_memory_gb
            )
            print(f"  Using tile size: {tile_height} × {tile_width} pixels")
    
    # Initialize model
    print(f"\n🤖 Initializing Cellpose model...")
    print(f"  Model type: {model_type}")
    print(f"  Device: {'GPU' if actual_use_gpu else 'CPU'}")
    
    try:
        model = models.CellposeModel(gpu=actual_use_gpu, model_type=model_type)
    except Exception as e:
        if actual_use_gpu:
            print(f"⚠️  GPU initialization failed: {e}")
            print(f"  Falling back to CPU...")
            actual_use_gpu = False
            model = models.CellposeModel(gpu=False, model_type=model_type)
        else:
            raise
    
    # Initialize tiling variables
    tiles = None
    overlap_pixels = None
    num_tiles = 0
    
    # Process with or without tiling
    if use_tiling:
        # Calculate overlap
        if overlap is None:
            # Auto-calculate overlap
            if diameter is not None:
                overlap_pixels = calculate_overlap(min(tile_height, tile_width), 
                                                  cell_diameter=diameter)
            else:
                overlap_pixels = calculate_overlap(min(tile_height, tile_width))
        elif isinstance(overlap, float) and 0 < overlap < 1:
            # Percentage
            overlap_pixels = int(min(tile_height, tile_width) * overlap)
        else:
            # Pixels
            overlap_pixels = int(overlap)
        
        print(f"  Tile overlap: {overlap_pixels} pixels")
        
        # Generate tiles
        tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
        num_tiles = len(tiles)
        print(f"  Generated {num_tiles} tiles")
        
        # Check GPU memory one more time before processing (in case other processes started)
        if actual_use_gpu:
            current_gpu_free = get_current_gpu_memory()
            if current_gpu_free is not None:
                # Estimate memory needed per tile (rough: tile size * 5x overhead)
                tile_pixels = tile_height * tile_width
                tile_memory_gb = (tile_pixels * 2 * 2 * 5) / (1024**3)  # channels * dtype * overhead
                
                if current_gpu_free < tile_memory_gb * 1.5:  # Need at least 1.5x tile size free
                    print(f"\n⚠️  WARNING: GPU memory is low ({current_gpu_free:.2f} GB free)")
                    print(f"  Estimated {tile_memory_gb:.2f} GB needed per tile")
                    print(f"  Other processes may be using GPU memory")
                    
                    # Try to reduce tile size if possible
                    if tile_height > 4096 and tile_width > 4096:
                        print(f"  🔄 Reducing tile size to fit available memory...")
                        reduction_factor = min(2, (current_gpu_free / tile_memory_gb) * 0.8)
                        tile_height = max(2048, int(tile_height / reduction_factor))
                        tile_width = max(2048, int(tile_width / reduction_factor))
                        # Round to multiples of 256
                        tile_height = (tile_height // 256) * 256
                        tile_width = (tile_width // 256) * 256
                        
                        # Recalculate overlap and tiles
                        overlap_pixels = calculate_overlap(min(tile_height, tile_width))
                        tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                        num_tiles = len(tiles)
                        print(f"  ✅ Reduced to {tile_height} × {tile_width} pixels, {num_tiles} tiles")
                    else:
                        print(f"  ⚠️  Tile size already small - may fail with OOM errors")
                        print(f"  💡 Consider: Set PYTORCH_ALLOC_CONF=expandable_segments:True")
                        print(f"  💡 Or: Wait for other GPU processes to finish")
                        print(f"  💡 Or: Use CPU instead (remove --gpu flag)")
        
        # Process all tiles with progress tracking
        print(f"\n🔄 Processing {num_tiles} tiles...")
        # Show initial time estimate
        time_min, time_max = estimate_initial_time(num_tiles, tile_height, tile_width, 
                                                    actual_use_gpu, channels=2)
        print(f"  ⏱️  Estimated time: {format_time(time_min)} - {format_time(time_max)} "
              f"(based on {'GPU' if actual_use_gpu else 'CPU'} processing)")
        
        progress_tracker = ProgressTracker(num_tiles)
        
        try:
            tile_results = process_tiles_with_progress(
                model, imgs, tiles, diameter, [0, 1], actual_use_gpu, progress_tracker,
                nuclear_file=nuclear_file if use_tiling else None,
                cyto_file=cyto_file if use_tiling else None
            )
            
            # Stitch tiles
            print(f"\n🔗 Stitching {num_tiles} tiles into full mask...")
            masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels)
            
            # Calculate average diameter from all tiles
            all_diams = []
            for _, _, _, diams, _ in tile_results:
                if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                    all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
            avg_diameter = np.mean(all_diams) if all_diams else None
            
        except Exception as e:
            print(f"\n❌ Error during tiled processing: {e}")
            # Try with smaller tiles as fallback
            if tile_height > 512 and tile_width > 512:
                print(f"  Attempting fallback with smaller tiles...")
                tile_height = tile_height // 2
                tile_width = tile_width // 2
                overlap_pixels = overlap_pixels // 2
                tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                print(f"  New tile size: {tile_height} × {tile_width}, {len(tiles)} tiles")
                
                progress_tracker = ProgressTracker(len(tiles))
                tile_results = process_tiles_with_progress(
                    model, imgs, tiles, diameter, [0, 1], actual_use_gpu, progress_tracker,
                    nuclear_file=nuclear_file if use_tiling else None,
                    cyto_file=cyto_file if use_tiling else None
                )
                masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels)
                
                all_diams = []
                for _, _, _, diams, _ in tile_results:
                    if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                        all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                avg_diameter = np.mean(all_diams) if all_diams else None
            else:
                raise
        
        flows = None  # Not stitched for now
        styles = None
        diams = avg_diameter
        
    else:
        # Process full image
        print(f"\n🔄 Running segmentation on full image...")
        # Show initial time estimate
        time_min, time_max = estimate_initial_time(1, None, None, actual_use_gpu, 
                                                    image_shape=image_shape, channels=2)
        print(f"  ⏱️  Estimated time: {format_time(time_min)} - {format_time(time_max)} "
              f"(based on {'GPU' if actual_use_gpu else 'CPU'} processing)")
        
        try:
            # Cellpose v4+ auto-detects channels for 2-channel images
            if imgs.shape[0] == 2:
                result = model.eval(imgs, diameter=diameter)
            else:
                result = model.eval(
                    imgs,
                    diameter=diameter,
                    channels=[0, 1] if len(imgs.shape) == 3 else None
                )
            # Normalize return shape (3 or 4 values)
            if isinstance(result, tuple):
                if len(result) == 4:
                    masks, flows, styles, diams = result
                elif len(result) == 3:
                    masks, flows, styles = result
                    diams = None
                else:
                    masks = result[0]
                    flows = styles = diams = None
            else:
                masks = result
                flows = styles = diams = None
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                print(f"\n⚠️  GPU memory error: {e}")
                print(f"  Falling back to CPU with tiling...")
                actual_use_gpu = False
                use_tiling = True
                model = models.CellposeModel(gpu=False, model_type=model_type)
                
                # Recalculate with CPU
                tile_height, tile_width = calculate_tile_size(
                    image_shape, available_gb, channels=2, dtype_size=2, 
                    model_memory_gb=1.0
                )
                overlap_pixels = calculate_overlap(min(tile_height, tile_width), 
                                                  cell_diameter=diameter)
                tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                num_tiles = len(tiles)
                
                print(f"  Using tiles: {tile_height} × {tile_width}, {num_tiles} tiles")
                progress_tracker = ProgressTracker(num_tiles)
                tile_results = process_tiles_with_progress(
                    model, None, tiles, diameter, [0, 1], False, progress_tracker,
                    nuclear_file=nuclear_file,
                    cyto_file=cyto_file
                )
                masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels)
                
                all_diams = []
                for _, _, _, diams, _ in tile_results:
                    if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                        all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                diams = np.mean(all_diams) if all_diams else None
                flows = None
                styles = None
            else:
                raise
    
    # Calculate results
    n_cells = len(np.unique(masks)) - 1  # Subtract 1 for background (0)
    print(f"\n✅ Segmentation complete!")
    print(f"  Found {n_cells} cells")
    if diams is not None:
        if isinstance(diams, (list, np.ndarray)):
            print(f"  Average cell diameter: {np.mean(diams):.2f} pixels")
        else:
            print(f"  Average cell diameter: {diams:.2f} pixels")
    
    # Save outputs
    masks_file = output_path / "cellpose_masks.tif"
    print(f"\n💾 Saving results...")
    print(f"  Masks: {masks_file}")
    imwrite(masks_file, masks)
    
    # Save summary
    summary_file = output_path / "segmentation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Cellpose Segmentation Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"GPU used: {actual_use_gpu}\n")
        f.write(f"Tiling used: {use_tiling}\n")
        if use_tiling and tiles is not None:
            f.write(f"Number of tiles: {num_tiles}\n")
            f.write(f"Tile size: {tile_height} × {tile_width}\n")
            f.write(f"Tile overlap: {overlap_pixels} pixels\n")
        f.write(f"Number of cells detected: {n_cells}\n")
        if imgs is not None:
            f.write(f"Image shape: {imgs.shape}\n")
        else:
            f.write(f"Image shape: {image_shape}\n")
        if diams is not None:
            if isinstance(diams, (list, np.ndarray)):
                f.write(f"Average cell diameter: {np.mean(diams):.2f} pixels\n")
            else:
                f.write(f"Average cell diameter: {diams:.2f} pixels\n")
        f.write(f"Masks saved to: {masks_file}\n")
    
    print(f"  Summary: {summary_file}")
    print(f"\n✅ All outputs saved to: {output_path}")
    
    # Cleanup
    if imgs is not None:
        del imgs
    gc.collect()
    
    return masks, flows, styles, diams


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Run Cellpose segmentation on DAPI and VIM channels with resource-aware tiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif -o results --gpu
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif -m cyto2 -d 30
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif --tile-size 2048 2048
  python cellpose_segmentation.py input/DAPI.tif input/VIM.tif --overlap 200 --gpu-memory-limit 4.0

Debug mode (for troubleshooting memory issues):
  CELLPOSE_DEBUG=1 python cellpose_segmentation.py input/DAPI.tif input/VIM.tif
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
                       help='Use GPU acceleration (if available and sufficient memory)')
    parser.add_argument('--tile-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                       default=None,
                       help='Manual tile size in pixels: HEIGHT WIDTH (e.g., --tile-size 2048 2048)')
    parser.add_argument('--overlap', type=float, default=None,
                       help='Tile overlap: pixels (int), percentage 0-1 (float), or None for auto')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum parallel workers (None = auto-detect, currently sequential)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU even if GPU is available')
    parser.add_argument('--gpu-memory-limit', type=float, default=2.0,
                       help='Minimum free GPU memory required to use GPU in GB (default: 2.0)')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Skip interactive prompts and continue automatically')
    
    args = parser.parse_args()
    
    # Parse tile size
    tile_size = None
    if args.tile_size:
        tile_size = tuple(args.tile_size)
    
    try:
        run_segmentation(
            nuclear_file=args.nuclear_file,
            cyto_file=args.cyto_file,
            output_dir=args.output,
            use_gpu=args.gpu,
            model_type=args.model,
            diameter=args.diameter,
            tile_size=tile_size,
            overlap=args.overlap,
            max_workers=args.max_workers,
            force_cpu=args.force_cpu,
            gpu_memory_limit=args.gpu_memory_limit
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()