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
    Get current GPU memory usage in GB.
    
    Returns:
    --------
    tuple or None: (free_gb, used_gb, total_gb) if available, or None
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            gpu_id = 0
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            free_memory = total_memory - reserved
            # Show allocated (active tensors) as "used" for more dynamic display
            # Reserved includes cached memory which stays constant
            used_memory = allocated if allocated > 0 else reserved
            return (free_memory, used_memory, total_memory)
        except Exception:
            pass
    
    # Try nvidia-smi as fallback
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', 
             '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            total_mb, free_mb, used_mb = map(float, line.split(', '))
            return (free_mb / 1024, used_mb / 1024, total_mb / 1024)
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
    """Print current GPU memory status and system RAM."""
    # Get system RAM status
    sys_mem = psutil.virtual_memory()
    sys_used_gb = sys_mem.used / (1024**3)
    sys_avail_gb = sys_mem.available / (1024**3)
    sys_total_gb = sys_mem.total / (1024**3)
    sys_pct = sys_mem.percent
    
    status = get_gpu_memory_status()
    if status:
        print(f"\n  📊 Memory Status:")
        print(f"     GPU VRAM: {status['used_gb']:.2f} GB used ({status['used_gb']/status['total_gb']*100:.1f}%) | "
              f"{status['free_gb']:.2f} GB free (Total: {status['total_gb']:.2f} GB)")
        print(f"     System RAM: {sys_used_gb:.1f} GB used ({sys_pct:.1f}%) | "
              f"{sys_avail_gb:.1f} GB available (Total: {sys_total_gb:.1f} GB)")
        
        if status['processes']:
            print(f"     Processes using GPU:")
            for proc in status['processes']:
                print(f"       PID {proc['pid']}: {proc['name']} ({proc['memory_gb']:.2f} GB)")
    else:
        # Fallback to simple check - now shows used instead of just free
        gpu_mem = get_current_gpu_memory()
        if gpu_mem is not None:
            free_gb, used_gb, total_gb = gpu_mem
            pct_used = (used_gb / total_gb * 100) if total_gb > 0 else 0
            print(f"\n  📊 Memory Status:")
            print(f"     GPU VRAM: {used_gb:.2f} GB used ({pct_used:.1f}%) | "
                  f"{free_gb:.2f} GB free (Total: {total_gb:.2f} GB)")
            print(f"     System RAM: {sys_used_gb:.1f} GB used ({sys_pct:.1f}%) | "
                  f"{sys_avail_gb:.1f} GB available (Total: {sys_total_gb:.1f} GB)")
        else:
            # No GPU info available, show system RAM only
            print(f"\n  📊 System RAM: {sys_used_gb:.1f} GB used ({sys_pct:.1f}%) | "
                  f"{sys_avail_gb:.1f} GB available (Total: {sys_total_gb:.1f} GB)")


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


def wait_for_gpu_memory(required_gb, timeout_seconds=300, check_interval=10, current_pid=None):
    """
    Wait for GPU memory to become available, checking if other processes are using it.
    
    Parameters:
    -----------
    required_gb : float
        Required GPU memory in GB
    timeout_seconds : int
        Maximum time to wait in seconds (default: 5 minutes)
    check_interval : int
        Seconds between checks (default: 10)
    current_pid : int or None
        Current process PID to exclude from "other processes" check
    
    Returns:
    --------
    tuple: (success, final_free_gb, waited_seconds)
        success: bool - True if enough memory became available
        final_free_gb: float - Final free GPU memory in GB
        waited_seconds: float - Total time waited
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False, 0.0, 0.0
    
    start_time = time.time()
    last_status_time = start_time
    status_interval = 30  # Print status every 30 seconds
    
    print(f"\n⏳ Waiting for GPU memory to become available...")
    print(f"   Required: {required_gb:.2f} GB")
    print(f"   Timeout: {timeout_seconds} seconds")
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed >= timeout_seconds:
            gpu_mem = get_current_gpu_memory()
            final_free = gpu_mem[0] if gpu_mem else None
            print(f"\n⏰ Timeout reached ({timeout_seconds}s). Proceeding with available memory.")
            return False, final_free if final_free else 0.0, elapsed
        
        # Check current GPU status
        status = get_gpu_memory_status()
        gpu_mem = get_current_gpu_memory()
        current_free = gpu_mem[0] if gpu_mem else None
        
        if current_free and current_free >= required_gb:
            print(f"\n✅ GPU memory available! ({current_free:.2f} GB free)")
            return True, current_free, elapsed
        
        # Print status periodically
        if time.time() - last_status_time >= status_interval:
            if status:
                print(f"\n  ⏳ Still waiting... ({elapsed:.0f}s elapsed)")
                print(f"     Current free: {status['free_gb']:.2f} GB (need {required_gb:.2f} GB)")
                if status['processes']:
                    other_procs = [p for p in status['processes'] 
                                 if current_pid is None or str(p['pid']) != str(current_pid)]
                    if other_procs:
                        print(f"     Other processes using GPU:")
                        for proc in other_procs:
                            print(f"       PID {proc['pid']}: {proc['name']} ({proc['memory_gb']:.2f} GB)")
            else:
                print(f"  ⏳ Still waiting... ({elapsed:.0f}s elapsed, {current_free:.2f} GB free)")
            last_status_time = time.time()
        
        time.sleep(check_interval)


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


def calculate_tile_size_from_grid(image_shape, grid_rows, grid_cols, overlap_pixels):
    """
    Calculate tile size from grid dimensions.
    
    Parameters:
    -----------
    image_shape : tuple
        Full image shape (height, width)
    grid_rows : int
        Number of rows in the grid
    grid_cols : int
        Number of columns in the grid
    overlap_pixels : int
        Overlap in pixels between adjacent tiles
    
    Returns:
    --------
    tuple: (tile_height, tile_width)
    """
    height, width = image_shape
    grid_rows = max(1, int(grid_rows))
    grid_cols = max(1, int(grid_cols))
    overlap_pixels = max(0, int(overlap_pixels))
    
    # Account for overlap when calculating tile sizes. Each adjacent boundary
    # is "covered twice" by overlap in the tile generation step size.
    effective_height = height + (overlap_pixels * (grid_rows - 1))
    effective_width = width + (overlap_pixels * (grid_cols - 1))
    
    tile_height = int(np.ceil(effective_height / grid_rows))
    tile_width = int(np.ceil(effective_width / grid_cols))
    
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


def generate_grid_tiles(image_shape, grid_rows, grid_cols, overlap_pixels):
    """
    Generate an exact grid_rows × grid_cols set of tiles covering the image,
    expanding each tile by ~overlap_pixels (split as half on each side)
    except at the outer image borders.
    
    This is used by the progressive grid strategy so that a 2×2 grid
    always produces exactly 4 tiles, 3×3 → 9 tiles, etc.
    """
    height, width = image_shape
    grid_rows = max(1, int(grid_rows))
    grid_cols = max(1, int(grid_cols))
    overlap_pixels = max(0, int(overlap_pixels))
    overlap_half = overlap_pixels // 2
    
    # Core (non-overlapping) partition
    base_h = int(np.ceil(height / grid_rows))
    base_w = int(np.ceil(width / grid_cols))
    
    tiles = []
    for r in range(grid_rows):
        core_y0 = r * base_h
        core_y1 = min((r + 1) * base_h, height)
        # Expand by overlap, but don't go outside image or beyond neighbors
        y0 = core_y0 if r == 0 else max(0, core_y0 - overlap_half)
        y1 = core_y1 if r == grid_rows - 1 else min(height, core_y1 + overlap_half)
        
        for c in range(grid_cols):
            core_x0 = c * base_w
            core_x1 = min((c + 1) * base_w, width)
            x0 = core_x0 if c == 0 else max(0, core_x0 - overlap_half)
            x1 = core_x1 if c == grid_cols - 1 else min(width, core_x1 + overlap_half)
            
            tiles.append((y0, y1, x0, x1))
    
    return tiles


def _is_oom_error(exc: BaseException) -> bool:
    """
    Best-effort detection of out-of-memory conditions.
    Used only for the progressive grid retry loop so we don't retry on unrelated bugs.
    """
    msg = str(exc).lower()
    markers = (
        "cuda out of memory",
        "out of memory",
        "gpu oom",
        "exit code 137",
        "return code 137",
        "killed",   # sometimes appears when the OS OOM-kills the process
        "signal 9",
    )
    return any(m in msg for m in markers)


def _cleanup_temp_dir(temp_dir):
    """Best-effort cleanup of temporary tile cache directory."""
    if not temp_dir:
        return
    try:
        if os.path.isdir(temp_dir):
            for name in os.listdir(temp_dir):
                p = os.path.join(temp_dir, name)
                try:
                    os.remove(p)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass
        elif os.path.exists(temp_dir):
            try:
                os.remove(temp_dir)
            except Exception:
                pass
    except Exception:
        pass


def process_with_progressive_grid(
    model,
    imgs,
    image_shape,
    dtype_size,
    diameter,
    actual_use_gpu,
    overlap,
    nuclear_file,
    cyto_file,
    output_dir,
    grid_size=None,
    max_grid_size=10,
):
    """
    Progressive grid tiling strategy:
    Start with N×N grid (default N=2). If an OOM occurs, increase to (N+1)×(N+1) and retry.
    Only retries on OOM-like errors. Other exceptions are re-raised immediately.
    
    Returns:
        (tile_results, tiles, tile_height, tile_width, overlap_pixels, use_disk_storage, temp_dir)
    """
    start_n = int(grid_size) if grid_size is not None else 2
    start_n = max(1, start_n)
    max_n = int(max_grid_size) if max_grid_size is not None else 10
    max_n = max(start_n, max(1, max_n))
    last_error = None

    for n in range(start_n, max_n + 1):
        # First pass: estimate core tile size without overlap, used only for overlap calculation.
        base_tile_h = int(np.ceil(image_shape[0] / n))
        base_tile_w = int(np.ceil(image_shape[1] / n))

        # Determine overlap for this attempt (depends on approximate tile size).
        if overlap is None:
            if diameter is not None:
                overlap_pixels = calculate_overlap(min(base_tile_h, base_tile_w), cell_diameter=diameter)
            else:
                overlap_pixels = calculate_overlap(min(base_tile_h, base_tile_w))
        elif isinstance(overlap, float) and 0 < overlap < 1:
            overlap_pixels = int(min(base_tile_h, base_tile_w) * overlap)
        else:
            overlap_pixels = int(overlap)
        overlap_pixels = max(0, int(overlap_pixels))

        # Generate an exact n×n grid of tiles with this overlap.
        tiles = generate_grid_tiles(image_shape, n, n, overlap_pixels)
        num_tiles = len(tiles)
        # Use the first tile to report tile size.
        if tiles:
            y0, y1, x0, x1 = tiles[0]
            tile_height = y1 - y0
            tile_width = x1 - x0
        else:
            tile_height = base_tile_h
            tile_width = base_tile_w

        print(f"\n🧩 Progressive grid attempt: {n}×{n} ({num_tiles} tiles)", flush=True)
        print(f"  Tile size: {tile_height} × {tile_width} pixels", flush=True)
        print(f"  Tile overlap: {overlap_pixels} pixels", flush=True)

        # Decide whether to use disk storage for this attempt (same heuristics as before).
        use_disk_storage = False
        temp_dir = None
        mem_check = psutil.virtual_memory()
        available_gb = mem_check.available / (1024**3)

        pixels_per_tile = tile_height * tile_width
        bytes_per_tile = pixels_per_tile * 4 * 2  # masks + flows estimate
        estimated_per_tile_gb = bytes_per_tile / (1024**3)
        estimated_tile_results_gb = estimated_per_tile_gb * num_tiles * 1.5

        should_use_disk = (
            estimated_tile_results_gb > available_gb * 0.5 or
            num_tiles > 50 or
            available_gb < 30.0
        )
        if should_use_disk:
            use_disk_storage = True
            import tempfile
            if output_dir:
                temp_dir = os.path.join(output_dir, '.tile_cache')
                os.makedirs(temp_dir, exist_ok=True)
            else:
                temp_dir = tempfile.mkdtemp(prefix='cellpose_tiles_')
            print(f"\n  💾 Auto-enabled disk storage for tile results:", flush=True)
            print(f"     Reason: Estimated {estimated_tile_results_gb:.1f} GB needed | "
                  f"{available_gb:.1f} GB available | {num_tiles} tiles", flush=True)
            print(f"     Location: {temp_dir}", flush=True)

        # Proactive memory check: if available memory is too low, proactively switch to denser grid
        mem_check = psutil.virtual_memory()
        available_gb = mem_check.available / (1024**3)
        # Estimate memory needed for processing (tile data + model + overhead)
        estimated_processing_gb = estimated_per_tile_gb * 2  # 2 tiles in memory at once (current + next)
        estimated_processing_gb += 5.0  # Model and overhead
        
        if available_gb < estimated_processing_gb * 1.5:  # Need 1.5x buffer
            print(f"\n  ⚠️  Proactive OOM prevention: Available memory ({available_gb:.1f} GB) may be insufficient", flush=True)
            print(f"     Estimated need: {estimated_processing_gb:.1f} GB for processing", flush=True)
            print(f"  🧹 Cleaning up and retrying with {(n+1)}×{(n+1)}...", flush=True)
            try:
                if actual_use_gpu:
                    clear_gpu_cache()
                gc.collect()
            except Exception:
                pass
            _cleanup_temp_dir(temp_dir)
            continue

        progress_tracker = ProgressTracker(num_tiles)

        try:
            tile_results = process_tiles_with_progress(
                model, imgs, tiles, diameter, [0, 1], actual_use_gpu, progress_tracker,
                nuclear_file=nuclear_file,
                cyto_file=cyto_file,
                use_disk_storage=use_disk_storage,
                temp_dir=temp_dir
            )
            return tile_results, tiles, tile_height, tile_width, overlap_pixels, use_disk_storage, temp_dir

        except Exception as e:
            last_error = e
            if _is_oom_error(e):
                print(f"\n  ❌ OOM detected during {n}×{n} attempt: {e}", flush=True)
                print(f"  🧹 Cleaning up and retrying with {(n+1)}×{(n+1)}...", flush=True)
                try:
                    if actual_use_gpu:
                        clear_gpu_cache()
                    gc.collect()
                except Exception:
                    pass
                _cleanup_temp_dir(temp_dir)
                continue

            _cleanup_temp_dir(temp_dir)
            raise

    raise RuntimeError(
        f"All progressive grid attempts failed with OOM from {start_n}×{start_n} to {max_n}×{max_n}. "
        f"Last error: {last_error}"
    )


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


def stitch_tiles(tile_results, tile_coords_list, image_shape, overlap_pixels, use_disk_storage=False, temp_dir=None):
    """
    Stitch tile results into full image mask.
    
    Parameters:
    -----------
    tile_results : list
        List of (masks, flows, styles, diams, coords) tuples OR file paths if use_disk_storage
    tile_coords_list : list
        List of tile coordinates
    image_shape : tuple
        Full image shape (height, width)
    overlap_pixels : int
        Overlap in pixels
    use_disk_storage : bool
        If True, tile_results contains file paths to load from disk
    temp_dir : str or None
        Temporary directory for disk storage (for cleanup)
    
    Returns:
    --------
    np.ndarray: Full stitched mask
    """
    height, width = image_shape
    num_tiles = len(tile_results)
    
    print(f"  📐 Creating full mask array ({height}×{width}, uint32)...", flush=True)
    sys.stdout.flush()
    create_start = time.time()
    full_mask = np.zeros((height, width), dtype=np.uint32)
    create_time = time.time() - create_start
    print(f"    ✅ Array created in {create_time:.2f} seconds", flush=True)
    sys.stdout.flush()
    
    # Track next available cell ID
    next_cell_id = 1
    
    print(f"  🔗 Stitching {num_tiles} tiles...", flush=True)
    if use_disk_storage:
        print(f"    💾 Loading tiles from disk as needed...", flush=True)
    sys.stdout.flush()
    
    # Detailed memory diagnostics before starting
    mem_check = psutil.virtual_memory()
    print(f"    [MEM] Available: {mem_check.available / (1024**3):.2f} GB | "
          f"Used: {mem_check.used / (1024**3):.2f} GB | "
          f"Percent: {mem_check.percent:.1f}%", flush=True)
    sys.stdout.flush()
    
    print(f"    [DEBUG] Starting iteration over {num_tiles} tile results...", flush=True)
    sys.stdout.flush()
    
    loop_start_time = time.time()
    last_progress_time = loop_start_time
    last_progress_tile = 0
    
    # Track files to clean up later
    files_to_cleanup = []
    
    for i in range(num_tiles):
        tile_start_time = time.time()
        
        # Load tile result (from disk or memory)
        if use_disk_storage:
            filepath = tile_results[i]
            masks, flows, styles, diams, (y_start, y_end, x_start, x_end) = load_tile_result_from_disk(filepath)
            files_to_cleanup.append(filepath)  # Mark for cleanup
        else:
            masks, flows, styles, diams, (y_start, y_end, x_start, x_end) = tile_results[i]
        
        if i == 0:
            unpack_time = tile_start_time - loop_start_time
            print(f"    [DEBUG] Successfully loaded first tile result (took {unpack_time:.2f}s)", flush=True)
            sys.stdout.flush()
        tile_start_time = time.time()
        
        if i == 1:
            unpack_time = tile_start_time - loop_start_time
            print(f"    [DEBUG] Successfully unpacked first tile result (took {unpack_time:.2f}s)", flush=True)
            sys.stdout.flush()
        
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
        
        # ALWAYS convert to uint32 early to handle large cell counts (>1M cells)
        # Cellpose may return uint16 masks (max 65535), but we need uint32 (max ~4.2B)
        # for stitching many tiles with many cells
        if mask_region.dtype != np.uint32:
            mask_region = mask_region.astype(np.uint32)
        
        # Remap cell IDs to be unique
        np_unique_start = time.time()
        unique_ids = np.unique(mask_region)
        np_unique_time = time.time() - np_unique_start
        unique_ids = unique_ids[unique_ids > 0]  # Exclude background (0)
        
        # remapped_region will be created from mask_region (already uint32) during remapping
        remapped_region = mask_region.copy()
        
        remap_start = time.time()
        
        # Fast vectorized remapping: Create a lookup array instead of looping
        if len(unique_ids) > 0:
            # Create mapping from old_id to new_id using vectorized operations
            # This is O(n) instead of O(n*m) where n=pixels, m=unique_ids
            
            # Method: Create an array where index=old_id, value=new_id
            # For uint16 masks, max ID is 65535, but we may have uint32 masks with larger IDs
            max_old_id = int(unique_ids.max())
            
            # Create lookup array (allocate for all IDs we need, including values > 65535)
            # Initialize with zeros (background stays 0)
            # Always allocate max_old_id + 1 to handle any ID value
            id_mapping = np.zeros(max_old_id + 1, dtype=np.uint32)
            
            # Fill in the mapping: old_id -> next_cell_id, next_cell_id+1, etc.
            new_ids = np.arange(next_cell_id, next_cell_id + len(unique_ids), dtype=np.uint32)
            id_mapping[unique_ids] = new_ids
            
            # Apply mapping: remapped_region = id_mapping[mask_region]
            # This is a single vectorized operation, much faster than loops
            remapped_region = id_mapping[mask_region]
            
            next_cell_id += len(unique_ids)
        
        remap_time = time.time() - remap_start
        
        # Place in full mask
        full_y_start = y_start + use_y_start
        full_y_end = y_start + use_y_end
        full_x_start = x_start + use_x_start
        full_x_end = x_start + use_x_end
        
        copy_start = time.time()
        full_mask[full_y_start:full_y_end, full_x_start:full_x_end] = remapped_region
        copy_time = time.time() - copy_start
        
        # Free tile memory immediately after copying to reduce fragmentation
        # This helps Python's allocator reclaim memory progressively
        # Delete all large arrays we've extracted from the tile
        # NOTE: be careful not to delete variables twice; that caused UnboundLocalError
        # in previous runs when use_disk_storage=True.
        del masks, mask_region, remapped_region
        # Also free flows and styles if they exist (they can be large)
        if flows is not None and isinstance(flows, np.ndarray):
            del flows
        if styles is not None and isinstance(styles, np.ndarray):
            del styles
        if len(unique_ids) > 0:
            del id_mapping, new_ids, unique_ids
        
        # Clean up file after processing if using disk storage
        if use_disk_storage:
            # Delete file after processing
            if os.path.exists(files_to_cleanup[i]):
                try:
                    os.remove(files_to_cleanup[i])
                except Exception:
                    pass  # Ignore deletion errors
        else:
            # Null out the tile_results entry to allow Python to free the tuple
            # This helps reduce fragmentation by freeing memory progressively
            tile_results[i] = None
        
        # Periodically force garbage collection to help with fragmentation
        # Do this every 10 tiles or when memory is tight
        if i % 10 == 9:  # i is 0-indexed
            gc.collect()
        
        tile_elapsed = time.time() - tile_start_time
        total_elapsed = time.time() - loop_start_time
        
        # Progress indicator every 10 tiles, every tile for first 5, or at the end
        # Also show if a tile takes >30 seconds (might indicate a problem)
        should_print = (
            i < 5 or  # First 5 tiles always show (i is 0-indexed)
            i % 10 == 9 or  # Every 10 tiles (9, 19, 29, ...)
            i == num_tiles - 1 or  # Last tile
            tile_elapsed > 30.0  # Slow tile warning
        )
        
        if should_print:
            pct = ((i + 1) / num_tiles) * 100  # i is 0-indexed, display as 1-indexed
            elapsed_since_last = time.time() - last_progress_time
            tiles_since_last = (i + 1) - last_progress_tile
            tiles_per_sec = tiles_since_last / elapsed_since_last if elapsed_since_last > 0 else 0
            
            status_line = f"    Progress: {i+1}/{num_tiles} tiles ({pct:.1f}%) | "
            status_line += f"Cell ID: {next_cell_id} | "
            status_line += f"Time: {tile_elapsed:.1f}s/tile | "
            
            if i > 0:
                status_line += f"Speed: {tiles_per_sec:.2f} tiles/s | "
            
            # Show slow operations
            if np_unique_time > 5.0 or remap_time > 5.0 or copy_time > 5.0:
                status_line += f"[np.unique: {np_unique_time:.1f}s, remap: {remap_time:.1f}s, copy: {copy_time:.1f}s]"
            
            print(status_line, flush=True)
            
            # Memory check every 10 tiles or if slow
            if i % 10 == 9 or tile_elapsed > 30.0:  # i is 0-indexed, so check at 9, 19, 29, ...
                mem_current = psutil.virtual_memory()
                print(f"      [MEM] Available: {mem_current.available / (1024**3):.2f} GB | "
                      f"Used: {mem_current.used / (1024**3):.2f} GB", flush=True)
            
            last_progress_time = time.time()
            last_progress_tile = i
        
        # Hang detection: if a single tile takes >2 minutes, warn
        if tile_elapsed > 120.0:
            print(f"    ⚠️  WARNING: Tile {i+1} took {tile_elapsed:.1f} seconds (>2 min) - possible memory issue", flush=True)
    
    # Clean up temporary directory if using disk storage
    if use_disk_storage and temp_dir and os.path.exists(temp_dir):
        try:
            # Remove any remaining files
            for filepath in files_to_cleanup:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
            # Try to remove directory (may fail if not empty, which is ok)
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass  # Directory may not be empty or already removed
        except Exception:
            pass  # Ignore cleanup errors
    
    print(f"  ✅ Stitching complete! Total cells: {next_cell_id - 1}")
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


def save_tile_result_to_disk(result, tile_idx, temp_dir):
    """
    Save a tile result to disk as npz file.
    
    Parameters:
    -----------
    result : tuple
        (masks, flows, styles, diams, coords) tuple
    tile_idx : int
        Tile index
    temp_dir : str
        Temporary directory for storing tile results
    
    Returns:
    --------
    str: Path to saved file
    """
    masks, flows, styles, diams, coords = result
    filepath = os.path.join(temp_dir, f'tile_{tile_idx:04d}.npz')
    
    # Save data (None values will be stored as None)
    # Convert coords tuple to numpy array for reliable saving
    save_dict = {
        'masks': masks,
        'coords': np.array(coords) if isinstance(coords, (tuple, list)) else coords
    }
    
    # Handle flows - might be tuple or array with inhomogeneous shapes
    if flows is not None:
        try:
            # Try to save as-is if it's already a numpy array
            if isinstance(flows, np.ndarray):
                save_dict['flows'] = flows
            elif isinstance(flows, (tuple, list)):
                # If it's a tuple/list, try to convert each element
                try:
                    save_dict['flows'] = np.array(flows)
                except (ValueError, TypeError):
                    # If conversion fails, save as object array
                    save_dict['flows'] = np.array(flows, dtype=object)
            else:
                save_dict['flows'] = flows
        except Exception as e:
            print(f"  ⚠️  Warning: Could not save flows for tile {tile_idx} ({e}), skipping")
    
    # Handle styles - might be tuple or array with inhomogeneous shapes
    if styles is not None:
        try:
            if isinstance(styles, np.ndarray):
                save_dict['styles'] = styles
            elif isinstance(styles, (tuple, list)):
                try:
                    save_dict['styles'] = np.array(styles)
                except (ValueError, TypeError):
                    save_dict['styles'] = np.array(styles, dtype=object)
            else:
                save_dict['styles'] = styles
        except Exception as e:
            print(f"  ⚠️  Warning: Could not save styles for tile {tile_idx} ({e}), skipping")
    
    # Handle diams which can be scalar, array, or inhomogeneous list/array
    if diams is not None:
        try:
            if isinstance(diams, np.ndarray):
                save_dict['diams'] = diams
            elif isinstance(diams, (int, float)):
                # Scalar value
                save_dict['diams'] = np.array(diams)
            else:
                # Try to convert to homogeneous array
                try:
                    save_dict['diams'] = np.array(diams)
                except (ValueError, TypeError):
                    # If conversion fails (inhomogeneous shapes), try object array
                    try:
                        save_dict['diams'] = np.array(diams, dtype=object)
                    except (ValueError, TypeError):
                        # If even object array fails, convert to list and save as object
                        if isinstance(diams, list):
                            save_dict['diams'] = np.array([np.array(d) if isinstance(d, (list, np.ndarray)) else d for d in diams], dtype=object)
                        else:
                            # Last resort: skip diams if it can't be saved
                            print(f"  ⚠️  Warning: Could not save diams for tile {tile_idx}, skipping")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not save diams for tile {tile_idx} ({e}), skipping")
    
    # Try to save, with progressive fallback if any field causes issues
    try:
        np.savez_compressed(filepath, **save_dict)
    except (ValueError, TypeError) as e:
        # If saving fails, try removing optional fields one by one
        error_msg = str(e)
        print(f"  ⚠️  Warning: Error saving tile {tile_idx} ({error_msg}), trying fallback...")
        
        # Try without diams
        if 'diams' in save_dict:
            try:
                save_dict_no_diams = {k: v for k, v in save_dict.items() if k != 'diams'}
                np.savez_compressed(filepath, **save_dict_no_diams)
                print(f"  ✅ Saved tile {tile_idx} without diams")
                return filepath
            except Exception:
                pass
        
        # Try without flows
        if 'flows' in save_dict:
            try:
                save_dict_minimal = {k: v for k, v in save_dict.items() if k not in ['diams', 'flows']}
                np.savez_compressed(filepath, **save_dict_minimal)
                print(f"  ✅ Saved tile {tile_idx} with minimal data (masks, coords, styles)")
                return filepath
            except Exception:
                pass
        
        # Try with only masks and coords (essential data)
        try:
            np.savez_compressed(filepath, masks=save_dict['masks'], coords=save_dict['coords'])
            print(f"  ✅ Saved tile {tile_idx} with essential data only (masks, coords)")
            return filepath
        except Exception as final_e:
            raise RuntimeError(f"Failed to save tile {tile_idx} even with minimal data: {final_e}")
    return filepath


def load_tile_result_from_disk(filepath):
    """
    Load a tile result from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to npz file
    
    Returns:
    --------
    tuple: (masks, flows, styles, diams, coords) tuple
    """
    data = np.load(filepath, allow_pickle=True)
    masks = data['masks']
    coords = tuple(data['coords'])
    flows = data.get('flows', None)
    styles = data.get('styles', None)
    diams = data.get('diams', None)
    data.close()
    return (masks, flows, styles, diams, coords)


def process_tiles_with_progress(model, imgs, tiles, diameter, channels, use_gpu, 
                                progress_tracker=None, nuclear_file=None, cyto_file=None,
                                use_disk_storage=False, temp_dir=None):
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
    use_disk_storage : bool
        If True, save tile results to disk instead of keeping in memory
    temp_dir : str or None
        Temporary directory for disk storage (created automatically if None)
    
    Returns:
    --------
    list: List of tile results (file paths if use_disk_storage, else data tuples)
    """
    results = []
    num_tiles = len(tiles)
    read_from_files = (nuclear_file is not None and cyto_file is not None)
    
    # Setup disk storage if enabled
    if use_disk_storage:
        if temp_dir is None:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='cellpose_tiles_')
            print(f"  💾 Using disk storage for tile results: {temp_dir}")
        else:
            os.makedirs(temp_dir, exist_ok=True)
            print(f"  💾 Using disk storage directory: {temp_dir}")
    
    # Debug: Track memory usage
    debug_mode = os.environ.get('CELLPOSE_DEBUG', '0') == '1'
    
    # GPU status monitoring
    last_gpu_status_time = time.time()
    gpu_status_interval = 30  # Print GPU status every 30 seconds
    gpu_status_tile_interval = 5  # Also print every 5 tiles
    
    for i, (y_start, y_end, x_start, x_end) in enumerate(tiles):
        tile_start = time.time()
        
        # Check memory before reading tile (always check, warn if low, raise if critical)
        mem_before = psutil.virtual_memory()
        available_gb = mem_before.available / (1024**3)
        
        # Raise exception if memory is critically low (< 1 GB available) to trigger retry with denser grid
        if available_gb < 1.0:
            error_msg = f"System out of memory: only {available_gb:.2f} GB available before tile {i+1}/{num_tiles}. Exit code 137 (OOM kill) likely imminent."
            print(f"\n❌ CRITICAL: {error_msg}", flush=True)
            raise RuntimeError(error_msg)
        
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
            gpu_mem = get_current_gpu_memory()
            gpu_free = gpu_mem[0] if gpu_mem else None
            if gpu_free is not None:
                if gpu_free < 0.5:  # Less than 500 MB free
                    print(f"\n⚠️  Low GPU memory before tile {i+1}/{num_tiles}: {gpu_free:.2f} GB free")
                    print(f"  Clearing GPU cache...")
                    clear_gpu_cache()
                    # Re-check after clearing
                    gpu_mem_after = get_current_gpu_memory()
                    gpu_free_after = gpu_mem_after[0] if gpu_mem_after else None
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
            
            # Save to disk or store in memory
            if use_disk_storage:
                filepath = save_tile_result_to_disk(result, i, temp_dir)
                results.append(filepath)  # Store file path instead of data
                # Free memory immediately after saving
                del result
            else:
                results.append(result)  # Store in memory
            
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
            if not use_disk_storage:
                # Note: result is still referenced in results list, so del result doesn't free memory
                # Memory will be freed progressively during stitching
                pass
            gc.collect()
            
            # Monitor memory usage and warn if getting high
            if (i + 1) % 5 == 0 or available_gb < 10.0:  # Every 5 tiles or if low
                mem_current = psutil.virtual_memory()
                available_current = mem_current.available / (1024**3)
                used_current = mem_current.used / (1024**3)
                percent_used = mem_current.percent
                
                if percent_used > 85.0:  # Warn if over 85% used
                    print(f"\n  ⚠️  High memory usage: {used_current:.1f} GB used ({percent_used:.1f}%) | "
                          f"{available_current:.1f} GB available | "
                          f"Tiles processed: {i+1}/{num_tiles}")
                    print(f"     💡 Memory will be freed progressively during stitching (after all tiles)")
                elif (i + 1) % 10 == 0:  # Show status every 10 tiles
                    print(f"  [MEM] {used_current:.1f} GB used ({percent_used:.1f}%) | "
                          f"{available_current:.1f} GB available")
            
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
                    max_retries = 5  # Increased from 3 to give more chances with cleanup
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
                        gpu_mem = get_current_gpu_memory()
                        gpu_free = gpu_mem[0] if gpu_mem else None
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
        sys.stdout.flush()
    
    print(f"  [DEBUG] Returning {len(results)} tile results", flush=True)
    sys.stdout.flush()
    return results


def run_segmentation(nuclear_file, cyto_file, output_dir='output', use_gpu=False, 
                    model_type='cyto3', diameter=None, tile_size=None, overlap=None,
                    max_workers=None, force_cpu=False, gpu_memory_limit=2.0,
                    grid_size=None, max_grid_size=10,
                    resample=True, flow_threshold=0.4, cellprob_threshold=0.0):
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
        # Check if progressive grid will be used (when tile_size is None)
        will_use_progressive_grid = (tile_size is None)
        
        if will_use_progressive_grid:
            print(f"  🔲 Will use progressive grid tiling (reading tiles on-demand from files)")
            # Don't calculate tile size here - progressive grid will handle it
            tile_height, tile_width = None, None  # Will be set by progressive grid
        else:
            print(f"  🔲 Will use automatic tiling (reading tiles on-demand from files)")
            
            # Calculate optimal tile size (legacy path only)
            memory_for_tiling = memory_limit_gb
            tile_height, tile_width = calculate_tile_size(
                image_shape, memory_for_tiling, channels=2, dtype_size=dtype_size, 
                model_memory_gb=model_memory_gb
            )
            print(f"  Calculated tile size: {tile_height} × {tile_width} pixels")
    elif estimated_total_memory_gb > 20.0 and tile_size is None:  # Force tiling for very large images (>20 GB estimated), unless tile_size explicitly set
        # Even if we have enough memory, very large images can cause issues
        # Force tiling to be safe
        use_tiling = True
        
        # Check if progressive grid will be used (when tile_size is None)
        # If so, don't show legacy tile size calculation - progressive grid will handle it
        will_use_progressive_grid = (tile_size is None)
        
        if will_use_progressive_grid:
            print(f"\n⚠️  Very large image detected (estimated {estimated_total_memory_gb:.2f} GB)")
            print(f"  🔲 Using progressive grid tiling for stability (even though {memory_limit_gb:.2f} GB available)")
            print(f"  This prevents memory allocation issues during processing")
            # Don't calculate tile size here - progressive grid will handle it
            tile_height, tile_width = None, None  # Will be set by progressive grid
        else:
            print(f"\n⚠️  Very large image detected (estimated {estimated_total_memory_gb:.2f} GB)")
            print(f"  🔲 Using tiling for stability (even though {memory_limit_gb:.2f} GB available)")
            print(f"  This prevents memory allocation issues during processing")
            
            # Use large tiles but still tile
            memory_for_tiling = memory_limit_gb * 0.8  # Use 80% of available
            tile_height, tile_width = calculate_tile_size(
                image_shape, memory_for_tiling, channels=2, dtype_size=dtype_size, 
                model_memory_gb=model_memory_gb
            )
            print(f"  Calculated tile size: {tile_height} × {tile_width} pixels")
    else:
        print(f"\n✅ Sufficient memory available (estimated {estimated_total_memory_gb:.2f} GB needed, {memory_limit_gb:.2f} GB available)")
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
        model = models.CellposeModel(gpu=actual_use_gpu)
    except Exception as e:
        if actual_use_gpu:
            print(f"⚠️  GPU initialization failed: {e}")
            print(f"  Falling back to CPU...")
            actual_use_gpu = False
            model = models.CellposeModel(gpu=False)
        else:
            raise
    
    # Initialize tiling variables
    tiles = None
    overlap_pixels = None
    num_tiles = 0
    
    # Process with or without tiling
    if use_tiling:
        processed_by_progressive_grid = False
        tile_results = None
        use_disk_storage = False
        temp_dir = None
        
        # If tile_size is not provided, use progressive grid strategy (2×2 → 3×3 → … on OOM).
        # This becomes the default auto-tiling behavior.
        if tile_size is None:
            (tile_results,
             tiles,
             tile_height,
             tile_width,
             overlap_pixels,
             use_disk_storage,
             temp_dir) = process_with_progressive_grid(
                model=model,
                imgs=imgs,
                image_shape=image_shape,
                dtype_size=dtype_size,
                diameter=diameter,
                actual_use_gpu=actual_use_gpu,
                overlap=overlap,
                nuclear_file=nuclear_file,
                cyto_file=cyto_file,
                output_dir=output_dir,
                grid_size=grid_size,
                max_grid_size=max_grid_size,
            )
            num_tiles = len(tiles)
            processed_by_progressive_grid = True
        
        if not processed_by_progressive_grid:
            # Legacy/manual tile-size path: calculate overlap and generate tiles
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
            overlap_pixels = max(0, int(overlap_pixels))
            
            print(f"  Tile overlap: {overlap_pixels} pixels")
            
            tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
            num_tiles = len(tiles)
            print(f"  Generated {num_tiles} tiles")
        
        # Check GPU memory one more time before processing (in case other processes started)
        if actual_use_gpu:
            gpu_mem = get_current_gpu_memory()
            current_gpu_free = gpu_mem[0] if gpu_mem else None
            if current_gpu_free is not None:
                # Estimate memory needed per tile (rough: tile size * 5x overhead)
                tile_pixels = tile_height * tile_width
                tile_memory_gb = (tile_pixels * 2 * 2 * 5) / (1024**3)  # channels * dtype * overhead
                
                if current_gpu_free < tile_memory_gb * 1.5:  # Need at least 1.5x tile size free
                    # Check if other processes are using significant GPU memory
                    status = get_gpu_memory_status()
                    other_processes_using_gpu = False
                    if status and status['processes']:
                        current_pid = os.getpid()
                        other_procs = [p for p in status['processes'] 
                                     if str(p['pid']) != str(current_pid)]
                        if other_procs:
                            total_other_memory = sum(p['memory_gb'] for p in other_procs)
                            if total_other_memory > 1.0:  # More than 1 GB used by others
                                other_processes_using_gpu = True
                    
                    if other_processes_using_gpu:
                        # Wait for other processes to free memory
                        print(f"\n⚠️  GPU memory is low ({current_gpu_free:.2f} GB free)")
                        print(f"  Estimated {tile_memory_gb:.2f} GB needed per tile")
                        print(f"  Other processes are using GPU memory")
                        
                        # Wait for memory (default 5 minutes timeout)
                        success, final_free, waited = wait_for_gpu_memory(
                            required_gb=tile_memory_gb * 1.5,
                            timeout_seconds=300,
                            current_pid=os.getpid()
                        )
                        
                        if success:
                            print(f"  ✅ Proceeding with original tile size")
                            # Keep original tile size
                        else:
                            # Still not enough after waiting - reduce tile size
                            print(f"  ⚠️  Still insufficient memory after waiting ({final_free:.2f} GB free)")
                            print(f"  🔄 Reducing tile size to fit available memory...")
                            reduction_factor = min(2, (final_free / tile_memory_gb) * 0.8)
                            tile_height = max(2048, int(tile_height / reduction_factor))
                            tile_width = max(2048, int(tile_width / reduction_factor))
                            tile_height = (tile_height // 256) * 256
                            tile_width = (tile_width // 256) * 256
                            
                            overlap_pixels = calculate_overlap(min(tile_height, tile_width))
                            tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                            num_tiles = len(tiles)
                            print(f"  ✅ Reduced to {tile_height} × {tile_width} pixels, {num_tiles} tiles")
                    else:
                        # No other processes - just reduce tile size immediately
                        print(f"\n⚠️  GPU memory is low ({current_gpu_free:.2f} GB free)")
                        print(f"  🔄 Reducing tile size to fit available memory...")
                        reduction_factor = min(2, (current_gpu_free / tile_memory_gb) * 0.8)
                        tile_height = max(2048, int(tile_height / reduction_factor))
                        tile_width = max(2048, int(tile_width / reduction_factor))
                        tile_height = (tile_height // 256) * 256
                        tile_width = (tile_width // 256) * 256
                        
                        overlap_pixels = calculate_overlap(min(tile_height, tile_width))
                        tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                        num_tiles = len(tiles)
                        print(f"  ✅ Reduced to {tile_height} × {tile_width} pixels, {num_tiles} tiles")
        
        # If progressive grid already processed tiles, skip processing and go straight to stitching.
        if processed_by_progressive_grid:
            # Extract diameters BEFORE stitching
            all_diams = []
            if use_disk_storage:
                for filepath in tile_results:
                    try:
                        _, _, _, diams_tile, _ = load_tile_result_from_disk(filepath)
                        if isinstance(diams_tile, (list, np.ndarray)) and len(diams_tile) > 0:
                            all_diams.extend(diams_tile if isinstance(diams_tile, list) else diams_tile.tolist())
                    except Exception:
                        pass
            else:
                for _, _, _, diams_tile, _ in tile_results:
                    if isinstance(diams_tile, (list, np.ndarray)) and len(diams_tile) > 0:
                        all_diams.extend(diams_tile if isinstance(diams_tile, list) else diams_tile.tolist())
            avg_diameter = np.mean(all_diams) if all_diams else None
            
            print(f"\n🔗 Stitching {num_tiles} tiles into full mask...", flush=True)
            sys.stdout.flush()
            stitch_start = time.time()
            masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels,
                                 use_disk_storage=use_disk_storage, temp_dir=temp_dir)
            stitch_time = time.time() - stitch_start
            print(f"  ⏱️  Stitching took {format_time(stitch_time)}")
            
            flows = None  # Not stitched for now
            styles = None
            diams = avg_diameter
            
            # Clean up tile cache directory after successful completion
            if use_disk_storage and temp_dir and os.path.exists(temp_dir):
                print(f"\n🧹 Cleaning up tile cache directory...", flush=True)
                try:
                    # Remove all files in the cache directory (including any old ones from previous runs)
                    for filename in os.listdir(temp_dir):
                        filepath = os.path.join(temp_dir, filename)
                        try:
                            if os.path.isfile(filepath):
                                os.remove(filepath)
                        except Exception:
                            pass
                    print(f"  ✅ Tile cache cleaned up", flush=True)
                except Exception as cleanup_err:
                    print(f"  ⚠️  Warning: Could not fully clean tile cache: {cleanup_err}", flush=True)
        else:
            # Process all tiles with progress tracking
            print(f"\n🔄 Processing {num_tiles} tiles...")
            # Show initial time estimate
            time_min, time_max = estimate_initial_time(num_tiles, tile_height, tile_width, 
                                                        actual_use_gpu, channels=2)
            print(f"  ⏱️  Estimated time: {format_time(time_min)} - {format_time(time_max)} "
                  f"(based on {'GPU' if actual_use_gpu else 'CPU'} processing)")
            
            progress_tracker = ProgressTracker(num_tiles)
            
            # Auto-enable disk storage if memory would be constrained
            use_disk_storage = False
            temp_dir = None
            mem_check = psutil.virtual_memory()
            available_gb = mem_check.available / (1024**3)
            total_gb = mem_check.total / (1024**3)
            
            # Estimate memory per tile (masks + flows + overhead)
            # Rough estimate: tile_height * tile_width * 4 bytes (uint32) * 2 (masks + flows overhead)
            if use_tiling:
                pixels_per_tile = tile_height * tile_width
                bytes_per_tile = pixels_per_tile * 4 * 2  # masks + flows estimate
                estimated_per_tile_gb = bytes_per_tile / (1024**3)
            else:
                estimated_per_tile_gb = 0.5  # Default estimate if no tiling
            
            # Estimate total memory needed for all tile results
            estimated_tile_results_gb = estimated_per_tile_gb * num_tiles * 1.5  # 1.5x overhead
            
            # Enable disk storage if:
            # 1. Estimated tile results > 50% of available memory, OR
            # 2. More than 50 tiles (many small tiles = many results), OR
            # 3. Less than 30 GB available (<25% of 120GB system)
            should_use_disk = (
                estimated_tile_results_gb > available_gb * 0.5 or
                num_tiles > 50 or
                available_gb < 30.0
            )
            
            if should_use_disk:
                use_disk_storage = True
                import tempfile
                # Create temp directory in output_dir if specified, else system temp
                if output_dir:
                    temp_dir = os.path.join(output_dir, '.tile_cache')
                    os.makedirs(temp_dir, exist_ok=True)
                else:
                    temp_dir = tempfile.mkdtemp(prefix='cellpose_tiles_')
                print(f"\n  💾 Auto-enabled disk storage for tile results:")
                print(f"     Reason: Estimated {estimated_tile_results_gb:.1f} GB needed | "
                      f"{available_gb:.1f} GB available | {num_tiles} tiles")
                print(f"     Location: {temp_dir}")
            
            try:
                tile_results = process_tiles_with_progress(
                    model, imgs, tiles, diameter, [0, 1], actual_use_gpu, progress_tracker,
                    nuclear_file=nuclear_file if use_tiling else None,
                    cyto_file=cyto_file if use_tiling else None,
                    use_disk_storage=use_disk_storage,
                    temp_dir=temp_dir
                )
                
                # Debug: Verify we got results and estimate memory usage
                print(f"\n✅ All {len(tile_results)} tiles processed successfully", flush=True)
                
                # Estimate memory usage of tile results (only if in-memory)
                if tile_results and not use_disk_storage:
                    first_result = tile_results[0]
                    if len(first_result) >= 1:  # At least masks
                        masks = first_result[0]
                        size_per_mask_mb = masks.nbytes / (1024**2)
                        total_masks_mb = size_per_mask_mb * len(tile_results)
                        print(f"  📊 Estimated tile results memory: ~{total_masks_mb:.1f} MB ({total_masks_mb/1024:.1f} GB) for masks alone", flush=True)
                elif use_disk_storage:
                    print(f"  💾 Tile results stored on disk (no memory accumulation)", flush=True)
                
                sys.stdout.flush()
                
                # Extract diameters BEFORE stitching (stitch_tiles sets tile_results entries to None if in-memory)
                all_diams = []
                if use_disk_storage:
                    # Load each tile from disk to extract diameters
                    for i, filepath in enumerate(tile_results):
                        try:
                            _, _, _, diams, _ = load_tile_result_from_disk(filepath)
                            if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                        except Exception:
                            pass  # Skip if file can't be read
                else:
                    for _, _, _, diams, _ in tile_results:
                        if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                            all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                avg_diameter = np.mean(all_diams) if all_diams else None
                
                # Stitch tiles
                print(f"\n🔗 Stitching {num_tiles} tiles into full mask...", flush=True)
                sys.stdout.flush()
                
                # Check memory before stitching with detailed diagnostics
                mem_before = psutil.virtual_memory()
                available_gb = mem_before.available / (1024**3)
                used_gb = mem_before.used / (1024**3)
                total_gb = mem_before.total / (1024**3)
                
                # Estimate memory needed for stitching
                # full_mask: 64624 × 73957 × 4 bytes = ~19GB
                # Plus working arrays during loop
                estimated_stitch_memory_gb = 25.0  # Conservative estimate
                
                print(f"  💾 Memory before stitching:", flush=True)
                print(f"    Available: {available_gb:.2f} GB / Total: {total_gb:.2f} GB ({mem_before.percent:.1f}% used)", flush=True)
                print(f"    Estimated need: ~{estimated_stitch_memory_gb:.1f} GB for stitching", flush=True)
                
                if available_gb < estimated_stitch_memory_gb:
                    print(f"    ⚠️  WARNING: Low available memory ({available_gb:.2f} GB < {estimated_stitch_memory_gb:.1f} GB estimated)", flush=True)
                    print(f"       Stitching may be slow or fail. Consider freeing memory.", flush=True)
                else:
                    print(f"    ✅ Sufficient memory available", flush=True)
                
                sys.stdout.flush()
                
                stitch_start = time.time()
                masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels, 
                                   use_disk_storage=use_disk_storage, temp_dir=temp_dir)
                stitch_time = time.time() - stitch_start
                print(f"  ⏱️  Stitching took {format_time(stitch_time)}")
                
                # Check memory after stitching
                mem_after = psutil.virtual_memory()
                print(f"  💾 Memory after stitching: {mem_after.available / (1024**3):.2f} GB available")
                
                flows = None  # Not stitched for now
                styles = None
                diams = avg_diameter
                
            except Exception as e:
                print(f"\n❌ Error during tiled processing: {e}")
                error_str = str(e)
                is_gpu_oom = ("CUDA out of memory" in error_str or 
                             "out of memory" in error_str.lower() or
                             "GPU OOM" in error_str)
                
                # CRITICAL: Release memory from failed processing before fallback
                # Clear tile_results to prevent memory accumulation from both runs
                try:
                    if 'tile_results' in locals():
                        print(f"  🧹 Clearing memory from previous tile processing ({len(tile_results) if tile_results else 0} tiles)...")
                        del tile_results
                    if actual_use_gpu:
                        clear_gpu_cache()
                    gc.collect()
                    print(f"  ✅ Memory cleaned up")
                except Exception as cleanup_err:
                    print(f"  ⚠️  Warning during cleanup: {cleanup_err}")
                
                # Try with smaller tiles as fallback
                if tile_height > 512 and tile_width > 512:
                    print(f"  Attempting fallback with smaller tiles...")
                    
                    # Calculate current tile count to limit fallback increase
                    overlap_pixels_current = calculate_overlap(min(tile_height, tile_width))
                    tiles_current = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels_current)
                    current_tile_count = len(tiles_current)
                    # Target ~25% increase (e.g., 96 → 120 tiles)
                    target_tile_increase = 1.25
                    max_target_tiles = int(current_tile_count * target_tile_increase)
                    
                    # If GPU OOM, calculate tile size based on available memory
                    if is_gpu_oom and actual_use_gpu:
                        try:
                            clear_gpu_cache()
                            gpu_mem = get_current_gpu_memory()
                            current_gpu_free = gpu_mem[0] if gpu_mem else None
                            if current_gpu_free is not None:
                                print(f"  Current GPU memory: {current_gpu_free:.2f} GB free")
                            
                            if current_gpu_free and current_gpu_free > 0.5:  # At least 500 MB available
                                # Calculate tile size based on available memory
                                # Reserve some memory for model and overhead
                                # Use more conservative calculation since OOM occurred despite retries
                                memory_for_tiling = current_gpu_free * 0.4  # Use 40% of free memory (very conservative due to fragmentation)
                                new_tile_height, new_tile_width = calculate_tile_size(
                                    image_shape, memory_for_tiling, channels=2, 
                                    dtype_size=dtype_size, model_memory_gb=model_memory_gb
                                )
                                
                                # Ensure new tiles are significantly smaller than current (at least 50% reduction)
                                # This accounts for memory fragmentation
                                if new_tile_height < tile_height * 0.5 and new_tile_width < tile_width * 0.5:
                                    tile_height = new_tile_height
                                    tile_width = new_tile_width
                                    print(f"  Calculated tile size based on available GPU memory: "
                                          f"{tile_height} × {tile_width} pixels")
                                else:
                                    # Force at least 50% reduction to account for fragmentation
                                    tile_height = max(2048, int(tile_height * 0.4))
                                    tile_width = max(2048, int(tile_width * 0.4))
                                    tile_height = (tile_height // 256) * 256  # Round to 256
                                    tile_width = (tile_width // 256) * 256
                                    print(f"  Reduced tile size by 60% to account for fragmentation: "
                                          f"{tile_height} × {tile_width} pixels")
                            elif current_gpu_free > 0.1:  # 100-500 MB available
                                # Very low memory - reduce aggressively
                                print(f"  ⚠️  Very low GPU memory ({current_gpu_free:.2f} GB) - reducing tiles aggressively")
                                tile_height = max(2048, tile_height // 4)
                                tile_width = max(2048, tile_width // 4)
                                print(f"  Aggressively reduced tile size: {tile_height} × {tile_width} pixels")
                            else:
                                # Extremely low memory (< 100 MB) - reduce very aggressively
                                print(f"  ⚠️  Extremely low GPU memory ({current_gpu_free:.2f} GB) - reducing tiles very aggressively")
                                tile_height = max(1024, tile_height // 8)
                                tile_width = max(1024, tile_width // 8)
                                print(f"  Very aggressively reduced tile size: {tile_height} × {tile_width} pixels")
                        except Exception as mem_err:
                            print(f"  ⚠️  Could not get GPU memory info: {mem_err}")
                            # Conservative fallback: reduce by ~10% to get ~25% more tiles (e.g., 96→120)
                            reduction_factor = 0.9  # Reduce to 90% (10% smaller)
                            tile_height = max(2048, int(tile_height * reduction_factor))
                            tile_width = max(2048, int(tile_width * reduction_factor))
                            tile_height = (tile_height // 256) * 256
                            tile_width = (tile_width // 256) * 256
                    else:
                        # Not GPU OOM or not using GPU - use conservative reduction (~10%)
                        # This targets ~25% tile count increase (e.g., 96→120), not 4x (96→400)
                        reduction_factor = 0.9  # Reduce to 90% (10% smaller)
                        tile_height = max(2048, int(tile_height * reduction_factor))
                        tile_width = max(2048, int(tile_width * reduction_factor))
                        tile_height = (tile_height // 256) * 256  # Round to 256
                        tile_width = (tile_width // 256) * 256
                    
                    # Recalculate overlap and check tile count
                    overlap_pixels = calculate_overlap(min(tile_height, tile_width))
                    tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                    new_tile_count = len(tiles)
                    
                    # If tile count exceeds max, reduce further
                    if new_tile_count > max_target_tiles:
                        print(f"  ⚠️  Would create {new_tile_count} tiles (max: {max_target_tiles})")
                        # Reduce further to target max tile count
                        # Estimate: reduce dimensions by sqrt(max_target_tiles / new_tile_count)
                        reduction = (max_target_tiles / new_tile_count) ** 0.5
                        tile_height = max(2048, int(tile_height * reduction))
                        tile_width = max(2048, int(tile_width * reduction))
                        tile_height = (tile_height // 256) * 256
                        tile_width = (tile_width // 256) * 256
                        overlap_pixels = calculate_overlap(min(tile_height, tile_width))
                        tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                        new_tile_count = len(tiles)
                        print(f"  Reduced further to stay within limit: {tile_height} × {tile_width}, {new_tile_count} tiles")
                    else:
                        print(f"  New tile size: {tile_height} × {tile_width}, {new_tile_count} tiles (was {current_tile_count})")
                    
                        progress_tracker = ProgressTracker(len(tiles))
                    try:
                        tile_results = process_tiles_with_progress(
                            model, imgs, tiles, diameter, [0, 1], actual_use_gpu, progress_tracker,
                            nuclear_file=nuclear_file if use_tiling else None,
                            cyto_file=cyto_file if use_tiling else None,
                            use_disk_storage=use_disk_storage,
                            temp_dir=temp_dir
                        )
                        # Extract diameters BEFORE stitching
                        all_diams = []
                        if use_disk_storage:
                            for filepath in tile_results:
                                try:
                                    _, _, _, diams, _ = load_tile_result_from_disk(filepath)
                                    if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                        all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                                except Exception:
                                    pass
                        else:
                            for _, _, _, diams, _ in tile_results:
                                if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                    all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                        avg_diameter = np.mean(all_diams) if all_diams else None
                        
                        masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels,
                                           use_disk_storage=use_disk_storage, temp_dir=temp_dir)
                        flows = None
                        styles = None
                        diams = avg_diameter
                    except Exception as retry_error:
                        # Retry also failed - try even smaller tiles or give up
                        print(f"\n  ❌ Retry with reduced tiles also failed: {retry_error}")
                        if tile_height > 1024 and tile_width > 1024:
                            print(f"  Attempting with even smaller tiles (quarter size)...")
                            tile_height = max(1024, tile_height // 4)
                            tile_width = max(1024, tile_width // 4)
                            overlap_pixels = calculate_overlap(min(tile_height, tile_width))
                            tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                            print(f"  Final tile size: {tile_height} × {tile_width}, {len(tiles)} tiles")
                            
                            # Force disk storage for fallback to avoid memory issues
                            if not use_disk_storage:
                                if temp_dir is None:
                                    import tempfile
                                    temp_dir = tempfile.mkdtemp(prefix='cellpose_tiles_')
                                use_disk_storage = True
                                print(f"  💾 Enabling disk storage for fallback processing")
                            
                            progress_tracker = ProgressTracker(len(tiles))
                            tile_results = process_tiles_with_progress(
                                model, imgs, tiles, diameter, [0, 1], actual_use_gpu, progress_tracker,
                                nuclear_file=nuclear_file if use_tiling else None,
                                cyto_file=cyto_file if use_tiling else None,
                                use_disk_storage=use_disk_storage,
                                temp_dir=temp_dir
                            )
                            # Extract diameters BEFORE stitching
                            all_diams = []
                            if use_disk_storage:
                                for filepath in tile_results:
                                    try:
                                        _, _, _, diams, _ = load_tile_result_from_disk(filepath)
                                        if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                            all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                                    except Exception:
                                        pass
                            else:
                                for _, _, _, diams, _ in tile_results:
                                    if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                        all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                            avg_diameter = np.mean(all_diams) if all_diams else None
                            
                            masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels,
                                               use_disk_storage=use_disk_storage, temp_dir=temp_dir)
                            flows = None
                            styles = None
                            diams = avg_diameter
                        else:
                            # Tiles are already very small - give up
                            print(f"  ❌ Cannot reduce tiles further. Current size: {tile_height} × {tile_width}")
                            print(f"  💡 Suggestions:")
                            try:
                                gpu_mem = get_current_gpu_memory()
                                gpu_free = gpu_mem[0] if gpu_mem else None
                                if gpu_free is not None:
                                    print(f"     - Free GPU memory from other processes (currently {gpu_free:.2f} GB free)")
                                else:
                                    print(f"     - Free GPU memory from other processes")
                            except:
                                print(f"     - Free GPU memory from other processes")
                            print(f"     - Use smaller manual tile size with --tile-size")
                            print(f"     - Process on CPU instead (--no-gpu)")
                            raise RuntimeError(f"GPU OOM error persisted even with very small tiles ({tile_height} × {tile_width}). "
                                             f"Consider freeing memory from other processes or using CPU.")
            else:
                raise
        
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
        except (RuntimeError, Exception) as e:
            error_str = str(e)
            if "CUDA out of memory" in error_str or "out of memory" in error_str.lower():
                print(f"\n⚠️  GPU memory error: {e}")
                print(f"  Falling back to GPU with large tiles...")
                # Try large tiles on GPU first (might be compute capability issue)
                use_tiling = True
                # Use very large tiles (close to full image) to minimize overhead
                tile_height = min(image_shape[0], 32000)  # Max 32k pixels per dimension
                tile_width = min(image_shape[1], 32000)
                overlap_pixels = calculate_overlap(min(tile_height, tile_width), 
                                                  cell_diameter=diameter)
                tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                num_tiles = len(tiles)
                
                print(f"  Using large GPU tiles: {tile_height} × {tile_width}, {num_tiles} tiles")
                # Force disk storage for fallback to avoid memory issues
                if not use_disk_storage:
                    if temp_dir is None:
                        import tempfile
                        if output_dir:
                            temp_dir = os.path.join(output_dir, '.tile_cache')
                            os.makedirs(temp_dir, exist_ok=True)
                        else:
                            temp_dir = tempfile.mkdtemp(prefix='cellpose_tiles_')
                    use_disk_storage = True
                    print(f"  💾 Enabling disk storage for GPU fallback")
                progress_tracker = ProgressTracker(num_tiles)
                try:
                    tile_results = process_tiles_with_progress(
                        model, None, tiles, diameter, [0, 1], actual_use_gpu, progress_tracker,
                        nuclear_file=nuclear_file,
                        cyto_file=cyto_file,
                        use_disk_storage=use_disk_storage,
                        temp_dir=temp_dir
                    )
                    # Extract diameters BEFORE stitching
                    all_diams = []
                    if use_disk_storage:
                        for filepath in tile_results:
                            try:
                                _, _, _, diams, _ = load_tile_result_from_disk(filepath)
                                if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                    all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                            except Exception:
                                pass
                    else:
                        for _, _, _, diams, _ in tile_results:
                            if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                    diams = np.mean(all_diams) if all_diams else None
                    
                    masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels,
                                       use_disk_storage=use_disk_storage, temp_dir=temp_dir)
                    
                    flows = None
                    styles = None
                except Exception as tile_error:
                    print(f"\n⚠️  GPU tiling also failed: {tile_error}")
                    print(f"  Falling back to CPU with tiling...")
                    actual_use_gpu = False
                    model = models.CellposeModel(gpu=False)
                    
                    # Recalculate with CPU
                    tile_height, tile_width = calculate_tile_size(
                        image_shape, available_gb, channels=2, dtype_size=2, 
                        model_memory_gb=1.0
                    )
                    overlap_pixels = calculate_overlap(min(tile_height, tile_width), 
                                                      cell_diameter=diameter)
                    tiles = generate_tiles(image_shape, tile_height, tile_width, overlap_pixels)
                    num_tiles = len(tiles)
                    
                    print(f"  Using CPU tiles: {tile_height} × {tile_width}, {num_tiles} tiles")
                    # Force disk storage for CPU fallback to avoid memory issues
                    if not use_disk_storage:
                        if temp_dir is None:
                            import tempfile
                            if output_dir:
                                temp_dir = os.path.join(output_dir, '.tile_cache')
                                os.makedirs(temp_dir, exist_ok=True)
                            else:
                                temp_dir = tempfile.mkdtemp(prefix='cellpose_tiles_')
                        use_disk_storage = True
                        print(f"  💾 Enabling disk storage for CPU fallback")
                    progress_tracker = ProgressTracker(num_tiles)
                    tile_results = process_tiles_with_progress(
                        model, None, tiles, diameter, [0, 1], False, progress_tracker,
                        nuclear_file=nuclear_file,
                        cyto_file=cyto_file,
                        use_disk_storage=use_disk_storage,
                        temp_dir=temp_dir
                    )
                    # Extract diameters BEFORE stitching
                    all_diams = []
                    if use_disk_storage:
                        for filepath in tile_results:
                            try:
                                _, _, _, diams, _ = load_tile_result_from_disk(filepath)
                                if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                    all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                            except Exception:
                                pass
                    else:
                        for _, _, _, diams, _ in tile_results:
                            if isinstance(diams, (list, np.ndarray)) and len(diams) > 0:
                                all_diams.extend(diams if isinstance(diams, list) else diams.tolist())
                    diams = np.mean(all_diams) if all_diams else None
                    
                    masks = stitch_tiles(tile_results, tiles, image_shape, overlap_pixels,
                                       use_disk_storage=use_disk_storage, temp_dir=temp_dir)
                    
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
    
    # Cleanup: Release image memory and clean up tile cache if it exists
    if imgs is not None:
        del imgs
    gc.collect()
    
    # Final cleanup of tile cache directory if it still exists (from previous runs or current run)
    if use_tiling and 'temp_dir' in locals() and temp_dir and os.path.exists(temp_dir):
        cache_dir = temp_dir if os.path.isdir(temp_dir) else None
        if cache_dir and os.path.exists(cache_dir):
            print(f"\n🧹 Final cleanup of tile cache directory...", flush=True)
            try:
                # Remove all remaining files (including any old ones from previous runs)
                for filename in os.listdir(cache_dir):
                    filepath = os.path.join(cache_dir, filename)
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                    except Exception:
                        pass
                print(f"  ✅ Tile cache directory cleaned up", flush=True)
            except Exception as cleanup_err:
                print(f"  ⚠️  Warning: Could not clean tile cache directory: {cleanup_err}", flush=True)
    
    return masks, flows, styles, diams


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Run Cellpose segmentation on DAPI and VIM channels with resource-aware tiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cellpose_segmentation.py data/DAPI.tif data/VIM.tif
  python cellpose_segmentation.py data/DAPI.tif data/VIM.tif -o output/cellpose_output --gpu
  python cellpose_segmentation.py data/DAPI.tif data/VIM.tif -m cyto2 -d 30
  python cellpose_segmentation.py data/DAPI.tif data/VIM.tif --tile-size 2048 2048
  python cellpose_segmentation.py data/DAPI.tif data/VIM.tif --overlap 200 --gpu-memory-limit 4.0

Debug mode (for troubleshooting memory issues):
  CELLPOSE_DEBUG=1 python cellpose_segmentation.py data/DAPI.tif data/VIM.tif
        """
    )
    
    parser.add_argument('nuclear_file', type=str,
                       help='Path to nuclear channel (DAPI) TIF file')
    parser.add_argument('cyto_file', type=str,
                       help='Path to cytoplasmic channel (VIM) TIF file')
    parser.add_argument('-o', '--output', type=str, default='output/cellpose_output',
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
    parser.add_argument('--grid-size', type=int, default=None,
                       help='Start grid size for progressive tiling: N means N×N (default: 2)')
    parser.add_argument('--max-grid-size', type=int, default=10,
                       help='Maximum grid size for progressive tiling retries (default: 10)')
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
            gpu_memory_limit=args.gpu_memory_limit,
            grid_size=args.grid_size,
            max_grid_size=args.max_grid_size
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