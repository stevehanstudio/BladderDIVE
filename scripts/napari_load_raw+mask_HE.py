#!/usr/bin/env python3
"""
Load raw CellDIVE image channels, optional H&E stain, and segmentation masks into Napari.

This script loads:
1. Raw multi-channel image from Zarr with proper channel names and colors
2. Optional H&E stain image (pre-registered or for manual registration)
3. One or more segmentation masks as outline-only labels layers

H&E options:
  - Pre-registered: Run register_he_to_celldive.py first; script will load H&E with correct alignment.
  - Manual registration: Set HE_IMAGE_PATH only; load H&E, then use Plugins > Manual Registration.

Usage:
  - Run with Napari: napari scripts/napari_load_raw+mask_HE.py
  - Or paste into the Napari console after opening Napari.
"""

import zarr
import dask.array as da
import napari
import tifffile
from pathlib import Path
try:
    # Optional: for closing dock widgets like "Channel Legend"
    from qtpy.QtWidgets import QDockWidget
except Exception:
    QDockWidget = None

try:
    from napari.utils import DirectLabelColormap
    HAS_DIRECT_COLORMAP = True
except ImportError:
    HAS_DIRECT_COLORMAP = False

import json
import os

# --- H&E CONFIG ---
WORKSPACE = Path(__file__).resolve().parent.parent
# Pre-registered: path to transform JSON (from register_he_to_celldive.py). If exists, H&E loads aligned.
HE_TRANSFORM_PATH = WORKSPACE / "data" / "he_celldive_registration" / "he_celldive_transform.json"
# Unregistered: H&E image path (for manual registration). Used if no transform, or as fallback.
HE_IMAGE_PATH = WORKSPACE / "data" / "HE_celldive_resolution.ome.tif"  # or HE_registered_to_celldive.tif

# CellDIVE spatial resolution (µm/pixel); masks must match for correct overlay
CELLDIVE_MPP = 0.325


def close_dock_widget(viewer, title: str) -> bool:
    """
    Close a Napari dock widget by its window title.

    Returns True if something was closed, else False.
    """
    # 1) If a helper from napari_print.py is in scope, use it.
    if title == "Channel Legend" and "hide_channel_legend" in globals():
        try:
            globals()["hide_channel_legend"]()
            return True
        except Exception:
            pass

    # 2) Try napari window's internal registry (varies by version).
    try:
        dock_map = getattr(viewer.window, "_dock_widgets", None)
        if isinstance(dock_map, dict) and title in dock_map:
            try:
                dock_map[title].close()
                return True
            except Exception:
                pass
    except Exception:
        pass

    # 3) Fall back: scan Qt dock widgets by title.
    if QDockWidget is not None:
        try:
            main = getattr(viewer.window, "_qt_window", None)
            if main is not None:
                for dw in main.findChildren(QDockWidget):
                    try:
                        if dw.windowTitle() == title:
                            dw.close()
                            return True
                    except Exception:
                        continue
        except Exception:
            pass

    return False


def prefer_zarr_if_available(mask_path: str) -> str:
    """
    Prefer a sibling `.zarr` pyramid over a `.tif/.tiff` mask if available.

    Rationale: Zarr pyramids are much faster/more stable to view in Napari than
    huge TIFF label images. This lets you keep configs in terms of the TIFFs you
    produced, while automatically using the better `.zarr` representation when
    it exists.
    """
    p = Path(mask_path)

    # If user points to a TIFF, but a same-stem .zarr exists, prefer it.
    if p.suffix.lower() in {".tif", ".tiff"}:
        z = p.with_suffix(".zarr")
        if z.is_dir():
            return str(z)

    # If user points to a Zarr that doesn't exist, but a same-stem TIFF exists, fall back.
    if p.suffix.lower() == ".zarr" and not p.exists():
        for ext in (".tif", ".tiff"):
            t = p.with_suffix(ext)
            if t.is_file():
                return str(t)

    return str(p)


def load_mask(viewer, mask_store, name, color_rgba, contour=1, tif_chunks=(1024, 1024), scale=(CELLDIVE_MPP, CELLDIVE_MPP)):
    """
    Load a mask (Zarr pyramid or TIFF) into the Napari viewer as an outline-only labels layer.

    Parameters
    ----------
    viewer : napari.Viewer
        Current Napari viewer.
    mask_store : str
        Path to the mask `.zarr` directory OR a `.tif/.tiff` labels file.
    name : str
        Layer name in the viewer.
    color_rgba : list of float
        RGBA color for all labels, e.g. [1, 1, 0.8, 1] for light yellow.
    contour : int
        Outline width in pixels (default 1).
    tif_chunks : tuple[int, int]
        Dask chunk size for TIFF masks (default (1024, 1024)).

    Returns
    -------
    napari.layers.Labels or None
        The added labels layer, or None if the mask file was not found.
    """
    if not os.path.exists(mask_store):
        print(f"Mask not found: {mask_store}")
        return None

    mask_path = Path(mask_store)

    # Zarr multiscale pyramid (preferred for large masks)
    if mask_path.suffix.lower() == ".zarr" and mask_path.is_dir():
        mask_group = zarr.open(str(mask_path), mode="r")
        level_keys = sorted([k for k in mask_group.keys() if str(k).isdigit()], key=lambda x: int(x))
        if not level_keys:
            print(f"No pyramid levels found in zarr: {mask_store}")
            return None
        data = [da.from_zarr(mask_group[k]) for k in level_keys]
        multiscale = True

    # TIFF labels (lazy via memmap + dask)
    elif mask_path.suffix.lower() in {".tif", ".tiff"} and mask_path.is_file():
        mm = tifffile.memmap(str(mask_path))
        # If this TIFF has extra singleton dims, squeeze them out (common for some writers)
        if mm.ndim > 2:
            mm = mm.squeeze()
        if mm.ndim != 2:
            raise ValueError(f"Expected 2D labels TIFF, got shape {mm.shape} from {mask_store}")
        data = da.from_array(mm, chunks=tif_chunks)
        multiscale = False

    else:
        raise ValueError(f"Unsupported mask path (expected .zarr or .tif/.tiff): {mask_store}")

    labels_layer = viewer.add_labels(
        data,
        name=name,
        multiscale=multiscale,
        opacity=1.0,
        scale=scale,
    )
    labels_layer.contour = contour

    try:
        if HAS_DIRECT_COLORMAP:
            labels_layer.colormap = DirectLabelColormap(color_dict={None: color_rgba})
        else:
            labels_layer.colormap = {None: color_rgba}
    except Exception as e:
        print(f"Could not set colormap for '{name}': {e}")

    print(f"Loaded: {name}")
    return labels_layer


def load_he(viewer):
    """
    Load H&E stain layer. If HE_TRANSFORM_PATH exists, applies pre-registration.
    Otherwise loads H&E for manual registration via Plugins > Manual Registration.
    """
    import numpy as np

    # Pre-registered: load with transform
    if HE_TRANSFORM_PATH.exists():
        try:
            with open(HE_TRANSFORM_PATH) as f:
                t = json.load(f)
            he_path = Path(t.get("he_path", ""))
            if not he_path.is_absolute():
                he_path = WORKSPACE / he_path
            if not he_path.exists():
                he_path = WORKSPACE / "data" / "he_celldive_registration" / "HE_registered_to_celldive.tif"
            if he_path.exists():
                he = tifffile.imread(str(he_path))
                if he.ndim == 2:
                    he = np.stack([he, he, he], axis=-1)
                elif he.ndim == 3 and he.shape[2] == 4:
                    he = he[:, :, :3]
                scale = t.get("scale", [0.325, 0.325])
                if isinstance(scale, (int, float)):
                    scale = [scale, scale]
                translate = t.get("translate", [0, 0])
                viewer.add_image(
                    he,
                    name="H&E",
                    rgb=True,
                    blending="translucent",
                    opacity=0.7,
                    scale=(scale[0], scale[1]),
                    translate=(translate[0], translate[1]),
                )
                print("Loaded: H&E (pre-registered)")
                return
        except Exception as e:
            print(f"Could not load pre-registered H&E: {e}")

    # Fallback: load H&E for manual registration (Plugins > Manual Registration)
    for cand in [HE_IMAGE_PATH, WORKSPACE / "data" / "he_celldive_registration" / "HE_registered_to_celldive.tif", WORKSPACE / "data" / "HE_celldive_resolution.tif"]:
        if cand.exists():
            he = tifffile.imread(str(cand))
            if he.ndim == 2:
                he = np.stack([he, he, he], axis=-1)
            elif he.ndim == 3 and he.shape[2] == 4:
                he = he[:, :, :3]
            viewer.add_image(
                he,
                name="H&E",
                rgb=True,
                blending="translucent",
                opacity=0.7,
                scale=(0.325, 0.325),
            )
            print("Loaded: H&E (use Plugins > Manual Registration to align)")
            return
    print("H&E not found. Set HE_IMAGE_PATH or run register_he_to_celldive.py first.")


# --- Napari viewer ---
# Use existing viewer (when pasted into console) or create one (when run via napari script.py)
viewer = napari.current_viewer()
if viewer is None:
    viewer = napari.Viewer()

# --- 0. CLEAR EXISTING LAYERS ---
# Clear UI elements that don't get removed by viewer.layers.clear()
close_dock_widget(viewer, "Channel Legend")
viewer.layers.clear()
print("Cleared existing layers")

# --- 1. LOAD RAW IMAGE ---
store_path = "data/CellDIVE_SLIDE-045.zarr"
group = zarr.open(store_path, mode='r')
pyramid = [da.from_zarr(group[str(i)]) for i in range(5)]
n_channels = pyramid[0].shape[0]

zattrs_path = os.path.join(store_path, ".zattrs")
names, colors = [], []
channels_meta = []
if os.path.exists(zattrs_path):
    with open(zattrs_path) as f:
        meta = json.load(f)
        channels_meta = meta.get("omero", {}).get("channels", [])
        if not channels_meta and "multiscales" in meta:
            channels_meta = meta["multiscales"][0].get("omero", {}).get("channels", [])

for i in range(n_channels):
    if i < len(channels_meta):
        names.append(channels_meta[i].get("label", f"Ch{i}"))
        c = channels_meta[i].get("color", "FFFFFF")
        colors.append(f"#{c}" if not c.startswith("#") else c)
    else:
        names.append(f"Ch{i}")
        colors.append("gray")

viewer.add_image(
    pyramid,
    channel_axis=0,
    name=names,
    colormap=colors,
    multiscale=True,
    blending='additive',
    contrast_limits=[[0, 3000]] * n_channels,
    scale=(0.325, 0.325),
)

# --- 2. LOAD H&E (optional) ---
load_he(viewer)

# --- 3. LOAD MASKS (add more entries to compare Cellpose outputs) ---
# Format: (path, layer_name, RGBA)
MASK_CONFIGS = [
    ("./output/cellpose_output/cellpose_masks_dapi_only_9tiles.zarr", "DAPI only Cell Segmentation", [1.0, 1.0, 1.0, 0.9]),   # white
    ("./output/cellpose_output/cellpose_masks_dapi+vim_9tiles.zarr", "DAPI + VIM Cell Segmentation", [1.0, 1.0, 0.0, 0.9]),   # yellow
    ("./output/cellpose_output/cellpose_masks_dapi+panck_9tiles.zarr", "DAPI + PanCK Cell Segmentation", [0.75, 1.0, 0.75, 0.9]),  # light green
    ("./output/cellpose_output/cellpose_masks_dapi+panck+cd45_9tiles.zarr", "DAPI + PanCK + CD45RO Cell Segmentation", [0.75, 0.88, 1.0, 0.9]),  # light blue
]

for path, layer_name, color in MASK_CONFIGS:
    load_mask(viewer, prefer_zarr_if_available(path), layer_name, color)

# Keep viewer open when run as entry point (e.g. python script.py or napari script.py)
napari.run()
