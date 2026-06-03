#!/usr/bin/env python3
"""
Load CellDIVE image + cell centroids colored by annotation (cell type, Leiden, etc.) in Napari.

Display modes:
  - "points": centroid only (one point per cell)
  - "cells": full cell area as solid color (requires mask)
  - "both": add both; toggle layer visibility to switch

The mask layer is the Cellpose segmentation; by default it shows outline only (transparent fill).

Usage from notebook:
  open_napari_with_adata(adata, "cell_type", display_mode="cells", mask_path=...)

SpatialData / table views: pass a concrete AnnData, e.g. ``open_napari_with_adata(sdata["table"].copy(), "leiden", ...)``.
"""

import json
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import pandas as pd
import zarr

try:
    from napari.utils import DirectLabelColormap
except ImportError:
    DirectLabelColormap = None


def _get_project_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def _load_image_pyramid(viewer, zarr_path: Path) -> None:
    """Load CellDIVE zarr as multiscale image with channel names and colors."""
    group = zarr.open(str(zarr_path), mode="r")
    pyramid = [da.from_zarr(group[str(i)]) for i in range(5)]
    n_channels = pyramid[0].shape[0]

    zattrs_path = zarr_path / ".zattrs"
    names, colors = [], []
    channels_meta = []
    if zattrs_path.exists():
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
        blending="additive",
        contrast_limits=[[0, 3000]] * n_channels,
    )


def _qualitative_colors(n: int) -> list:
    """Return n visually distinct colors for categorical data."""
    import matplotlib.pyplot as plt

    try:
        cmap = plt.colormaps["tab20"]
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")
    if n <= 20:
        return [plt.matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
    try:
        cmap2 = plt.colormaps["tab20b"]
    except AttributeError:
        cmap2 = plt.cm.get_cmap("tab20b")
    colors = [plt.matplotlib.colors.to_hex(cmap(i / 19)) for i in range(20)]
    for i in range(min(n - 20, 20)):
        colors.append(plt.matplotlib.colors.to_hex(cmap2(i / 19)))
    if n > 40:
        try:
            cmap3 = plt.colormaps["Set3"]
        except AttributeError:
            cmap3 = plt.cm.get_cmap("Set3")
        for i in range(min(n - 40, 12)):
            colors.append(plt.matplotlib.colors.to_hex(cmap3(i / 11)))
    return colors[:n]


def _hex_to_rgba(hex_color: str) -> tuple:
    """Convert '#RRGGBB' to (r, g, b, a) in [0, 1]."""
    import matplotlib.colors as mcolors

    rgb = mcolors.hex2color(hex_color if hex_color.startswith("#") else f"#{hex_color}")
    return (*rgb, 1.0)


def _cell_ids_for_mask(adata) -> np.ndarray:
    """Integer IDs per row matching Cellpose label indices (``obs['cell_id']`` or numeric ``obs_names``)."""
    if "cell_id" in adata.obs.columns:
        return np.asarray(adata.obs["cell_id"], dtype=np.int64)
    try:
        return np.asarray(pd.to_numeric(pd.Index(adata.obs_names), errors="raise"), dtype=np.int64)
    except (ValueError, TypeError) as e:
        raise ValueError(
            "Cells layer needs integer segmentation IDs: use obs['cell_id'] (recommended) or numeric obs_names. "
            f"obs.columns: {list(adata.obs.columns)[:20]}"
        ) from e


def _load_cells_labels_layer(
    viewer: napari.Viewer,
    mask_path: Path,
    adata,
    color_column: str,
    scale_level: int = 0,
    per_type: bool = False,
    color_mapping: dict[str, str] | None = None,
    background_color: tuple[float, float, float, float] = (0, 0, 0, 1),
    layer_name: str | None = None,
) -> None:
    """
    Create Labels layer(s) with cells filled by cell-type color.

    per_type: If True, one layer per cell type (toggle each on/off).
              If False, one composite layer with all types.
    scale_level: 0=full res (memory heavy), 2=4x down (default), etc.
    """
    mask_group = zarr.open(str(mask_path), mode="r")
    level_keys = sorted([k for k in mask_group.keys() if str(k).isdigit()], key=int)
    if scale_level >= len(level_keys):
        scale_level = 0
    mask_da = da.from_zarr(mask_group[level_keys[scale_level]])

    # Get scale from mask metadata so labels align with image
    scale_xy = (1.0, 1.0)
    zattrs_path = mask_path / ".zattrs"
    if zattrs_path.exists():
        with open(zattrs_path) as f:
            meta = json.load(f)
        datasets = meta.get("multiscales", [{}])[0].get("datasets", [])
        if scale_level < len(datasets):
            scale_xy = tuple(datasets[scale_level]["coordinateTransformations"][0]["scale"])

    cell_to_type = dict(zip(_cell_ids_for_mask(adata), adata.obs[color_column].astype(str)))
    unique = pd.unique(adata.obs[color_column].astype(str))
    if color_mapping:
        colors = [color_mapping.get(str(cat), "#808080") for cat in unique]
    else:
        colors = _qualitative_colors(len(unique))
    max_label = int(mask_da.max().compute())

    if per_type:
        # One Labels layer per cell type — each toggleable
        type_to_cell_ids = {t: set() for t in unique}
        for cid, ct in cell_to_type.items():
            if ct in type_to_cell_ids:
                type_to_cell_ids[ct].add(cid)

        for i, cat in enumerate(unique):
            cell_ids = type_to_cell_ids[cat]
            if not cell_ids:
                continue
            mapping = np.zeros(max_label + 1, dtype=np.uint8)
            for cid in cell_ids:
                if 0 <= cid <= max_label:
                    mapping[cid] = 1

            def remap_block(block, m=mapping):
                out = np.take(m, np.clip(block.ravel(), 0, max_label)).reshape(block.shape)
                return out.astype(np.uint8)

            binary = da.map_blocks(remap_block, mask_da, dtype=np.uint8, drop_axis=[], new_axis=[])
            color_dict = {0: background_color, 1: _hex_to_rgba(colors[i]), None: background_color}
            layer = viewer.add_labels(binary, name=cat, opacity=0.7, scale=scale_xy)
            if DirectLabelColormap is not None:
                layer.colormap = DirectLabelColormap(color_dict=color_dict)
            else:
                layer.colormap = color_dict

        print(f"Loaded {len(unique)} cell-type layers (scale level {scale_level})")
    else:
        # One composite Labels layer
        type_to_idx = {t: i + 1 for i, t in enumerate(unique)}
        mapping = np.zeros(max_label + 1, dtype=np.uint16)
        for cid, ct in cell_to_type.items():
            if 0 <= cid <= max_label:
                mapping[cid] = type_to_idx.get(ct, 0)

        def remap_block(block):
            out = np.take(mapping, np.clip(block.ravel(), 0, max_label)).reshape(block.shape)
            return out.astype(np.uint16)

        remapped = da.map_blocks(remap_block, mask_da, dtype=np.uint16, drop_axis=[], new_axis=[])
        color_dict = {0: background_color, None: background_color}
        for i, cat in enumerate(unique):
            color_dict[i + 1] = _hex_to_rgba(colors[i])

        name = layer_name if layer_name is not None else f"Cells ({color_column})"
        labels_layer = viewer.add_labels(
            remapped,
            name=name,
            opacity=0.7,
            scale=scale_xy,
        )
        if DirectLabelColormap is not None:
            labels_layer.colormap = DirectLabelColormap(color_dict=color_dict)
        else:
            labels_layer.colormap = color_dict

        print(f"Loaded cells as filled Labels (scale level {scale_level})")


def _load_mask_layer(
    viewer: napari.Viewer,
    mask_path: Path,
    scale_level: int = 2,
    contour_only: bool = True,
    contour_width: int | float = 1,
) -> None:
    """Load the Cellpose segmentation mask as a Labels layer (outline only by default).

    Uses full multiscale pyramid (like napari_load_raw+mask.py) so contour=1 stays thin
    at all zoom levels; single-level loading at scale_level=2 made outlines appear thick.
    """
    mask_group = zarr.open(str(mask_path), mode="r")
    level_keys = sorted([k for k in mask_group.keys() if str(k).isdigit()], key=int)
    if not level_keys:
        raise ValueError(f"No pyramid levels in mask zarr: {mask_path}")

    # Load full pyramid (same as napari_load_raw+mask.py) for thin outline at all zooms
    data = [da.from_zarr(mask_group[k]) for k in level_keys]
    layer = viewer.add_labels(data, name="Cellpose segmentation (outline)", opacity=0.9, multiscale=True)
    if contour_only:
        layer.contour = contour_width  # 1 = thin 1px outline
    # Match napari_load_raw+mask.py: {None: color} only (label 0 is transparent by default)
    if DirectLabelColormap is not None:
        layer.colormap = DirectLabelColormap(color_dict={None: (1.0, 1.0, 1.0, 0.9)})
    print(f"Loaded Cellpose segmentation as Labels layer (multiscale pyramid, contour={contour_width})")


def open_napari_with_adata(
    adata,
    color_column: str,
    project_root: Path | None = None,
    image_zarr: str | Path | None = None,
    mask_path: str | Path | None = None,
    display_mode: str = "both",
    max_points: int = 100_000,
    point_size: float = 2.0,
    mask_scale_level: int = 2,
    cells_per_type: bool = False,
    add_mask_layer: bool = True,
    mask_contour_only: bool = True,
    mask_contour_width: int | float = 1,
    color_mapping: dict[str, str] | None = None,
    cells_background_color: tuple[float, float, float, float] = (0, 0, 0, 1),
    cells_layer_name: str | None = None,
) -> napari.Viewer:
    """
    Open Napari with CellDIVE image and cell annotations.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with obs[color_column] and centroids in obs or obsm['spatial'].
    color_column : str
        Column in adata.obs (e.g. 'cell_type', 'leiden').
    project_root : Path, optional
        Project root.
    image_zarr : str | Path, optional
        Path to CellDIVE zarr.
    mask_path : str | Path, optional
        Path to segmentation mask zarr.
        Required for display_mode "cells" or "both".
    display_mode : str
        "points" = centroid only; "cells" = full cell area; "both" = both (toggle visibility).
    max_points : int
        Subsample points if larger.
    point_size : float
        Size of centroid points.
    mask_scale_level : int
        Mask pyramid level for cells (0=full res, 2=4x down for memory).
    cells_per_type : bool
        If True, one Labels layer per cell type (toggle each on/off).
        If False, one composite layer.
    add_mask_layer : bool
        If True, add the Cellpose segmentation mask as a Labels layer.
    mask_contour_only : bool
        If True, show only outline (transparent fill); if False, filled.
    mask_contour_width : int | float
        Outline width in pixels (default 1; try 0.5 for thinner).
    color_mapping : dict[str, str], optional
        Map category -> hex color (e.g. {"DAPI": "#00FF00", "cell": "#808080"}).
    cells_background_color : tuple[float, float, float, float]
        RGBA for non-cell areas (default (0,0,0,1) = black).
    cells_layer_name : str, optional
        Custom name for the cells Labels layer (e.g. "Specks / noise (by dominant channel)").

    Returns
    -------
    napari.Viewer
    """
    if getattr(adata, "is_view", False):
        adata = adata.copy()

    if color_column not in adata.obs.columns:
        cols = list(adata.obs.columns)
        preview = cols[:35] if len(cols) > 35 else cols
        more = f" (+{len(cols) - 35} more)" if len(cols) > 35 else ""
        raise KeyError(
            f"obs[{color_column!r}] not found. Run clustering / cell typing first, or pass an existing column. "
            f"For cell_type labels, use output/celldive_protein_matrix_celltypes.h5ad (not celldive_protein_matrix.h5ad). "
            f"Available obs columns: {preview}{more}"
        )

    root = project_root or _get_project_root()
    zarr_path = Path(image_zarr) if image_zarr else root / "data" / "CellDIVE_SLIDE-045.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Image zarr not found: {zarr_path}")

    mask_path_resolved = Path(mask_path) if mask_path else root / "output" / "cellpose_output" / "cellpose_masks_dapi_only_9tiles.zarr"
    if display_mode in ("cells", "both") and not mask_path_resolved.exists():
        raise FileNotFoundError(f"Mask not found for cells mode: {mask_path_resolved}")
    if add_mask_layer and not mask_path_resolved.exists():
        raise FileNotFoundError(f"Mask not found for add_mask_layer: {mask_path_resolved}")

    viewer = napari.current_viewer()
    if viewer is None:
        viewer = napari.Viewer()

    _load_image_pyramid(viewer, zarr_path)

    if add_mask_layer:
        _load_mask_layer(viewer, mask_path_resolved, mask_scale_level, contour_only=mask_contour_only, contour_width=mask_contour_width)

    labels = np.asarray(adata.obs[color_column].astype(str))
    unique = pd.unique(labels)
    colors = _qualitative_colors(len(unique))

    if display_mode in ("cells", "both"):
        _load_cells_labels_layer(
            viewer, mask_path_resolved, adata, color_column, mask_scale_level,
            per_type=cells_per_type, color_mapping=color_mapping,
            background_color=cells_background_color,
            layer_name=cells_layer_name,
        )

    if display_mode in ("points", "both"):
        if "centroid_x" in adata.obs.columns and "centroid_y" in adata.obs.columns:
            x = np.asarray(adata.obs["centroid_x"])
            y = np.asarray(adata.obs["centroid_y"])
        elif "spatial" in adata.obsm:
            xy = np.asarray(adata.obsm["spatial"])
            x, y = xy[:, 0], xy[:, 1]
        else:
            raise ValueError("No centroids in obs (centroid_x/y) or obsm['spatial']")

        points_all = np.column_stack([y, x])
        n = len(points_all)
        if n > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, max_points, replace=False)
            points_all = points_all[idx]
            labels = labels[idx]

        total_added = 0
        for i, cat in enumerate(unique):
            mask = labels == cat
            if mask.sum() == 0:
                continue
            pts = points_all[mask]
            viewer.add_points(pts, face_color=colors[i], size=point_size, name=f"{cat} (pts)")
            total_added += len(pts)

        print(f"Loaded {total_added} cells as {len(unique)} point layers")

    return viewer


def main():
    import argparse

    import anndata as ad

    parser = argparse.ArgumentParser(description="Load CellDIVE image + cell points in Napari")
    parser.add_argument("h5ad", nargs="?", default=None, help="Path to H5AD with cell annotations")
    parser.add_argument("column", nargs="?", default="cell_type", help="obs column for colors")
    parser.add_argument("--max-points", type=int, default=100_000, help="Max points to display")
    parser.add_argument("--mode", choices=["points", "cells", "both"], default="both", help="Display mode")
    parser.add_argument("--cells-per-type", action="store_true", help="One Labels layer per cell type")
    parser.add_argument("--no-mask-layer", action="store_true", help="Do not add DAPI mask as a layer")
    args = parser.parse_args()

    root = _get_project_root()
    h5ad_path = Path(args.h5ad) if args.h5ad else root / "output" / "celldive_protein_matrix_celltypes.h5ad"
    if not h5ad_path.exists():
        h5ad_path = root / "output" / "celldive_protein_matrix.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"No H5AD found. Tried: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path)
    if args.column not in adata.obs.columns:
        raise ValueError(f"Column '{args.column}' not in adata.obs. Available: {list(adata.obs.columns)}")

    open_napari_with_adata(
        adata,
        args.column,
        project_root=root,
        max_points=args.max_points,
        display_mode=args.mode,
        cells_per_type=args.cells_per_type,
        add_mask_layer=not args.no_mask_layer,
    )
    napari.run()


if __name__ == "__main__":
    main()
