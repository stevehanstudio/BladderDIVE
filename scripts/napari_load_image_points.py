#!/usr/bin/env python3
"""Backward-compatible alias for Jupyter/SpatialData workflows.

Prefer importing ``open_napari_with_adata`` from ``napari_load_image_adata``;
this module exists so ``from scripts.napari_load_image_points import open_napari_with_points`` works.
"""

from scripts.napari_load_image_adata import open_napari_with_adata as open_napari_with_points

__all__ = ["open_napari_with_points"]
