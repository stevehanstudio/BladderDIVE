#!/usr/bin/env python3
"""
Napari utility functions for viewing, legends, and bookmarks.

This module is intended to be imported or pasted into the Napari console, e.g.:

```python
import napari
from scripts import napari_utils as nu

tools = nu.NapariTools()  # uses napari.current_viewer()
tools.config_scalebar()
tools.register_bookmarks()  # 'b' to save, 'n' to next
tools.add_zoom_box()
tools.increase_zoom_box(percent=25)   # 25% bigger
tools.increase_zoom_box(percent=-10)  # 10% smaller
tools.save_view()    # ./screenshots/view.png
tools.save_legend()  # ./screenshots/channel_legend.png (single-line legend PNG)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _get_viewer(viewer=None):
    """
    Return a Napari viewer.

    If `viewer` is None, attempts to use `napari.current_viewer()`.
    """
    if viewer is not None:
        return viewer

    try:
        import napari  # imported lazily so this module can be imported outside napari

        v = napari.current_viewer()
        if v is None:
            raise RuntimeError("No active Napari viewer found.")
        return v
    except Exception as e:
        raise RuntimeError(
            "No viewer provided and no active Napari viewer found. "
            "Pass `viewer=...` or run from the Napari console with an open viewer."
        ) from e


def save_view(viewer=None, path: str = "./screenshots/view.png", *, canvas_only: bool = True) -> str:
    """
    Save the current Napari viewport to a PNG.

    Parameters
    ----------
    viewer : napari.Viewer
        Active Napari viewer.
    path : str
        Output PNG path.
    canvas_only : bool
        If True, saves only the canvas/viewport (no UI). If False, saves the full window.

    Returns
    -------
    str
        The output path.
    """
    viewer = _get_viewer(viewer)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    viewer.screenshot(path=str(out), canvas_only=canvas_only)
    return str(out)

def save_view_png(viewer=None, path: str = "./screenshots/view.png", *, canvas_only: bool = True) -> str:
    """Backward-compatible alias for `save_view`."""
    return save_view(viewer, path, canvas_only=canvas_only)


def add_zoom_box(
    viewer=None,
    *,
    side_length: Optional[float] = None,
    border_width: float = 400.0,
    fill_alpha: float = 0.0,
    layer_name: str = "Zoom Box",
    debug: bool = False,
) -> Any:
    """
    Create a new Shapes layer with a centered square overlay.

    The square is placed at the center of the current viewport/camera and is:
    - A perfect square in data coordinates (side_length by side_length)
    - Translucent fill (white with alpha = fill_alpha)
    - Solid white border with thickness = border_width (in screen pixels)

    Parameters
    ----------
    viewer : napari.Viewer
        Active Napari viewer.
    side_length : float
        Side length of the square in **world coordinates**. If None (default), uses
        \(\\min(\\text{viewport width}, \\text{viewport height}) / 5\\) based on the
        current camera zoom and canvas size.
    border_width : float
        Border thickness baseline in pixels (default 80).
        If side_length is auto-computed, the border is scaled by the fraction of the
        full image currently visible, so it gets thinner as you zoom in.
    fill_alpha : float
        Alpha for the fill color (0..1). Use 0 for fully transparent fill.
    layer_name : str
        Name of the new Shapes layer.
    debug : bool
        If True, prints diagnostic information about viewport sizing and zoom.

    Returns
    -------
    napari.layers.Shapes
        The created Shapes layer.
    """
    viewer = _get_viewer(viewer)

    if debug:
        print("[add_zoom_box] debug=True")

    def _get_canvas_size_px(v):
        """
        Best-effort retrieval of the napari canvas size in screen pixels.

        Returns
        -------
        (w_px, h_px, source) or (None, None, source)
        """
        # Try a few napari/Qt layouts across versions
        candidates = []
        candidates.append(getattr(v.window, "_qt_viewer", None))
        qt_win = getattr(v.window, "_qt_window", None)
        if qt_win is not None:
            candidates.append(getattr(qt_win, "_qt_viewer", None))

        for obj in candidates:
            if obj is None:
                continue
            canvas = getattr(obj, "canvas", None)
            if canvas is None:
                continue
            try:
                size = canvas.size  # typically (w_px, h_px)
                w_px, h_px = float(size[0]), float(size[1])
                return w_px, h_px, "canvas.size"
            except Exception:
                pass

        return None, None, "unavailable"

    # Compute a default side length from the current viewport (in world coords).
    # NOTE: viewer.camera.center/zoom are in world coordinates.
    if side_length is None:
        w_px, h_px, canvas_src = _get_canvas_size_px(viewer)

        zoom = float(getattr(viewer.camera, "zoom", 1.0) or 1.0)  # px per world unit
        if w_px is not None and h_px is not None and zoom > 0:
            world_w = float(w_px) / zoom
            world_h = float(h_px) / zoom
            side_length = min(world_w, world_h) / 5.0
        else:
            # Fallback if canvas size isn't accessible
            side_length = 1000.0

        if debug or w_px is None or h_px is None:
            print(
                "[add_zoom_box] canvas_size_px="
                f"({w_px}, {h_px}) (src={canvas_src}), zoom={zoom:.3f}, "
                f"computed side_length={side_length:.3f} (world units)"
            )

    # Napari's camera center is in world coordinates. For 2D overlays, use last two dims (Y, X).
    center = viewer.camera.center
    cy, cx = float(center[-2]), float(center[-1])

    half = float(side_length) / 2.0

    # Border width scaling:
    # We want the border to be "40 px of the whole image" when zoomed in, i.e.
    # get thinner as the visible field-of-view shrinks. We approximate this by
    # scaling border_width by the fraction of the full image currently visible.
    zoom = float(getattr(viewer.camera, "zoom", 1.0) or 1.0)
    effective_border_width = float(border_width)
    try:
        w_px, h_px, _ = _get_canvas_size_px(viewer)
        if w_px is not None and h_px is not None and zoom > 0:
            viewport_world_min = min(float(w_px) / zoom, float(h_px) / zoom)

            from napari.layers import Image

            img_layers = [lyr for lyr in viewer.layers if isinstance(lyr, Image)]
            if img_layers:
                ext = img_layers[0].extent.world  # (2, ndim)
                span = ext[1] - ext[0]
                full_world_min = float(min(abs(span[-2]), abs(span[-1])))
                if full_world_min > 0:
                    frac_visible = max(0.0, min(1.0, viewport_world_min / full_world_min))
                    effective_border_width = float(border_width) * frac_visible
    except Exception:
        # If we can't determine extents, fall back to a simple zoom-based thinning.
        effective_border_width = float(border_width) / max(1.0, zoom)

    # Keep the border visible but not overpowering
    effective_border_width = max(1.0, min(float(border_width), effective_border_width))
    if debug:
        print(
            f"[add_zoom_box] border_width base={border_width:.1f}px, "
            f"effective={effective_border_width:.2f}px at zoom={zoom:.3f}"
        )

    # Rectangle vertices in (y, x) order
    rect = [
        [cy - half, cx - half],
        [cy - half, cx + half],
        [cy + half, cx + half],
        [cy + half, cx - half],
    ]

    layer = viewer.add_shapes(
        [rect],
        shape_type="rectangle",
        name=layer_name,
        edge_color=[1.0, 1.0, 1.0, 1.0],  # solid white
        face_color=[1.0, 1.0, 1.0, float(fill_alpha)],  # translucent white fill
        edge_width=float(effective_border_width),
        opacity=0.8,
    )

    return layer


def increase_zoom_box(
    viewer=None,
    *,
    percent: float = 10.0,
    layer_name: str = "Zoom Box",
    shape_index: Optional[int] = None,
    debug: bool = False,
) -> Any:
    """
    Increase (or decrease) the size of an existing zoom box by a percentage.

    This edits an existing Shapes layer (default name: "Zoom Box") in-place by
    scaling its rectangle about its center.

    By default, it will:
      1. Prefer the *active* Shapes layer (if its name contains `layer_name`)
      2. Use the currently selected shape in that layer (if any)
      3. Otherwise fall back to the first shape in the layer

    Parameters
    ----------
    viewer : napari.Viewer
        Active Napari viewer.
    percent : float
        Percentage change in side length. Positive increases size, negative decreases
        (e.g. 25 -> 1.25x, -20 -> 0.8x).
    layer_name : str
        Name (or name substring) of the Shapes layer containing the zoom box.
    shape_index : int, optional
        Which shape within the layer to modify. If None (default), uses the
        currently selected shape (if any), else the first shape.
    debug : bool
        If True, prints diagnostic information about the resize.

    Returns
    -------
    napari.layers.Shapes
        The modified Shapes layer.
    """
    viewer = _get_viewer(viewer)

    layer = None

    try:
        from napari.layers import Shapes
    except Exception:
        Shapes = None  # type: ignore[assignment]

    # 1) Prefer the active Shapes layer whose name matches/contains layer_name
    active = getattr(getattr(viewer, "layers", None), "selection", None)
    active = getattr(active, "active", None)
    if Shapes is not None and isinstance(active, Shapes):
        name = getattr(active, "name", "") or ""
        if not layer_name or layer_name in name:
            layer = active

    # 2) Fallback: search all layers for a (Shapes) layer matching layer_name
    if layer is None:
        for lyr in viewer.layers:
            if Shapes is not None and not isinstance(lyr, Shapes):
                continue
            name = getattr(lyr, "name", "") or ""
            if (layer_name and (name == layer_name or name.startswith(layer_name))) or (
                not layer_name
            ):
                layer = lyr
                break
    if layer is None:
        raise RuntimeError(f'No layer named "{layer_name}" found.')

    data = getattr(layer, "data", None)
    if data is None or len(data) == 0:
        raise RuntimeError(f'Layer "{layer_name}" has no shapes to resize.')

    # Decide which shape index to modify.
    if shape_index is None:
        # Prefer a selected shape, if available.
        selected = getattr(layer, "selected_data", None)
        if selected:
            idx = min(selected)
        else:
            idx = 0
    else:
        idx = int(shape_index)

    if idx < 0 or idx >= len(data):
        raise RuntimeError(
            f'Layer "{layer_name}" has {len(data)} shapes; shape_index={idx} is invalid.'
        )

    rect = data[idx]
    if rect is None or len(rect) < 4:
        raise RuntimeError(
            f'Expected a rectangle-like shape (>=4 vertices) at index {shape_index} in "{layer_name}".'
        )

    # Vertices are in (y, x) order for 2D. Center = mean of vertices.
    cy = float(sum(float(p[0]) for p in rect)) / float(len(rect))
    cx = float(sum(float(p[1]) for p in rect)) / float(len(rect))

    half_y = max(abs(float(p[0]) - cy) for p in rect)
    half_x = max(abs(float(p[1]) - cx) for p in rect)
    half = float(max(half_y, half_x))

    scale = 1.0 + (float(percent) / 100.0)
    if scale <= 0:
        raise RuntimeError(f"percent={percent} results in non-positive scale={scale}.")

    new_half = half * scale
    new_rect = [
        [cy - new_half, cx - new_half],
        [cy - new_half, cx + new_half],
        [cy + new_half, cx + new_half],
        [cy + new_half, cx - new_half],
    ]

    if debug:
        old_side = 2.0 * half
        new_side = 2.0 * new_half
        print(
            f'[increase_zoom_box] layer="{layer_name}" idx={idx} '
            f"center=({cy:.3f},{cx:.3f}) side {old_side:.3f} -> {new_side:.3f} (world units), "
            f"percent={percent:.2f}"
        )

    # Update in-place. Assigning via list() helps across napari versions.
    new_data = list(data)
    new_data[idx] = new_rect
    layer.data = new_data
    return layer

# ----------------------------
# Channel color legend (dock widget + PNG export)
# ----------------------------

try:
    from qtpy.QtCore import Qt
    from qtpy.QtCore import QPoint
    from qtpy.QtCore import QObject, QEvent
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
    from qtpy.QtGui import QImage, QPainter, QColor, QFont, QFontMetrics
except Exception:
    QWidget = None


def _rgba_to_hex(rgba) -> str:
    """Convert RGBA float[4] or int[4] to #RRGGBB."""
    if rgba is None:
        return "#CCCCCC"
    r, g, b = rgba[:3]
    if max(r, g, b) <= 1.0:
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _layer_swatch_hex(layer) -> str:
    """Best-effort single-color swatch for an Image layer's colormap."""
    try:
        cmap = getattr(layer, "colormap", None)
        # napari Colormap typically has `.colors` as an Nx4 array
        if hasattr(cmap, "colors") and len(cmap.colors) > 0:
            return _rgba_to_hex(cmap.colors[-1])
    except Exception:
        pass
    return "#CCCCCC"


@dataclass
class ChannelLegendState:
    """Internal state for the docked channel legend widget."""

    widget: Any
    layout: Any
    dock: Any
    anchor_filter: Optional[Any] = None


_CHANNEL_LEGEND_STATE: Dict[int, ChannelLegendState] = {}


def _position_channel_legend(viewer) -> None:
    """Anchor the floating legend near the upper-right of the Napari main window."""
    state = _CHANNEL_LEGEND_STATE.get(id(viewer))
    if state is None or state.dock is None:
        return
    try:
        main = getattr(viewer.window, "_qt_window", None)
        if main is None:
            return

        fg = main.frameGeometry()  # global coords
        margin_right = 20
        margin_top = 60
        state.dock.adjustSize()
        state.dock.move(
            fg.topRight() - QPoint(state.dock.width() + margin_right, -margin_top)
        )
    except Exception:
        pass


class _LegendAnchorFilter(QObject):
    """Reposition legend on main window move/resize."""

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if event.type() in (QEvent.Resize, QEvent.Move):
            _position_channel_legend(self._viewer)
        return False


def _simplify_channel_label(name: str) -> str:
    """
    Shorten verbose channel names for the legend.

    Examples:
      - "CD3E-AF555-CST" -> "CD3E"
      - "SMA-AF488" -> "SMA"
      - "DAPI_R01" -> "DAPI"
    """
    if not name:
        return name
    # Prefer the biomarker part before any dye/vendor suffixes.
    base = name.split("-", 1)[0]
    # For names like DAPI_R01, keep just DAPI.
    base = base.split("_", 1)[0]
    return base.strip() or name


def _rgba_any_to_qcolor(rgba) -> QColor:
    """
    Convert RGBA in either 0-1 floats or 0-255 ints to QColor.
    """
    if rgba is None:
        return QColor(0, 0, 0, 0)
    r, g, b, a = rgba
    if max(r, g, b, a) <= 1.0:
        r, g, b, a = [int(v * 255) for v in (r, g, b, a)]
    return QColor(int(r), int(g), int(b), int(a))


def save_legend(
    viewer=None,
    path: str = "./screenshots/channel_legend.png",
    *,
    font_size: int = 12,
    swatch_px: int = 14,
    pad_px: int = 8,
    gap_px: int = 30,
    text_swatch_gap_px: int = 6,
    background_rgba=(1.0, 1.0, 1.0, 1.0),
):
    """
    Save a single-line (horizontal) channel legend as a PNG.

    The legend includes ONLY currently visible Image layers. Each entry is:
      [color swatch] [label]

    Notes:
    - This does NOT overlay anything on the Napari viewport.
    - Labels are simplified (e.g. \"CD3E-AF555-CST\" -> \"CD3E\").
    """
    if QWidget is None:
        raise RuntimeError("qtpy is not available; cannot render legend PNG.")

    from napari.layers import Image

    viewer = _get_viewer(viewer)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    visible_imgs = [lyr for lyr in viewer.layers if isinstance(lyr, Image) and lyr.visible]
    entries = [(_layer_swatch_hex(lyr), _simplify_channel_label(lyr.name)) for lyr in visible_imgs]

    if not entries:
        entries = [("#CCCCCC", "No visible image channels")]

    font = QFont()
    font.setPointSize(int(font_size))
    fm = QFontMetrics(font)

    # Compute total width for one-line layout
    widths = []
    for _, label in entries:
        text_w = fm.horizontalAdvance(label)
        widths.append(swatch_px + text_swatch_gap_px + text_w)
    total_w = pad_px * 2 + sum(widths) + gap_px * (len(widths) - 1)
    total_h = pad_px * 2 + max(swatch_px, fm.height())

    img = QImage(int(total_w), int(total_h), QImage.Format_ARGB32)
    img.fill(_rgba_any_to_qcolor(background_rgba))

    painter = QPainter(img)
    painter.setFont(font)
    painter.setRenderHint(QPainter.Antialiasing, True)

    x = pad_px
    swatch_y = int((total_h - swatch_px) / 2)
    text_baseline = int((total_h + fm.ascent() - fm.descent()) / 2)

    for hex_color, label in entries:
        # swatch
        swatch = QColor(hex_color)
        painter.fillRect(int(x), int(swatch_y), int(swatch_px), int(swatch_px), swatch)
        painter.setPen(QColor("#333333"))
        painter.drawRect(int(x), int(swatch_y), int(swatch_px), int(swatch_px))

        # label
        # Match text color to the channel color
        painter.setPen(QColor(hex_color))
        painter.drawText(int(x + swatch_px + text_swatch_gap_px), int(text_baseline), label)

        x += swatch_px + text_swatch_gap_px + fm.horizontalAdvance(label) + gap_px

    painter.end()
    img.save(str(out_path))
    print(f"Saved channel legend PNG: {out_path}")
    return str(out_path)

def save_channel_legend_png(
    viewer=None,
    path: str = "./screenshots/channel_legend.png",
    *,
    font_size: int = 12,
    swatch_px: int = 14,
    pad_px: int = 8,
    gap_px: int = 30,
    text_swatch_gap_px: int = 6,
    background_rgba=(1.0, 1.0, 1.0, 1.0),
):
    """Backward-compatible alias for `save_legend`."""
    return save_legend(
        viewer,
        path,
        font_size=font_size,
        swatch_px=swatch_px,
        pad_px=pad_px,
        gap_px=gap_px,
        text_swatch_gap_px=text_swatch_gap_px,
        background_rgba=background_rgba,
    )


def _legend_clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()


def update_channel_legend():
    """Refresh the dock legend to match currently visible Image layers."""
    # Note: kept for backwards-compatibility; prefer update_channel_legend_for_viewer
    raise RuntimeError("Use update_channel_legend_for_viewer(viewer) instead.")


def update_channel_legend_for_viewer(viewer) -> None:
    """Refresh the dock legend to match currently visible Image layers."""
    viewer = _get_viewer(viewer)
    state = _CHANNEL_LEGEND_STATE.get(id(viewer))
    if state is None or state.layout is None:
        return

    from napari.layers import Image

    _legend_clear(state.layout)

    visible_imgs = [lyr for lyr in viewer.layers if isinstance(lyr, Image) and lyr.visible]
    if not visible_imgs:
        lbl = QLabel("No visible image channels")
        lbl.setStyleSheet("color: #666;")
        state.layout.addWidget(lbl)
        return

    for lyr in visible_imgs:
        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row.setLayout(row_layout)

        swatch = QLabel("")
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(
            f"background-color: {_layer_swatch_hex(lyr)}; border: 1px solid #333;"
        )

        name = QLabel(_simplify_channel_label(lyr.name))
        name.setTextInteractionFlags(Qt.TextSelectableByMouse)

        row_layout.addWidget(swatch)
        row_layout.addWidget(name, stretch=1)
        state.layout.addWidget(row)


def ensure_channel_legend():
    """
    Create/show the channel legend dock widget and keep it updated.

    Run once per Napari session (safe to re-run).
    """
    raise RuntimeError("Use ensure_channel_legend_for_viewer(viewer) instead.")


def ensure_channel_legend_for_viewer(viewer) -> None:
    """
    Create/show the channel legend dock widget for a specific viewer and keep it updated.

    This is a dock widget (UI), not an overlay; it will not appear in
    `viewer.screenshot(canvas_only=True)`.
    """
    viewer = _get_viewer(viewer)
    if QWidget is None:
        print("qtpy not available; cannot create dock legend widget.")
        return

    if id(viewer) not in _CHANNEL_LEGEND_STATE:
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)  # reduce vertical spacing between legend rows
        w.setLayout(layout)
        dock = viewer.window.add_dock_widget(w, name="Channel Legend", area="right")
        _CHANNEL_LEGEND_STATE[id(viewer)] = ChannelLegendState(widget=w, layout=layout, dock=dock)

        # Make it a compact floating panel anchored to upper-right.
        try:
            # Remove title bar to reduce wasted space.
            dock.setTitleBarWidget(QWidget())
            dock.setFloating(True)
            dock.adjustSize()

            # Position now…
            _position_channel_legend(viewer)

            # …and keep it anchored on resize/move.
            main = getattr(viewer.window, "_qt_window", None)
            state = _CHANNEL_LEGEND_STATE[id(viewer)]
            if main is not None and state.anchor_filter is None:
                state.anchor_filter = _LegendAnchorFilter(viewer)
                main.installEventFilter(state.anchor_filter)
        except Exception:
            # If any of the Qt internals change, we still keep the legend docked.
            pass

        # Update on layer list changes
        viewer.layers.events.inserted.connect(lambda e: update_channel_legend_for_viewer(viewer))
        viewer.layers.events.removed.connect(lambda e: update_channel_legend_for_viewer(viewer))
        viewer.layers.events.reordered.connect(lambda e: update_channel_legend_for_viewer(viewer))

    # Update on visibility/name/colormap changes for existing layers
    for lyr in list(viewer.layers):
        try:
            lyr.events.visible.connect(lambda e: update_channel_legend_for_viewer(viewer))
            lyr.events.name.connect(lambda e: update_channel_legend_for_viewer(viewer))
            if hasattr(lyr.events, "colormap"):
                lyr.events.colormap.connect(lambda e: update_channel_legend_for_viewer(viewer))
        except Exception:
            pass

    update_channel_legend_for_viewer(viewer)
    _position_channel_legend(viewer)


# Convenience helpers (run from Napari console)
def hide_channel_legend():
    raise RuntimeError("Use hide_channel_legend_for_viewer(viewer) instead.")


def hide_channel_legend_for_viewer(viewer) -> None:
    """Hide/remove the Channel Legend panel for a specific viewer if it exists."""
    viewer = _get_viewer(viewer)
    state = _CHANNEL_LEGEND_STATE.pop(id(viewer), None)
    if state is None:
        return
    try:
        if state.dock is not None:
            state.dock.close()
    except Exception:
        pass
    try:
        viewer.window.remove_dock_widget(state.widget)
    except Exception:
        pass

def config_scalebar(
    viewer=None,
    *,
    pixel_size_um: Optional[float] = None,
    unit: str = "µm",
    length: Optional[float] = None,
    font_size: int = 12,
    color: str = "white",
    box: bool = True,
    box_color: Tuple[float, float, float, float] = (0, 0, 0, 0.35),
) -> None:
    """
    Configure the Napari scale bar for accurate physical units.

    Parameters
    ----------
    viewer : napari.Viewer
        The active Napari viewer.
    pixel_size_um : float
        Microns per pixel (µm/px). Applied to the first Image layer scale.
    unit : str
        Unit string to display (default: "µm").
    length : float | None
        If None, Napari auto-selects a readable length based on zoom/FOV.
        If set, fixes the scale bar length in physical units.
    font_size : int
        Font size for scale bar text.
    color : str
        Color for the scale bar + text (e.g. "white").
    box : bool
        Whether to draw a background box behind the text/line.
    box_color : tuple
        RGBA background color for the box.
    """
    from napari.layers import Image

    viewer = _get_viewer(viewer)

    img_layers = [lyr for lyr in viewer.layers if isinstance(lyr, Image)]
    if img_layers and pixel_size_um is not None:
        img_layers[0].scale = [pixel_size_um, pixel_size_um]
    elif img_layers and pixel_size_um is None:
        # If no pixel size is provided, keep the current layer scale (do not override).
        pass

    viewer.scale_bar.unit = unit
    viewer.scale_bar.length = length
    viewer.scale_bar.visible = True
    viewer.scale_bar.font_size = int(font_size)
    viewer.scale_bar.colored = True
    viewer.scale_bar.color = color
    viewer.scale_bar.box = bool(box)
    viewer.scale_bar.box_color = box_color

def configure_scale_bar(
    viewer=None,
    *,
    pixel_size_um: Optional[float] = None,
    unit: str = "µm",
    length: Optional[float] = None,
    font_size: int = 12,
    color: str = "white",
    box: bool = True,
    box_color: Tuple[float, float, float, float] = (0, 0, 0, 0.35),
) -> None:
    """Backward-compatible alias for `config_scalebar`."""
    return config_scalebar(
        viewer,
        pixel_size_um=pixel_size_um,
        unit=unit,
        length=length,
        font_size=font_size,
        color=color,
        box=box,
        box_color=box_color,
    )

def register_bookmark_keys(
    viewer=None,
    *,
    save_key: str = "b",
    next_key: str = "n",
    store: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = True,
) -> List[Dict[str, Any]]:
    """
    Register a simple “bookmark” system on a viewer.

    - Press **save_key** (default: `b`) to save the current camera view.
    - Press **next_key** (default: `n`) to cycle through saved bookmarks.

    Parameters
    ----------
    viewer : napari.Viewer
        Active Napari viewer.
    save_key : str
        Key binding to save a bookmark (default: "b").
    next_key : str
        Key binding to jump to the next bookmark (default: "n").
    store : list[dict] | None
        Optional existing list to hold bookmark dicts. If None, a new list is created.
    overwrite : bool
        If True (default), overwrites any existing keybindings for save_key/next_key.

    Returns
    -------
    list[dict]
        The bookmark store list that will be appended/rotated.
    """
    viewer = _get_viewer(viewer)
    if store is None:
        store = []

    @viewer.bind_key(save_key, overwrite=overwrite)
    def _save_bookmark(v):
        """Save the current camera view into the bookmark store."""
        state = {
            "center": v.camera.center,
            "zoom": v.camera.zoom,
            "angles": v.camera.angles,
        }
        store.append(state)
        print(f"Bookmark {len(store)} saved!")

    @viewer.bind_key(next_key, overwrite=overwrite)
    def _next_bookmark(v):
        """Cycle to the next saved bookmark (rotating the store)."""
        if not store:
            print(f"No bookmarks yet! Press '{save_key}' to save one.")
            return

        state = store.pop(0)
        store.append(state)
        v.camera.center = state["center"]
        v.camera.zoom = state["zoom"]
        v.camera.angles = state["angles"]
        print("Moved to next bookmark.")

    return store

# ----------------------------
# User-friendly class wrapper
# ----------------------------


class NapariTools:
    """
    Convenience wrapper around common Napari utility actions.

    This class is designed to make interactive use in the Napari console simple:

    ```python
    import napari
    from scripts.napari_utils import NapariTools

    viewer = napari.current_viewer()
    tools = NapariTools(viewer)
    tools.register_bookmarks()
    tools.save_view_png("./screenshots/view.png")
    ```

    Notes
    -----
    - No UI is created at import time. Everything happens only after you pass a viewer.
    - The bookmark store is kept on the instance (`tools.bookmarks`).
    """

    def __init__(self, viewer=None):
        self.viewer = _get_viewer(viewer)
        self.bookmarks: List[Dict[str, Any]] = []

    # ---- screenshots ----
    def save_view(self, path: str = "./screenshots/view.png", *, canvas_only: bool = True) -> str:
        """Save the current viewport to a PNG."""
        return save_view(self.viewer, path, canvas_only=canvas_only)

    def save_view_png(self, path: str = "./screenshots/view.png", *, canvas_only: bool = True) -> str:
        """Backward-compatible alias for `save_view`."""
        return self.save_view(path, canvas_only=canvas_only)

    # ---- scale bar ----
    def config_scalebar(
        self,
        *,
        pixel_size_um: Optional[float] = None,
        unit: str = "µm",
        length: Optional[float] = None,
        font_size: int = 12,
        color: str = "white",
        box: bool = True,
        box_color: Tuple[float, float, float, float] = (0, 0, 0, 0.35),
    ) -> None:
        """Configure the scale bar using the instance viewer."""
        config_scalebar(
            self.viewer,
            pixel_size_um=pixel_size_um,
            unit=unit,
            length=length,
            font_size=font_size,
            color=color,
            box=box,
            box_color=box_color,
        )

    def configure_scale_bar(self, *, pixel_size_um: Optional[float] = None, **kwargs) -> None:
        """Backward-compatible alias for `config_scalebar`."""
        return self.config_scalebar(pixel_size_um=pixel_size_um, **kwargs)

    # ---- channel legend ----
    def save_legend(
        self,
        path: str = "./screenshots/channel_legend.png",
        *,
        font_size: int = 12,
        swatch_px: int = 14,
        pad_px: int = 8,
        gap_px: int = 30,
        text_swatch_gap_px: int = 6,
        background_rgba=(1.0, 1.0, 1.0, 1.0),
    ) -> str:
        """Write a single-line channel legend PNG for visible image channels."""
        return save_legend(
            self.viewer,
            path,
            font_size=font_size,
            swatch_px=swatch_px,
            pad_px=pad_px,
            gap_px=gap_px,
            text_swatch_gap_px=text_swatch_gap_px,
            background_rgba=background_rgba,
        )

    def save_channel_legend_png(self, path: str = "./screenshots/channel_legend.png", **kwargs) -> str:
        """Backward-compatible alias for `save_legend`."""
        return self.save_legend(path, **kwargs)

    def show_channel_legend(self) -> None:
        """Show the floating channel legend dock widget (upper-right)."""
        ensure_channel_legend_for_viewer(self.viewer)

    def hide_channel_legend(self) -> None:
        """Hide/remove the floating channel legend dock widget."""
        hide_channel_legend_for_viewer(self.viewer)

    # ---- bookmarks ----
    def register_bookmarks(
        self,
        *,
        save_key: str = "b",
        next_key: str = "n",
        overwrite: bool = True,
    ) -> None:
        """
        Register bookmark hotkeys on this viewer.

        - Press save_key (default 'b') to save the current camera view
        - Press next_key (default 'n') to cycle through saved bookmarks
        """
        self.bookmarks = register_bookmark_keys(
            self.viewer,
            save_key=save_key,
            next_key=next_key,
            store=self.bookmarks,
            overwrite=overwrite,
        )

    # ---- overlays ----
    def add_zoom_box(
        self,
        *,
        side_length: Optional[float] = None,
        border_width: float = 160.0,
        fill_alpha: float = 0.0,
        layer_name: str = "Zoom Box",
        debug: bool = False,
    ) -> Any:
        """Wrapper for `add_zoom_box` using this instance's viewer."""
        return add_zoom_box(
            self.viewer,
            side_length=side_length,
            border_width=border_width,
            fill_alpha=fill_alpha,
            layer_name=layer_name,
            debug=debug,
        )

    def increase_zoom_box(
        self,
        *,
        percent: float = 10.0,
        layer_name: str = "Zoom Box",
        shape_index: Optional[int] = None,
        debug: bool = False,
    ) -> Any:
        """Wrapper for `increase_zoom_box` using this instance's viewer."""
        return increase_zoom_box(
            self.viewer,
            percent=percent,
            layer_name=layer_name,
            shape_index=shape_index,
            debug=debug,
        )
