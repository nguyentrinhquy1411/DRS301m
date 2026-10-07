"""Jet heatmap rendering and alpha-blended overlays.

Colour convention: every image here is uint8 **BGR** (OpenCV). ``cv2.applyColorMap`` already returns
BGR, so heatmaps can be blended with BGR photos directly. Convert to RGB only at the very end if an
RGB-based library (matplotlib, PIL) needs to display the result.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import Geometry

ALPHA_PRESETS = (0.2, 0.3, 0.4, 0.5, 0.6)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _validate_heatmap(heatmap: np.ndarray) -> np.ndarray:
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2-D, got shape {heatmap.shape}")
    if not np.isfinite(heatmap).all():
        raise ValueError("heatmap contains NaN/Inf")
    return np.clip(heatmap, 0.0, 1.0)


def create_heatmap(heatmap: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """[0, 1] float heatmap -> uint8 BGR image with the Jet colormap.

    ``size`` is ``(width, height)`` (OpenCV order). Resizing happens on the scalar map *before*
    colouring so that no colours are interpolated.
    """
    heat = _validate_heatmap(heatmap).astype(np.float32)
    if size is not None and (heat.shape[1], heat.shape[0]) != tuple(size):
        heat = np.clip(cv2.resize(heat, tuple(size), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    heat_u8 = np.uint8(np.round(heat * 255))
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR


def overlay_heatmap(original_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """``overlay = (1 - alpha) * original + alpha * jet(heatmap)``; heatmap is resized to the original size."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    h, w = original_bgr.shape[:2]
    jet = create_heatmap(heatmap, size=(w, h))
    return cv2.addWeighted(original_bgr, 1.0 - alpha, jet, alpha, 0.0)


def _clip_to_image(image: np.ndarray, geometry: Geometry):
    """Intersection of ``geometry`` with the image: (dst slices, src offsets inside the geometry)."""
    x1, y1, x2, y2 = geometry
    ih, iw = image.shape[:2]
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, iw), min(y2, ih)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return (cy1, cy2, cx1, cx2), (cy1 - y1, cy2 - y1, cx1 - x1, cx2 - x1)


def overlay_heatmap_in_image(image_bgr: np.ndarray, heatmap: np.ndarray, geometry: Geometry,
                             alpha: float = 0.4) -> np.ndarray:
    """MODE 2: blend a face-level heatmap into the full image at its original position.

    The heatmap is stretched to the crop rectangle only (never to the whole image). Parts of the
    rectangle outside the image (replicated padding during cropping) are discarded.
    """
    out = image_bgr.copy()
    clipped = _clip_to_image(out, geometry)
    if clipped is None:
        return out
    (dy1, dy2, dx1, dx2), (sy1, sy2, sx1, sx2) = clipped
    x1, y1, x2, y2 = geometry
    jet = create_heatmap(heatmap, size=(x2 - x1, y2 - y1))[sy1:sy2, sx1:sx2]
    out[dy1:dy2, dx1:dx2] = cv2.addWeighted(out[dy1:dy2, dx1:dx2], 1.0 - alpha, jet, alpha, 0.0)
    return out


def paste_heatmap_in_image(image_shape: tuple[int, int], heatmap: np.ndarray, geometry: Geometry,
                           canvas: np.ndarray | None = None) -> np.ndarray:
    """MODE 2: Jet heatmap placed at the face position on a black canvas (full-image heatmap view)."""
    if canvas is None:
        canvas = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    clipped = _clip_to_image(canvas, geometry)
    if clipped is not None:
        (dy1, dy2, dx1, dx2), (sy1, sy2, sx1, sx2) = clipped
        x1, y1, x2, y2 = geometry
        canvas[dy1:dy2, dx1:dx2] = create_heatmap(heatmap, size=(x2 - x1, y2 - y1))[sy1:sy2, sx1:sx2]
    return canvas


def draw_face_label(image: np.ndarray, box: tuple[int, int, int, int], text: str,
                    color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw a bounding box and a text label above it (in place, BGR colour)."""
    x, y, w, h = box
    thickness = max(1, round(min(image.shape[:2]) / 400))
    scale = max(0.4, min(image.shape[:2]) / 900)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness + 1, cv2.LINE_AA)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    ty = y - 6 if y - th - base - 6 > 0 else y + h + th + 6
    cv2.rectangle(image, (x, ty - th - base // 2), (x + tw + 4, ty + base), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, ty), FONT, scale, color, thickness, cv2.LINE_AA)


def make_comparison(original: np.ndarray, heatmap_bgr: np.ndarray, overlay: np.ndarray,
                    title_lines: list[str], panel_height: int = 320) -> np.ndarray:
    """Original | Grad-CAM | Overlay side by side with a title strip on top."""
    def fit(img: np.ndarray) -> np.ndarray:
        scale = panel_height / img.shape[0]
        return cv2.resize(img, (max(1, round(img.shape[1] * scale)), panel_height), interpolation=cv2.INTER_AREA
                          if scale < 1 else cv2.INTER_CUBIC)

    panels = [fit(p) for p in (original, heatmap_bgr, overlay)]
    gap = 8
    width = sum(p.shape[1] for p in panels) + gap * 2
    line_h, caption_h = 30, 30
    title_h = 14 + line_h * max(1, len(title_lines))
    canvas = np.full((title_h + panel_height + caption_h, max(width, 480), 3), 255, dtype=np.uint8)

    for i, line in enumerate(title_lines):
        cv2.putText(canvas, line, (10, 14 + line_h * (i + 1) - 8), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    x = (canvas.shape[1] - width) // 2
    for panel, caption in zip(panels, ("Original", "Grad-CAM", "Overlay")):
        canvas[title_h:title_h + panel_height, x:x + panel.shape[1]] = panel
        cv2.putText(canvas, caption, (x + 6, title_h + panel_height + 22), FONT, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        x += panel.shape[1] + gap
    return canvas


def check_overlay(original: np.ndarray, overlay: np.ndarray) -> None:
    """Sanity check: overlay has the same size and dtype as the original and is not blank."""
    if overlay.shape != original.shape:
        raise ValueError(f"Overlay shape {overlay.shape} != original shape {original.shape}")
    if overlay.dtype != np.uint8:
        raise ValueError(f"Overlay dtype must be uint8, got {overlay.dtype}")
