"""
pointcloud_processor.py — Front-view projection and image rendering.

Pipeline:
  1. render_front_view(points, w, h) → PIL.Image (RGBA)
       - Projects X (horizontal) and Z (vertical) axes.
       - Rasterize via np.histogram2d for efficiency.
       - Colors pixels by average Z-height (blue→cyan→green→yellow→red).
  2. auto_crop_913(image) → PIL.Image
       - Finds bounding box of non-background pixels.
       - Centers a 9:13 rectangle on that bounding box.
       - Returns the cropped PIL.Image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Aspect ratio constants (width : height = 9 : 13)
ASPECT_W = 9
ASPECT_H = 13

# Background colour (RGBA) used when no points project to a pixel
_BG_RGBA = (0, 0, 0, 255)

# ── Colormap ──────────────────────────────────────────────────────────────────


def _height_colormap(t: np.ndarray) -> np.ndarray:
    """Map normalised values in [0, 1] to RGBA using a smooth HSV sweep.

    Hue sweeps continuously from 0° (red, lowest Z) to 270° (purple, highest Z)
    at full saturation and value, producing a seamless spectrum:
      red → orange → yellow → green → cyan → blue → purple
    """
    t = np.clip(t, 0.0, 1.0)

    # Map t to hue in [0, 0.75] of the colour wheel, then to the 6-sector space
    h6 = t * 4.5              # [0.0, 4.5]  (= 0°..270° / 60°)
    i  = np.floor(h6).astype(np.int32)
    f  = h6 - np.floor(h6)   # fractional position within each 60° sector
    q  = 1.0 - f              # complement

    r = np.empty_like(t)
    g = np.empty_like(t)
    b = np.empty_like(t)

    # HSV(h, S=1, V=1) sector formulas  (p=0 throughout since S=V=1)
    # Sector 0  [0°,  60°):  R=1,  G=f,  B=0   — red → yellow
    m = (i == 0); r[m] = 1.0;   g[m] = f[m]; b[m] = 0.0
    # Sector 1  [60°, 120°): R=q,  G=1,  B=0   — yellow → green
    m = (i == 1); r[m] = q[m];  g[m] = 1.0;  b[m] = 0.0
    # Sector 2  [120°,180°): R=0,  G=1,  B=f   — green → cyan
    m = (i == 2); r[m] = 0.0;   g[m] = 1.0;  b[m] = f[m]
    # Sector 3  [180°,240°): R=0,  G=q,  B=1   — cyan → blue
    m = (i == 3); r[m] = 0.0;   g[m] = q[m]; b[m] = 1.0
    # Sector 4  [240°,270°]: R=f,  G=0,  B=1   — blue → purple  (partial)
    m = (i >= 4); r[m] = f[m];  g[m] = 0.0;  b[m] = 1.0

    rgba = np.stack(
        [(r * 255).astype(np.uint8), (g * 255).astype(np.uint8),
         (b * 255).astype(np.uint8), np.full_like(r, 255, dtype=np.uint8)],
        axis=-1,
    )
    return rgba


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_front_view(
    points: np.ndarray,
    img_w: int = 800,
    img_h: int = 800,
    point_size: int = 1,
    z_min_clamp: float | None = None,
    z_max_clamp: float | None = None,
) -> Image.Image:
    """Render a front-view (XZ) projection of *points* into an RGBA image.

    Parameters
    ----------
    points:
        (N, 3) float32 array with columns X, Y, Z.
    img_w, img_h:
        Output image dimensions in pixels.
    point_size:
        Radius of each rendered point in pixels (>= 1). Values above 1
        dilate each pixel outward using a square maximum filter.
    z_min_clamp, z_max_clamp:
        Explicit Z range for the colour map.  Points outside this range are
        clamped to the nearest colour stop.  When either value is None the
        range is derived automatically from the data in the current frame.

    Returns
    -------
    PIL.Image.Image (mode 'RGBA').
    """
    if points is None or len(points) == 0:
        return Image.new('RGBA', (img_w, img_h), _BG_RGBA)

    x = points[:, 0].astype(np.float64)
    z = points[:, 2].astype(np.float64)

    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    # Guard against degenerate (flat) point clouds
    x_range = x_max - x_min if x_max != x_min else 1.0
    z_range = z_max - z_min if z_max != z_min else 1.0

    # Colour-map Z range: use user-supplied clamp values, fall back to auto
    color_z_min = z_min_clamp if z_min_clamp is not None else z_min
    color_z_max = z_max_clamp if z_max_clamp is not None else z_max
    color_z_range = color_z_max - color_z_min if color_z_max != color_z_min else 1.0

    # ── Rasterise: count points and accumulate Z per pixel bin ───────────
    # Spatial binning uses the data extent so no pixels are wasted.
    counts, x_edges, z_edges = np.histogram2d(
        x, z,
        bins=[img_w, img_h],
        range=[[x_min, x_max], [z_min, z_max]],
    )  # shape: (img_w, img_h)

    # Accumulate normalised Z per pixel using the colour-map range
    z_norm = np.clip((z - color_z_min) / color_z_range, 0.0, 1.0)
    z_accum, _, _ = np.histogram2d(
        x, z,
        bins=[img_w, img_h],
        range=[[x_min, x_max], [z_min, z_max]],
        weights=z_norm,
    )

    occupied = counts > 0
    avg_z = np.where(occupied, z_accum / np.where(occupied, counts, 1), 0.0)

    # ── Colour mapping ─────────────────────────────────────────────────────
    # avg_z is already normalised; apply colourmap only to occupied pixels
    flat_z = avg_z.flatten()
    flat_occ = occupied.flatten()

    pixel_rgba = np.full((img_w * img_h, 4), list(_BG_RGBA), dtype=np.uint8)
    if flat_occ.any():
        pixel_rgba[flat_occ] = _height_colormap(flat_z[flat_occ])

    # Shape: (img_w, img_h, 4) — but PIL Image needs (height, width, 4)
    # histogram2d returns (X bins, Z bins), so axis-0 = X (→ column), axis-1 = Z (→ row)
    img_array = pixel_rgba.reshape(img_w, img_h, 4)

    # Transpose so rows = Z axis (vertical), cols = X axis (horizontal)
    # Also flip Z so positive-Z is at the top of the image
    img_array = img_array.transpose(1, 0, 2)[::-1, :, :].copy()

    image = Image.fromarray(img_array, mode='RGBA')

    # Dilate points if size > 1: MaxFilter expands coloured pixels outward
    if point_size > 1:
        kernel = 2 * point_size - 1  # always odd: 1→1, 2→3, 3→5 …
        image = image.filter(ImageFilter.MaxFilter(kernel))

    return image


# ── Auto-crop 9:13 ────────────────────────────────────────────────────────────


def auto_crop_913(image: Image.Image) -> Image.Image:
    """Crop *image* to a 9:13 rectangle centred on the point-cloud content.

    The bounding box of non-background pixels is found first.  A 9:13
    rectangle is then grown (if necessary) to fully enclose that bbox while
    remaining centred on it, then clamped to the image boundary.

    If the image contains no foreground content the full image is returned
    after a centre-crop to 9:13.
    """
    bg = _BG_RGBA[:3]  # compare only RGB
    arr = np.array(image)  # (H, W, 4)
    rgb = arr[:, :, :3]

    # Find rows / cols that contain at least one non-background pixel
    mask = ~np.all(rgb == bg, axis=2)  # (H, W) bool
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    img_h, img_w = arr.shape[:2]

    if rows.any() and cols.any():
        row_min, row_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
        col_min, col_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
        cx = (col_min + col_max) // 2
        cy = (row_min + row_max) // 2
        content_w = col_max - col_min + 1
        content_h = row_max - row_min + 1
    else:
        cx, cy = img_w // 2, img_h // 2
        content_w, content_h = img_w, img_h

    # Determine crop dimensions that enclose content and satisfy 9:13 ratio
    # Scale up from content bbox so neither axis is clipped
    if content_w * ASPECT_H >= content_h * ASPECT_W:
        # Width is the limiting axis
        crop_w = content_w
        crop_h = (content_w * ASPECT_H + ASPECT_W - 1) // ASPECT_W
    else:
        crop_h = content_h
        crop_w = (content_h * ASPECT_W + ASPECT_H - 1) // ASPECT_H

    # Clamp to image dimensions while keeping 9:13 ratio
    if crop_w > img_w or crop_h > img_h:
        scale = min(img_w / crop_w, img_h / crop_h)
        crop_w = int(crop_w * scale)
        crop_h = int(crop_h * scale)
        # Snap to exact 9:13
        if crop_w * ASPECT_H > crop_h * ASPECT_W:
            crop_h = (crop_w * ASPECT_H) // ASPECT_W
        else:
            crop_w = (crop_h * ASPECT_W) // ASPECT_H

    # Centre crop around (cx, cy)
    left = max(0, cx - crop_w // 2)
    top = max(0, cy - crop_h // 2)
    right = left + crop_w
    bottom = top + crop_h

    # Shift if we hit the right/bottom edge
    if right > img_w:
        right = img_w
        left = right - crop_w
    if bottom > img_h:
        bottom = img_h
        top = bottom - crop_h

    left = max(0, left)
    top = max(0, top)

    return image.crop((left, top, right, bottom))


# ── Shape masking for export ───────────────────────────────────────────────────


def apply_ellipse_mask(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Return *image* cropped to *bbox* with an ellipse alpha mask applied.

    Pixels inside the inscribed ellipse are kept; pixels outside are made
    transparent (alpha = 0).  The image is then cropped to the ellipse
    bounding box so the output dimensions clearly reflect the ellipse shape
    rather than returning a full-size image with invisible transparent corners.

    Parameters
    ----------
    image:
        Source PIL.Image (will be converted to RGBA internally).
    bbox:
        (left, top, right, bottom) in image pixel coordinates defining the
        bounding box of the ellipse.

    Returns
    -------
    PIL.Image in RGBA mode, cropped to *bbox* dimensions.
    """
    from PIL import ImageDraw

    img = image.convert('RGBA')

    # Build an L-mode mask: white (255) inside the ellipse, black (0) outside
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(bbox, fill=255)

    # Replace alpha channel with the ellipse mask
    r, g, b, a = img.split()
    a = Image.fromarray(
        np.minimum(np.array(a), np.array(mask)), mode='L'
    )
    masked = Image.merge('RGBA', (r, g, b, a))

    # Crop to the bounding box so the output is visibly elliptical
    return masked.crop(bbox)


def crop_to_box(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop *image* to *bbox* rectangle.

    Parameters
    ----------
    bbox:
        (left, top, right, bottom) in image pixel coordinates.
    """
    return image.crop(bbox)
