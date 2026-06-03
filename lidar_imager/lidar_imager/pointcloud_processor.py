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
    h_fov: float = 90.0,
    min_depth: float = 0.1,
) -> Image.Image:
    """Render a perspective (pinhole-camera) front view of *points*.

    The camera sits at the origin looking in the +Y direction.  A **fixed**
    horizontal field of view (h_fov) defines a stable frustum so the image
    never bounces or rescales between frames.  The vertical FOV is derived
    from h_fov and the image aspect ratio so every pixel is square.

    Parameters
    ----------
    points:
        (N, 3) float32 array with columns X, Y, Z.
    img_w, img_h:
        Output image dimensions in pixels.
    point_size:
        Radius of each rendered point in pixels (>= 1).
    z_min_clamp, z_max_clamp:
        Explicit Z range for the colour map (None = auto from surviving data).
    h_fov:
        Horizontal field of view in degrees (default 90°).
    min_depth:
        Minimum Y distance; points at or behind this threshold are discarded.
    """
    if points is None or len(points) == 0:
        return Image.new('RGBA', (img_w, img_h), _BG_RGBA)

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    z = points[:, 2].astype(np.float64)

    # ── Depth filter ──────────────────────────────────────────────────────
    valid = y > min_depth
    if not np.any(valid):
        return Image.new('RGBA', (img_w, img_h), _BG_RGBA)
    x, y, z = x[valid], y[valid], z[valid]

    # ── Perspective projection ────────────────────────────────────────────
    # Standard pinhole: u = X/Y (horizontal), v = Z/Y (vertical, +Z = up)
    u = x / y
    v = z / y

    # Fixed frustum — half-tangents derived from FOV; never changes per-frame
    tan_h = np.tan(np.radians(h_fov * 0.5))
    tan_v = tan_h * (img_h / img_w)   # square pixels, no aspect distortion

    # Frustum cull — discard anything outside the camera's view
    in_fov = (u >= -tan_h) & (u <= tan_h) & (v >= -tan_v) & (v <= tan_v)
    u, v, z = u[in_fov], v[in_fov], z[in_fov]
    if len(u) == 0:
        return Image.new('RGBA', (img_w, img_h), _BG_RGBA)

    # ── Colour-map Z range ────────────────────────────────────────────────
    color_z_min = z_min_clamp if z_min_clamp is not None else z.min()
    color_z_max = z_max_clamp if z_max_clamp is not None else z.max()
    color_z_range = color_z_max - color_z_min if color_z_max != color_z_min else 1.0

    # ── Rasterise via bincount (2-3× faster than histogram2d) ────────────
    # u ∈ [-tan_h, tan_h] → column index [0, img_w-1]
    # v ∈ [-tan_v, tan_v] → row    index [0, img_h-1]  (+v at top = row 0)
    ix = np.clip(
        ((u + tan_h) / (2.0 * tan_h) * img_w).astype(np.int32), 0, img_w - 1
    )
    iy = np.clip(
        ((tan_v - v) / (2.0 * tan_v) * img_h).astype(np.int32), 0, img_h - 1
    )

    flat_idx = iy * img_w + ix
    n_pixels  = img_h * img_w

    counts = np.bincount(flat_idx, minlength=n_pixels).astype(np.float64)
    z_norm = np.clip((z - color_z_min) / color_z_range, 0.0, 1.0)
    z_accum = np.bincount(flat_idx, weights=z_norm, minlength=n_pixels)

    occupied = counts > 0
    avg_z    = np.where(occupied, z_accum / np.where(occupied, counts, 1.0), 0.0)

    # ── Colour mapping ────────────────────────────────────────────────────
    pixel_rgba = np.full((n_pixels, 4), list(_BG_RGBA), dtype=np.uint8)
    if occupied.any():
        pixel_rgba[occupied] = _height_colormap(avg_z[occupied])

    # flat_idx = iy*img_w + ix  →  reshape directly to (img_h, img_w, 4);
    # no transpose or axis-flip needed — iy already encodes +Z-at-top.
    image = Image.fromarray(pixel_rgba.reshape(img_h, img_w, 4), mode='RGBA')

    if point_size > 1:
        kernel = 2 * point_size - 1   # always odd: 1→1, 2→3, 3→5 …
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
