"""
app.py — Tkinter GUI for the LiDAR Imager.

Layout
------
  ┌────────────────────────────────────────────────────────────┐
  │  [Freeze/Live]   Pt Size: [slider]            [Export PNG] │  toolbar
  ├─────────────────────────────────────────────────────────── ┤
  │  [Set Export Folder]  Folder: /path/to/folder              │  folder bar
  ├──────────────────────────┬─────────────────────────────────┤
  │                          │                                 │
  │   9:13 Preview           │   Circle Crop                   │
  │   (circle overlay)       │   (live auto-crop)              │
  │                          │                                 │
  ├──────────────────────────┴─────────────────────────────────┤
  │  topic: /pointcloud   frames: 0   pts: 0   t: --           │  status
  └────────────────────────────────────────────────────────────┘

Crops are fixed and automatic:
  - Left canvas: 9:13 auto-crop with circle overlay showing crop boundary.
  - Right canvas: live circle crop (largest circle centred on 9:13 image).

Export saves two PNG files to the configured folder:
  lidar_TIMESTAMP_9x13.png   — the full 9:13 rectangle
  lidar_TIMESTAMP_circle.png — the circle crop with alpha outside the circle
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageTk

from .pointcloud_processor import (
    apply_ellipse_mask,
    auto_crop_913,
    render_front_view,
)

if TYPE_CHECKING:
    from .ros_node import PointCloudNode

# ── Constants ──────────────────────────────────────────────────────────────────

_REFRESH_MS = 100          # live update interval
_RENDER_SIZE = 800         # internal render resolution (square)
_PREVIEW_SIZE = 480        # canvas pixel size (square, both panels)
_BG_COLOUR = '#1e1e1e'
_CANVAS_BG = '#111111'
_CROP_COLOUR = '#00e5ff'   # overlay outline colour
_CROP_WIDTH = 2            # overlay line width


class LidarImagerApp(tk.Tk):
    """Main application window."""

    def __init__(self, node: PointCloudNode) -> None:
        super().__init__()
        self._node = node
        self._frozen = False
        self._current_rect_image: Image.Image | None = None    # latest 9:13 image
        self._current_circle_image: Image.Image | None = None  # latest circle crop
        self._frozen_rect_image: Image.Image | None = None
        self._frozen_circle_image: Image.Image | None = None
        self._export_dir: str | None = None
        self._export_dir_var = tk.StringVar(value='(none — Export will prompt for folder)')

        self.title('LiDAR Imager')
        self.configure(bg=_BG_COLOUR)
        self.resizable(False, False)

        self._build_ui()
        self._after_id = self.after(_REFRESH_MS, self._update_loop)

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Toolbar ────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg=_BG_COLOUR, pady=6, padx=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self._freeze_btn = tk.Button(
            toolbar, text='Freeze', width=8,
            command=self._toggle_freeze,
            bg='#333333', fg='white', relief=tk.FLAT,
            activebackground='#555555', activeforeground='white',
        )
        self._freeze_btn.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(toolbar, text='Z Min:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        self._z_min_var = tk.StringVar()
        tk.Entry(
            toolbar, textvariable=self._z_min_var, width=6,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(toolbar, text='Z Max:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        self._z_max_var = tk.StringVar()
        tk.Entry(
            toolbar, textvariable=self._z_max_var, width=6,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(side=tk.LEFT, padx=(2, 0))

        tk.Label(toolbar, text='(blank = auto)', bg=_BG_COLOUR,
                 fg='#666666', font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(4, 0))

        # Point size slider (packed right-to-left so it stays near the export btn)
        self._export_btn = tk.Button(
            toolbar, text='Export PNG', width=10,
            command=self._export_png,
            bg='#005f87', fg='white', relief=tk.FLAT,
            activebackground='#007aad', activeforeground='white',
        )
        self._export_btn.pack(side=tk.RIGHT, padx=(0, 4))

        tk.Label(toolbar, text='Pt Size:', bg=_BG_COLOUR, fg='white').pack(
            side=tk.RIGHT, padx=(16, 2)
        )
        self._point_size = tk.IntVar(value=2)
        pt_slider = tk.Scale(
            toolbar, variable=self._point_size,
            from_=1, to=10, resolution=1, orient=tk.HORIZONTAL,
            length=120, showvalue=True,
            bg=_BG_COLOUR, fg='white', troughcolor='#444444',
            highlightthickness=0, bd=0,
            activebackground=_BG_COLOUR,
        )
        pt_slider.pack(side=tk.RIGHT)

        # ── Export-folder bar ──────────────────────────────────────────────
        folder_bar = tk.Frame(self, bg='#252525', pady=3, padx=8)
        folder_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            folder_bar, text='Set Export Folder', width=16,
            command=self._set_export_folder,
            bg='#333333', fg='white', relief=tk.FLAT,
            activebackground='#555555', activeforeground='white',
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(
            folder_bar, text='Folder:', bg='#252525', fg='#888888',
            font=('TkDefaultFont', 9),
        ).pack(side=tk.LEFT)
        tk.Label(
            folder_bar, textvariable=self._export_dir_var,
            bg='#252525', fg='#cccccc', font=('TkDefaultFont', 9),
            anchor='w',
        ).pack(side=tk.LEFT, padx=(4, 0))

        # ── Canvas area ────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=_BG_COLOUR)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8)

        # Left: 9:13 preview with circle overlay
        left_frame = tk.Frame(canvas_frame, bg=_BG_COLOUR)
        left_frame.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(left_frame, text='9:13 Preview', bg=_BG_COLOUR,
                 fg='#aaaaaa', font=('TkDefaultFont', 9)).pack()
        self._preview_canvas = tk.Canvas(
            left_frame, width=_PREVIEW_SIZE, height=_PREVIEW_SIZE,
            bg=_CANVAS_BG, highlightthickness=1, highlightbackground='#444444',
        )
        self._preview_canvas.pack()
        self._preview_photo: ImageTk.PhotoImage | None = None

        # Right: live circle crop
        right_frame = tk.Frame(canvas_frame, bg=_BG_COLOUR)
        right_frame.pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(right_frame, text='Circle Crop', bg=_BG_COLOUR,
                 fg='#aaaaaa', font=('TkDefaultFont', 9)).pack()
        self._circle_canvas = tk.Canvas(
            right_frame, width=_PREVIEW_SIZE, height=_PREVIEW_SIZE,
            bg=_CANVAS_BG, highlightthickness=1, highlightbackground='#444444',
        )
        self._circle_canvas.pack()
        self._circle_photo: ImageTk.PhotoImage | None = None

        # ── Status bar ─────────────────────────────────────────────────────
        status_frame = tk.Frame(self, bg='#2a2a2a', pady=3, padx=8)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_var = tk.StringVar(value='Waiting for data on /pointcloud …')
        tk.Label(
            status_frame, textvariable=self._status_var,
            bg='#2a2a2a', fg='#888888', font=('TkDefaultFont', 9),
            anchor='w',
        ).pack(fill=tk.X)

    # ── Live update loop ───────────────────────────────────────────────────────

    def _update_loop(self) -> None:
        """Poll the ROS node and refresh both canvases."""
        if not self._frozen:
            points = self._node.get_latest_points()
            if points is not None:
                rendered = render_front_view(
                    points, _RENDER_SIZE, _RENDER_SIZE,
                    point_size=self._point_size.get(),
                    z_min_clamp=self._parse_z('min'),
                    z_max_clamp=self._parse_z('max'),
                )
                rect_img = auto_crop_913(rendered)
                circle_bbox = self._get_circle_bbox(rect_img)
                circle_img = apply_ellipse_mask(rect_img, circle_bbox)
                self._current_rect_image = rect_img
                self._current_circle_image = circle_img
                self._refresh_preview(rect_img)
                self._refresh_circle_canvas(circle_img)

        # Update status bar regardless of freeze state
        status = self._node.get_status()
        import time
        ts = (
            f't: {status["last_stamp"]:.2f}s'
            if status['last_stamp'] > 0
            else 't: --'
        )
        self._status_var.set(
            f'topic: {status["topic"]}   '
            f'frames: {status["frame_count"]}   '
            f'pts: {status["point_count"]}   '
            f'{ts}'
        )

        self._after_id = self.after(_REFRESH_MS, self._update_loop)

    # ── Canvas helpers ─────────────────────────────────────────────────────────

    def _pil_to_canvas(
        self, canvas: tk.Canvas, pil_image: Image.Image
    ) -> ImageTk.PhotoImage:
        """Scale *pil_image* to fit *canvas* and return a PhotoImage."""
        cw = canvas.winfo_width() or _PREVIEW_SIZE
        ch = canvas.winfo_height() or _PREVIEW_SIZE
        img = pil_image.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _refresh_preview(self, pil_image: Image.Image) -> None:
        """Display the 9:13 image on the left canvas."""
        photo = self._pil_to_canvas(self._preview_canvas, pil_image)
        self._preview_photo = photo
        cw = self._preview_canvas.winfo_width() or _PREVIEW_SIZE
        ch = self._preview_canvas.winfo_height() or _PREVIEW_SIZE
        iw, ih = photo.width(), photo.height()
        ox = (cw - iw) // 2
        oy = (ch - ih) // 2
        self._preview_canvas.delete('all')
        self._preview_canvas.create_image(ox, oy, anchor=tk.NW, image=photo)

    def _refresh_circle_canvas(self, pil_image: Image.Image) -> None:
        """Display the circle-cropped image on the right canvas."""
        photo = self._pil_to_canvas(self._circle_canvas, pil_image)
        self._circle_photo = photo
        cw = self._circle_canvas.winfo_width() or _PREVIEW_SIZE
        ch = self._circle_canvas.winfo_height() or _PREVIEW_SIZE
        iw, ih = photo.width(), photo.height()
        self._circle_canvas.delete('all')
        self._circle_canvas.create_image(
            (cw - iw) // 2, (ch - ih) // 2, anchor=tk.NW, image=photo
        )

    # ── Freeze / Live ──────────────────────────────────────────────────────────

    def _toggle_freeze(self) -> None:
        self._frozen = not self._frozen
        if self._frozen:
            self._frozen_rect_image = self._current_rect_image
            self._frozen_circle_image = self._current_circle_image
            self._freeze_btn.config(
                text='Live', bg='#5a3a00', activebackground='#7a5200'
            )
        else:
            self._frozen_rect_image = None
            self._frozen_circle_image = None
            self._freeze_btn.config(
                text='Freeze', bg='#333333', activebackground='#555555'
            )
    # ── Z clamp helpers ────────────────────────────────────────────────────────

    def _parse_z(self, which: str) -> float | None:
        """Parse the Z min/max entry field; return None if blank or invalid."""
        raw = (self._z_min_var if which == 'min' else self._z_max_var).get().strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    # ── Circle bbox helper ────────────────────────────────────────────────────

    @staticmethod
    def _get_circle_bbox(image: Image.Image) -> tuple[int, int, int, int]:
        """Return the bbox of the largest circle centred on *image*.

        The circle diameter equals the image width (the shorter side for a 9:13
        portrait image) and is centred vertically and horizontally.
        """
        w, h = image.size
        d = w  # diameter = width (shorter axis)
        top = (h - d) // 2
        return (0, top, w, top + d)

    # ── Export folder ──────────────────────────────────────────────────────────

    def _set_export_folder(self) -> None:
        folder = filedialog.askdirectory(title='Select export folder')
        if not folder:
            return
        self._export_dir = folder
        self._export_dir_var.set(folder)
        self._status_var.set(f'Export folder set → {folder}')

    def _next_export_base(self) -> str:
        """Return a timestamped base path (no suffix/extension) in the export folder."""
        import os
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]  # ms precision
        return os.path.join(self._export_dir, f'lidar_{stamp}')

    # ── Export ─────────────────────────────────────────────────────────────────

    def _export_png(self) -> None:
        rect_src = self._frozen_rect_image or self._current_rect_image
        circle_src = self._frozen_circle_image or self._current_circle_image
        if rect_src is None or circle_src is None:
            self._status_var.set('No image to export — waiting for point cloud data.')
            return

        # Ensure we have a folder — ask if not yet set
        if self._export_dir is None:
            folder = filedialog.askdirectory(title='Select export folder')
            if not folder:
                return
            self._export_dir = folder
            self._export_dir_var.set(folder)

        base = self._next_export_base()
        rect_path = f'{base}_9x13.png'
        circle_path = f'{base}_circle.png'

        rect_src.convert('RGBA').save(rect_path, format='PNG')
        circle_src.save(circle_path, format='PNG')

        self._status_var.set(
            f'Exported → {rect_path}  +  {circle_path}'
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        if hasattr(self, '_after_id'):
            self.after_cancel(self._after_id)
        super().destroy()
