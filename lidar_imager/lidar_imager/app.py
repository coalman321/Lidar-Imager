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

import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import colorchooser, filedialog, ttk
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

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
_CONFIG_PATH = os.path.expanduser('~/.lidar_imager.json')


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a TTF font at *size* pt, falling back to PIL's built-in."""
    import os
    candidates = [
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _get_system_fonts() -> list[tuple[str, str]]:
    """Return a sorted [(display_name, path)] list of every TTF/OTF font
    found by fc-list on the system.  Returns [] if fc-list is unavailable."""
    import os
    import subprocess
    try:
        result = subprocess.run(
            ['fc-list', '--format', '%{file}\t%{family}\t%{style}\n'],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    fonts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split('\t')
        path = parts[0].strip()
        if not path.lower().endswith(('.ttf', '.otf')):
            continue
        if not os.path.isfile(path):
            continue
        family = parts[1].split(',')[0].strip() if len(parts) > 1 else ''
        style  = parts[2].split(',')[0].strip() if len(parts) > 2 else ''
        display = f'{family} {style}'.strip() if style else family
        if not display:
            display = os.path.splitext(os.path.basename(path))[0]
        if path not in seen:
            seen.add(path)
            fonts.append((display, path))
    return sorted(fonts, key=lambda x: x[0].lower())


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
        self._export_dir_var = tk.StringVar(value='')

        self._bg_image: Image.Image | None = None
        self._bg_file_path: str = ''
        self._bg_path_var = tk.StringVar(value='(none)')
        self._bg_x_var = tk.StringVar(value='30')
        self._bg_y_var = tk.StringVar(value='225')
        self._bg_scale_var = tk.DoubleVar(value=160.0)
        self._name_var = tk.StringVar()
        self._color_mode = tk.StringVar(value='z')   # 'z' | 'range' | 'x' | 'y'
        self._val_min_var = tk.StringVar()
        self._val_max_var = tk.StringVar()
        self._point_size = tk.IntVar(value=2)
        self._custom_font_path: str | None = '/usr/share/fonts/opentype/urw-base35/C059-Bold.otf'
        self._font_display: str = 'C059 Bold'  # display name for font button label
        self._text_color: str = '#FFC500'  # hex colour for name text
        self._name_x_offset = tk.IntVar(value=0)
        self._name_y_offset = tk.IntVar(value=-85)
        self._fonts_cache: list[tuple[str, str]] | None = None
        self._config_dlg: tk.Toplevel | None = None

        self._load_config()

        self.title('LiDAR Imager')
        self.configure(bg=_BG_COLOUR)
        self.resizable(True, True)
        self.minsize(800, 560)

        self._build_ui()
        if self._bg_file_path:
            self.after(0, lambda p=self._bg_file_path: self._load_bg_from_path(p))
        self._after_id = self.after(_REFRESH_MS, self._update_loop)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

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
        self._freeze_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            toolbar, text='Config', width=8,
            command=self._open_config,
            bg='#333333', fg='white', relief=tk.FLAT,
            activebackground='#555555', activeforeground='white',
        ).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(toolbar, text='Name:', bg=_BG_COLOUR, fg='white').pack(
            side=tk.LEFT, padx=(0, 2)
        )
        tk.Entry(
            toolbar, textvariable=self._name_var, width=22,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(side=tk.LEFT)

        # Point size slider and export button (packed right-to-left)
        self._export_btn = tk.Button(
            toolbar, text='Export', width=10,
            command=self._export_png,
            bg='#005f87', fg='white', relief=tk.FLAT,
            activebackground='#007aad', activeforeground='white',
        )
        self._export_btn.pack(side=tk.RIGHT, padx=(0, 4))

        # ── Canvas area ────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=_BG_COLOUR)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8)

        # Left: 9:13 preview
        left_frame = tk.Frame(canvas_frame, bg=_BG_COLOUR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(left_frame, text='Image Preview', bg=_BG_COLOUR,
                 fg='#aaaaaa', font=('TkDefaultFont', 9)).pack()
        self._preview_canvas = tk.Canvas(
            left_frame, width=_PREVIEW_SIZE, height=_PREVIEW_SIZE,
            bg=_CANVAS_BG, highlightthickness=1, highlightbackground='#444444',
        )
        self._preview_canvas.pack(fill=tk.BOTH, expand=True)
        self._preview_photo: ImageTk.PhotoImage | None = None

        # Right: live circle crop
        right_frame = tk.Frame(canvas_frame, bg=_BG_COLOUR)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        tk.Label(right_frame, text='Button Preview', bg=_BG_COLOUR,
                 fg='#aaaaaa', font=('TkDefaultFont', 9)).pack()
        self._circle_canvas = tk.Canvas(
            right_frame, width=_PREVIEW_SIZE, height=_PREVIEW_SIZE,
            bg=_CANVAS_BG, highlightthickness=1, highlightbackground='#444444',
        )
        self._circle_canvas.pack(fill=tk.BOTH, expand=True)
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

    # ── Config popup ────────────────────────────────────────────────────────────

    def _open_config(self) -> None:
        """Open (or raise) the singleton configuration window."""
        if self._config_dlg is not None and self._config_dlg.winfo_exists():
            self._config_dlg.lift()
            self._config_dlg.focus_force()
            return

        dlg = tk.Toplevel(self)
        dlg.title('Configuration')
        dlg.configure(bg=_BG_COLOUR)
        dlg.geometry('520x620')
        dlg.transient(self)
        dlg.resizable(True, True)
        self._config_dlg = dlg

        def _sep(parent: tk.Frame, label: str) -> None:
            """Draw a labelled horizontal separator."""
            f = tk.Frame(parent, bg=_BG_COLOUR)
            f.pack(fill=tk.X, padx=8, pady=(14, 4))
            tk.Label(
                f, text=label, bg=_BG_COLOUR, fg='#aaaaaa',
                font=('TkDefaultFont', 8, 'bold'),
            ).pack(side=tk.LEFT)
            tk.Frame(f, bg='#444444', height=1).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0)
            )

        # ── Scrollable body ───────────────────────────────────────────────
        _scroll_canvas = tk.Canvas(dlg, bg=_BG_COLOUR, highlightthickness=0)
        _scrollbar = tk.Scrollbar(dlg, orient=tk.VERTICAL, command=_scroll_canvas.yview)
        _scroll_canvas.configure(yscrollcommand=_scrollbar.set)
        _scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        _scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        body = tk.Frame(_scroll_canvas, bg=_BG_COLOUR)
        _body_window = _scroll_canvas.create_window((0, 0), window=body, anchor='nw')

        def _on_body_configure(event: tk.Event) -> None:
            _scroll_canvas.configure(scrollregion=_scroll_canvas.bbox('all'))

        def _on_canvas_configure(event: tk.Event) -> None:
            _scroll_canvas.itemconfig(_body_window, width=event.width)

        body.bind('<Configure>', _on_body_configure)
        _scroll_canvas.bind('<Configure>', _on_canvas_configure)

        # Mouse-wheel scrolling
        def _on_mousewheel(event: tk.Event) -> None:
            _scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        dlg.bind('<MouseWheel>', _on_mousewheel)
        dlg.bind('<Button-4>', lambda e: _scroll_canvas.yview_scroll(-1, 'units'))
        dlg.bind('<Button-5>', lambda e: _scroll_canvas.yview_scroll(1, 'units'))

        # ── Z Range ───────────────────────────────────────────────────────
        _sep(body, 'COLOR MAPPING')
        # ── Mode selection ────────────────────────────────────────────────
        mode_f = tk.Frame(body, bg=_BG_COLOUR)
        mode_f.pack(fill=tk.X, padx=20, pady=(0, 6))
        for label, val in [('Z height', 'z'), ('Range', 'range'), ('X depth', 'x'), ('Y lateral', 'y')]:
            tk.Radiobutton(
                mode_f, text=label, variable=self._color_mode, value=val,
                bg=_BG_COLOUR, fg='white', selectcolor='#333333',
                activebackground=_BG_COLOUR, activeforeground='white',
            ).pack(side=tk.LEFT, padx=(0, 10))
        # ── Min / max clamp (applies to whichever mode is selected) ────────
        tk.Label(body, text='Min / max  (blank = auto):', bg=_BG_COLOUR, fg='#aaaaaa',
                 font=('TkDefaultFont', 8)).pack(anchor='w', padx=20, pady=(4, 0))
        vf = tk.Frame(body, bg=_BG_COLOUR)
        vf.pack(fill=tk.X, padx=20)
        tk.Entry(
            vf, textvariable=self._val_min_var, width=8,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(side=tk.LEFT, padx=(0, 14))
        tk.Label(vf, text='Max:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        tk.Entry(
            vf, textvariable=self._val_max_var, width=8,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(side=tk.LEFT, padx=(2, 10))
        tk.Label(
            vf, text='(m or units)', bg=_BG_COLOUR, fg='#666666',
            font=('TkDefaultFont', 8),
        ).pack(side=tk.LEFT)

        # ── Export Folder ─────────────────────────────────────────────────
        _sep(body, 'EXPORT FOLDER')
        ef = tk.Frame(body, bg=_BG_COLOUR)
        ef.pack(fill=tk.X, padx=20)
        tk.Button(
            ef, text='Set Folder', width=10,
            command=self._set_export_folder,
            bg='#333333', fg='white', relief=tk.FLAT,
            activebackground='#555555', activeforeground='white',
        ).pack(side=tk.LEFT)
        tk.Label(
            ef, textvariable=self._export_dir_var,
            bg=_BG_COLOUR, fg='#cccccc', font=('TkDefaultFont', 9), anchor='w',
        ).pack(side=tk.LEFT, padx=(10, 0))

        # ── Background Image ──────────────────────────────────────────────
        _sep(body, 'BACKGROUND IMAGE')
        bgf = tk.Frame(body, bg=_BG_COLOUR)
        bgf.pack(fill=tk.X, padx=20)
        tk.Button(
            bgf, text='Set Image', width=10,
            command=self._set_background,
            bg='#333355', fg='white', relief=tk.FLAT,
            activebackground='#555577', activeforeground='white',
        ).pack(side=tk.LEFT)
        tk.Button(
            bgf, text='Clear', width=6,
            command=self._clear_background,
            bg='#333333', fg='white', relief=tk.FLAT,
            activebackground='#555555', activeforeground='white',
        ).pack(side=tk.LEFT, padx=(4, 12))
        tk.Label(
            bgf, textvariable=self._bg_path_var,
            bg=_BG_COLOUR, fg='#cccccc', font=('TkDefaultFont', 9), anchor='w',
        ).pack(side=tk.LEFT)

        off_f = tk.Frame(body, bg=_BG_COLOUR)
        off_f.pack(fill=tk.X, padx=20, pady=(6, 0))
        tk.Label(off_f, text='Offset  X:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        tk.Entry(
            off_f, textvariable=self._bg_x_var, width=6,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(side=tk.LEFT, padx=(2, 14))
        tk.Label(off_f, text='Y:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        tk.Entry(
            off_f, textvariable=self._bg_y_var, width=6,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(side=tk.LEFT, padx=(2, 10))
        tk.Label(
            off_f, text='px  (top-left of 9:13 on background)',
            bg=_BG_COLOUR, fg='#666666', font=('TkDefaultFont', 8),
        ).pack(side=tk.LEFT)

        scale_f = tk.Frame(body, bg=_BG_COLOUR)
        scale_f.pack(fill=tk.X, padx=20, pady=(4, 0))
        tk.Label(scale_f, text='Scale %:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        tk.Scale(
            scale_f, variable=self._bg_scale_var,
            from_=10, to=400, resolution=5, orient=tk.HORIZONTAL,
            length=220, showvalue=True,
            bg=_BG_COLOUR, fg='white', troughcolor='#444444',
            highlightthickness=0, bd=0,
            activebackground=_BG_COLOUR,
        ).pack(side=tk.LEFT, padx=(4, 0))
        # ── Point Size ───────────────────────────────────────────────
        _sep(body, 'POINT SIZE')
        psf = tk.Frame(body, bg=_BG_COLOUR)
        psf.pack(fill=tk.X, padx=20, pady=(0, 4))
        tk.Scale(
            psf, variable=self._point_size,
            from_=1, to=10, resolution=1, orient=tk.HORIZONTAL,
            length=200, showvalue=True,
            bg=_BG_COLOUR, fg='white', troughcolor='#444444',
            highlightthickness=0, bd=0,
            activebackground=_BG_COLOUR,
        ).pack(side=tk.LEFT)
        tk.Label(psf, text='px dilation radius', bg=_BG_COLOUR,
                 fg='#666666', font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(6, 0))
        # ── Text Overlay ──────────────────────────────────────────────────
        _sep(body, 'TEXT OVERLAY')
        tf = tk.Frame(body, bg=_BG_COLOUR)
        tf.pack(fill=tk.X, padx=20)
        self._font_btn = tk.Button(
            tf, text=f'Font: {self._font_display}', width=18,
            command=self._pick_font,
            bg='#333333', fg='white', relief=tk.FLAT,
            activebackground='#555555', activeforeground='white',
        )
        self._font_btn.pack(side=tk.LEFT)
        self._color_swatch = tk.Button(
            tf, text='  ', width=2,
            command=self._pick_text_color,
            bg=self._text_color, relief=tk.RAISED, bd=1,
        )
        self._color_swatch.pack(side=tk.LEFT, padx=(10, 4))
        tk.Label(tf, text='Text colour', bg=_BG_COLOUR, fg='#888888',
                 font=('TkDefaultFont', 9)).pack(side=tk.LEFT)
        # Name offset
        nof = tk.Frame(body, bg=_BG_COLOUR)
        nof.pack(fill=tk.X, padx=20, pady=(6, 0))
        tk.Label(nof, text='Name offset  X:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        tk.Spinbox(
            nof, textvariable=self._name_x_offset, from_=-2000, to=2000,
            increment=1, width=6,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
            buttonbackground='#444444',
        ).pack(side=tk.LEFT, padx=(2, 14))
        tk.Label(nof, text='Y:', bg=_BG_COLOUR, fg='white').pack(side=tk.LEFT)
        tk.Spinbox(
            nof, textvariable=self._name_y_offset, from_=-2000, to=2000,
            increment=1, width=6,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
            buttonbackground='#444444',
        ).pack(side=tk.LEFT, padx=(2, 10))
        tk.Label(nof, text='px  (relative to centred position)',
                 bg=_BG_COLOUR, fg='#666666', font=('TkDefaultFont', 8)).pack(side=tk.LEFT)

    # ── Live update loop ───────────────────────────────────────────────────────

    def _update_loop(self) -> None:
        """Poll the ROS node and refresh both canvases."""
        if not self._frozen:
            points = self._node.get_latest_points()
            if points is not None:
                rendered = render_front_view(
                    points, _RENDER_SIZE, _RENDER_SIZE,
                    point_size=self._point_size.get(),
                    color_mode=self._color_mode.get(),
                    val_min_clamp=self._parse_val('min'),
                    val_max_clamp=self._parse_val('max'),
                )
                rect_img = auto_crop_913(rendered)
                circle_bbox = self._get_circle_bbox(rect_img)
                circle_img = apply_ellipse_mask(rect_img, circle_bbox)
                self._current_rect_image = rect_img
                self._current_circle_image = circle_img
                self._refresh_preview(self._composite_onto_bg(rect_img))
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


    def _parse_val(self, which: str) -> float | None:
        """Parse the generic min/max field (range/X/Y); return None if blank or invalid."""
        raw = (self._val_min_var if which == 'min' else self._val_max_var).get().strip()
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

    # ── Font / colour pickers ──────────────────────────────────────────────────

    def _pick_font(self) -> None:
        """Open a searchable dialog listing all system fonts via fc-list."""
        if self._fonts_cache is None:
            self._status_var.set('Loading system fonts …')
            self.update_idletasks()
            self._fonts_cache = _get_system_fonts()
        if not self._fonts_cache:
            self._status_var.set(
                'No system fonts found — ensure fontconfig (fc-list) is installed.'
            )
            return

        # Prepend a "(default)" sentinel entry so users can reset
        all_fonts: list[tuple[str, str | None]] = [('(default)', None)] + self._fonts_cache  # type: ignore[operator]

        dlg = tk.Toplevel(self)
        dlg.title('Select Font')
        dlg.configure(bg=_BG_COLOUR)
        dlg.geometry('440x540')
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(True, True)

        # ── Search bar ────────────────────────────────────────────────────
        tk.Label(dlg, text='Search:', bg=_BG_COLOUR, fg='white').pack(
            anchor='w', padx=8, pady=(8, 2)
        )
        search_var = tk.StringVar()
        tk.Entry(
            dlg, textvariable=search_var,
            bg='#333333', fg='white', insertbackground='white',
            relief=tk.FLAT, highlightthickness=1, highlightbackground='#555555',
        ).pack(fill=tk.X, padx=8, pady=(0, 6))

        # ── Font listbox ──────────────────────────────────────────────────
        list_frame = tk.Frame(dlg, bg=_BG_COLOUR)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            bg='#222222', fg='white',
            selectbackground='#005f87', selectforeground='white',
            relief=tk.FLAT, bd=0, font=('TkDefaultFont', 10),
            activestyle='none',
        )
        listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        # Mutable list of currently visible (name, path) entries
        visible: list[tuple[str, str | None]] = list(all_fonts)

        def _repopulate(items: list) -> None:
            listbox.delete(0, tk.END)
            for name, _ in items:
                listbox.insert(tk.END, f'  {name}')

        _repopulate(visible)

        def _on_search(*_) -> None:
            q = search_var.get().lower()
            visible.clear()
            visible.extend((n, p) for n, p in all_fonts if q in n.lower())
            _repopulate(visible)

        search_var.trace_add('write', _on_search)

        # ── Font preview canvas ───────────────────────────────────────────
        preview_canvas = tk.Canvas(
            dlg, height=52, bg='#111111', highlightthickness=1,
            highlightbackground='#333333',
        )
        preview_canvas.pack(fill=tk.X, padx=8, pady=(6, 0))
        _preview_photo: list = [None]  # keep reference to avoid GC

        def _update_preview(*_) -> None:
            sel = listbox.curselection()
            preview_canvas.delete('all')
            if not sel:
                return
            _, path = visible[sel[0]]
            try:
                font = ImageFont.truetype(path, 26) if path else _load_font(26)
            except Exception:
                preview_canvas.create_text(
                    8, 26, text='(preview unavailable)',
                    fill='#555555', anchor='w',
                )
                return
            w = preview_canvas.winfo_width() or 420
            img = Image.new('RGBA', (w, 52), (17, 17, 17, 255))
            ImageDraw.Draw(img).text((8, 10), 'AaBbCc 123', font=font, fill=(255, 255, 255, 255))
            photo = ImageTk.PhotoImage(img)
            _preview_photo[0] = photo
            preview_canvas.create_image(0, 0, anchor='nw', image=photo)

        listbox.bind('<<ListboxSelect>>', _update_preview)

        # ── Buttons ───────────────────────────────────────────────────────
        count_var = tk.StringVar(value=f'{len(self._fonts_cache)} fonts found')
        tk.Label(dlg, textvariable=count_var, bg=_BG_COLOUR,
                 fg='#666666', font=('TkDefaultFont', 8)).pack(anchor='w', padx=8)

        def _on_search_count(*_) -> None:
            _on_search()
            n = len([x for x in visible if x[1] is not None])
            count_var.set(f'{n} fonts shown')

        search_var.trace_add('write', _on_search_count)

        btn_frame = tk.Frame(dlg, bg=_BG_COLOUR)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        def _apply() -> None:
            sel = listbox.curselection()
            if not sel:
                return
            name, path = visible[sel[0]]
            self._custom_font_path = path  # None → resets to default
            label = '(default)' if path is None else name[:16]
            self._font_display = label
            self._font_btn.config(text=f'Font: {label}')
            dlg.destroy()

        tk.Button(
            btn_frame, text='Cancel', width=8, command=dlg.destroy,
            bg='#333333', fg='white', relief=tk.FLAT,
            activebackground='#555555', activeforeground='white',
        ).pack(side=tk.RIGHT, padx=(4, 0))
        tk.Button(
            btn_frame, text='OK', width=8, command=_apply,
            bg='#005f87', fg='white', relief=tk.FLAT,
            activebackground='#007aad', activeforeground='white',
        ).pack(side=tk.RIGHT)

        listbox.bind('<Double-1>', lambda _e: _apply())
        dlg.bind('<Return>', lambda _e: _apply())
        dlg.bind('<Escape>', lambda _e: dlg.destroy())

    def _pick_text_color(self) -> None:
        result = colorchooser.askcolor(color=self._text_color, title='Text colour')
        if result[1] is None:
            return
        self._text_color = result[1]  # hex string e.g. '#ff8800'
        self._color_swatch.config(bg=self._text_color)

    def _get_name_font(
        self, size: int
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load the user-selected font at *size*, falling back to the default."""
        if self._custom_font_path:
            try:
                return ImageFont.truetype(self._custom_font_path, size)
            except Exception:
                pass
        return _load_font(size)

    def _text_fill(self, alpha: int = 230) -> tuple[int, int, int, int]:
        """Return the current text colour as an RGBA tuple."""
        c = self._text_color.lstrip('#')
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha)

    # ── Name overlay helpers ───────────────────────────────────────────────────

    def _apply_name_to_rect(self, img: Image.Image) -> Image.Image:
        """Return a copy of *img* with the user name drawn in the top half."""
        name = self._name_var.get().strip()
        if not name:
            return img
        out = img.convert('RGBA').copy()
        draw = ImageDraw.Draw(out)
        font_size = max(24, out.width // 14)
        font = self._get_name_font(font_size)
        bbox = draw.textbbox((0, 0), name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (out.width - tw) // 2 + self._name_x_offset.get()
        y = out.height // 8 + self._name_y_offset.get()   # upper portion of image
        # Dark outline for readability on any background
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (0, 2), (-2, 0), (2, 0)):
            draw.text((x + dx, y + dy), name, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), name, font=font, fill=self._text_fill())
        return out

    def _make_circle_with_name(self, circle_img: Image.Image) -> Image.Image:
        """Return *circle_img* with the user name panel placed to its right."""
        name = self._name_var.get().strip()
        if not name:
            return circle_img
        cw, ch = circle_img.size
        panel_w = max(180, cw // 2)
        # Start fully transparent so alpha_composite preserves the ellipse edges
        out = Image.new('RGBA', (cw + panel_w, ch), (0, 0, 0, 0))
        out.alpha_composite(circle_img.convert('RGBA'), dest=(0, 0))
        # Fill only the text panel with an opaque background
        draw = ImageDraw.Draw(out)
        draw.rectangle((cw, 0, cw + panel_w - 1, ch - 1), fill=(0, 0, 0, 255))
        font_size = max(20, panel_w // 7)
        font = self._get_name_font(font_size)
        bbox = draw.textbbox((0, 0), name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = cw + (panel_w - tw) // 2
        ty = (ch - th) // 2
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((tx + dx, ty + dy), name, font=font, fill=(0, 0, 0, 200))
        draw.text((tx, ty), name, font=font, fill=self._text_fill())
        return out

    # ── Background image ───────────────────────────────────────────────────────

    def _load_bg_from_path(self, path: str) -> None:
        """Load *path* as the background image, updating state and status bar."""
        try:
            self._bg_image = Image.open(path).convert('RGBA')
            self._bg_image.load()
            self._bg_file_path = path
            self._bg_path_var.set(os.path.basename(path))
            self._status_var.set(
                f'Background loaded: {os.path.basename(path)}  '
                f'({self._bg_image.width}×{self._bg_image.height} px)'
            )
        except Exception as exc:
            self._status_var.set(f'Failed to load background: {exc}')

    def _set_background(self) -> None:
        """Open a file dialog to select a background PNG and load it."""
        path = filedialog.askopenfilename(
            title='Select background image',
            filetypes=[('PNG images', '*.png'), ('All files', '*.*')],
        )
        if not path:
            return
        self._load_bg_from_path(path)

    def _clear_background(self) -> None:
        self._bg_image = None
        self._bg_file_path = ''
        self._bg_path_var.set('(none)')
        self._status_var.set('Background cleared.')

    def _composite_onto_bg(self, rect_img: Image.Image) -> Image.Image:
        """Paste *rect_img* onto a copy of the background at the configured offset.

        Returns the composited RGBA image, or *rect_img* unchanged if no
        background has been loaded.
        """
        if self._bg_image is None:
            return rect_img.convert('RGBA')
        try:
            ox = int(self._bg_x_var.get())
        except ValueError:
            ox = 0
        try:
            oy = int(self._bg_y_var.get())
        except ValueError:
            oy = 0
        scale = self._bg_scale_var.get() / 100.0
        if scale != 1.0 and scale > 0:
            new_w = max(1, round(self._bg_image.width * scale))
            new_h = max(1, round(self._bg_image.height * scale))
            bg = self._bg_image.resize((new_w, new_h), Image.LANCZOS)
        else:
            bg = self._bg_image.copy()
        fg = rect_img.convert('RGBA')
        bg.paste(fg, (ox, oy), fg)
        return bg

    # ── Config persistence ─────────────────────────────────────────────────────

    def _load_config(self) -> None:
        """Populate settings from ~/.lidar_imager.json (silently skipped if absent)."""
        try:
            with open(_CONFIG_PATH) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if data.get('export_dir'):
            self._export_dir = data['export_dir']
            self._export_dir_var.set(data['export_dir'])
        if data.get('bg_file_path'):
            self._bg_file_path = data['bg_file_path']
        if 'bg_x' in data:
            self._bg_x_var.set(str(data['bg_x']))
        if 'bg_y' in data:
            self._bg_y_var.set(str(data['bg_y']))
        if 'bg_scale' in data:
            self._bg_scale_var.set(float(data['bg_scale']))
        if 'color_mode' in data:
            self._color_mode.set(data['color_mode'])
        if 'val_min' in data:
            self._val_min_var.set(data['val_min'])
        if 'val_max' in data:
            self._val_max_var.set(data['val_max'])
        if 'point_size' in data:
            self._point_size.set(int(data['point_size']))
        if 'font_path' in data:
            self._custom_font_path = data['font_path'] or None
        if 'font_display' in data:
            self._font_display = data['font_display']
        if 'text_color' in data:
            self._text_color = data['text_color']
        if 'name_x_offset' in data:
            self._name_x_offset.set(int(data['name_x_offset']))
        if 'name_y_offset' in data:
            self._name_y_offset.set(int(data['name_y_offset']))

    def _save_config(self) -> None:
        """Write current settings to ~/.lidar_imager.json."""
        data = {
            'export_dir': self._export_dir or '',
            'bg_file_path': self._bg_file_path,
            'bg_x': self._bg_x_var.get(),
            'bg_y': self._bg_y_var.get(),
            'bg_scale': self._bg_scale_var.get(),
            'color_mode': self._color_mode.get(),
            'val_min': self._val_min_var.get(),
            'val_max': self._val_max_var.get(),
            'point_size': self._point_size.get(),
            'font_path': self._custom_font_path or '',
            'font_display': self._font_display,
            'text_color': self._text_color,
            'name_x_offset': self._name_x_offset.get(),
            'name_y_offset': self._name_y_offset.get(),
        }
        try:
            with open(_CONFIG_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _on_close(self) -> None:
        """Save config then destroy the window."""
        self._save_config()
        self.destroy()

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
        import os, re
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]  # ms precision
        name = self._name_var.get().strip()
        name_part = ('_' + re.sub(r'[^\w]', '_', name).strip('_')) if name else ''
        return os.path.join(self._export_dir, f'lidar_{stamp}{name_part}')

    # ── Export ─────────────────────────────────────────────────────────────────

    def _export_circle_pdf(self, circle_src: Image.Image, base_path: str) -> str:
        """Export a PDF containing two circles (2.25" and 1.25") plus name text.

        Both circles are rendered from *circle_src* (RGBA, transparent outside
        the ellipse) at 300 DPI so the printed sizes are exact.
        Returns the saved PDF path.
        """
        DPI     = 300
        LARGE_D = round(2.25 * DPI)   # 675 px  → 2.25" diameter
        SMALL_D = round(1.25 * DPI)   # 375 px  → 1.25" diameter
        MARGIN  = round(0.20 * DPI)   # 60 px   → 0.20" margin
        GAP     = round(0.25 * DPI)   # 75 px   → 0.25" gap between circles

        name = self._name_var.get().strip()
        name_area = round(0.42 * DPI) if name else round(0.10 * DPI)

        page_w = MARGIN + LARGE_D + GAP + SMALL_D + MARGIN
        page_h = MARGIN + LARGE_D + name_area + MARGIN
        page   = Image.new('RGB', (page_w, page_h), (255, 255, 255))

        def _paste(src: Image.Image, diameter: int, x: int, y: int) -> None:
            rgba  = src.convert('RGBA').resize((diameter, diameter), Image.LANCZOS)
            patch = Image.new('RGB', (diameter, diameter), (255, 255, 255))
            patch.paste(rgba.convert('RGB'), (0, 0), rgba)
            page.paste(patch, (x, y))

        # Large circle — left, top-aligned with margin
        _paste(circle_src, LARGE_D, MARGIN, MARGIN)

        # Small circle — right, vertically centred relative to large
        _paste(
            circle_src, SMALL_D,
            MARGIN + LARGE_D + GAP,
            MARGIN + (LARGE_D - SMALL_D) // 2,
        )

        # Name text — centred below both circles, scaled to fit
        if name:
            draw = ImageDraw.Draw(page)
            font_size = round(0.25 * DPI)   # start ~75 px ≈ 18 pt
            font = self._get_name_font(font_size)
            max_w = page_w - 2 * MARGIN
            while font_size > 20:
                bbox = draw.textbbox((0, 0), name, font=font)
                if (bbox[2] - bbox[0]) <= max_w:
                    break
                font_size -= 4
                font = self._get_name_font(font_size)
            bbox = draw.textbbox((0, 0), name, font=font)
            tx = (page_w - (bbox[2] - bbox[0])) // 2
            ty = MARGIN + LARGE_D + round(0.09 * DPI)
            c  = self._text_color.lstrip('#')
            fill = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
            draw.text((tx, ty), name, font=font, fill=fill)

        pdf_path = f'{base_path}_circles.pdf'
        page.save(pdf_path, format='PDF', resolution=DPI)
        return pdf_path

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

        self._composite_onto_bg(self._apply_name_to_rect(rect_src)).save(rect_path, format='PNG')
        pdf_path = self._export_circle_pdf(circle_src, base)

        self._status_var.set(f'Exported → {rect_path}  +  {pdf_path}')

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        if hasattr(self, '_after_id'):
            self.after_cancel(self._after_id)
        super().destroy()
