"""Measure tab -- K-space hyperspectral (per-pixel TWINS spectra).

A TWINS wedge scan stores the full 2-D ROI at every position (a datacube), then
HyperspectralProcessor runs an independent DFT per pixel -> a spectrum cube
(n_freq, h, w). You can scrub wavelength to see the spatial map at each λ and
click any pixel to see its spectrum.

The wedge stage is the ONLY scanned axis: one Acquire = one sweep = one cube.
The cube list / z_values plumbing below is kept single-entry (z = None) so the
saved .npz keeps the same layout the viewer and the analysis app already read.

Controls live in the sidebar tab; the map+spectrum open in a separate resizable
window (HyperViewer), like the repo's standalone measurement windows.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

# Disk-space guards for saving hypercubes (large files).
LOW_DISK_WARN_GB = 3.0     # warn once when free space drops below this during a scan
LOW_DISK_ABORT_GB = 0.5    # abort the scan to avoid failed / truncated saves


def _free_gb(path):
    """Free space (GB) on the volume that holds `path`, walking up to the nearest
    existing directory. Returns None if it can't be determined."""
    try:
        p = os.path.abspath(path)
        while p and not os.path.isdir(p):
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return shutil.disk_usage(p or ".").free / 1e9
    except Exception:  # noqa: BLE001
        return None

from instruments.subtwinslv import TwinsScanner
from instruments.h5_writer import (
    h5_filename, load_measurement_h5, save_measurement_h5,
)
from instruments.hyperspectral import (
    HyperspectralProcessor, DEFAULT_START_MM, DEFAULT_STOP_MM, DEFAULT_N_STEPS,
    DEFAULT_APODIZATION, DEFAULT_WL_START, DEFAULT_WL_STOP,
    ZEROFILL_FACTOR, ZEROFILL_MIN, ZEROFILL_MAX, resolve_n_points,
)
from instruments.dsp import APOD_TYPES

# QSettings scope. Deliberately NOT "MIR_CAMERA" -- that is the MWIR app's scope,
# and sharing it makes the two apps overwrite each other's saved scan parameters.
SETTINGS_ORG = "SWIR_CAMERA"

# Entries of the Measure tab's Format box.
FORMAT_NPZ = "NumPy (.npz)"
FORMAT_H5 = "HDF5 (.h5)"


# grey and jet aren't bundled in pyqtgraph and need matplotlib (absent here), so
# build them by hand; viridis/inferno/magma/turbo are pyqtgraph built-ins.
_MANUAL_CMAPS = {
    "grey": ([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]),
    "jet": ([0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
            [(0, 0, 128), (0, 0, 255), (0, 255, 255),
             (255, 255, 0), (255, 0, 0), (128, 0, 0)]),
}


def _get_cmap(name):
    """Return a pyqtgraph ColorMap by name, or None if unavailable."""
    key = str(name).lower()
    if key == "gray":
        key = "grey"
    if key in _MANUAL_CMAPS:
        pos, cols = _MANUAL_CMAPS[key]
        return pg.ColorMap(pos=pos, color=cols)
    try:
        cm = pg.colormap.get(key)        # pyqtgraph built-ins (viridis, etc.)
        if cm is not None:
            return cm
    except Exception:  # noqa: BLE001
        pass
    return None


# ===========================================================================
# Pop-out viewer: wavelength-scrub map + per-pixel spectrum + z selector
# ===========================================================================
class HyperViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("K-Space Hyperspectral Viewer")
        self.resize(900, 520)
        self.wavelengths = None
        self.cubes = []          # list of (n_freq, h, w)  (in-RAM mode)
        self.z_values = []       # list of z (mm) or None
        self.sat_masks = []      # list of (h, w) bool or None
        self.sel_px = None       # (row, col) in ROI
        self._cont_regions = []  # shaded band overlays on the spectrum plot
        # Lazy mode: a callable z_index -> cube, used for big Z-series files so
        # only the current position's cube is held in RAM.
        self.cube_loader = None
        self._lazy_cache = {}    # z_index -> cube (kept tiny)

        root = QHBoxLayout(self)

        # left: spatial map with wavelength slider (ImageView time axis = λ)
        left = QVBoxLayout()
        self.lbl_cal = QLabel("")          # wedge-axis calibration status (from metadata)
        self.lbl_cal.setStyleSheet("font-weight:600;")
        left.addWidget(self.lbl_cal)
        self.map_view = pg.ImageView()
        self.map_view.view.invertY(True)     # match live-view orientation
        self.map_view.scene.sigMouseClicked.connect(self._on_click)
        # Hide pyqtgraph's built-in timeline/ROI/menu -- we drive the wavelength
        # with our own slider below. (Leaving the built-in one connected made the
        # two sliders chase each other / drift.)
        for _w in (self.map_view.ui.roiBtn, self.map_view.ui.menuBtn,
                   self.map_view.ui.roiPlot):
            _w.hide()
        left.addWidget(self.map_view, 1)

        ctl = QHBoxLayout()
        self.combo_map = QComboBox()
        self.combo_map.addItems(["λ scrub", "Peak λ", "Peak intensity",
                                 "SAM (selected px)", "Continuum line"])
        self.combo_map.setToolTip("λ scrub: spatial map at each wavelength.\n"
                                  "Peak λ/intensity: per-pixel spectral peak.\n"
                                  "SAM: spectral-angle distance to the selected pixel.\n"
                                  "Continuum line: per-pixel line emission with the\n"
                                  "  broadband (thermal) continuum subtracted -- a linear\n"
                                  "  baseline from two shoulder bands is removed under the\n"
                                  "  line band (see CONTINUUM_SUBTRACTION.md).")
        self.combo_map.currentTextChanged.connect(self._refresh_map)
        ctl.addWidget(QLabel("Map:")); ctl.addWidget(self.combo_map, 1)
        self.combo_deriv = QComboBox()
        self.combo_deriv.addItems(["raw", "d/dλ", "d²/dλ²"])
        self.combo_deriv.setToolTip("Spectral derivative of the pixel spectrum.")
        self.combo_deriv.currentTextChanged.connect(self._update_spectrum)
        ctl.addWidget(QLabel("Spectrum:")); ctl.addWidget(self.combo_deriv, 1)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["grey", "inferno", "viridis", "magma", "turbo", "jet"])
        self.combo_cmap.setToolTip("Map colormap.")
        self.combo_cmap.currentTextChanged.connect(self._apply_cmap)
        ctl.addWidget(QLabel("Colors:")); ctl.addWidget(self.combo_cmap, 1)
        left.addLayout(ctl)

        # Continuum-line bands (line + two shoulders, µm). Only relevant/visible
        # when Map = "Continuum line"; a linear continuum estimated from the two
        # shoulders is subtracted under the line band, per pixel.
        def _band(lo, hi):
            a = QDoubleSpinBox(); b = QDoubleSpinBox()
            for s, v in ((a, lo), (b, hi)):
                s.setDecimals(3); s.setRange(0.1, 100.0); s.setSuffix(" µm")
                s.setSingleStep(0.005); s.setValue(v)
                s.valueChanged.connect(self._on_continuum_changed)
            return a, b
        self.cont_row_w = QWidget()
        crow = QHBoxLayout(self.cont_row_w); crow.setContentsMargins(0, 0, 0, 0)
        self.spin_line0, self.spin_line1 = _band(3.965, 4.050)
        self.spin_lsh0, self.spin_lsh1 = _band(3.900, 3.955)
        self.spin_rsh0, self.spin_rsh1 = _band(4.060, 4.120)
        crow.addWidget(QLabel("Line")); crow.addWidget(self.spin_line0); crow.addWidget(self.spin_line1)
        crow.addWidget(QLabel("L")); crow.addWidget(self.spin_lsh0); crow.addWidget(self.spin_lsh1)
        crow.addWidget(QLabel("R")); crow.addWidget(self.spin_rsh0); crow.addWidget(self.spin_rsh1)
        self.cont_row_w.setVisible(False)
        left.addWidget(self.cont_row_w)

        # Remember the colormap choice across sessions (and in the standalone viewer).
        self._cmap_settings = QtCore.QSettings(SETTINGS_ORG, "HyperViewer")
        saved_cmap = self._cmap_settings.value("cmap", "grey")
        self.combo_cmap.blockSignals(True)
        self.combo_cmap.setCurrentText(str(saved_cmap))
        self.combo_cmap.blockSignals(False)
        self._apply_cmap()

        # dedicated wavelength slider (active in λ-scrub mode)
        wlrow = QHBoxLayout()
        self.wl_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.wl_slider.valueChanged.connect(self._on_wl_slider)
        self.lbl_wl_val = QLabel("-- µm")
        self.lbl_wl_val.setStyleSheet("font-weight:600;")
        wlrow.addWidget(QLabel("Wavelength:"))
        wlrow.addWidget(self.wl_slider, 1)
        wlrow.addWidget(self.lbl_wl_val)
        left.addLayout(wlrow)

        # delay (z) slider -- hidden entirely unless a Z-scan was acquired
        self.zrow_w = QWidget()
        zrow = QHBoxLayout(self.zrow_w); zrow.setContentsMargins(0, 0, 0, 0)
        self.lbl_z = QLabel("Z (mm):")
        self.z_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        self.lbl_z_val = QLabel("-")
        zrow.addWidget(self.lbl_z)
        zrow.addWidget(self.z_slider, 1)
        zrow.addWidget(self.lbl_z_val)
        self.zrow_w.setVisible(False)
        left.addWidget(self.zrow_w)
        root.addLayout(left, 2)

        # right: spectrum of the selected pixel
        right = QVBoxLayout()
        self.spec_plot = pg.PlotWidget(title="Pixel spectrum")
        self.spec_plot.setLabel("bottom", "Wavelength", units="µm")
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen("#e8590c", width=2))
        right.addWidget(self.spec_plot, 1)
        self.lbl_pixel = QLabel("Click the map to pick a pixel")
        right.addWidget(self.lbl_pixel)
        root.addLayout(right, 1)

    def set_calibration_note(self, meta) -> None:
        """Show whether the loaded spectra were computed on the motor-calibrated
        wedge axis, read from the file/scan metadata."""
        meta = meta or {}
        if meta.get("position_axis_used") == "calibrated" or meta.get("motor_calibration_applied"):
            f = meta.get("motor_calibration_file") or "parameters_int.txt"
            self.lbl_cal.setText(f"Wedge axis: CALIBRATED  ({f})")
            self.lbl_cal.setStyleSheet("font-weight:600; color:#2f9e44;")
        elif meta:
            self.lbl_cal.setText("Wedge axis: raw measured (no motor calibration)")
            self.lbl_cal.setStyleSheet("font-weight:600; color:#e8590c;")
        else:
            self.lbl_cal.setText("")

    def set_result(self, wavelengths, cubes, z_values, sat_masks=None) -> None:
        self.wavelengths = np.asarray(wavelengths)
        self.cubes = list(cubes)
        self.z_values = list(z_values)
        self.sat_masks = list(sat_masks) if sat_masks else [None] * len(self.cubes)
        nf = len(self.wavelengths)
        self.wl_slider.blockSignals(True)
        self.wl_slider.setMinimum(0)
        self.wl_slider.setMaximum(max(0, nf - 1))
        self.wl_slider.setValue(0)
        self.wl_slider.blockSignals(False)
        self.cube_loader = None      # in-RAM mode
        self._lazy_cache = {}
        self._configure_sliders(len(self.cubes))

    def set_result_lazy(self, wavelengths, z_values, cube_loader,
                        sat_masks=None) -> None:
        """Z-series mode: cubes are loaded on demand via cube_loader(z_index),
        so only the current position is held in RAM (for big per-Z files)."""
        self.wavelengths = np.asarray(wavelengths)
        self.cubes = []
        self.z_values = list(z_values)
        self.sat_masks = list(sat_masks) if sat_masks else [None] * len(z_values)
        self.cube_loader = cube_loader
        self._lazy_cache = {}
        nf = len(self.wavelengths)
        self.wl_slider.blockSignals(True)
        self.wl_slider.setMinimum(0); self.wl_slider.setMaximum(max(0, nf - 1))
        self.wl_slider.setValue(0)
        self.wl_slider.blockSignals(False)
        self._configure_sliders(len(z_values))

    def _configure_sliders(self, n_z: int) -> None:
        self.zrow_w.setVisible(n_z > 1)
        self.z_slider.blockSignals(True)
        self.z_slider.setMinimum(0); self.z_slider.setMaximum(max(0, n_z - 1))
        self.z_slider.setValue(0)
        self.z_slider.blockSignals(False)
        self._show_cube(0)

    def _current_cube(self):
        zi = self.z_slider.value()
        if self.cube_loader is not None:
            if zi not in self._lazy_cache:
                self._lazy_cache.clear()          # keep only the current cube
                try:
                    self._lazy_cache[zi] = self.cube_loader(zi)
                except Exception:  # noqa: BLE001
                    return None
            return self._lazy_cache.get(zi)
        if not self.cubes:
            return None
        return self.cubes[zi]

    def _current_mask(self):
        zi = self.z_slider.value()
        if self.sat_masks and zi < len(self.sat_masks):
            return self.sat_masks[zi]
        return None

    def _reference_spectrum(self, cube):
        """Reference spectrum for SAM: the selected pixel, else the ROI mean."""
        if self.sel_px is not None:
            row, col = self.sel_px
            _, h, w = cube.shape
            if 0 <= row < h and 0 <= col < w:
                return cube[:, row, col]
        from instruments.analysis import roi_average
        return roi_average(cube, self._current_mask())[0]

    def _show_cube(self, z_idx: int) -> None:
        z = self.z_values[z_idx] if z_idx < len(self.z_values) else None
        self.lbl_z_val.setText("-" if z is None else f"{z:.4f} mm")
        self._refresh_map()
        self._update_spectrum()

    def _refresh_map(self) -> None:
        cube = self._current_cube()
        if cube is None:
            return
        mode = self.combo_map.currentText()
        self.cont_row_w.setVisible(mode == "Continuum line")
        self._update_continuum_overlay()
        if mode == "λ scrub":
            # cube is (n_freq, h, w); axis 0 = wavelength, driven by our slider.
            self.map_view.setImage(cube, xvals=self.wavelengths, autoLevels=True)
            self.map_view.ui.roiPlot.hide()   # setImage re-shows it; keep it hidden
            self.wl_slider.setEnabled(True)
            idx = max(0, min(self.wl_slider.value(), len(self.wavelengths) - 1))
            self.map_view.setCurrentIndex(idx)
            self._set_wl_label(idx)
            return
        from instruments import analysis as A
        if mode == "Peak λ":
            img = A.peak_wavelength_map(cube, self.wavelengths)
        elif mode == "Peak intensity":
            img = A.peak_intensity_map(cube)
        elif mode == "Continuum line":
            ln, ls, rs = self._continuum_bands()
            img = A.continuum_line_image(cube, self.wavelengths, ln, ls, rs)
            if img is None:
                self.wl_slider.setEnabled(False)
                self.lbl_wl_val.setText("bands out of range")
                return
        else:  # SAM
            img = A.spectral_angle_map(cube, self._reference_spectrum(cube))
        img = np.asarray(img, dtype=float)
        mask = self._current_mask()
        if mask is not None:
            img = img.copy(); img[np.asarray(mask, bool)] = np.nan
        self.map_view.setImage(img, autoLevels=True)
        # No wavelength axis on a 2-D map -> the slider doesn't apply.
        self.wl_slider.setEnabled(False)
        self.lbl_wl_val.setText(f"{mode} map")

    def _continuum_bands(self):
        """(line, left_shoulder, right_shoulder) bands (µm), each (lo, hi)."""
        return (tuple(sorted((self.spin_line0.value(), self.spin_line1.value()))),
                tuple(sorted((self.spin_lsh0.value(), self.spin_lsh1.value()))),
                tuple(sorted((self.spin_rsh0.value(), self.spin_rsh1.value()))))

    def _on_continuum_changed(self, *args) -> None:
        if self.combo_map.currentText() == "Continuum line":
            self._refresh_map()

    def _update_continuum_overlay(self) -> None:
        """Shade the line + shoulder bands on the pixel spectrum plot so they can
        be positioned on the resonance / clean continuum. Cleared in other modes."""
        for reg in self._cont_regions:
            self.spec_plot.removeItem(reg)
        self._cont_regions = []
        if self.combo_map.currentText() != "Continuum line":
            return
        ln, ls, rs = self._continuum_bands()
        for (lo, hi), rgba in ((ln, (232, 89, 12, 45)),      # line = orange
                               (ls, (77, 171, 247, 40)),     # shoulders = blue
                               (rs, (77, 171, 247, 40))):
            reg = pg.LinearRegionItem([lo, hi], movable=False,
                                      brush=pg.mkBrush(*rgba), pen=pg.mkPen(None))
            reg.setZValue(-10)
            self.spec_plot.addItem(reg)
            self._cont_regions.append(reg)

    def _on_z_changed(self, idx: int) -> None:
        # In lazy mode self.cubes is empty -> count by z_values instead.
        n = len(self.z_values)
        if 0 <= idx < n:
            self._show_cube(idx)

    def _apply_cmap(self, *args) -> None:
        cm = _get_cmap(self.combo_cmap.currentText())
        if cm is not None:
            self.map_view.setColorMap(cm)
        self._cmap_settings.setValue("cmap", self.combo_cmap.currentText())

    def _set_wl_label(self, idx: int) -> None:
        if self.wavelengths is None or len(self.wavelengths) == 0:
            return
        idx = max(0, min(idx, len(self.wavelengths) - 1))
        self.lbl_wl_val.setText(f"{self.wavelengths[idx]:.3f} µm")

    def _on_wl_slider(self, idx: int) -> None:
        """Dedicated wavelength slider -> drive the ImageView's scrub index."""
        if self.combo_map.currentText() != "λ scrub":
            return
        self.map_view.setCurrentIndex(int(idx))
        self._set_wl_label(int(idx))

    def _on_click(self, ev) -> None:
        cube = self._current_cube()
        if cube is None:
            return
        ii = self.map_view.getImageItem()
        pos = ev.scenePos()
        if not ii.sceneBoundingRect().contains(pos):
            return
        p = ii.mapFromScene(pos)
        col, row = int(p.x()), int(p.y())     # row-major image: x=col, y=row
        _, h, w = cube.shape
        if 0 <= row < h and 0 <= col < w:
            self.sel_px = (row, col)
            self._update_spectrum()
            # SAM is referenced to the selected pixel -> recompute the map.
            if self.combo_map.currentText().startswith("SAM"):
                self._refresh_map()

    def _update_spectrum(self) -> None:
        cube = self._current_cube()
        if cube is None or self.sel_px is None or self.wavelengths is None:
            return
        row, col = self.sel_px
        _, h, w = cube.shape
        if not (0 <= row < h and 0 <= col < w):
            return
        y = np.asarray(cube[:, row, col], dtype=float)
        order = {"raw": 0, "d/dλ": 1, "d²/dλ²": 2}.get(self.combo_deriv.currentText(), 0)
        for _ in range(order):
            y = np.gradient(y, self.wavelengths)
        self.spec_curve.setData(self.wavelengths, y)
        self.lbl_pixel.setText(f"Pixel (row {row}, col {col}) in ROI")


# ===========================================================================
# Live interferogram preview (ROI mean vs wedge position, during the scan)
# ===========================================================================
class LiveInterferogram(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live interferogram")
        self.resize(560, 320)
        lay = QVBoxLayout(self)
        self.plot = pg.PlotWidget(title="ROI mean vs wedge position")
        self.plot.setLabel("bottom", "Wedge position", units="mm")
        self.plot.setLabel("left", "ROI mean")
        self.curve = self.plot.plot(pen=pg.mkPen("#1c7ed6", width=2),
                                    symbol="o", symbolSize=3, symbolBrush="#1c7ed6")
        lay.addWidget(self.plot)
        self._x: list = []
        self._y: list = []

    def reset(self) -> None:
        self._x, self._y = [], []
        self.curve.setData([], [])

    def add_point(self, pos: float, value: float) -> None:
        self._x.append(pos); self._y.append(value)
        self.curve.setData(self._x, self._y)


def load_kspace_npz(path: str):
    """Read a saved K-space .npz -> (wavelengths, cubes, z_values, sat_masks).

    Works for files saved by MeasurePanel (spectral cube, optional saturation
    masks and raw interferogram). Returns None if the spectral cube is absent.
    """
    d = np.load(path, allow_pickle=True)
    if "wavelengths" not in d or "spectrum_cubes" not in d:
        return None
    wl = np.asarray(d["wavelengths"])
    # A cube saved in complex form is displayed as its magnitude.
    cubes = [np.abs(c) if np.iscomplexobj(c) else np.asarray(c)
             for c in d["spectrum_cubes"]]
    zv = d["z_values"] if "z_values" in d else np.full(len(cubes), np.nan)
    z_values = [None if np.isnan(v) else float(v) for v in np.asarray(zv).ravel()]
    masks = None
    if "saturation_masks" in d:
        masks = [np.asarray(m, bool) for m in d["saturation_masks"]]
    return wl, cubes, z_values, masks


def kspace_metadata(path: str) -> dict:
    """The embedded metadata dict of a saved K-space .npz (or {}). Lets a viewer
    show whether the spectra were computed on the calibrated wedge axis."""
    try:
        with np.load(path, allow_pickle=True) as d:
            if "metadata" in d.files:
                return dict(d["metadata"].item())
    except Exception:  # noqa: BLE001
        pass
    return {}


# ===========================================================================
# Sidebar controls
# ===========================================================================
class MeasurePanel(QWidget):
    sig_status = QtCore.pyqtSignal(str)
    sig_done = QtCore.pyqtSignal(object, object, object, object)  # wl, cubes, z, sat_masks
    sig_point = QtCore.pyqtSignal(float, float, bool)  # pos, ROI-mean, is_first_of_scan
    sig_warn = QtCore.pyqtSignal(str, str)  # (title, message) -> modal warning on the GUI thread

    def __init__(self, stages_panel, frame_source, roi_provider,
                 roi_show=None, bg_provider=None, save_dir_provider=None,
                 meta_provider=None, save_dir: str = r"D:\CAMERA\kspace") -> None:
        super().__init__()
        self.sp = stages_panel
        self.frame_source = frame_source
        self.roi_provider = roi_provider      # () -> (r0,r1,c0,c1) or None (full frame)
        self.roi_show = roi_show              # (bool) -> toggle the on-image ROI box
        self.bg_provider = bg_provider        # () -> (background_frame|None, subtract_bool)
        self.save_dir_provider = save_dir_provider  # () -> current save folder (camera)
        self.meta_provider = meta_provider    # () -> dict of camera metadata
        self.save_dir = save_dir              # fallback if no provider
        self._scan_meta = {}                  # scan parameters captured at scan start
        self.background_map = None             # binned-ROI background saved with the cube
        self.background_subtracted = False
        self._abort = False
        self._paused = False                   # Pause holds the worker at loop boundaries
        self._low_disk_warned = False
        self.viewer = None
        self.wavelengths = None
        self.cubes = []                        # ALWAYS real (magnitude) -- display
        # Complex spectra, populated only when "Save complex spectrum" is on.
        # The saved file uses these; every display path uses self.cubes.
        self.complex_cubes = []
        self.z_values = []
        self.sat_masks = []
        # Raw interferogram cubes + positions (per z), kept for optional saving
        # and for walk-off calibration from a sharp-sample scan.
        self.raw_cubes = []
        self.raw_positions = []
        self._last_datacube = None
        self._last_positions = None
        # Per-run save folder (set on each Acquire in _start); all of a run's
        # files land in <run-stamp>.<filename>/ under the camera folder.
        self._run_folder = None
        self._run_stamp = None
        self._save_folder = None
        self._save_fname = None
        self.live_monitor = None
        # A processor instance used only for live scan-parameter estimation
        # (resolution / max-step); loads the calibration once.
        self.est_proc = HyperspectralProcessor()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_roi_group())
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_spectrum_group())
        layout.addWidget(self._build_postproc_group())
        layout.addWidget(self._build_walkoff_group())
        layout.addWidget(self._build_run_group())

        # Persist scan/spectrum params (incl. the wavelength window) between
        # measurements. Restore first, THEN bind saves so restoring doesn't
        # immediately rewrite the same values.
        self._settings = QtCore.QSettings(SETTINGS_ORG, "KSpace")
        self._restore_settings()
        for widget, _cast in self._persisted_spins().values():
            widget.valueChanged.connect(self._save_settings)
        self.combo_apod.currentTextChanged.connect(self._save_settings)
        self.chk_walkoff.toggled.connect(self._save_settings)
        self.combo_save.currentTextChanged.connect(self._save_settings)
        self.combo_format.currentTextChanged.connect(self._save_settings)
        self.chk_save_raw.toggled.connect(self._save_settings)
        for chk in self._persisted_checks().values():   # sat, svd
            chk.toggled.connect(self._save_settings)
        self.edit_filename.editingFinished.connect(self._save_settings)

        self.sig_status.connect(self._on_status)
        self.sig_done.connect(self._on_done)
        self.sig_point.connect(self._on_point)
        self.sig_warn.connect(self._on_warn)
        self._update_step()

        # Keep the ROI px readout live while the box is dragged.
        self._roi_timer = QtCore.QTimer(self)
        self._roi_timer.timeout.connect(self._refresh_roi_label)
        self._roi_timer.start(700)

    # -- groups --------------------------------------------------------------
    def _build_roi_group(self) -> QGroupBox:
        g = QGroupBox("ROI + binning")
        grid = QGridLayout(g)
        hint = QLabel("Set the ROI on the live view (tick \"Show measurement ROI\" "
                      "and drag the box). It is saved with the measurement.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(hint, 0, 0, 1, 2)

        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 64)
        self.spin_bin.setValue(1)
        self.spin_bin.setToolTip("Bin NxN pixels into one super-pixel (better SNR, "
                                 "smaller cube). 1 = no binning.")
        self.spin_bin.valueChanged.connect(self._refresh_roi_label)
        grid.addWidget(QLabel("Binning (NxN)"), 1, 0)
        grid.addWidget(self.spin_bin, 1, 1)

        self.lbl_roi = QLabel("ROI: full frame")
        self.lbl_roi.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(self.lbl_roi, 2, 0, 1, 2)
        return g

    def _refresh_roi_label(self) -> None:
        roi = self.roi_provider() if self.roi_provider else None
        b = self.spin_bin.value()
        if roi is None:
            self.lbl_roi.setText("ROI: full frame")
        else:
            r0, r1, c0, c1 = roi
            h, w = (r1 - r0) // b, (c1 - c0) // b
            self.lbl_roi.setText(f"ROI: {r1-r0}×{c1-c0} px  →  {h}×{w} after bin {b}")

    def _build_scan_group(self) -> QGroupBox:
        g = QGroupBox("TWINS cube scan")
        grid = QGridLayout(g)
        self.spin_start = self._mm_spin(DEFAULT_START_MM)
        self.spin_stop = self._mm_spin(DEFAULT_STOP_MM)
        self.spin_steps = QSpinBox(); self.spin_steps.setRange(3, 10000)
        self.spin_steps.setValue(DEFAULT_N_STEPS)
        for s in (self.spin_start, self.spin_stop, self.spin_steps):
            s.valueChanged.connect(self._update_step)
        self.lbl_step = QLabel("-- µm"); self.lbl_step.setStyleSheet("font-weight:600;")
        self.spin_frames = QSpinBox(); self.spin_frames.setRange(1, 200); self.spin_frames.setValue(1)
        grid.addWidget(QLabel("Start"), 0, 0); grid.addWidget(self.spin_start, 0, 1)
        grid.addWidget(QLabel("Stop"), 1, 0); grid.addWidget(self.spin_stop, 1, 1)
        grid.addWidget(QLabel("Steps"), 2, 0); grid.addWidget(self.spin_steps, 2, 1)
        grid.addWidget(QLabel("Step size"), 3, 0); grid.addWidget(self.lbl_step, 3, 1)
        grid.addWidget(QLabel("Frames/point"), 4, 0); grid.addWidget(self.spin_frames, 4, 1)
        return g

    def _build_spectrum_group(self) -> QGroupBox:
        g = QGroupBox("Spectrum (per-pixel DFT)")
        grid = QGridLayout(g)
        # Apodization window: 'gaussian' is the NIREOS position-space window
        # (uses Gauss width below); the rest are standard FTIR windows.
        self.combo_apod = QComboBox(); self.combo_apod.addItems(APOD_TYPES)
        self.combo_apod.setCurrentText("gaussian")
        self.combo_apod.setToolTip("Apodization window. Affects the spectral "
                                   "lineshape and the resolution estimate below.")
        self.combo_apod.currentTextChanged.connect(self._update_step)
        self.spin_apod = QDoubleSpinBox(); self.spin_apod.setRange(0.01, 5.0)
        self.spin_apod.setSingleStep(0.05); self.spin_apod.setValue(DEFAULT_APODIZATION)
        self.spin_apod.setToolTip("Gaussian apodization width (only for the "
                                  "'gaussian' window type).")
        # MWIR band for this InSb camera (repo defaults are LWIR 8-14 µm).
        # Overridden by the last-used value once a measurement has been run.
        self.spin_wl0 = self._um_spin(DEFAULT_WL_START)
        self.spin_wl1 = self._um_spin(DEFAULT_WL_STOP)
        self.spin_wl0.valueChanged.connect(self._update_step)
        self.spin_wl1.valueChanged.connect(self._update_step)
        # 0 = "Auto": ZEROFILL_FACTOR x steps, clamped to [MIN, MAX]. This is
        # interpolation/oversampling of the spectrum, NOT true resolution (that
        # is fixed by the scan length). A manual value > 0 overrides Auto.
        self.spin_nfreq = QSpinBox(); self.spin_nfreq.setRange(0, ZEROFILL_MAX)
        self.spin_nfreq.setValue(0); self.spin_nfreq.setSpecialValueText("Auto")
        self.spin_nfreq.setToolTip(
            f"Spectral output points (interpolation only -- does not change the\n"
            f"true resolution, which is set by the scan length).\n"
            f"Auto = {ZEROFILL_FACTOR}x steps, clamped to [{ZEROFILL_MIN}, {ZEROFILL_MAX}].")
        self.spin_nfreq.valueChanged.connect(self._update_step)
        self.lbl_nfreq_auto = QLabel("")
        self.lbl_nfreq_auto.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(QLabel("Apod type"), 0, 0); grid.addWidget(self.combo_apod, 0, 1)
        grid.addWidget(QLabel("Gauss width"), 1, 0); grid.addWidget(self.spin_apod, 1, 1)
        grid.addWidget(QLabel("λ start"), 2, 0); grid.addWidget(self.spin_wl0, 2, 1)
        grid.addWidget(QLabel("λ stop"), 3, 0); grid.addWidget(self.spin_wl1, 3, 1)
        grid.addWidget(QLabel("N freq"), 4, 0); grid.addWidget(self.spin_nfreq, 4, 1)
        grid.addWidget(self.lbl_nfreq_auto, 5, 0, 1, 2)

        # Calibration-aware estimates: spectral resolution from scan range +
        # apodization window, and the max stage step that still samples the
        # shortest λ at 5 pts/cycle.
        self.lbl_resolution = QLabel("-- nm"); self.lbl_resolution.setStyleSheet("font-weight:600;")
        self.lbl_min_step = QLabel("-- µm"); self.lbl_min_step.setStyleSheet("font-weight:600;")
        grid.addWidget(QLabel("Resolution"), 6, 0); grid.addWidget(self.lbl_resolution, 6, 1)
        grid.addWidget(QLabel("Max step (5/cyc)"), 7, 0); grid.addWidget(self.lbl_min_step, 7, 1)

        # Apodization ZPD centre. The WHOLE acquired interferogram is always
        # transformed, so this only sets where the apodization window is centred:
        # an independent I² barycentre per pixel (DEFAULT -- follows a zero-path
        # position that varies across the field of view), or one field-wide
        # envelope centre-burst (signed spatial sum). Either way the centre comes
        # from the acquired data; no expected-ZPD position is assumed.
        self.combo_center = QComboBox()
        self.combo_center.addItems(["barycentre (per-pixel)", "envelope (field)",
                                    "geometric centre"])
        self.combo_center.setCurrentText("barycentre (per-pixel)")
        self.combo_center.setToolTip(
            "How the apodization centre (ZPD) is located in the acquired scan:\n"
            "  barycentre (per-pixel) = each pixel's own I² centroid (default)\n"
            "  envelope (field)       = one centre-burst for the whole frame\n"
            "  geometric centre       = the midpoint sample of the scan, ignoring\n"
            "                           the signal (use when the scan is already\n"
            "                           centred on ZPD)")
        self.combo_center.currentTextChanged.connect(self._save_settings)
        grid.addWidget(QLabel("Apod centre"), 8, 0); grid.addWidget(self.combo_center, 8, 1)

        # Keep the complex DFT instead of its magnitude. The viewer and the
        # ROI-average CSV still show |spectrum|; only the SAVED cube differs.
        self.chk_complex = QCheckBox("Save complex spectrum (keep phase)")
        self.chk_complex.setToolTip(
            "Save the cube as complex64 -- float32 real + float32 imag -- so the "
            "interferometric PHASE is kept alongside the amplitude, instead of "
            "the float32 magnitude alone.\n"
            "Same float32 precision either way; the file grows only because two "
            "numbers per element are stored instead of one (8 vs 4 bytes).\n"
            "The viewer, the maps and the ROI-average CSV always display "
            "|spectrum|, so nothing changes on screen.")
        self.chk_complex.toggled.connect(self._save_settings)
        grid.addWidget(self.chk_complex, 9, 0, 1, 2)
        return g

    def _center_method(self) -> str:
        """Apodization-centre method for the processor: 'barycenter' (per-pixel,
        default), 'envelope' (field-wide) or 'geometric' (scan midpoint)."""
        text = self.combo_center.currentText()
        if text.startswith("bary"):
            return "barycenter"
        return "geometric" if text.startswith("geom") else "envelope"

    def _build_postproc_group(self) -> QGroupBox:
        g = QGroupBox("Post-processing")
        grid = QGridLayout(g)
        self.chk_sat = QCheckBox("Mask saturated pixels")
        self.chk_sat.setChecked(True)
        self.chk_sat.setToolTip("Exclude pixels that clipped at any wedge "
                                "position from the ROI average and the maps.")
        grid.addWidget(self.chk_sat, 0, 0, 1, 2)
        self.spin_sat = QSpinBox(); self.spin_sat.setRange(1, 65535)
        self.spin_sat.setValue(16383); self.spin_sat.setSuffix(" cts")
        self.spin_sat.setToolTip("Saturation count level (14-bit full scale = 16383).")
        grid.addWidget(QLabel("Saturation level"), 1, 0); grid.addWidget(self.spin_sat, 1, 1)

        return g

    def _build_walkoff_group(self) -> QGroupBox:
        g = QGroupBox("Walk-off correction (TWINS image drift)")
        grid = QGridLayout(g)
        hint = QLabel("Calibrate the per-frame image shift ONCE on a sharp, "
                      "high-contrast target, then apply it to every scan.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(hint, 0, 0, 1, 2)

        self.chk_walkoff = QCheckBox("Apply walk-off correction")
        grid.addWidget(self.chk_walkoff, 1, 0, 1, 2)

        self.spin_wo_y = self._wo_spin(); self.spin_wo_x = self._wo_spin()
        grid.addWidget(QLabel("Rate Y (px/mm)"), 2, 0); grid.addWidget(self.spin_wo_y, 2, 1)
        grid.addWidget(QLabel("Rate X (px/mm)"), 3, 0); grid.addWidget(self.spin_wo_x, 3, 1)

        self.btn_wo_cal = QPushButton("Calibrate from last scan")
        self.btn_wo_cal.setToolTip("Register the frames of the most recent scan "
                                   "(use a sharp sample) and fit the shift rate.")
        self.btn_wo_cal.clicked.connect(self._calibrate_walkoff)
        grid.addWidget(self.btn_wo_cal, 4, 0, 1, 2)

        self.lbl_wo = QLabel("not calibrated")
        self.lbl_wo.setWordWrap(True); self.lbl_wo.setStyleSheet("color:#888; font-size:11px;")
        grid.addWidget(self.lbl_wo, 5, 0, 1, 2)
        return g

    def _wo_spin(self):
        s = QDoubleSpinBox(); s.setRange(-1000.0, 1000.0); s.setDecimals(4)
        s.setSingleStep(0.1); s.setValue(0.0); s.setSuffix(" px/mm"); return s

    def _calibrate_walkoff(self) -> None:
        cube = self._last_datacube
        pos = self._last_positions
        if cube is None or pos is None:
            self.lbl_wo.setText("Run a scan on a sharp sample first, then calibrate.")
            return
        self.lbl_wo.setText("calibrating (registering frames)...")
        self.btn_wo_cal.setEnabled(False)
        try:
            from instruments.walkoff import estimate_shift_rate
            est = estimate_shift_rate(cube, pos)
            self.spin_wo_y.setValue(est["rate_y"])
            self.spin_wo_x.setValue(est["rate_x"])
            self.chk_walkoff.setChecked(True)
            self.lbl_wo.setText(
                f"rate_y={est['rate_y']:.3f} (r²={est['r2_y']:.2f}), "
                f"rate_x={est['rate_x']:.3f} (r²={est['r2_x']:.2f}) px/mm — "
                f"low r² ⇒ not a clean linear drift / use a sharper sample.")
        except Exception as e:  # noqa: BLE001
            self.lbl_wo.setText(f"calibration error: {e}")
        finally:
            self.btn_wo_cal.setEnabled(True)

    def _build_run_group(self) -> QGroupBox:
        g = QGroupBox("Run")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Acquire")
        self.btn_run.clicked.connect(self._start)
        self.btn_pause = QPushButton("Pause"); self.btn_pause.setEnabled(False)
        self.btn_pause.setToolTip("Hold the scan at the next interferogram / DFT "
                                  "boundary (the current cube finishes first). "
                                  "Press again to resume.")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_run); row.addWidget(self.btn_pause)
        row.addWidget(self.btn_stop)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_recompute = QPushButton("Recompute")
        self.btn_recompute.clicked.connect(self._recompute)
        self.btn_recompute.setToolTip("Re-run the DFT on the LAST scan's raw "
                                      "interferogram with the current settings (apodization "
                                      "type/width, apod centre, λ window, N freq) -- no re-scan.")
        self.btn_view = QPushButton("Open Viewer"); self.btn_view.clicked.connect(self._open_viewer)
        self.btn_load = QPushButton("Load"); self.btn_load.clicked.connect(self._load)
        self.btn_load.setToolTip("Open a saved K-space .npz in the viewer.")
        self.btn_save = QPushButton("Save"); self.btn_save.clicked.connect(self._save)
        row2.addWidget(self.btn_recompute); row2.addWidget(self.btn_view)
        row2.addWidget(self.btn_load); row2.addWidget(self.btn_save)
        v.addLayout(row2)
        row3 = QHBoxLayout()
        self.combo_save = QComboBox()
        self.combo_save.addItems(["Full cube (every pixel)", "ROI average", "Both"])
        self.combo_save.setCurrentText("Both")
        self.combo_save.setToolTip(
            "What 'Save' writes:\n"
            "  Full cube  -> every pixel's spectrum (.npz, λ×h×w)\n"
            "  ROI average -> mean±std spectrum over the acquired ROI (.csv)\n"
            "  Both -> both files.")
        row3.addWidget(QLabel("Save as")); row3.addWidget(self.combo_save, 1)
        v.addLayout(row3)
        row3b = QHBoxLayout()
        self.combo_format = QComboBox()
        self.combo_format.addItems([FORMAT_NPZ, FORMAT_H5])
        self.combo_format.setToolTip(
            "File format for the full cube:\n"
            f"  {FORMAT_NPZ}  -> one .npz, the default; every tool here reads it\n"
            f"  {FORMAT_H5}  -> ScopeFoundry layout, t0/c0/image + t0/c0/position_mm,\n"
            "                named <timestamp>_hyperspectral_<filename>.h5\n"
            "The ROI-average CSV is unaffected.")
        row3b.addWidget(QLabel("Format")); row3b.addWidget(self.combo_format, 1)
        v.addLayout(row3b)
        row4 = QHBoxLayout()
        self.edit_filename = QLineEdit("kspace")
        self.edit_filename.setToolTip("Base filename; files are saved as "
                                      "<date>.<filename> in the camera's save folder.")
        row4.addWidget(QLabel("Filename")); row4.addWidget(self.edit_filename, 1)
        v.addLayout(row4)
        self.chk_save_raw = QCheckBox("Raw interferogram saved for reprocessing (always)")
        self.chk_save_raw.setToolTip("The raw (positions, datacube) is ALWAYS stored "
                                     "in the .npz so you can reprocess offline (FT window / "
                                     "apodization / ZPD) without re-scanning. It is small "
                                     "next to the spectrum (n_pos << n_freq).")
        self.chk_save_raw.setChecked(True)
        self.chk_save_raw.setEnabled(False)
        v.addWidget(self.chk_save_raw)
        self.progress = QProgressBar(); v.addWidget(self.progress)
        self.lbl_status = QLabel("idle"); self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_status.setWordWrap(True); v.addWidget(self.lbl_status)
        return g

    # -- small spin helpers --------------------------------------------------
    def _mm_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.0, 50.0); s.setDecimals(3)
        s.setSingleStep(0.1); s.setValue(val); s.setSuffix(" mm"); return s

    def _um_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.1, 100.0); s.setDecimals(2)
        s.setSingleStep(0.1); s.setValue(val); s.setSuffix(" µm"); return s

    # -- persistence (remember params between measurements) ------------------
    def _persisted_spins(self) -> dict:
        """key -> (widget, cast). QSettings stores strings, so cast on restore."""
        return {
            "ks_start": (self.spin_start, float),
            "ks_stop": (self.spin_stop, float),
            "ks_steps": (self.spin_steps, int),
            "ks_frames": (self.spin_frames, int),
            "ks_bin": (self.spin_bin, int),
            "ks_apod": (self.spin_apod, float),
            "ks_wl0": (self.spin_wl0, float),
            "ks_wl1": (self.spin_wl1, float),
            "ks_nfreq": (self.spin_nfreq, int),
            "ks_wo_y": (self.spin_wo_y, float),
            "ks_wo_x": (self.spin_wo_x, float),
            "ks_sat_level": (self.spin_sat, int),
        }

    def _persisted_checks(self) -> dict:
        """key -> checkbox widgets persisted between measurements."""
        return {"ks_sat_on": self.chk_sat, "ks_complex": self.chk_complex}

    def _restore_settings(self) -> None:
        for key, (widget, cast) in self._persisted_spins().items():
            val = self._settings.value(key, None)
            if val is None:
                continue
            try:
                widget.setValue(cast(val))
            except (TypeError, ValueError):
                pass
        apod = self._settings.value("ks_apod_type", None)
        if apod is not None:
            self.combo_apod.setCurrentText(str(apod))
        ctr = self._settings.value("ks_apod_center", None)
        if ctr is not None:
            self.combo_center.setCurrentText(str(ctr))
        wo = self._settings.value("ks_walkoff_on", None)
        if wo is not None:
            self.chk_walkoff.setChecked(str(wo).lower() == "true")
        save_mode = self._settings.value("ks_save_mode", None)
        if save_mode is not None:
            self.combo_save.setCurrentText(str(save_mode))
        save_fmt = self._settings.value("ks_save_format", None)
        if save_fmt is not None:
            self.combo_format.setCurrentText(str(save_fmt))
        for key, chk in self._persisted_checks().items():
            v = self._settings.value(key, None)
            if v is not None:
                chk.setChecked(str(v).lower() == "true")
        fn = self._settings.value("ks_filename", None)
        if fn is not None:
            self.edit_filename.setText(str(fn))

    def _save_settings(self, *args) -> None:
        for key, (widget, _cast) in self._persisted_spins().items():
            self._settings.setValue(key, widget.value())
        self._settings.setValue("ks_apod_type", self.combo_apod.currentText())
        self._settings.setValue("ks_apod_center", self.combo_center.currentText())
        self._settings.setValue("ks_walkoff_on", self.chk_walkoff.isChecked())
        self._settings.setValue("ks_save_mode", self.combo_save.currentText())
        self._settings.setValue("ks_save_format", self.combo_format.currentText())
        for key, chk in self._persisted_checks().items():
            self._settings.setValue(key, chk.isChecked())
        self._settings.setValue("ks_filename", self.edit_filename.text())

    def _update_step(self) -> None:
        n = self.spin_steps.value()
        start, stop = self.spin_start.value(), self.spin_stop.value()
        step_um = None
        if n > 1:
            step_um = abs(stop - start) / (n - 1) * 1000
            self.lbl_step.setText(f"{step_um:.1f} µm")
        else:
            self.lbl_step.setText("-- µm")
        # Show what "Auto" resolves to (interpolation bins), so the readout
        # tracks the scan length live.
        if hasattr(self, "lbl_nfreq_auto"):
            manual = self.spin_nfreq.value()
            resolved = resolve_n_points(n, manual=manual)
            if manual > 0:
                self.lbl_nfreq_auto.setText(f"N freq = {resolved} (manual)")
            else:
                self.lbl_nfreq_auto.setText(f"Auto → {resolved} bins ({ZEROFILL_FACTOR}×{n} steps)")

        # Calibration-aware resolution + sampling-adequacy check.
        if hasattr(self, "lbl_resolution"):
            wl0, wl1 = self.spin_wl0.value(), self.spin_wl1.value()
            wl_center = 0.5 * (wl0 + wl1)
            apod_type = self.combo_apod.currentText()
            res_nm = self.est_proc.estimate_resolution_nm(
                abs(stop - start), wl_center, apod_type=apod_type)
            self.lbl_resolution.setText(
                "-- nm" if res_nm is None else f"~{res_nm:.0f} nm @ {wl_center:.1f} µm")

            wl_short = min(wl0, wl1)
            max_um = self.est_proc.max_step_um(wl_short, samples_per_cycle=5)
            if max_um is None:
                self.lbl_min_step.setText("-- µm")
                self.lbl_min_step.setStyleSheet("font-weight:600;")
            else:
                self.lbl_min_step.setText(f"{max_um:.2f} µm @ {wl_short:.1f} µm")
                # Red if the chosen step under-samples the shortest λ, else green.
                if step_um is not None and step_um > max_um:
                    self.lbl_min_step.setStyleSheet("font-weight:600; color:#f44336;")
                else:
                    self.lbl_min_step.setStyleSheet("font-weight:600; color:#4CAF50;")

    # -- run -----------------------------------------------------------------
    def _start(self) -> None:
        if not getattr(self.sp.twins, "is_connected", False):
            self.sig_status.emit("TWINS stage not connected")
            return
        if self.frame_source() is None:
            self.sig_status.emit("No live frame -- start the camera first")
            return
        roi = self.roi_provider() if self.roi_provider else None   # None = full frame
        self._scan_roi = roi          # saved with the measurement
        self._scan_bin = self.spin_bin.value()
        if roi is not None:
            r0, r1, c0, c1 = roi
            self.sig_status.emit(f"ROI saved for scan: rows {r0}-{r1}, cols {c0}-{c1}")

        walkoff = (dict(rate_y=self.spin_wo_y.value(), rate_x=self.spin_wo_x.value())
                   if self.chk_walkoff.isChecked() else None)
        # Snapshot the captured background (full frame) + whether to subtract it,
        # taken now so it can't change mid-scan.
        bg, bg_sub = self.bg_provider() if self.bg_provider else (None, False)
        bg = None if bg is None else np.asarray(bg, dtype=np.float32)
        if roi is not None and bg is not None:
            self.sig_status.emit("background " + ("subtracted" if bg_sub else "saved (not subtracted)"))
        params = dict(
            start=self.spin_start.value(), stop=self.spin_stop.value(),
            n=self.spin_steps.value(), frames=self.spin_frames.value(), roi=roi,
            bin=self.spin_bin.value(),
            apod=self.spin_apod.value(), apod_type=self.combo_apod.currentText(),
            wl0=self.spin_wl0.value(),
            wl1=self.spin_wl1.value(), nfreq=self.spin_nfreq.value(),
            walkoff=walkoff,
            background=bg, bg_subtract=bool(bg_sub and bg is not None),
            sat_on=self.chk_sat.isChecked(), sat_level=self.spin_sat.value(),
            center_method=self._center_method(),
            complex_out=self.chk_complex.isChecked(),
        )
        # Capture scan parameters as metadata (no arrays) for the saved hypercube.
        nsteps = params["n"]
        step_um = (abs(params["stop"] - params["start"]) / (nsteps - 1) * 1000
                   if nsteps > 1 else None)
        self._scan_meta = dict(
            start_mm=params["start"], stop_mm=params["stop"], n_steps=nsteps,
            step_um=step_um, frames_per_point=params["frames"], binning=params["bin"],
            roi=list(roi) if roi is not None else None,
            apodization=params["apod_type"], apod_width=params["apod"],
            wl_start_um=params["wl0"], wl_stop_um=params["wl1"],
            n_freq_setting=params["nfreq"],
            walkoff=walkoff, background_subtracted=params["bg_subtract"],
            saturation_masking=params["sat_on"], saturation_level=params["sat_level"],
            ft_region="full",
            apod_center=params["center_method"],
            complex_spectrum=params["complex_out"],
            filename=self.edit_filename.text().strip() or "kspace",
        )
        # Each Acquire = one experiment "run": save ALL its files into a folder
        # named <run-timestamp>.<filename> under the camera folder.
        camera_folder = (self.save_dir_provider() if self.save_dir_provider
                         else None) or self.save_dir
        # --- Disk-space check: estimate the data size and warn if the save volume
        # is low (do this BEFORE freezing the UI / starting the thread). ---
        frame = self.frame_source()
        if roi is not None:
            hh, ww = roi[1] - roi[0], roi[3] - roi[2]
        elif frame is not None:
            hh, ww = int(frame.shape[0]), int(frame.shape[1])
        else:
            hh = ww = 0
        binf = max(1, params["bin"])
        hb, wb = max(1, hh // binf), max(1, ww // binf)
        n_freq_est = resolve_n_points(params["n"], manual=params["nfreq"])
        # per cube = raw (n_pos planes) + spectrum (n_freq planes), float32. Files
        # are saved UNCOMPRESSED, so this matches actual on-disk size closely.
        est_gb = (params["n"] + n_freq_est) * hb * wb * 4 / 1e9
        free_gb = _free_gb(camera_folder)
        if free_gb is not None and (free_gb < est_gb * 1.1 or free_gb < LOW_DISK_ABORT_GB):
            drive = os.path.splitdrive(os.path.abspath(camera_folder))[0] or camera_folder
            ans = QMessageBox.warning(
                self, "Low disk space",
                f"This acquisition needs roughly {est_gb:.1f} GB, but only "
                f"{free_gb:.1f} GB is free on {drive}\\.\n\nData may fail to save "
                f"mid-scan. Proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                self.sig_status.emit(
                    f"acquisition cancelled -- only {free_gb:.1f} GB free (need ~{est_gb:.1f} GB)")
                return
        self._save_fname = self.edit_filename.text().strip() or "kspace"
        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_folder = os.path.join(camera_folder, f"{self._run_stamp}.{self._save_fname}")
        self._save_folder = self._run_folder   # the run's files go here
        self._cam_meta = (self.meta_provider() or {}) if self.meta_provider else {}
        self._save_raw_flag = self.chk_save_raw.isChecked()
        self._abort = False
        self._paused = False
        self.raw_cubes, self.raw_positions = [], []
        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)
        self.sp.freeze(True)
        # One TWINS wedge sweep per Acquire -> a single hyperspectral cube, so the
        # progress bar tracks the wedge steps of that sweep.
        self.progress.setMaximum(max(1, params["n"]))
        self.progress.setValue(0)
        # Live interferogram preview (centerburst forming as the wedge scans).
        if self.live_monitor is None:
            self.live_monitor = LiveInterferogram()
        self.live_monitor.reset()
        self.live_monitor.show(); self.live_monitor.raise_()
        # Keep the handle so shutdown() can abort + join this scan before the
        # stages are disconnected (else close-during-scan races the stage DLLs).
        self._scan_thread = threading.Thread(target=self._worker, args=(params,), daemon=True)
        self._scan_thread.start()

    @QtCore.pyqtSlot(float, float, bool)
    def _on_point(self, pos: float, value: float, is_first: bool) -> None:
        if self.live_monitor is None:
            return
        if is_first:
            self.live_monitor.reset()
        self.live_monitor.add_point(pos, value)

    def _stop(self) -> None:
        self._abort = True
        self._paused = False   # release the worker if it's parked in a pause
        self.sig_status.emit("stopping...")

    def _toggle_pause(self) -> None:
        """Pause/resume the running scan. The worker parks at the next loop
        boundary (finishing the in-flight cube / DFT first), so the stages are
        never stopped mid-move. Stop still works while paused."""
        self._paused = not self._paused
        self.btn_pause.setText("Resume" if self._paused else "Pause")
        self.sig_status.emit("paused" if self._paused else "resuming...")

    def _wait_if_paused(self) -> None:
        """Block the worker thread while paused (polling so Stop still aborts)."""
        while self._paused and not self._abort:
            time.sleep(0.1)

    def _recompute(self) -> None:
        """Re-run the DFT on the last scan's RAW interferogram with the current
        settings (apodization type/width, apod centre, λ window, N freq) -- no
        re-scan. Lets you compare e.g. barycentre vs envelope centring on
        already-acquired data."""
        if not getattr(self, "raw_cubes", None):
            self.lbl_status.setText("no raw data to recompute -- run a scan first")
            return
        p = dict(
            wl0=self.spin_wl0.value(), wl1=self.spin_wl1.value(),
            nfreq=self.spin_nfreq.value(), apod=self.spin_apod.value(),
            apod_type=self.combo_apod.currentText(),
            walkoff=(dict(rate_y=self.spin_wo_y.value(), rate_x=self.spin_wo_x.value())
                     if self.chk_walkoff.isChecked() else None),
            sat_on=self.chk_sat.isChecked(), sat_level=self.spin_sat.value(),
            center_method=self._center_method(),
            complex_out=self.chk_complex.isChecked(),
        )
        # Keep the saved metadata in step with what was recomputed.
        self._scan_meta.update(
            ft_region="full",
            apodization=p["apod_type"], apod_width=p["apod"],
            wl_start_um=p["wl0"], wl_stop_um=p["wl1"], n_freq_setting=p["nfreq"],
            apod_center=p["center_method"],
            complex_spectrum=p["complex_out"],
            recomputed=True)
        self.btn_run.setEnabled(False)
        self.btn_recompute.setEnabled(False)
        self.lbl_status.setText(
            f"recomputing from raw (apod centre: {p['center_method']})...")
        threading.Thread(target=self._recompute_worker, args=(p,), daemon=True).start()

    def _recompute_worker(self, p: dict) -> None:
        try:
            from instruments.analysis import saturation_mask
            proc = HyperspectralProcessor()
            cubes, masks, wls = [], [], None
            complex_cubes = []
            for positions, datacube in zip(self.raw_positions, self.raw_cubes):
                datacube = np.asarray(datacube)
                sat_mask = None
                if p["sat_on"]:
                    sat_src = datacube
                    if self.background_subtracted and self.background_map is not None:
                        sat_src = datacube + self.background_map[None, :, :]
                    sat_mask = saturation_mask(sat_src, p["sat_level"])
                n_freq = resolve_n_points(len(positions), manual=p["nfreq"])
                wl, cube = proc.compute_hyperspectral(
                    positions, datacube, wl_start=p["wl0"], wl_stop=p["wl1"],
                    apod_width=p["apod"], n_freq=n_freq,
                    apod_type=p["apod_type"], walkoff=p["walkoff"],
                    center_method=p["center_method"],
                    complex_output=p["complex_out"])
                if cube is None:
                    continue
                # Keep the complex cube for saving; everything that DISPLAYS or
                # averages the cube works on the magnitude, so the viewer, maps
                # and ROI-average CSV are unaffected by the choice.
                if p["complex_out"]:
                    complex_cubes.append(np.asarray(cube, dtype=np.complex64))
                    cube = np.abs(cube).astype(np.float32)
                wls = wl
                cubes.append(cube)
                masks.append(sat_mask)
            self.complex_cubes = complex_cubes
            self.sig_done.emit(wls, cubes, self.z_values, masks)
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"recompute error: {e}")
            self.sig_done.emit(None, None, None, None)

    def _position_calibration(self, positions):
        """(calibrated_axis, info) for a raw measured wedge axis -- the SAME
        motor-nonlinearity correction compute_hyperspectral applies internally,
        recomputed here so the calibrated axis can be saved alongside the raw one.
        `info` records whether it was applied, the cal file, and the peak shift."""
        info = {"motor_calibration_applied": False, "motor_calibration_file": None,
                "position_axis_used": "raw_measured"}
        if positions is None:
            return None, info
        raw = np.asarray(positions, dtype=float)
        try:
            from instruments.calibration import (calibrate_position_axis,
                                                  position_calibration_status)
            avail, fname = position_calibration_status()
            cal = np.asarray(calibrate_position_axis(raw), dtype=float)
            info["motor_calibration_applied"] = bool(avail)
            info["motor_calibration_file"] = fname
            # Which axis the saved spectra were actually computed on. The DFT
            # runs on the calibrated axis whenever the cal file is available.
            info["position_axis_used"] = "calibrated" if avail else "raw_measured"
            if avail and cal.shape == raw.shape:
                info["motor_calibration_max_dev_um"] = float(
                    np.max(np.abs(cal - raw)) * 1e3)
            return cal, info
        except Exception as e:  # noqa: BLE001
            info["motor_calibration_error"] = str(e)
            return raw, info

    def _worker(self, p: dict) -> None:
        try:
            from instruments.analysis import saturation_mask
            from instruments.subtwinslv import bin_image
            scanner = TwinsScanner(self.sp.twins, self.frame_source)
            proc = HyperspectralProcessor()
            cubes, zvals, masks, wls = [], [], [], None
            complex_cubes = []
            raw_cubes, raw_positions = [], []
            # Background: subtract from each frame (if enabled) and record the
            # binned-ROI background in the cube geometry for saving.
            bg = p["background"]
            bg_for_scan = bg if p["bg_subtract"] else None
            self.background_subtracted = bool(p["bg_subtract"] and bg is not None)
            if bg is not None:
                _roi = p["roi"]
                _crop = bg if _roi is None else bg[_roi[0]:_roi[1], _roi[2]:_roi[3]]
                self.background_map = bin_image(np.asarray(_crop, dtype=np.float32), p["bin"])
            else:
                self.background_map = None
            # One Acquire = ONE TWINS wedge sweep -> one raw interferogram cube.
            # (There is no Z / angle grid in this app: the SLC-1750 wedge stage is
            # the only scanned axis.)
            def prog(i, tot, pos, value):
                self.sig_status.emit(f"pt {i}/{tot}: wedge step")
                self.sig_point.emit(float(pos), float(value), i == 1)

            acquired = []   # [{positions, datacube}] -- 0 or 1 entry
            self._wait_if_paused()
            if not self._abort:
                positions, datacube = scanner.scan_cube(
                    p["start"], p["stop"], p["n"], p["roi"],
                    frames_avg=p["frames"], bin_factor=p["bin"], background=bg_for_scan,
                    progress=prog, should_abort=lambda: self._abort,
                    status_cb=lambda m: self.sig_status.emit(m))

                if datacube is not None and len(positions) >= 3:
                    # Keep the raw cube so walk-off can be calibrated from it
                    # later (use a sharp-sample scan).
                    self._last_datacube = datacube
                    self._last_positions = positions
                    acquired.append({"positions": np.asarray(positions),
                                     "datacube": np.asarray(datacube)})
                    # Disk-space guard. Nothing further is acquired after this
                    # point, so warn rather than abort -- aborting here would only
                    # throw away the cube we just spent the whole sweep acquiring.
                    # The final write is guarded by the pre-flight check in _start
                    # and reports failure via the auto-save path.
                    free = _free_gb(self._save_folder)
                    if free is not None and free < LOW_DISK_WARN_GB:
                        self.sig_warn.emit(
                            "Low disk space",
                            f"Only {free:.1f} GB free on the save drive — free up "
                            f"space or this acquisition may fail to save.")

            # ---- Phase 2: transform the acquired cube (per-pixel DFT).
            for item in acquired:
                self._wait_if_paused()   # park before the DFT if paused
                if self._abort:
                    break
                positions, datacube = item["positions"], item["datacube"]
                # Saturation mask from the RAW counts: if a background was
                # subtracted, add it back so a clipped pixel still reads >= the
                # saturation level (otherwise subtraction would mask it).
                sat_mask = None
                if p["sat_on"]:
                    sat_src = datacube
                    if self.background_subtracted and self.background_map is not None:
                        sat_src = datacube + self.background_map[None, :, :]
                    sat_mask = saturation_mask(sat_src, p["sat_level"])
                # "Auto" (nfreq == 0) -> ZEROFILL_FACTOR x steps, clamped.
                n_freq = resolve_n_points(len(positions), manual=p["nfreq"])
                self.sig_status.emit(f"per-pixel DFT ({n_freq} bins)...")
                wl, cube = proc.compute_hyperspectral(
                    positions, datacube, wl_start=p["wl0"], wl_stop=p["wl1"],
                    apod_width=p["apod"], n_freq=n_freq,
                    apod_type=p["apod_type"], walkoff=p["walkoff"],
                    center_method=p["center_method"],
                    complex_output=p["complex_out"])
                if cube is None:
                    continue
                # Keep the complex cube for saving; everything that DISPLAYS or
                # averages the cube works on the magnitude, so the viewer, maps
                # and ROI-average CSV are unaffected by the choice.
                if p["complex_out"]:
                    complex_cubes.append(np.asarray(cube, dtype=np.complex64))
                    cube = np.abs(cube).astype(np.float32)
                wls = wl
                cubes.append(cube)
                # No Z / angle axis in this app -> the viewer's delay slider stays
                # hidden, and the saved .npz records z_values = [NaN].
                zvals.append(None)
                masks.append(sat_mask)
                raw_cubes.append(datacube)
                raw_positions.append(positions)
                # The finished cube is auto-saved once by _on_done via
                # _autosave_cube(), so there is no per-grid-point save here.
            self.raw_cubes = raw_cubes
            self.raw_positions = raw_positions
            self.complex_cubes = complex_cubes
            self.sig_done.emit(wls, cubes, zvals, masks)
        except Exception as e:  # noqa: BLE001
            self.sig_status.emit(f"error: {e}")
            self.sig_done.emit(None, None, None, None)

    # -- signal slots --------------------------------------------------------
    @QtCore.pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)
        # Acquisition progress: "pt <i>/<total> ..." marks each grid point.
        if msg.startswith("pt "):
            try:
                cur = int(msg.split()[1].split("/")[0])
                self.progress.setValue(cur)
            except Exception:  # noqa: BLE001
                pass

    @QtCore.pyqtSlot(str, str)
    def _on_warn(self, title: str, message: str) -> None:
        """Modal warning raised from the scan thread (via sig_warn)."""
        QMessageBox.warning(self, title, message)

    @QtCore.pyqtSlot(object, object, object, object)
    def _on_done(self, wavelengths, cubes, z_values, sat_masks) -> None:
        self.btn_run.setEnabled(True)
        self.btn_recompute.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self._paused = False
        self.btn_stop.setEnabled(False)
        self.sp.freeze(False)
        if not cubes:
            self.lbl_status.setText("no data (aborted or scan failed)")
            return
        self.wavelengths, self.cubes, self.z_values = wavelengths, cubes, z_values
        self.sat_masks = sat_masks or [None] * len(cubes)
        nsat = sum(int(m.sum()) for m in self.sat_masks if m is not None)
        sat_txt = f", {nsat} saturated px" if nsat else ""
        msg = f"done: cube {cubes[0].shape} (λ×h×w){sat_txt}"
        # Auto-save the hypercube (+ metadata) so no scan is ever lost.
        try:
            path = self._autosave_cube()
            msg += f"  |  auto-saved {os.path.basename(path)}"
        except Exception as e:  # noqa: BLE001
            msg += f"  |  AUTO-SAVE FAILED: {e}"
        self.lbl_status.setText(msg)
        self._open_viewer()

    def _run_target(self):
        """(folder, stamp, fname) for the current experiment's files. Creates the
        <run-stamp>.<filename> run folder under the camera folder. Falls back to a
        fresh run folder if there is no active scan (e.g. a manual save after Load)."""
        folder = getattr(self, "_run_folder", None)
        stamp = getattr(self, "_run_stamp", None)
        fname = getattr(self, "_save_fname", None) or self.edit_filename.text().strip() or "kspace"
        if not folder or not stamp:
            base = (self.save_dir_provider() if self.save_dir_provider else None) or self.save_dir
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(base, f"{stamp}.{fname}")
        os.makedirs(folder, exist_ok=True)
        return folder, stamp, fname

    def _autosave_cube(self) -> str:
        """Write the full hypercube into the run folder in the selected format --
        called automatically after every scan."""
        folder, stamp, fname = self._run_target()
        return self._save_cube(os.path.join(folder, f"{stamp}.{fname}"))

    def _open_viewer(self) -> None:
        if not self.cubes:
            self.lbl_status.setText("nothing to view yet")
            return
        if self.viewer is None:
            self.viewer = HyperViewer()
        self.viewer.set_result(self.wavelengths, self.cubes, self.z_values,
                               getattr(self, "sat_masks", None))
        try:
            self.viewer.set_calibration_note(self._build_metadata())
        except Exception:  # noqa: BLE001
            pass
        self.viewer.show()
        self.viewer.raise_()

    def _roi_average(self):
        """Spatial mean & std spectrum over each cube (the acquired ROI),
        excluding saturated pixels. Returns (wavelengths, means, stds) with
        means/stds shaped (n_z, n_freq)."""
        from instruments.analysis import roi_average
        wl = np.asarray(self.wavelengths)
        masks = getattr(self, "sat_masks", None) or [None] * len(self.cubes)
        ms = [roi_average(c, masks[i] if i < len(masks) else None)
              for i, c in enumerate(self.cubes)]
        means = np.array([m[0] for m in ms])
        stds = np.array([m[1] for m in ms])
        return wl, means, stds

    def _save_roi_average_csv(self, stem: str) -> str:
        """Write the ROI-average spectrum/spectra as a CSV: wavelength_um, then a
        mean & std column per z map."""
        wl, means, stds = self._roi_average()
        cols = [wl]
        header = ["wavelength_um"]
        for zi, z in enumerate(self.z_values):
            tag = "" if z is None else f"_z{z:.4f}mm"
            cols += [means[zi], stds[zi]]
            header += [f"mean{tag}", f"std{tag}"]
        np.savetxt(stem + "_roi_avg.csv", np.column_stack(cols),
                   delimiter=",", header=",".join(header), comments="")
        return stem + "_roi_avg.csv"

    def _build_metadata(self) -> dict:
        """Scan parameters + camera state describing the saved hypercube."""
        meta = dict(self._scan_meta or {})
        if self.cubes:
            c0 = np.asarray(self.cubes[0])
            meta["cube_shape"] = list(c0.shape)        # (n_freq, h, w)
            meta["n_freq"] = int(c0.shape[0])
            meta["n_maps"] = len(self.cubes)
        if self.wavelengths is not None and len(self.wavelengths):
            meta["wl_min_um"] = float(np.min(self.wavelengths))
            meta["wl_max_um"] = float(np.max(self.wavelengths))
        meta["z_values_mm"] = [None if z is None else float(z) for z in self.z_values]
        meta["saturated_px"] = sum(int(m.sum()) for m in (self.sat_masks or [])
                                   if m is not None)
        # Whether the spectra were computed on the motor-nonlinearity-corrected
        # wedge axis (parameters_int.txt), with the peak shift it introduced.
        rawpos0 = (self.raw_positions[0] if getattr(self, "raw_positions", None)
                   else None)
        _, cal_info = self._position_calibration(rawpos0)
        meta.update(cal_info)
        meta["saved_local_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.meta_provider:
            try:
                meta["camera"] = self.meta_provider() or {}
            except Exception:  # noqa: BLE001
                pass
        return meta

    def _save_cube(self, stem: str) -> str:
        """Write the full cube in the format picked in the Format box."""
        if self.combo_format.currentText() == FORMAT_H5:
            return self._save_cube_h5(stem)
        return self._save_cube_npz(stem)

    def _h5_path(self, stem: str) -> str:
        """`<run folder>/<yymmdd_HHMMSS>_hyperspectral[_<filename>].h5`.

        The timestamp is derived from the run stamp (not taken fresh) so the .h5
        and the run folder that holds it always agree; ScopeFoundry's format is
        %y%m%d_%H%M%S, this app's run stamp is %Y%m%d_%H%M%S, hence the slice."""
        folder = os.path.dirname(stem)
        stamp = getattr(self, "_run_stamp", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
        sample = getattr(self, "_save_fname", None) or self.edit_filename.text().strip()
        return os.path.join(folder, h5_filename(stamp[2:], sample=sample))

    def _save_cube_h5(self, stem: str) -> str:
        """Write the measurement as ScopeFoundry-layout HDF5. The raw
        interferogram IS the file's primary dataset here (t0/c0/image), so a
        cube with no raw data (e.g. one loaded from an old .npz) cannot be
        written this way -- the caller surfaces that error."""
        roi = getattr(self, "_scan_roi", None)
        cal = None
        if getattr(self, "raw_positions", None):
            cal = [self._position_calibration(pp)[0] for pp in self.raw_positions]
        base = (self.save_dir_provider() if self.save_dir_provider else None) or self.save_dir
        return save_measurement_h5(
            self._h5_path(stem),
            raw_cubes=getattr(self, "raw_cubes", None) or [],
            raw_positions=getattr(self, "raw_positions", None) or [],
            raw_positions_calibrated=cal,
            wavelengths=self.wavelengths,
            spectrum_cubes=self._spectrum_for_saving() if self.cubes else None,
            sat_masks=getattr(self, "sat_masks", None),
            background=getattr(self, "background_map", None),
            background_subtracted=bool(getattr(self, "background_subtracted", False)),
            roi=roi,
            binning=int(getattr(self, "_scan_bin", 1)),
            metadata=self._build_metadata(),
            sample=(getattr(self, "_save_fname", None)
                    or self.edit_filename.text().strip()),
            save_dir=base)

    def _spectrum_for_saving(self):
        """The cube array to write: complex64 when "Save complex spectrum" was on
        for this result, else the float32 magnitude in self.cubes."""
        cx = getattr(self, "complex_cubes", None)
        if cx and len(cx) == len(self.cubes):
            # complex64 = float32 real + float32 imag: the phase costs one extra
            # float32 per element, never float64. Pinned so nothing upcasts.
            return np.asarray(cx, dtype=np.complex64)
        return np.asarray(self.cubes, dtype=np.float32)

    def _save_cube_npz(self, stem: str) -> str:
        roi = getattr(self, "_scan_roi", None)
        meta = self._build_metadata()
        kw = dict(
            wavelengths=self.wavelengths,
            spectrum_cubes=self._spectrum_for_saving(),
            z_values=np.asarray([np.nan if z is None else z for z in self.z_values]),
            z_unit="mm",
            roi=np.asarray(roi if roi is not None else [], dtype=float),
            binning=int(getattr(self, "_scan_bin", 1)),
            metadata=np.array(meta, dtype=object),          # dict (load allow_pickle=True)
            metadata_json=json.dumps(meta, default=str, indent=2))  # portable, human-readable
        masks = getattr(self, "sat_masks", None)
        if masks and any(m is not None for m in masks):
            h, w = np.asarray(self.cubes[0]).shape[1:]
            kw["saturation_masks"] = np.array([
                (np.zeros((h, w), bool) if m is None else np.asarray(m, bool))
                for m in masks])
        # The captured background (binned-ROI geometry, aligns with the cube) +
        # whether it was subtracted from the interferogram.
        if getattr(self, "background_map", None) is not None:
            kw["background"] = np.asarray(self.background_map)
            kw["background_subtracted"] = bool(self.background_subtracted)
        # ALWAYS store the raw interferogram cube(s) + positions, so the file can
        # be reprocessed offline (different apodization / ZPD / FT window) without
        # re-scanning. The raw is small vs the spectrum (n_pos << n_freq).
        if getattr(self, "raw_cubes", None):
            try:
                kw["raw_interferograms"] = np.asarray(self.raw_cubes)
                kw["raw_positions"] = np.asarray(self.raw_positions)
                # The calibrated (motor-corrected) axis per z, as actually used.
                kw["raw_positions_calibrated"] = np.asarray(
                    [self._position_calibration(p)[0] for p in self.raw_positions])
            except Exception:  # noqa: BLE001 (ragged per-z shapes -> skip raw)
                pass
        np.savez(stem + ".npz", **kw)  # uncompressed: fast write, cube barely compresses
        return stem + ".npz"

    def _save(self) -> None:
        if not self.cubes:
            self.lbl_status.setText("nothing to save")
            return
        mode = self.combo_save.currentText()
        want_cube = mode.startswith("Full cube") or mode == "Both"
        want_roi = mode.startswith("ROI average") or mode == "Both"
        # Save into the run folder (<run-stamp>.<filename>/) so all of this
        # experiment's files stay together.
        try:
            folder, stamp, fname = self._run_target()
            stem = os.path.join(folder, f"{stamp}.{fname}")
            saved = []
            if want_cube:
                saved.append(os.path.basename(self._save_cube(stem)))
            if want_roi:
                saved.append(os.path.basename(self._save_roi_average_csv(stem)))
            self.lbl_status.setText(f"saved {', '.join(saved)} to {folder}")
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"save error: {e}")

    def _load(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        folder = (self.save_dir_provider() if self.save_dir_provider
                  else None) or self.save_dir
        start = folder if os.path.isdir(folder) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load K-space measurement", start,
            "Measurements (*.npz *.h5);;NumPy archive (*.npz);;HDF5 (*.h5)")
        if not path:
            return
        try:
            res = (load_measurement_h5(path) if path.lower().endswith(".h5")
                   else load_kspace_npz(path))
            if res is None:
                self.lbl_status.setText("file has no spectral cube")
                return
            wl, cubes, z_values, masks = res
            self.wavelengths, self.cubes, self.z_values = wl, cubes, z_values
            self.sat_masks = masks or [None] * len(cubes)
            self.raw_cubes, self.raw_positions = [], []  # not reloaded for viewing
            self.complex_cubes = []
            self.lbl_status.setText(
                f"loaded {os.path.basename(path)}: {len(cubes)} map(s), cube {cubes[0].shape}")
            self._open_viewer()
        except Exception as e:  # noqa: BLE001
            self.lbl_status.setText(f"load error: {e}")

    def shutdown(self) -> None:
        # Abort a running acquisition and wait for its worker thread to unwind
        # BEFORE the caller (MainWindow.closeEvent) disconnects the stages -- the
        # scan thread drives the TWINS stage, so releasing the DLL out
        # from under an in-flight move would race. should_abort=lambda: self._abort
        # is polled per step, so the worker stops at the next step boundary.
        self._abort = True
        t = getattr(self, "_scan_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=10.0)
        if self.viewer is not None:
            self.viewer.close()
        if self.live_monitor is not None:
            self.live_monitor.close()
