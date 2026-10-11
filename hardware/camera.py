"""
Teledyne Retiga E7 camera wrapper (PVCAM via PyVCAM).

Thin wrapper around PVcamDevice (Teledyne_ScopeFoundry/CameraDevice.py) that
exposes the SAME device functions used by CameraHW.py / hyperspectral_measure.py
(RetigaE7_PI branch), behind a clean open/close API the rest of the app shares.
Every call forwards 1:1 to the matching PVcamDevice method; nothing here changes
camera behaviour.

Device methods forwarded (see Teledyne_ScopeFoundry/CameraDevice.py):
    set_exposure / get_exposure            (integer milliseconds)
    set_binning / get_binning              (1 / 2 / 4, square)
    set_roi / getSubarray*                 (h0, v0, width, height in sensor px)
    set_readout / get_readout              (0 Speed, 1 Long Exposure, 2 Long Exposure EDR)
    set_gain / get_gain                    (1 Full Well, 2 Sensitivity; only 1 for readout 0)
    set_trigger_mode / get_trigger_mode    ('Internal Trigger', 'Edge Trigger', ...)
    get_temperature / get_temperature_setpoint / get_idname / get_width / get_height
    acq_start            -> cam.start_live()   (circular buffer, 16 frames)
    acq_start_seq(n)     -> cam.start_seq(n)   (non-circular, n frames)
    get_nparray          -> cam.poll_frame()['pixel_data']
    acq_stop             -> cam.finish()

The only direct PyVCAM calls that bypass PVcamDevice are poll_frame(timeout_ms)
and abort(): PVcamDevice.get_nparray waits forever, which would hang a scan
thread on a missed hardware trigger. They are the same underlying calls.
"""

from __future__ import annotations

import sys
import os
import time
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, '..')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def add_path(path):
    """Add a folder under the 'ScopeFoundry Projects' root to sys.path.

    This file is at  <ScopeFoundry Projects>/HyperMeasurementApp/hardware/camera.py
    so the projects root is three levels up; `path` (e.g. 'Teledyne_ScopeFoundry')
    is a sibling of HyperMeasurementApp.
    """
    here = os.path.abspath(__file__)
    projects_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    target = os.path.join(projects_root, path)
    if not os.path.isdir(target):
        print(f"[Camera] WARNING: expected device folder not found: {target}")
    if target not in sys.path:
        sys.path.append(target)


# PVCAM exposure (trigger) modes exposed by CameraHW.py, in the same order.
TRIGGER_MODES = ['Internal Trigger', 'Edge Trigger', 'Trigger First',
                 'Software Trigger Edge', 'Software Trigger First']

# Readout ports of the Retiga E7 (CameraDevice.py): index -> label.
READOUT_PORTS = {0: 'Speed', 1: 'Long Exposure', 2: 'Long Exposure EDR'}

# Gain indices (CameraHW.py): 1 Full Well, 2 Sensitivity (readout 1 and 2 only).
GAINS = {1: 'Full Well', 2: 'Sensitivity'}

# Retiga E7 sensor, sensor pixels (width, height).
SENSOR_W, SENSOR_H = 3200, 2200

# Readout time vs ROI, from hyperspectral_measure.set_motor_velocity (RetigaE7_PI):
#   readout_s = 1 / (a / (H - b) + c)    for ROI width W in (lo, hi]
_READOUT_FIT = [
    (2900, 3200, 1.029e+05, -35.78, -4.887),
    (2600, 2900, 1.089e+05, -42.89, -0.969),
    (0,    2600, 1.085e+05, -43.19,  4.604),
]


def readout_time_s(roi_width: int, roi_height: int) -> float:
    """Empirical Retiga E7 readout time (s) for the given ROI, as in set_motor_velocity."""
    W, H = int(roi_width), int(roi_height)
    a, b, c = _READOUT_FIT[0][2:]
    for lo, hi, fa, fb, fc in _READOUT_FIT:
        if lo < W <= hi:
            a, b, c = fa, fb, fc
            break
    denom = a / max(H - b, 1e-6) + c
    return 1.0 / denom if denom > 0 else 0.0


def max_frame_rate_hz(roi_width: int, roi_height: int, exposure_s: float) -> float:
    """Maximum internal-trigger frame rate (Hz) for the ROI + exposure (set_motor_velocity)."""
    t = exposure_s + readout_time_s(roi_width, roi_height)
    return 1.0 / t if t > 0 else 0.0


class RetigaCamera:
    """
    Wrapper around PVcamDevice (Teledyne Retiga E7).

    Keeps a small amount of cached state (exposure, binning, ROI, readout, gain,
    trigger mode) so the UI can show values before the camera is opened; once
    opened the cached values are pushed to the device and every getter reads
    back from the hardware.
    """

    def __init__(self,
                 exposure_ms: int = 20,
                 binning: int = 1,
                 readout_port: int = 0,
                 gain: int = 1,
                 trigger_mode: str = 'Internal Trigger',
                 roi: tuple[int, int, int, int] | None = None):
        self.sensor_h = SENSOR_W           # horizontal size (width)
        self.sensor_v = SENSOR_H           # vertical size (height)

        # Cached settings (mirrored to hardware once open).
        self.exposure_ms = int(exposure_ms)
        self.binning = int(binning)
        self.readout_port = int(readout_port)
        self.gain = int(gain)
        self.trigger_mode = trigger_mode
        h0, v0, w, h = roi if roi else (0, 0, SENSOR_W, SENSOR_H)
        self.subarrayh, self.subarrayv = int(w), int(h)
        self.subarrayh_pos, self.subarrayv_pos = int(h0), int(v0)

        self.device = None          # PVcamDevice
        self._open = False
        self._acquiring = False
        self._acq_mode = None       # 'live' | 'seq' | None

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    def open(self) -> bool:
        try:
            add_path('Teledyne_ScopeFoundry')
            from CameraDevice import PVcamDevice
            self.device = PVcamDevice()        # init_pvcam + first camera + defaults
            self._open = True
            try:
                self.sensor_h = int(self.device.get_width())
                self.sensor_v = int(self.device.get_height())
            except Exception:
                pass
            # Push the cached settings (readout -> gain, binning, ROI, exposure, trigger).
            self.set_readout_port(self.readout_port)
            self.set_gain(self.gain)
            self.set_binning(self.binning)
            self.set_roi(self.subarrayh, self.subarrayv,
                         self.subarrayh_pos, self.subarrayv_pos)
            self.set_exposure(self.exposure_ms)
            self.set_trigger_mode(self.trigger_mode)
            print(f"[Camera] Opened: {self.get_model_info()}")
            return True
        except Exception as e:
            print(f"[Camera] open() failed: {e}")
            self.device = None
            self._open = False
            return False

    def close(self):
        if self.device and self._open:
            try:
                self.stop_acquisition()
            except Exception:
                pass
            try:
                self.device.close()          # cam.close() + pvc.uninit_pvcam()
            except Exception as e:
                print(f"[Camera] close error: {e}")
        self._open = False
        self._acquiring = False
        self._acq_mode = None
        self.device = None

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def is_acquiring(self) -> bool:
        return self._acquiring

    def get_model_info(self) -> str:
        if self._open:
            try:
                return str(self.device.get_idname())
            except Exception:
                pass
        return "Teledyne Retiga E7"

    # ------------------------------------------------------------------ #
    #  Exposure (integer ms, PVCAM exp_time)
    # ------------------------------------------------------------------ #

    def set_exposure(self, ms: int):
        """PVcamDevice.set_exposure (milliseconds, integer)."""
        self.exposure_ms = int(round(ms))
        if self._open:
            self.device.set_exposure(self.exposure_ms)
            self.exposure_ms = int(self.device.get_exposure())

    def get_exposure(self) -> int:
        if self._open:
            self.exposure_ms = int(self.device.get_exposure())
        return self.exposure_ms

    def get_exposure_s(self) -> float:
        return self.get_exposure() * 1e-3

    def get_temperature(self) -> float:
        if self._open:
            try:
                return float(self.device.get_temperature())
            except Exception:
                return float('nan')
        return float('nan')

    def get_temperature_setpoint(self) -> float:
        if self._open:
            try:
                return float(self.device.get_temperature_setpoint())
            except Exception:
                return float('nan')
        return float('nan')

    def get_max_frame_rate(self) -> float:
        """Estimated max internal-trigger frame rate for the current ROI/exposure."""
        return max_frame_rate_hz(self.subarrayh, self.subarrayv, self.get_exposure_s())

    # ------------------------------------------------------------------ #
    #  Binning
    # ------------------------------------------------------------------ #

    def set_binning(self, binning: int):
        """PVcamDevice.set_binning (square). Only valid while idle."""
        self.binning = int(binning)
        if self._open:
            self.device.set_binning(self.binning)
            self.binning = int(self.device.get_binning())

    def get_binning(self) -> int:
        if self._open:
            self.binning = int(self.device.get_binning())
        return self.binning

    # ------------------------------------------------------------------ #
    #  ROI (sensor pixels) — PVcamDevice.set_roi(h0, v0, width, height)
    # ------------------------------------------------------------------ #

    def set_roi(self, hsize: int, vsize: int, hpos: int = 0, vpos: int = 0):
        """Set the camera ROI (unbinned sensor px). Must be called while idle."""
        hsize = int(np.clip(hsize, 1, self.sensor_h))
        vsize = int(np.clip(vsize, 1, self.sensor_v))
        hpos = int(np.clip(hpos, 0, self.sensor_h - hsize))
        vpos = int(np.clip(vpos, 0, self.sensor_v - vsize))
        self.subarrayh, self.subarrayv = hsize, vsize
        self.subarrayh_pos, self.subarrayv_pos = hpos, vpos
        if not self._open:
            return
        self.device.set_roi(hpos, vpos, hsize, vsize)
        self._refresh_roi()

    def set_full_frame(self):
        self.set_roi(self.sensor_h, self.sensor_v, 0, 0)

    def get_roi(self) -> dict:
        """Returns {'hsize','vsize','hpos','vpos'} in unbinned sensor pixels."""
        if self._open:
            self._refresh_roi()
        return {'hsize': self.subarrayh, 'vsize': self.subarrayv,
                'hpos': self.subarrayh_pos, 'vpos': self.subarrayv_pos}

    def _refresh_roi(self):
        try:
            self.subarrayh = int(self.device.getSubarrayH())
            self.subarrayv = int(self.device.getSubarrayV())
            self.subarrayh_pos = int(self.device.getSubarrayHpos())
            self.subarrayv_pos = int(self.device.getSubarrayVpos())
        except Exception:
            pass

    @property
    def eff_h(self) -> int:
        """Effective frame width after binning."""
        return int(self.subarrayh // self.binning)

    @property
    def eff_v(self) -> int:
        """Effective frame height after binning."""
        return int(self.subarrayv // self.binning)

    # ------------------------------------------------------------------ #
    #  Readout port / gain (speed table)
    # ------------------------------------------------------------------ #

    def set_readout_port(self, port: int):
        """PVcamDevice.set_readout: sets readout_port, speed=0 and resets gain to 1."""
        self.readout_port = int(port)
        if self._open:
            self.device.set_readout(self.readout_port)
            self.readout_port = int(self.device.get_readout())
            self.gain = int(self.device.get_gain())
        else:
            self.gain = 1

    def get_readout_port(self) -> int:
        if self._open:
            self.readout_port = int(self.device.get_readout())
        return self.readout_port

    def set_gain(self, gain: int):
        """PVcamDevice.set_gain (1 Full Well; 2 Sensitivity, readout 1/2 only)."""
        gain = int(gain)
        if self.readout_port == 0 and gain != 1:
            print("[Camera] gain 2 is not available on readout port 0 (Speed); using 1")
            gain = 1
        self.gain = gain
        if self._open:
            self.device.set_gain(gain)
            self.gain = int(self.device.get_gain())

    def get_gain(self) -> int:
        if self._open:
            self.gain = int(self.device.get_gain())
        return self.gain

    # ------------------------------------------------------------------ #
    #  Trigger (PVCAM exposure mode)
    # ------------------------------------------------------------------ #

    def set_trigger_mode(self, mode: str):
        """mode: one of TRIGGER_MODES (PVcamDevice.set_trigger_mode)."""
        if mode not in TRIGGER_MODES:
            raise ValueError(f"Unknown trigger mode {mode!r}; expected one of {TRIGGER_MODES}")
        self.trigger_mode = mode
        if self._open:
            self.device.set_trigger_mode(mode)

    def get_trigger_mode(self) -> str:
        if self._open:
            try:
                m = self.device.get_trigger_mode()
                if isinstance(m, str):
                    self.trigger_mode = m
            except Exception:
                pass
        return self.trigger_mode

    # ------------------------------------------------------------------ #
    #  Acquisition control (1:1 passthroughs to PVcamDevice)
    # ------------------------------------------------------------------ #

    def start_live(self):
        """PVcamDevice.acq_start -> cam.start_live() (circular buffer)."""
        if self._open:
            self.device.acq_start()
            self._acquiring = True
            self._acq_mode = 'live'

    def start_seq(self, n_frames: int):
        """PVcamDevice.acq_start_seq(n) -> cam.start_seq(num_frames=n)."""
        if self._open:
            self.device.acq_start_seq(int(n_frames))
            self._acquiring = True
            self._acq_mode = 'seq'

    def poll_frame(self, timeout_ms: int | None = None) -> np.ndarray | None:
        """
        Oldest buffered frame as a 2-D uint16 array (rows, cols).

        timeout_ms=None -> PVcamDevice.get_nparray() (waits forever, exactly as
        the ScopeFoundry code). With a timeout the same cam.poll_frame call is
        used; it raises on timeout so the caller can decide what to do.
        """
        if not self._open:
            return None
        if timeout_ms is None:
            return self.device.get_nparray()
        frame, _fps, _count = self.device.cam.poll_frame(timeout_ms=int(timeout_ms))
        return frame['pixel_data']

    def stop_acquisition(self):
        """PVcamDevice.acq_stop -> cam.finish(). Idempotent."""
        if self._open and self._acquiring:
            try:
                self.device.acq_stop()
            except Exception as e:
                print(f"[Camera] acq_stop error: {e}")
            self._acquiring = False
            self._acq_mode = None

    def abort(self):
        """pl_exp_abort: breaks a pending poll_frame (used to interrupt a scan/live)."""
        if self._open and self._acquiring:
            try:
                self.device.cam.abort()
            except Exception as e:
                print(f"[Camera] abort error: {e}")

    # ------------------------------------------------------------------ #
    #  Convenience helpers (used by the live view + camera panel)
    # ------------------------------------------------------------------ #

    def snap(self) -> np.ndarray | None:
        """Single frame: acq_start_seq(1) -> get_nparray -> acq_stop (as measure())."""
        if not self._open:
            return None
        try:
            self.start_seq(1)
            img = self.poll_frame(timeout_ms=self.exposure_ms + 10000)
            self.stop_acquisition()
            return img
        except Exception as e:
            print(f"[Camera] snap() error: {e}")
            self.stop_acquisition()
            return None

    def get_live_frame(self, timeout_ms: int | None = None) -> np.ndarray | None:
        if not self._open:
            return None
        return self.poll_frame(timeout_ms)

    def stop_live(self):
        self.stop_acquisition()

    def settings_dict(self) -> dict:
        """Current camera settings (stamped into the HDF5 file for provenance)."""
        return dict(camera=self.get_model_info(),
                    exposure_ms=self.exposure_ms,
                    binning=self.binning,
                    roi_hpos=self.subarrayh_pos, roi_vpos=self.subarrayv_pos,
                    roi_hsize=self.subarrayh, roi_vsize=self.subarrayv,
                    readout_port=self.readout_port,
                    gain=self.gain,
                    trigger_mode=self.trigger_mode)


# ------------------------------------------------------------------ #
#  Mock camera — for UI development without hardware
# ------------------------------------------------------------------ #

class MockCamera:
    """Animated synthetic Retiga-like frames for offline UI testing.

    Mirrors the RetigaCamera API (ms exposure, binning, ROI, readout/gain,
    trigger mode, live/sequence acquisition) so every panel works identically
    with or without hardware. Frames carry a slow cosine modulation so the
    interferogram plot shows fringes during a mock scan.
    """

    def __init__(self, frame_x=SENSOR_W, frame_y=SENSOR_H, binning=1, exposure_ms=20):
        self.sensor_h = frame_x
        self.sensor_v = frame_y
        self.subarrayh = frame_x
        self.subarrayv = frame_y
        self.subarrayh_pos = 0
        self.subarrayv_pos = 0
        self.binning = int(binning)
        self.exposure_ms = int(exposure_ms)
        self.readout_port = 0
        self.gain = 1
        self.trigger_mode = 'Internal Trigger'
        self._open = False
        self._acquiring = False
        self._acq_mode = None
        self._seq_left = 0
        self._frame_count = 0

    # --- connection ---
    def open(self) -> bool:
        self._open = True
        print("[MockCamera] Opened (simulation — Teledyne Retiga E7)")
        return True

    def close(self):
        self._open = False
        self._acquiring = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def is_acquiring(self) -> bool:
        return self._acquiring

    def get_model_info(self) -> str:
        return "Teledyne Retiga E7 (MOCK)"

    # --- exposure / status ---
    def set_exposure(self, ms: int):
        self.exposure_ms = max(1, int(round(ms)))

    def get_exposure(self) -> int:
        return self.exposure_ms

    def get_exposure_s(self) -> float:
        return self.exposure_ms * 1e-3

    def get_temperature(self) -> float:
        return -20.0

    def get_temperature_setpoint(self) -> float:
        return -20.0

    def get_max_frame_rate(self) -> float:
        return max_frame_rate_hz(self.subarrayh, self.subarrayv, self.get_exposure_s())

    # --- binning ---
    def set_binning(self, binning: int):
        self.binning = int(binning)

    def get_binning(self) -> int:
        return self.binning

    # --- ROI ---
    def set_roi(self, hsize, vsize, hpos=0, vpos=0):
        hsize = int(np.clip(hsize, 1, self.sensor_h))
        vsize = int(np.clip(vsize, 1, self.sensor_v))
        self.subarrayh, self.subarrayv = hsize, vsize
        self.subarrayh_pos = int(np.clip(hpos, 0, self.sensor_h - hsize))
        self.subarrayv_pos = int(np.clip(vpos, 0, self.sensor_v - vsize))

    def set_full_frame(self):
        self.set_roi(self.sensor_h, self.sensor_v, 0, 0)

    def get_roi(self) -> dict:
        return {'hsize': self.subarrayh, 'vsize': self.subarrayv,
                'hpos': self.subarrayh_pos, 'vpos': self.subarrayv_pos}

    @property
    def eff_h(self) -> int:
        return int(self.subarrayh // self.binning)

    @property
    def eff_v(self) -> int:
        return int(self.subarrayv // self.binning)

    # --- readout / gain ---
    def set_readout_port(self, port: int):
        self.readout_port = int(port)
        self.gain = 1

    def get_readout_port(self) -> int:
        return self.readout_port

    def set_gain(self, gain: int):
        self.gain = 1 if self.readout_port == 0 else int(gain)

    def get_gain(self) -> int:
        return self.gain

    # --- trigger ---
    def set_trigger_mode(self, mode: str):
        if mode not in TRIGGER_MODES:
            raise ValueError(f"Unknown trigger mode {mode!r}")
        self.trigger_mode = mode

    def get_trigger_mode(self) -> str:
        return self.trigger_mode

    # --- acquisition passthroughs ---
    def start_live(self):
        self._acquiring = True
        self._acq_mode = 'live'

    def start_seq(self, n_frames: int):
        self._acquiring = True
        self._acq_mode = 'seq'
        self._seq_left = int(n_frames)

    def poll_frame(self, timeout_ms: int | None = None) -> np.ndarray | None:
        if not self._open or not self._acquiring:
            raise RuntimeError("MockCamera: poll_frame called while not acquiring")
        if self._acq_mode == 'seq':
            if self._seq_left <= 0:
                raise RuntimeError("MockCamera: sequence finished (no more frames)")
            self._seq_left -= 1
        # Emulate exposure + readout (capped so the mock stays responsive).
        time.sleep(min(self.exposure_ms, 40) * 1e-3 + 0.005)
        self._frame_count += 1
        return self._make_frame()

    def stop_acquisition(self):
        self._acquiring = False
        self._acq_mode = None

    def abort(self):
        pass

    # --- convenience ---
    def snap(self) -> np.ndarray | None:
        if not self._open:
            return None
        self.start_seq(1)
        img = self.poll_frame()
        self.stop_acquisition()
        return img

    def get_live_frame(self, timeout_ms: int | None = None) -> np.ndarray | None:
        if not self._open:
            return None
        return self.poll_frame(timeout_ms)

    def stop_live(self):
        self.stop_acquisition()

    def settings_dict(self) -> dict:
        return dict(camera=self.get_model_info(),
                    exposure_ms=self.exposure_ms,
                    binning=self.binning,
                    roi_hpos=self.subarrayh_pos, roi_vpos=self.subarrayv_pos,
                    roi_hsize=self.subarrayh, roi_vsize=self.subarrayv,
                    readout_port=self.readout_port,
                    gain=self.gain,
                    trigger_mode=self.trigger_mode)

    def _make_frame(self) -> np.ndarray:
        t = self._frame_count * 0.05
        h, w = self.eff_v, self.eff_h

        # Work on a small grid then upscale for speed.
        scale = max(1, min(h, w) // 64)
        sh, sw = max(1, h // scale), max(1, w // scale)
        frame = np.random.normal(200, 20, (sh, sw)).astype(np.float32)

        # Interferogram-like modulation (cosine fringe vs frame index).
        fringe = 0.5 * (1.0 + np.cos(2 * np.pi * self._frame_count / 12.0))
        y, x = np.ogrid[:sh, :sw]
        for amp, sigma, cx_f, cy_f, fx, fy in [
            (40000, 0.10, 0.5, 0.5, 0.5, 0.7),
            (15000, 0.06, 0.3, 0.35, 0.4, 0.3),
        ]:
            cx = sw * (cx_f + 0.05 * np.cos(t * fx))
            cy = sh * (cy_f + 0.05 * np.sin(t * fy))
            sig = sigma * min(sh, sw)
            frame += amp * (0.3 + 0.7 * fringe) * np.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))

        frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
        frame = frame[:h, :w]
        if frame.shape != (h, w):  # pad if rounding left it short
            out = np.zeros((h, w), dtype=np.float32)
            out[:frame.shape[0], :frame.shape[1]] = frame
            frame = out
        return np.clip(frame, 0, 65535).astype(np.uint16)
