"""Teledyne FLIR **Forge 1GigE SWIR** (1.3 MP, C-mount) backend, via Spinnaker/PySpin.

The Forge SWIR is a standard GigE Vision camera built around Sony's IMX990
SenSWIR InGaAs sensor (1280x1024, 5 um pitch, ~400-1700 nm). It is driven here
through Teledyne's Spinnaker SDK Python bindings (`import PySpin`), which is the
only supported API for this camera -- unlike the Goldeye/IRC806 there is no
Pleora eBUS path.

Install:
    pip install spinnaker_python-4.4.0.246-cp312-cp312-win_amd64.whl
The wheel bundles the Spinnaker runtime DLLs, so `import PySpin` works on its
own. For GigE streaming you still want the matching **Spinnaker SDK installer**
(drivers + VS redistributables) so the NIC filter driver and jumbo frames are
available -- without it enumeration can come up empty or the stream drops frames.

Frames are pulled as **Mono16** so the rest of the app keeps its uint16 pipeline.
The sensor ADC is 12-bit, so counts occupy the low 12 bits (0..4095) unless the
camera is configured otherwise -- the actual pixel format and ADC depth are
reported in the status message and recorded per-connect.

Device selection: picks the camera whose model/vendor name looks like a Forge
(see MODEL_HINTS) so it never grabs another Spinnaker device on the same host;
falls back to the first camera found.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from .camera_interface import CameraInterface, CameraStatus, copy_camera_status

DEFAULT_EXPOSURE_MS = 1.0
# Fallback limits, used only if the camera doesn't expose the AutoExposure limit
# nodes. The real range is read from the hardware on connect (see
# _read_exposure_limits): the IMX990 datasheet is 15 us .. 30 s.
EXP_MIN_MS = 0.001       # 1 us floor; the hardware limit replaces this on connect
EXP_MAX_MS = 1000.0      # 1 s ceiling;  the hardware limit replaces this on connect
FETCH_TIMEOUT_MS = 1000  # GetNextImage timeout

# Model/vendor substrings that identify the Forge. "FGE" is the Forge model
# prefix (e.g. FGE-13S6M-C); DeviceModelName varies with firmware.
MODEL_HINTS = ("forge", "fge")

# Preferred pixel formats, best first. Mono16 keeps the app's uint16 pipeline.
PIXEL_FORMAT_PREFS = ("Mono16", "Mono12p", "Mono12Packed", "Mono8")


class ForgeSwirCamera(CameraInterface):
    """Forge 1GigE SWIR streamed over Spinnaker/PySpin."""

    def __init__(self, fetch_timeout_ms: int = FETCH_TIMEOUT_MS) -> None:
        self.fetch_timeout_ms = int(fetch_timeout_ms)
        self._spin = None          # the PySpin module
        self._system = None        # PySpin.System instance
        self._cam_list = None
        self._cam = None           # PySpin.CameraPtr
        self._nodemap = None       # GenICam nodemap of the device
        self._desired_exposure_ms = DEFAULT_EXPOSURE_MS
        self._average_count = 1
        self._binning = 1          # on-sensor (hardware) NxN binning; re-applied on connect
        self._binning_options = (1,)   # supported factors, read from the node on connect
        # True exposure range, read from the hardware on connect; defaults until then.
        self._exp_min_ms = EXP_MIN_MS
        self._exp_max_ms = EXP_MAX_MS
        self.status = CameraStatus(backend="forge", message="Forge SWIR idle",
                                   exposure_ms=DEFAULT_EXPOSURE_MS,
                                   exposure_min_ms=EXP_MIN_MS,
                                   exposure_max_ms=EXP_MAX_MS,
                                   binning=1, binning_options=(1,))

    # -- PySpin bootstrap ----------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._spin is not None:
            return
        import PySpin  # imported lazily so the app still runs without the wheel
        self._spin = PySpin

    # -- CameraInterface -----------------------------------------------------
    def connect(self) -> CameraStatus:
        try:
            self._ensure_loaded()
        except Exception as e:  # noqa: BLE001
            self.status.connected = False
            self.status.message = (f"PySpin import failed: {e}. Install the "
                                   "spinnaker_python wheel into this interpreter.")
            logger.error(self.status.message)
            return self.get_status()

        PySpin = self._spin
        try:
            self._system = PySpin.System.GetInstance()
            self._cam_list = self._system.GetCameras()
        except Exception as e:  # noqa: BLE001
            self.status.connected = False
            self.status.message = f"Spinnaker init failed: {e}"
            logger.error(self.status.message)
            return self.get_status()

        n = self._cam_list.GetSize()
        if n == 0:
            self._release_system()
            self.status.connected = False
            self.status.message = ("No Spinnaker camera found (powered? correct NIC / "
                                   "subnet? SpinView or another app holding it?)")
            logger.warning(self.status.message)
            return self.get_status()

        index = self._pick_device(n)
        try:
            self._cam = self._cam_list.GetByIndex(index)
            self._cam.Init()
            self._nodemap = self._cam.GetNodeMap()
        except Exception as e:  # noqa: BLE001
            self._cam = None
            self._release_system()
            self.status.connected = False
            self.status.message = f"Camera Init failed: {e}"
            logger.error(self.status.message)
            return self.get_status()

        self.status.connected = True
        self.status.selected_device_index = index
        self._configure()
        # Read the sensor's true exposure range from the hardware BEFORE applying
        # the exposure, so the clamp uses real limits (15 us .. 30 s), not guesses.
        self._read_exposure_limits()
        # Re-apply the remembered exposure (a fresh process starts at DEFAULT).
        self.set_exposure(self._desired_exposure_ms)
        self.refresh_temperatures()
        self.status.message = (
            f"Forge SWIR connected ({self.status.width}x{self.status.height}, "
            f"{self.status.startup_profile})")
        logger.info(self.status.message)
        return self.get_status()

    def disconnect(self) -> None:
        try:
            self.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._cam is not None:
                self._cam.DeInit()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"DeInit: {e}")
        # The camera pointer MUST be dropped before ReleaseInstance(), otherwise
        # Spinnaker raises "System reference count is not zero".
        self._cam = None
        self._nodemap = None
        self._release_system()
        self.status.connected = False
        self.status.acquiring = False
        self.status.message = "Forge SWIR disconnected"

    def _release_system(self) -> None:
        try:
            if self._cam_list is not None:
                self._cam_list.Clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._system is not None:
                self._system.ReleaseInstance()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ReleaseInstance: {e}")
        self._cam_list = None
        self._system = None

    def start_acquisition(self) -> None:
        if self._cam is None or self.status.acquiring:
            return
        try:
            self._cam.BeginAcquisition()
            self.status.acquiring = True
            self.status.message = "Forge SWIR streaming"
            logger.info(self.status.message)
        except Exception as e:  # noqa: BLE001
            self.status.acquiring = False
            self.status.message = f"start_acquisition failed: {e}"
            logger.error(self.status.message)

    def stop_acquisition(self) -> None:
        if self._cam is None or not self.status.acquiring:
            return
        try:
            self._cam.EndAcquisition()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"stop_acquisition: {e}")
        self.status.acquiring = False
        self.status.message = "Forge SWIR acquisition stopped"

    def get_frame(self) -> np.ndarray | None:
        """One frame, or the mean of `average_count` frames when averaging is on.

        Spinnaker has no on-camera frame averaging, so N frames are pulled and
        averaged in software -- the effective rate drops by N.
        """
        if self._cam is None or not self.status.acquiring:
            return None
        n = max(1, self._average_count)
        if n == 1:
            frame = self._grab_one()
        else:
            acc = None
            got = 0
            for _ in range(n):
                f = self._grab_one()
                if f is None:
                    continue
                acc = f.astype(np.float32) if acc is None else acc + f
                got += 1
            frame = None if not got else acc / got
        if frame is None:
            return None
        if frame.dtype != np.uint16:
            frame = np.rint(frame).astype(np.uint16)
        self.status.frame_counter += 1
        self.status.raw_peak_count = float(frame.max())
        return frame

    def _grab_one(self) -> np.ndarray | None:
        """Retrieve a single complete image as a uint16 array (copied)."""
        image = None
        try:
            image = self._cam.GetNextImage(self.fetch_timeout_ms)
            if image.IsIncomplete():
                return None
            # GetNDArray() is a VIEW into Spinnaker's buffer -- copy before
            # Release() or the data is recycled underneath us.
            arr = np.array(image.GetNDArray(), copy=True)
            if arr.ndim != 2:
                return None
            return arr if arr.dtype == np.uint16 else arr.astype(np.uint16)
        except Exception:  # noqa: BLE001  (never let a bad frame crash the worker)
            return None
        finally:
            if image is not None:
                try:
                    image.Release()
                except Exception:  # noqa: BLE001
                    pass

    def _read_exposure_limits(self) -> None:
        """Read the sensor's true exposure range (µs) from the AutoExposure limit
        nodes and record it in ms. These report the hardware capability (IMX990:
        15 µs .. 30 s) independent of the current frame rate -- unlike
        ExposureTime.GetMax(), which shrinks as the frame rate rises. Falls back to
        the module defaults if the nodes aren't present."""
        PySpin = self._spin
        lo_ms, hi_ms = EXP_MIN_MS, EXP_MAX_MS
        if self._cam is not None and PySpin is not None:
            try:
                lo = PySpin.CFloatPtr(
                    self._nodemap.GetNode("AutoExposureExposureTimeLowerLimit"))
                if PySpin.IsAvailable(lo) and PySpin.IsReadable(lo):
                    lo_ms = float(lo.GetMin()) / 1000.0
            except Exception as e:  # noqa: BLE001
                logger.debug(f"AutoExposureExposureTimeLowerLimit unavailable: {e}")
            try:
                hi = PySpin.CFloatPtr(
                    self._nodemap.GetNode("AutoExposureExposureTimeUpperLimit"))
                if PySpin.IsAvailable(hi) and PySpin.IsReadable(hi):
                    hi_ms = float(hi.GetMax()) / 1000.0
            except Exception as e:  # noqa: BLE001
                logger.debug(f"AutoExposureExposureTimeUpperLimit unavailable: {e}")
        self._exp_min_ms, self._exp_max_ms = lo_ms, hi_ms
        self.status.exposure_min_ms = lo_ms
        self.status.exposure_max_ms = hi_ms
        logger.info(f"Exposure range from hardware: {lo_ms:.4f} .. {hi_ms:.1f} ms")

    def set_exposure(self, exposure_ms: float) -> None:
        ms = float(np.clip(exposure_ms, self._exp_min_ms, self._exp_max_ms))
        self._desired_exposure_ms = ms
        self.status.exposure_ms = ms
        if self._cam is None:
            return
        PySpin = self._spin
        try:
            # ExposureTime is read-only unless exposure is in manual mode.
            self._set_enum("ExposureAuto", "Off")
            node = PySpin.CFloatPtr(self._nodemap.GetNode("ExposureTime"))
            if not PySpin.IsWritable(node):
                self.status.message = "ExposureTime not writable"
                return
            # Standard GenICam ExposureTime is in MICROSECONDS. Clamp to the
            # camera's own limits so an out-of-range value isn't silently ignored.
            us = float(np.clip(ms * 1000.0, node.GetMin(), node.GetMax()))
            node.SetValue(us)
            actual = node.GetValue() / 1000.0
            self.status.exposure_ms = actual
            self._refresh_frame_rate()   # exposure changes the resulting rate
            self.status.message = f"Exposure {actual:.3f} ms"
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set_exposure failed: {e}"
            logger.warning(self.status.message)

    def set_average(self, average_count: int) -> None:
        self._average_count = max(1, int(average_count))
        self.status.average_count = self._average_count
        self.status.message = (f"Software averaging {self._average_count} frame(s)"
                               if self._average_count > 1 else "Averaging off")

    def _apply_binning_nodes(self, n: int) -> int:
        """Select binning and set BinningHorizontal == BinningVertical == n (kept
        equal). Each axis is clamped to its own node limits. Returns the value
        actually applied. Assumes the stream is stopped (binning changes the frame
        geometry)."""
        PySpin = self._spin
        n = max(1, int(n))
        # "All" is the only binning engine this camera exposes (vs Sensor/ISP).
        self._set_enum("BinningSelector", "All")
        applied = n
        for axis in ("BinningVertical", "BinningHorizontal"):
            try:
                node = PySpin.CIntegerPtr(self._nodemap.GetNode(axis))
                if PySpin.IsWritable(node):
                    node.SetValue(int(np.clip(n, node.GetMin(), node.GetMax())))
                    applied = int(node.GetValue())
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[forge] set {axis} failed: {e}")
        return applied

    def _read_binning_options(self) -> tuple:
        """Enumerate the supported binning factors from the BinningHorizontal node
        (min..max stepped by its increment), with BinningSelector already set.
        Falls back to (1,) if the node isn't present."""
        PySpin = self._spin
        opts = (1,)
        try:
            node = PySpin.CIntegerPtr(self._nodemap.GetNode("BinningHorizontal"))
            if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                lo, hi = int(node.GetMin()), int(node.GetMax())
                try:
                    inc = int(node.GetInc())
                except Exception:  # noqa: BLE001
                    inc = 1
                inc = inc if inc > 0 else 1
                opts = tuple(range(lo, hi + 1, inc)) or (max(1, lo),)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"BinningHorizontal options unavailable: {e}")
        self._binning_options = opts
        self.status.binning_options = opts
        return opts

    def set_binning(self, binning: int) -> None:
        """Set the NxN binning factor (H and V kept equal), using the camera's
        "All" binning engine. Binning changes the frame size, so it is applied
        with the stream stopped and Width/Height are re-read afterwards; streaming
        resumes if it was on."""
        if self._cam is None:
            self._binning = max(1, int(binning))
            return
        PySpin = self._spin
        was_streaming = self.status.acquiring
        try:
            if was_streaming:
                self.stop_acquisition()
            applied = self._apply_binning_nodes(binning)
            self._binning = applied
            self.status.binning = applied
            try:
                nm = self._nodemap
                self.status.width = int(PySpin.CIntegerPtr(nm.GetNode("Width")).GetValue())
                self.status.height = int(PySpin.CIntegerPtr(nm.GetNode("Height")).GetValue())
            except Exception:  # noqa: BLE001
                pass
            self.status.message = (f"Binning {applied}x{applied} "
                                   f"-> {self.status.width}x{self.status.height}")
            logger.info(f"[forge] {self.status.message}")
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set_binning failed: {e}"
            logger.warning(self.status.message)
        finally:
            if was_streaming:
                self.start_acquisition()
            self._refresh_frame_rate()   # binning changes the resulting rate

    def _set_int_node(self, name: str, value) -> int | None:
        """Clamp `value` to an integer node's [min, max], snap DOWN to its
        increment, apply it, and return the value actually set (None if the node
        isn't writable)."""
        PySpin = self._spin
        try:
            node = PySpin.CIntegerPtr(self._nodemap.GetNode(name))
            if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                return None
            lo, hi = int(node.GetMin()), int(node.GetMax())
            try:
                inc = int(node.GetInc())
            except Exception:  # noqa: BLE001
                inc = 1
            inc = inc if inc > 0 else 1
            v = int(np.clip(int(value), lo, hi))
            v -= (v - lo) % inc                 # align to lo + k*inc
            node.SetValue(int(np.clip(v, lo, hi)))
            return int(node.GetValue())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[forge] set {name} failed: {e}")
            return None

    def _refresh_geometry(self) -> None:
        PySpin = self._spin
        for attr, name in (("width", "Width"), ("height", "Height"),
                           ("offset_x", "OffsetX"), ("offset_y", "OffsetY")):
            try:
                node = PySpin.CIntegerPtr(self._nodemap.GetNode(name))
                if PySpin.IsReadable(node):
                    setattr(self.status, attr, int(node.GetValue()))
            except Exception:  # noqa: BLE001
                pass

    def _refresh_gain(self) -> None:
        PySpin = self._spin
        try:
            node = PySpin.CFloatPtr(self._nodemap.GetNode("Gain"))
            if PySpin.IsReadable(node):
                self.status.gain_db = float(node.GetValue())
        except Exception:  # noqa: BLE001
            pass

    def _refresh_frame_rate(self) -> None:
        """Read the RESULTING (achievable) frame rate at the current exposure/ROI/
        binning/bandwidth into status.frame_rate_hz. Falls back to the settable
        AcquisitionFrameRate node if the resulting one isn't exposed."""
        if self._cam is None:
            return
        PySpin = self._spin
        for name in ("AcquisitionResultingFrameRate", "AcquisitionFrameRate"):
            try:
                node = PySpin.CFloatPtr(self._nodemap.GetNode(name))
                if PySpin.IsReadable(node):
                    self.status.frame_rate_hz = float(node.GetValue())
                    return
            except Exception:  # noqa: BLE001
                continue

    def _refresh_camera_settings(self) -> None:
        """Read the current values of every exposed camera setting into status, so
        the saved metadata reflects what the camera is ACTUALLY doing (not just
        what the UI last sent)."""
        PySpin = self._spin
        nm = self._nodemap

        def enum_sym(name):
            try:
                node = PySpin.CEnumerationPtr(nm.GetNode(name))
                if PySpin.IsReadable(node):
                    return str(node.GetCurrentEntry().GetSymbolic())
            except Exception:  # noqa: BLE001
                pass
            return ""

        def read_bool(name):
            try:
                node = PySpin.CBooleanPtr(nm.GetNode(name))
                if PySpin.IsReadable(node):
                    return bool(node.GetValue())
            except Exception:  # noqa: BLE001
                pass
            return False

        self.status.pixel_format = enum_sym("PixelFormat")
        self.status.adc_bit_depth = enum_sym("AdcBitDepth")
        self._refresh_frame_rate()   # RESULTING (achievable) rate, read-only
        self.status.reverse_x = read_bool("ReverseX")
        self.status.reverse_y = read_bool("ReverseY")
        self.status.exposure_auto = enum_sym("ExposureAuto")
        self.status.gain_auto = enum_sym("GainAuto")
        self._refresh_gain()

    def reset_roi(self) -> None:
        """Full-frame readout: offsets 0, Width/Height at maximum."""
        if self._cam is None:
            return
        PySpin = self._spin
        was = self.status.acquiring
        try:
            if was:
                self.stop_acquisition()
            self._set_int_node("OffsetX", 0)     # offsets first so size can grow
            self._set_int_node("OffsetY", 0)
            self._set_int_node("Width", int(PySpin.CIntegerPtr(
                self._nodemap.GetNode("Width")).GetMax()))
            self._set_int_node("Height", int(PySpin.CIntegerPtr(
                self._nodemap.GetNode("Height")).GetMax()))
            self._refresh_geometry()
            self.status.message = f"ROI full frame {self.status.width}x{self.status.height}"
            logger.info(f"[forge] {self.status.message}")
        finally:
            if was:
                self.start_acquisition()

    def set_roi(self, row0: int, row1: int, col0: int, col1: int) -> None:
        """Crop readout to the drawn ROI via Width/Height/OffsetX/OffsetY, so the
        sensor reads out and transmits only that region. Frames are used as read
        from the sensor (no software flip), so the ROI maps straight through:
        OffsetX=col0, OffsetY=row0. Applied with the stream stopped (these nodes
        are locked while streaming)."""
        if self._cam is None:
            return
        PySpin = self._spin
        was = self.status.acquiring
        try:
            if was:
                self.stop_acquisition()
            # Reset to full first so the incoming coordinates are absolute.
            self._set_int_node("OffsetX", 0)
            self._set_int_node("OffsetY", 0)
            wmax = int(PySpin.CIntegerPtr(self._nodemap.GetNode("Width")).GetMax())
            hmax = int(PySpin.CIntegerPtr(self._nodemap.GetNode("Height")).GetMax())
            self._set_int_node("Width", wmax)
            self._set_int_node("Height", hmax)

            r0, r1 = max(0, int(row0)), min(hmax, int(row1))
            c0, c1 = max(0, int(col0)), min(wmax, int(col1))
            if r1 <= r0 or c1 <= c0:
                logger.warning("[forge] set_roi: empty ROI ignored")
                self._refresh_geometry()
                return
            # No software flip: the ROI maps straight to the sensor.
            self._set_int_node("Width", c1 - c0)     # size first (offsets are 0)
            self._set_int_node("Height", r1 - r0)
            self._set_int_node("OffsetX", c0)
            self._set_int_node("OffsetY", r0)
            self._refresh_geometry()
            self.status.message = (f"ROI {self.status.width}x{self.status.height} "
                                   f"@ ({c0},{r0})")
            logger.info(f"[forge] {self.status.message}")
        finally:
            if was:
                self.start_acquisition()
            self._refresh_frame_rate()   # ROI changes the resulting rate

    def set_option(self, name: str, value) -> None:
        """Set a GenICam node by name (ExposureAuto, GainAuto, PixelFormat,
        AcquisitionFrameRate, Gain, ...). The node's own type decides how the
        value is interpreted, so enums take their symbolic string."""
        if self._cam is None:
            return
        PySpin = self._spin
        try:
            node = self._nodemap.GetNode(str(name))
            if node is None:
                self.status.message = f"{name}: no such node"
                return
            base = PySpin.CNodePtr(node)
            if not PySpin.IsWritable(base):
                self.status.message = f"{name} is not writable"
                return
            itype = base.GetPrincipalInterfaceType()
            if itype == PySpin.intfIEnumeration:
                self._set_enum(name, str(value))
            elif itype == PySpin.intfIFloat:
                f = PySpin.CFloatPtr(node)
                f.SetValue(float(np.clip(float(value), f.GetMin(), f.GetMax())))
            elif itype == PySpin.intfIInteger:
                i = PySpin.CIntegerPtr(node)
                i.SetValue(int(np.clip(int(value), i.GetMin(), i.GetMax())))
            elif itype == PySpin.intfIBoolean:
                PySpin.CBooleanPtr(node).SetValue(bool(value))
            elif itype == PySpin.intfICommand:
                PySpin.CCommandPtr(node).Execute()
            else:
                PySpin.CStringPtr(node).SetValue(str(value))
            self._refresh_camera_settings()   # keep status in step for the metadata
            self.status.message = f"{name} = {value}"
            logger.info(f"[forge] {name} = {value}")
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set {name} failed: {e}"
            logger.warning(self.status.message)

    def refresh_temperatures(self) -> None:
        """Read the camera's internal temperature into status.board_temp_c."""
        if self._cam is None:
            return
        PySpin = self._spin
        for node_name in ("DeviceTemperature", "TemperatureAbs"):
            try:
                node = PySpin.CFloatPtr(self._nodemap.GetNode(node_name))
                if PySpin.IsReadable(node):
                    self.status.board_temp_c = float(node.GetValue())
                    return
            except Exception:  # noqa: BLE001
                continue

    def reconnect(self) -> None:
        """Full re-open after a GigE link drop (the worker calls this)."""
        was_streaming = self.status.acquiring
        self.disconnect()
        self.connect()
        if self.status.connected and was_streaming:
            self.start_acquisition()

    def get_status(self) -> CameraStatus:
        return copy_camera_status(self.status)

    # -- helpers -------------------------------------------------------------
    def _pick_device(self, count: int) -> int:
        """Index of the first camera that looks like a Forge, else 0."""
        for i in range(count):
            try:
                cam = self._cam_list.GetByIndex(i)
                tl = cam.GetTLDeviceNodeMap()
                text = " ".join(
                    self._read_str(tl, n) or ""
                    for n in ("DeviceModelName", "DeviceVendorName", "DeviceDisplayName")
                ).lower()
                del cam
                if any(h in text for h in MODEL_HINTS):
                    return i
            except Exception:  # noqa: BLE001
                continue
        return 0

    def _read_str(self, nodemap, name):
        PySpin = self._spin
        try:
            node = PySpin.CStringPtr(nodemap.GetNode(name))
            return node.GetValue() if PySpin.IsReadable(node) else None
        except Exception:  # noqa: BLE001
            return None

    def _set_enum(self, name: str, entry: str) -> bool:
        """Select `entry` on enumeration node `name`. Returns True if applied."""
        PySpin = self._spin
        try:
            node = PySpin.CEnumerationPtr(self._nodemap.GetNode(name))
            if not PySpin.IsWritable(node):
                return False
            item = node.GetEntryByName(entry)
            if item is None or not PySpin.IsReadable(item):
                return False
            node.SetIntValue(item.GetValue())
            return True
        except Exception:  # noqa: BLE001
            return False

    def _configure(self) -> None:
        """Put the camera into free-running continuous mode with manual exposure
        and gain, and pick the widest available mono pixel format.

        Every step is best-effort: node availability varies with firmware, and a
        missing node must not stop the camera from streaming."""
        PySpin = self._spin
        nm = self._nodemap

        # Free-run: no trigger, continuous acquisition.
        self._set_enum("TriggerMode", "Off")
        self._set_enum("AcquisitionMode", "Continuous")

        # Manual exposure + gain, so the TWINS interferogram is radiometrically
        # comparable across wedge positions (auto anything would fight the scan).
        self._set_enum("ExposureAuto", "Off")
        self._set_enum("ExposureMode", "Timed")
        self._set_enum("GainAuto", "Off")

        # No manual frame-rate cap: the camera free-runs at the RESULTING rate
        # allowed by exposure/ROI/binning/bandwidth (the UI shows that value,
        # read-only). Disabling the cap keeps the reported rate == delivered rate.
        try:
            node = PySpin.CBooleanPtr(nm.GetNode("AcquisitionFrameRateEnable"))
            if PySpin.IsWritable(node):
                node.SetValue(False)
        except Exception:  # noqa: BLE001
            pass

        # Pixel format: prefer Mono16 so the app's uint16 path is exact.
        chosen = ""
        for fmt in PIXEL_FORMAT_PREFS:
            if self._set_enum("PixelFormat", fmt):
                chosen = fmt
                break
        adc = ""
        try:
            node = PySpin.CEnumerationPtr(nm.GetNode("AdcBitDepth"))
            if PySpin.IsReadable(node):
                adc = node.GetCurrentEntry().GetSymbolic()
        except Exception:  # noqa: BLE001
            pass
        profile = chosen or "default"
        self.status.startup_profile = f"{profile}/{adc}" if adc else profile

        # GigE transport: jumbo frames (9000 B) cut per-packet overhead and
        # resends. Falls back silently to whatever the NIC negotiated if the
        # adapter has jumbo frames disabled.
        try:
            node = PySpin.CIntegerPtr(nm.GetNode("GevSCPSPacketSize"))
            if PySpin.IsWritable(node):
                node.SetValue(int(min(9000, node.GetMax())))
        except Exception:  # noqa: BLE001
            pass
        # Keep the GigE throughput limit ON so the camera paces itself to what the
        # 1 GigE link can actually carry -- this prevents dropped/incomplete frames
        # AND makes AcquisitionResultingFrameRate report the REAL working rate
        # (with the limit off it reports the sensor-timing max, e.g. >700 Hz, which
        # the link can't deliver). With AcquisitionFrameRateEnable=False the camera
        # free-runs at exactly this resulting rate.
        self._set_enum("DeviceLinkThroughputLimitMode", "On")

        # Binning: drive the "All" engine and re-apply the remembered factor
        # (H == V). Done before the geometry read below so Width/Height reflect
        # the binned frame.
        self.status.binning = self._apply_binning_nodes(self._binning)
        self._binning = self.status.binning
        # The factors the sensor supports (min..max), for the UI drop-down.
        self._read_binning_options()

        # Always hand back the most recent frame: the live view and the TWINS
        # scan both want "now", never a backlog queued behind a slow consumer.
        try:
            s_nm = self._cam.GetTLStreamNodeMap()
            node = PySpin.CEnumerationPtr(s_nm.GetNode("StreamBufferHandlingMode"))
            if PySpin.IsWritable(node):
                item = node.GetEntryByName("NewestOnly")
                if item is not None:
                    node.SetIntValue(item.GetValue())
        except Exception:  # noqa: BLE001
            pass

        # Geometry (incl. ROI offsets) + every exposed setting, for the status bar
        # and the saved metadata.
        self._refresh_geometry()
        self._refresh_camera_settings()
        try:
            tl = self._cam.GetTLDeviceNodeMap()
            self.status.serial_number = (self._read_str(tl, "DeviceSerialNumber")
                                         or self._read_str(tl, "DeviceModelName")
                                         or "Forge")
        except Exception:  # noqa: BLE001
            pass
