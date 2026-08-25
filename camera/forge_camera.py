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
EXP_MIN_MS = 0.001       # 1 us floor; the node's own minimum wins if larger
EXP_MAX_MS = 1000.0      # 1 s ceiling for dim scenes (frame rate drops to match)
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
        self.status = CameraStatus(backend="forge", message="Forge SWIR idle",
                                   exposure_ms=DEFAULT_EXPOSURE_MS)

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

    def set_exposure(self, exposure_ms: float) -> None:
        ms = float(np.clip(exposure_ms, EXP_MIN_MS, EXP_MAX_MS))
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
            self.status.message = f"Exposure {actual:.3f} ms"
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set_exposure failed: {e}"
            logger.warning(self.status.message)

    def set_average(self, average_count: int) -> None:
        self._average_count = max(1, int(average_count))
        self.status.average_count = self._average_count
        self.status.message = (f"Software averaging {self._average_count} frame(s)"
                               if self._average_count > 1 else "Averaging off")

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
            self.status.message = f"{name} = {value}"
            logger.info(f"[forge] {name} = {value}")
        except Exception as e:  # noqa: BLE001
            self.status.message = f"set {name} failed: {e}"
            logger.warning(self.status.message)

    def refresh_temperatures(self) -> None:
        """Read the camera's internal temperature into status.board_temp_c.

        The Forge SWIR has no separate cooled-FPA readout, so fpa_temp_k stays
        NaN (the UI hides it)."""
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
        # Don't let a throughput cap pace the stream below the sensor rate.
        self._set_enum("DeviceLinkThroughputLimitMode", "Off")

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

        # Geometry + identity for the status bar and the saved metadata.
        try:
            self.status.width = int(PySpin.CIntegerPtr(nm.GetNode("Width")).GetValue())
            self.status.height = int(PySpin.CIntegerPtr(nm.GetNode("Height")).GetValue())
        except Exception:  # noqa: BLE001
            pass
        try:
            tl = self._cam.GetTLDeviceNodeMap()
            self.status.serial_number = (self._read_str(tl, "DeviceSerialNumber")
                                         or self._read_str(tl, "DeviceModelName")
                                         or "Forge")
        except Exception:  # noqa: BLE001
            pass
