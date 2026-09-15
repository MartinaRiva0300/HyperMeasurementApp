from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True)
class CameraStatus:
    connected: bool = False
    acquiring: bool = False
    backend: str = "unknown"
    message: str = ""
    width: int = 0
    height: int = 0
    serial_number: str = ""
    requested_mode: str = ""
    startup_profile: str = ""
    selected_device_index: int = -1
    host_visible: bool = False
    job_file_path: str = ""
    average_count: int = 1
    exposure_ms: float = 10.0
    frame_counter: int = 0
    raw_peak_count: float = 0.0
    board_temp_c: float = float("nan")   # FPGA/electronics board temperature (°C)
    exposure_min_ms: float = 0.001       # sensor's shortest exposure (from hardware)
    exposure_max_ms: float = 1000.0      # sensor's longest exposure (from hardware)
    binning: int = 1                     # on-sensor (hardware) NxN binning factor
    binning_options: tuple = (1,)        # the NxN factors the sensor supports
    gain_db: float = float("nan")        # current sensor gain (dB); NaN = unknown
    offset_x: int = 0                    # ROI OffsetX on the sensor (camera units)
    offset_y: int = 0                    # ROI OffsetY on the sensor (camera units)
    pixel_format: str = ""               # e.g. "Mono16"
    adc_bit_depth: str = ""              # e.g. "Bit12"
    frame_rate_hz: float = float("nan")  # AcquisitionFrameRate
    reverse_x: bool = False              # camera horizontal mirror
    reverse_y: bool = False              # camera vertical mirror
    exposure_auto: str = ""              # ExposureAuto (Off/Once/Continuous)
    gain_auto: str = ""                  # GainAuto (Off/Once/Continuous)


def copy_camera_status(status: CameraStatus) -> CameraStatus:
    return CameraStatus(**asdict(status))


class CameraInterface(ABC):
    @abstractmethod
    def connect(self) -> CameraStatus:
        """Initialize the backend and return the current camera status."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release all backend resources."""

    @abstractmethod
    def start_acquisition(self) -> None:
        """Start streaming frames."""

    @abstractmethod
    def stop_acquisition(self) -> None:
        """Stop streaming frames."""

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """Capture and return a single frame, or None if unavailable."""

    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None:
        """Update the exposure in milliseconds."""

    def set_average(self, average_count: int) -> None:
        """Update the averaging count if supported."""

    def set_option(self, name: str, value) -> None:
        """Set a named camera option (e.g. GenICam node). No-op if unsupported."""

    def set_binning(self, binning: int) -> None:
        """Set the on-sensor (hardware) NxN binning factor. No-op if unsupported."""

    def set_roi(self, row0: int, row1: int, col0: int, col1: int) -> None:
        """Crop readout to the given ROI (Width/Height/OffsetX/OffsetY). Rows/cols
        are in DISPLAYED-frame coordinates (top-left origin). No-op if unsupported."""

    def reset_roi(self) -> None:
        """Restore full-frame readout (offset 0, maximum Width/Height)."""

    @abstractmethod
    def get_status(self) -> CameraStatus:
        """Return the latest backend status."""
