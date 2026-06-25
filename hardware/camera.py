"""
Hamamatsu Orca Flash camera wrapper — uses HamamatsuDevice from CameraDevice.py exactly.

Copy CameraDevice.py into the hardware/ folder (or anywhere on sys.path).
"""

import sys
import os
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, '..')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def add_path(path):
    import sys
    import os
    # add path to ospath list, assuming that the path is in a sybling folder
    from os.path import dirname
    sys.path.append(os.path.abspath(os.path.join(dirname(dirname(__file__)),path)))




class HamamatsuCamera:
    """
    Wrapper around HamamatsuDevice that adds a clean open/close/snap API
    compatible with the rest of the app, using your existing device methods:
      - hamamatsu.startAcquisition()
      - hamamatsu.getLastFrame()   -> [HCamData, [x, y]]
      - frame.getData()            -> numpy uint16 array
      - hamamatsu.stopAcquisition()
    """

    def __init__(self,
                 camera_id: int = 0,
                 frame_x: int = 2048,
                 frame_y: int = 2048,
                 exposure: float = 0.01,         # seconds
                 binning: int = 1,
                 trsource: str = 'internal',
                 trmode: str = 'normal',
                 trpolarity: str = 'positive',
                 tractive: str = 'edge',
                 subarrayh_pos: int = 0,
                 subarrayv_pos: int = 0):

        self.camera_id = camera_id
        self.frame_x = frame_x
        self.frame_y = frame_y
        self.exposure = exposure
        self.binning = binning
        self.trsource = trsource
        self.trmode = trmode
        self.trpolarity = trpolarity
        self.tractive = tractive
        self.subarrayh_pos = subarrayh_pos
        self.subarrayv_pos = subarrayv_pos

        self.hamamatsu = None
        self._open = False

        # Effective array dimensions (after binning)
        self.eff_h = frame_x // binning
        self.eff_v = frame_y // binning

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    def open(self) -> bool:
        try:
            add_path('Hamamatsu_ScopeFoundry')
            from CameraDevice import HamamatsuDevice
            self.hamamatsu = HamamatsuDevice(
                camera_id=self.camera_id,
                frame_x=self.frame_x,
                frame_y=self.frame_y,
                acquisition_mode='fixed_length',
                number_frames=1,
                exposure=self.exposure,
                trsource=self.trsource,
                trmode=self.trmode,
                trpolarity=self.trpolarity,
                tractive=self.tractive,
                subarrayh_pos=self.subarrayh_pos,
                subarrayv_pos=self.subarrayv_pos,
                binning=self.binning,
                hardware=None,
            )
            self._open = True
            print(f"[Camera] Opened: {self.hamamatsu.getModelInfo()}")
            return True
        except Exception as e:
            print(f"[Camera] open() failed: {e}")
            self._open = False
            return False

    def close(self):
        if self.hamamatsu and self._open:
            try:
                self.hamamatsu.shutdown()
            except Exception as e:
                print(f"[Camera] shutdown error: {e}")
        self._open = False
        self.hamamatsu = None

    @property
    def is_open(self) -> bool:
        return self._open

    # ------------------------------------------------------------------ #
    #  Settings
    # ------------------------------------------------------------------ #

    def set_exposure(self, seconds: float):
        """Set exposure time in seconds (matching HamamatsuDevice.setExposure)."""
        self.exposure = seconds
        if self._open and self.hamamatsu:
            self.hamamatsu.setExposure(seconds)

    def get_exposure(self) -> float:
        """Returns exposure in seconds."""
        return self.exposure

    def set_trigger_source(self, source: str):
        """source: 'internal' or 'external'"""
        if self._open and self.hamamatsu:
            self.hamamatsu.setTriggerSource(source)
        self.trsource = source

    def set_acquisition_mode(self, mode: str, n_frames: int = 1):
        """mode: 'fixed_length' or 'run_till_abort'"""
        if self._open and self.hamamatsu:
            self.hamamatsu.acquisition_mode = mode
            self.hamamatsu.number_frames = n_frames

    # ------------------------------------------------------------------ #
    #  Acquisition
    # ------------------------------------------------------------------ #

    def snap(self) -> np.ndarray | None:
        """
        Capture one frame using fixed_length mode (internal trigger).
        Returns a 2-D uint16 numpy array shaped (eff_v, eff_h).
        """
        if not self._open:
            return None
        try:
            # Match the pattern from hyperspectral_measure.py internal trigger branch
            self.hamamatsu.acquisition_mode = 'fixed_length'
            self.hamamatsu.number_frames = 1
            self.hamamatsu.startAcquisition()
            [frame, dims] = self.hamamatsu.getLastFrame()
            np_data = frame.getData()
            self.hamamatsu.stopAcquisition()
            # dims = [frame_x, frame_y] — reshape to (v, h) as in your measure()
            img = np.reshape(np_data, (dims[1], dims[0]))
            return img
        except Exception as e:
            print(f"[Camera] snap() error: {e}")
            try:
                self.hamamatsu.stopAcquisition()
            except Exception:
                pass
            return None

    def start_live(self):
        """Start run_till_abort acquisition for live preview."""
        if not self._open:
            return
        self.hamamatsu.acquisition_mode = 'run_till_abort'
        self.hamamatsu.number_frames = 100   # large buffer
        self.hamamatsu.startAcquisition()

    def get_live_frame(self) -> np.ndarray | None:
        """Get the most recent frame during live acquisition."""
        if not self._open:
            return None
        try:
            [frame, dims] = self.hamamatsu.getLastFrame()
            np_data = frame.getData()
            return np.reshape(np_data, (dims[1], dims[0]))
        except Exception as e:
            print(f"[Camera] get_live_frame() error: {e}")
            return None

    def stop_live(self):
        """Stop run_till_abort acquisition."""
        if self._open and self.hamamatsu:
            try:
                self.hamamatsu.stopAcquisition()
            except Exception as e:
                print(f"[Camera] stopAcquisition error: {e}")


# ------------------------------------------------------------------ #
#  Mock camera — for UI development without hardware
# ------------------------------------------------------------------ #

class MockCamera:
    """Animated synthetic Hamamatsu-like frames for offline UI testing."""

    def __init__(self, frame_x=2048, frame_y=2048, binning=1):
        self.frame_x = frame_x
        self.frame_y = frame_y
        self.binning = binning
        self.eff_h = frame_x // binning
        self.eff_v = frame_y // binning
        self._exposure = 0.01
        self._open = False
        self._frame_count = 0

    def open(self) -> bool:
        self._open = True
        print("[MockCamera] Opened (simulation mode — Hamamatsu Orca Flash 4V3)")
        return True

    def close(self):
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def set_exposure(self, seconds: float):
        self._exposure = seconds

    def get_exposure(self) -> float:
        return self._exposure

    def set_trigger_source(self, source: str):
        pass

    def set_acquisition_mode(self, mode: str, n_frames: int = 1):
        pass

    def snap(self) -> np.ndarray | None:
        if not self._open:
            return None
        self._frame_count += 1
        return self._make_frame()

    def start_live(self):
        pass

    def get_live_frame(self) -> np.ndarray | None:
        if not self._open:
            return None
        self._frame_count += 1
        return self._make_frame()

    def stop_live(self):
        pass

    def _make_frame(self) -> np.ndarray:
        t = self._frame_count * 0.05
        h, w = self.eff_v, self.eff_h

        # Use a smaller working grid then upscale for speed
        scale = 8
        sh, sw = h // scale, w // scale
        frame = np.random.normal(200, 20, (sh, sw)).astype(np.float32)

        y, x = np.ogrid[:sh, :sw]
        for amp, sigma, cx_f, cy_f, fx, fy in [
            (50000, 0.10, 0.5, 0.5, 0.5, 0.7),
            (20000, 0.06, 0.3, 0.35, 0.4, 0.3),
        ]:
            cx = sw * (cx_f + 0.1 * np.cos(t * fx))
            cy = sh * (cy_f + 0.1 * np.sin(t * fy))
            sig = sigma * min(sh, sw)
            frame += amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sig**2))

        # Upscale back to full size with nearest-neighbour
        frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
        frame = frame[:h, :w]
        return np.clip(frame, 0, 65535).astype(np.uint16)
