"""
Acquisition workers — all hardware I/O in background QThreads.

Uses:
  - camera.start_live() / camera.get_live_frame() / camera.stop_live()
  - camera.snap()
  - stage.get_position(), stage.move_absolute(), stage.move_relative()

All of these map to the real HamamatsuDevice and PI_VC_Device methods
via the wrapper classes in hardware/.
"""

import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


# ------------------------------------------------------------------ #
#  Live-preview worker — uses run_till_abort + getLastFrame
# ------------------------------------------------------------------ #

class LiveViewWorker(QThread):
    """
    Calls camera.start_live(), then repeatedly calls camera.get_live_frame()
    — mirroring the run() loop in hyperspectral_measure.py.
    """

    new_frame = pyqtSignal(np.ndarray)
    error     = pyqtSignal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._running = False

    def run(self):
        self._running = True
        self.camera.start_live()
        while self._running:
            frame = self.camera.get_live_frame()
            if frame is not None:
                self.new_frame.emit(frame)
            else:
                self.error.emit("Camera returned no frame.")
                break
        self.camera.stop_live()

    def stop(self):
        self._running = False
        self.wait()


# ------------------------------------------------------------------ #
#  Stage position poller
# ------------------------------------------------------------------ #

class StagePoller(QThread):
    """
    Periodically reads stage.get_position() -> dict{axis: float}
    and emits position_updated so the UI labels update smoothly.
    """

    position_updated = pyqtSignal(dict)

    def __init__(self, stage, interval_ms: int = 250, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.interval_ms = interval_ms
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            if self.stage.is_connected:
                pos = self.stage.get_position()
                self.position_updated.emit(pos)
            self.msleep(self.interval_ms)

    def stop(self):
        self._running = False
        self.wait()


# ------------------------------------------------------------------ #
#  Jog worker — single non-blocking move
# ------------------------------------------------------------------ #

class JogWorker(QThread):
    """
    Executes one move (relative µm or absolute mm) in a background thread.
    Delegates to stage.move_relative() or stage.move_absolute(),
    which call PI_VC_Device.move_relative() / move_absolute() + wait_on_target().
    """

    move_done = pyqtSignal()
    error     = pyqtSignal(str)

    def __init__(self, stage, move_type: str, params: dict, parent=None):
        """
        move_type: 'relative'  -> params = {axis: µm_delta}
                   'absolute'  -> params = {axis: mm_position}
        """
        super().__init__(parent)
        self.stage = stage
        self.move_type = move_type
        self.params = params

    def run(self):
        try:
            if self.move_type == 'relative':
                self.stage.move_relative(self.params, wait=True)
            else:
                self.stage.move_absolute(self.params, wait=True)
            self.move_done.emit()
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------------ #
#  Acquisition worker — move → settle → snap sequence
# ------------------------------------------------------------------ #

class AcquisitionWorker(QThread):
    """
    For each position in the list:
      1. stage.move_absolute(pos)          — uses PI_VC_Device.move_absolute + wait_on_target
      2. wait settle_ms
      3. camera.snap()                     — uses HamamatsuDevice fixed_length single-frame
      4. emit new_image + progress

    Mirrors the internal-trigger branch of hyperspectral_measure.measure().
    """

    progress  = pyqtSignal(int, int)           # (current, total)
    new_image = pyqtSignal(dict, np.ndarray)   # (position_dict, frame)
    finished  = pyqtSignal(list)               # list of (pos_dict, frame)
    error     = pyqtSignal(str)

    def __init__(self, stage, camera, positions: list,
                 settle_ms: int = 200, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.camera = camera
        self.positions = positions
        self.settle_ms = settle_ms
        self._abort = False

    def run(self):
        total = len(self.positions)
        results = []

        for i, pos in enumerate(self.positions):
            if self._abort:
                break

            # 1. Move stage to position (mm, absolute)
            print(f"[Acq] Step {i+1}/{total} — moving to {pos}")
            self.stage.move_absolute(pos, wait=True)

            # 2. Settle
            self.msleep(self.settle_ms)

            # 3. Capture (fixed_length, 1 frame, internal trigger)
            frame = self.camera.snap()
            if frame is None:
                self.error.emit(f"Camera returned no frame at step {i+1}")
                return

            results.append((pos, frame))
            self.new_image.emit(pos, frame)
            self.progress.emit(i + 1, total)

        self.finished.emit(results)

    def abort(self):
        self._abort = True
