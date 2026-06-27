"""
Background workers — all hardware I/O runs in QThreads so the Qt event loop
(and the live image) stays responsive.

Threading note (requirement 3): the heavy work here is hardware I/O — DCAM
camera reads and PI GCS moves. Those spend their time inside C extension calls
that release the Python GIL, so these QThreads genuinely run concurrently with
the UI on separate cores. (Pure-Python CPU work would still be GIL-bound; none
of the hot paths here are pure Python.)

Workers:
    LiveViewWorker      run_till_abort live preview -> emits frames
    StagePoller         periodic position read -> emits {axis: mm}
    JogWorker           one move (relative um / absolute mm)
    HyperSpectralWorker the line scan from hyperspectral_measure.py
                        (internal + external trigger branches, optional H5)
"""

import os
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import h5py
except Exception:                      # pragma: no cover - only needed for saving
    h5py = None


# ------------------------------------------------------------------ #
#  Live-preview worker
# ------------------------------------------------------------------ #

class LiveViewWorker(QThread):
    """camera.start_live() then repeatedly camera.get_live_frame().

    Throttled to ~`interval_ms` between frames so it never busy-spins (which in
    mock mode would flood the GUI event queue and freeze the app, and on real
    hardware would re-pull the same buffer wastefully).
    """

    new_frame = pyqtSignal(np.ndarray)
    error     = pyqtSignal(str)

    def __init__(self, camera, interval_ms: int = 33, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.interval_ms = max(1, int(interval_ms))
        self._running = False

    def run(self):
        self._running = True
        try:
            self.camera.start_live()
        except Exception as e:
            self.error.emit(f"start_live failed: {e}")
            return
        try:
            while self._running:
                frame = self.camera.get_live_frame()
                if frame is not None:
                    self.new_frame.emit(frame)
                else:
                    self.error.emit("Camera returned no frame.")
                    break
                self.msleep(self.interval_ms)
        finally:
            self.camera.stop_live()

    def stop(self):
        """Signal the loop to exit and wait (bounded) for it to finish."""
        self._running = False
        if not self.wait(2000):
            # Last resort so the UI can never hang on a stuck driver call.
            self.terminate()
            self.wait()


# ------------------------------------------------------------------ #
#  Stage position poller
# ------------------------------------------------------------------ #

class StagePoller(QThread):
    """Periodically reads the stage position (non-blocking) -> {axis: mm}."""

    position_updated = pyqtSignal(dict)

    def __init__(self, stage, interval_ms: int = 250, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.interval_ms = interval_ms
        self._running = False

    def run(self):
        self._running = True
        read = getattr(self.stage, 'get_position_no_wait', self.stage.get_position)
        while self._running:
            if self.stage.is_connected:
                try:
                    self.position_updated.emit(read())
                except Exception:
                    pass
            self.msleep(self.interval_ms)

    def stop(self):
        self._running = False
        self.wait()


# ------------------------------------------------------------------ #
#  Jog worker — single non-blocking move
# ------------------------------------------------------------------ #

class JogWorker(QThread):
    """One move in a background thread (relative um or absolute mm)."""

    move_done = pyqtSignal()
    error     = pyqtSignal(str)

    def __init__(self, stage, move_type: str, params: dict, parent=None):
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
#  Hyperspectral scan worker — port of hyperspectral_measure.measure()
# ------------------------------------------------------------------ #

# Hamamatsu C11440-42U40 readout constant (see set_motor_velocity Version 4)
_H_LINE_TIME = 32.4812e-6      # s per horizontal line

# Ring-buffer size for the continuous internal-trigger scan. Frames are consumed
# continuously, so a small ring suffices; if a long move lets it overrun, only
# stale frames (which we discard anyway) are overwritten.
_SCAN_RING_BUFFERS = 16


class HyperSpectralWorker(QThread):
    """
    Performs the hyperspectral line scan, faithfully porting both the internal-
    and external-trigger branches of hyperspectral_measure.measure().

    scan : single-axis PIStageWrapper (the scan axis)
    camera : HamamatsuCamera
    params : dict with keys
        start_pos (mm), step_um (um), step_num (int), trigger ('internal'/'external'),
        save_h5 (bool), save_dir, sample, posx, posy, refresh_ms,
        fast_velocity (mm/s), auto_velocity (bool), motor_velocity (mm/s),
        xsampling, ysampling, zsampling
    """

    new_image = pyqtSignal(np.ndarray)        # latest image
    point     = pyqtSignal(float, float)      # (position_mm, intensity)
    progress  = pyqtSignal(int, int)          # (current, total)
    status    = pyqtSignal(str)
    finished_scan = pyqtSignal(str)           # h5 path, or '' if not saved
    error     = pyqtSignal(str)

    def __init__(self, scan, camera, params: dict, parent=None):
        super().__init__(parent)
        self.scan = scan
        self.camera = camera
        self.p = dict(params)
        self._abort = False
        self._h5 = None

    # -- public --
    def abort(self):
        self._abort = True

    # -- helpers --
    def _intensity(self, img: np.ndarray) -> float:
        r = int(np.clip(self.p.get('posx', 0), 0, img.shape[0] - 1))
        c = int(np.clip(self.p.get('posy', 0), 0, img.shape[1] - 1))
        return float(img[r, c])

    def _scan_velocity(self) -> float:
        """Automatic velocity (set_motor_velocity Version 4) or the manual value."""
        step_mm = self.p['step_um'] * 1e-3
        if not self.p.get('auto_velocity', True):
            return max(self.p.get('motor_velocity', 0.125), 1e-6)
        exp = self.camera.get_exposure()
        V = self.camera.subarrayv
        frame_rate = 1.0 / (V / 2 * _H_LINE_TIME + exp + 10 * _H_LINE_TIME)
        vel = step_mm / (4 * exp)
        required = 1.0 / (step_mm / vel)
        self.status.emit(f"Max frame rate ~{frame_rate:.1f} Hz; required ~{required:.1f} Hz")
        if required > 0.85 * frame_rate:
            vel = 0.75 * step_mm * frame_rate
            self.status.emit(f"Velocity capped to {vel:.4f} mm/s for current camera settings")
        return max(vel, 1e-6)

    # ------------------------------------------------------------------ #
    def run(self):
        try:
            if self.p['trigger'] == 'internal':
                self._run_internal()
            elif self.p['trigger'] == 'external':
                self._run_external()
            else:
                self.error.emit("Trigger must be 'internal' or 'external'.")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                self.camera.stop_acquisition()
            except Exception:
                pass
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
                self._h5 = None

    # ---- internal trigger: stepwise move -> grab a fresh frame -------- #
    def _run_internal(self):
        start = self.p['start_pos']
        step_mm = self.p['step_um'] * 1e-3
        n = self.p['step_num']

        self.camera.set_trigger_source('internal')
        # ONE continuous acquisition for the whole scan: starting/stopping (and
        # reallocating the full-frame buffer + captureSetup) per step is what made
        # each step take far longer than the exposure. A small ring buffer is
        # enough because we consume frames continuously.
        self.camera.set_acquisition('run_till_abort', _SCAN_RING_BUFFERS)

        self.scan.set_velocity(self.p.get('fast_velocity', 5.0))
        self.status.emit(f"Moving to start {start:.4f} mm")
        self.scan.move_absolute(start, wait=True)

        self.camera.start_acquisition()
        img_h5 = pos_h5 = None
        try:
            for i in range(n):
                if self._abort:
                    self.status.emit("Aborted.")
                    break

                # The camera free-runs, so the newest buffered frame may have been
                # exposed during the move. Drop one in-progress frame, then take the
                # next one — guaranteed to be exposed after the stage settled.
                self.camera.get_last_frame()
                img = self.camera.get_last_frame()
                if img is None:
                    self.error.emit(f"No frame at step {i+1}")
                    return

                # Read the encoder right after the frame so the saved position
                # matches the frame's exposure window as closely as possible.
                cur = self.scan.get_position_no_wait()

                self.new_image.emit(img)
                self.point.emit(cur, self._intensity(img))
                self.progress.emit(i + 1, n)

                if self.p['save_h5']:
                    if img_h5 is None:
                        path, img_h5, pos_h5 = self._create_h5(img, n)
                        self._saved_path = path
                    img_h5[i, :, :] = img
                    pos_h5[i] = cur * 1000.0          # um
                    self._h5.flush()

                if i < n - 1:
                    self.scan.move_absolute(start + (i + 1) * step_mm, wait=True)
        finally:
            self.camera.stop_acquisition()

        saved = getattr(self, '_saved_path', '') if self.p['save_h5'] else ''
        self.finished_scan.emit(saved or '')

    # ---- external trigger: PI hardware trigger + run_till_abort ------- #
    def _run_external(self):
        start = self.p['start_pos']
        step_mm = self.p['step_um'] * 1e-3
        n = self.p['step_num']
        refresh = max(self.p.get('refresh_ms', 50), 1)
        correction = 0.05
        ch, ch_tot = 1, 4

        target = start + n * step_mm
        forward = target >= start
        if not forward:
            correction = -correction

        self.camera.set_trigger_source('external')
        self.camera.set_acquisition('run_till_abort', n + 10)

        self.scan.set_velocity(self.p.get('fast_velocity', 5.0))
        self.status.emit(f"Moving to start {start - correction:.4f} mm")
        self.scan.move_absolute(start - correction, wait=True)

        vel = self._scan_velocity()
        self.scan.set_velocity(vel)
        self.status.emit(f"Scan velocity {vel:.4f} mm/s")

        self.camera.start_acquisition()
        self.status.emit("Acquisition started (external trigger)")
        self.scan.trigger(step_mm, start, target, ch, ch_tot)
        self.scan.move_absolute(target + correction, wait=False)

        # Wait for the stage to traverse, polling only the position for progress.
        # IMPORTANT: do NOT call get_last_frame()/get_live_frame() here. That
        # advances the camera's internal frame counter, so the final getFrames()
        # would return only the last few frames and the rest of the saved stack
        # (image + position_mm) would be zeros. All frames are collected at once
        # after the move instead.
        while not self._abort:
            pos = self.scan.get_position_no_wait()
            self.status.emit(f"Scanning… {pos:.4f} mm")
            if (forward and pos >= target) or (not forward and pos <= target):
                break
            self.msleep(int(refresh))

        self.scan.wait_on_target()
        self.status.emit("Stage on target; collecting frames")

        self.camera.stop_acquisition_not_releasing()
        frames, _ = self.camera.get_frames()
        n_frames = len(frames)
        self.status.emit(f"Acquired {n_frames} frames")
        positions = np.linspace(start, target, max(n_frames, 1)) * 1000.0   # um

        saved = ''
        if self.p['save_h5'] and n_frames > 0 and not self._abort:
            # Size the datasets to the frames we actually got — no trailing zeros.
            path, img_h5, pos_h5 = self._create_h5(frames[0], n_frames)
            saved = path
            for i in range(n_frames):
                img_h5[i, :, :] = frames[i]
                pos_h5[i] = positions[i]
            self._h5.flush()

        # Replay the acquired frames to the live view + interferogram (display).
        for i in range(n_frames):
            if self._abort:
                break
            self.new_image.emit(frames[i])
            self.point.emit(positions[i] / 1000.0, self._intensity(frames[i]))
            self.progress.emit(i + 1, max(n_frames, 1))

        # Restore safe state (mirrors measure()).
        self.scan.trigger_disable(ch_tot)
        self.camera.set_trigger_source('internal')
        self.scan.set_velocity(self.p.get('fast_velocity', 5.0))

        if frames:
            self.new_image.emit(frames[-1])
        self.finished_scan.emit(saved)

    # ---- HDF5 helpers ------------------------------------------------- #
    def _create_h5(self, sample_img: np.ndarray, length: int):
        if h5py is None:
            raise RuntimeError("h5py is not installed — cannot save .h5")
        save_dir = self.p.get('save_dir') or os.path.expanduser('~/hyper_data')
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.p.get('sample', '') or ''
        parts = [timestamp, 'hyper'] + ([sample] if sample else [])
        fname = os.path.join(save_dir, '_'.join(parts) + '.h5')

        self._h5 = h5py.File(fname, 'w')
        grp = self._h5.create_group('measurement/hyper')
        h, w = sample_img.shape
        img_h5 = grp.create_dataset('t0/c0/image', shape=(length, h, w),
                                    dtype=sample_img.dtype)
        img_h5.attrs['element_size_um'] = [self.p.get('zsampling', 1.0),
                                           self.p.get('ysampling', 1.0),
                                           self.p.get('xsampling', 1.0)]
        pos_h5 = grp.create_dataset('t0/c0/position_mm', shape=(length,),
                                    dtype=np.float32)
        # Stamp the key scan settings for provenance.
        for k in ('start_pos', 'step_um', 'step_num', 'trigger'):
            grp.attrs[k] = self.p.get(k)
        return fname, img_h5, pos_h5
