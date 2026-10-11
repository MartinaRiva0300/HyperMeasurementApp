"""
Background workers — all hardware I/O runs in QThreads so the Qt event loop
(and the live image) stays responsive.

Threading note: the heavy work here is hardware I/O — PVCAM frame polls and PI
GCS moves. Those spend their time inside C extension calls that release the
Python GIL, so these QThreads genuinely run concurrently with the UI.

Workers:
    LiveViewWorker      start_live (circular buffer) + poll_frame -> emits frames
    StagePoller         periodic position read -> emits {axis: mm}
    JogWorker           one move (relative um / absolute mm)
    HyperSpectralWorker the line scan from hyperspectral_measure.py (RetigaE7_PI):
                        'Internal' (step -> 1-frame sequence) and 'Edge'
                        (PI position trigger -> n-frame sequence), optional H5
"""

from __future__ import annotations

import os
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import h5py
except Exception:                      # pragma: no cover - only needed for saving
    h5py = None

from hardware.camera import max_frame_rate_hz


# ------------------------------------------------------------------ #
#  Live-preview worker
# ------------------------------------------------------------------ #

class LiveViewWorker(QThread):
    """camera.start_live() then repeatedly camera.get_live_frame().

    poll_frame blocks until the next frame is available (the oldest frame is
    popped from the PVCAM circular buffer), so the loop paces itself at the
    camera frame rate. Frames are forwarded to the GUI at most every
    `interval_ms` so the event queue is never flooded.
    """

    new_frame = pyqtSignal(np.ndarray)
    error     = pyqtSignal(str)

    def __init__(self, camera, interval_ms: int = 33, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.interval_ms = max(1, int(interval_ms))
        self._running = False

    def _timeout_ms(self) -> int:
        return int(self.camera.get_exposure()) + 2000

    def run(self):
        self._running = True
        try:
            self.camera.start_live()
        except Exception as e:
            self.error.emit(f"start_live failed: {e}")
            return
        last_emit = 0.0
        failures = 0
        try:
            while self._running:
                try:
                    frame = self.camera.get_live_frame(timeout_ms=self._timeout_ms())
                except Exception as e:
                    if not self._running:
                        break
                    failures += 1
                    if failures >= 3:
                        self.error.emit(f"poll_frame failed: {e}")
                        break
                    continue
                failures = 0
                if frame is None:
                    self.error.emit("Camera returned no frame.")
                    break
                now = time.monotonic()
                if (now - last_emit) * 1000.0 >= self.interval_ms:
                    self.new_frame.emit(frame)
                    last_emit = now
        finally:
            self.camera.stop_live()

    def stop(self):
        """Signal the loop to exit and wait (bounded) for it to finish."""
        self._running = False
        try:
            self.camera.abort()            # break a pending poll_frame
        except Exception:
            pass
        if not self.wait(self._timeout_ms() + 1000):
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

# PI trigger output channel used by the Edge-trigger scan (measure()).
_TRIG_CH, _TRIG_CH_TOT = 1, 4
# Overshoot (mm) before the start / after the end of the triggered travel, to
# compensate the delay between trigger and movement start (measure()).
_TRIG_CORRECTION_MM = 0.05


class HyperSpectralWorker(QThread):
    """
    Performs the hyperspectral line scan, porting both branches of
    hyperspectral_measure.measure() (RetigaE7_PI):

      'internal'  move -> wait on target -> acq_start_seq(1) -> get_nparray ->
                  acq_stop, for every position (step grid, or a loaded list).
      'edge'      camera in 'Edge Trigger', acq_start_seq(n); the PI stage
                  fires a trigger every `step` between start and stop while
                  travelling at the scan velocity; frames are polled as they
                  arrive.

    scan : single-axis PIStageWrapper (the scan axis)
    camera : RetigaCamera (or MockCamera)
    params : dict with keys
        start_pos (mm), step_um (um), step_num (int), trigger ('internal'/'edge'),
        positions (optional 1-D array of mm targets; internal trigger only),
        save_h5 (bool), save_dir, sample, posx, posy, refresh_ms,
        fast_velocity (mm/s), auto_velocity (bool), motor_velocity (mm/s),
        xsampling, ysampling, zsampling, camera_settings (dict, provenance)
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
        self._saved_path = ''

    # -- public --
    def abort(self):
        self._abort = True
        try:
            self.camera.abort()            # break a pending poll_frame
        except Exception:
            pass

    # -- helpers --
    def _intensity(self, img: np.ndarray) -> float:
        r = int(np.clip(self.p.get('posx', 0), 0, img.shape[0] - 1))
        c = int(np.clip(self.p.get('posy', 0), 0, img.shape[1] - 1))
        return float(img[r, c])

    def _scan_velocity(self) -> float:
        """Automatic velocity (set_motor_velocity, RetigaE7_PI) or the manual value."""
        step_mm = self.p['step_um'] * 1e-3
        if not self.p.get('auto_velocity', True):
            return max(self.p.get('motor_velocity', 0.125), 1e-6)
        exp = self.camera.get_exposure_s()
        roi = self.camera.get_roi()
        frame_rate = max_frame_rate_hz(roi['hsize'], roi['vsize'], exp)
        vel = step_mm / (4 * exp) if exp > 0 else 0.0
        required = vel / step_mm if step_mm > 0 else 0.0
        self.status.emit(
            f"Max frame rate for ROI ({roi['hsize']}×{roi['vsize']}) and exposure "
            f"{exp*1e3:.0f} ms ~{frame_rate:.1f} Hz; required ~{required:.1f} Hz")
        if required > 0.85 * frame_rate:           # empirical threshold
            vel = 0.75 * step_mm * frame_rate
            self.status.emit(
                f"Velocity too high for the camera settings; capped to {vel:.4f} mm/s")
        return max(vel, 1e-6)

    # ------------------------------------------------------------------ #
    def run(self):
        try:
            if self.p['trigger'] == 'internal':
                self._run_internal()
            elif self.p['trigger'] == 'edge':
                self._run_edge()
            else:
                self.error.emit("Trigger must be 'internal' or 'edge'.")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                self.camera.stop_acquisition()
            except Exception:
                pass
            self._close_h5()

    # ---- internal trigger: step -> 1-frame sequence ------------------ #
    def _run_internal(self):
        start = self.p['start_pos']
        step_mm = self.p['step_um'] * 1e-3
        n = int(self.p['step_num'])

        loaded = self.p.get('positions')
        if loaded is not None and len(loaded) > 0:
            positions = np.asarray(loaded, dtype=float).ravel()
            n = len(positions)
            self.status.emit(f"Using {n} loaded positions ({positions[0]:.4f} → {positions[-1]:.4f} mm)")
        else:
            # start + i*step, i = 0..n-1 (final position start + n*step excluded)
            positions = start + np.arange(n) * step_mm

        self.camera.set_trigger_mode('Internal Trigger')
        self.scan.set_velocity(self.p.get('fast_velocity', 5.0))
        self.status.emit(f"Moving to start {positions[0]:.4f} mm")

        timeout_ms = int(self.camera.get_exposure()) + 10000
        img_h5 = pos_h5 = None
        count = 0
        for i in range(n):
            if self._abort:
                self.status.emit("Interrupted.")
                break

            self.scan.move_absolute(float(positions[i]), wait=True)
            cur = self.scan.get_position()          # waits on target, as measure()

            # Exactly measure(): acq_start_seq(1) -> get_nparray -> acq_stop
            self.camera.start_seq(1)
            try:
                img = self.camera.poll_frame(timeout_ms=timeout_ms)
            finally:
                self.camera.stop_acquisition()
            if img is None:
                self.error.emit(f"No frame at step {i+1}")
                return

            self.new_image.emit(img)
            self.point.emit(cur, self._intensity(img))
            self.progress.emit(i + 1, n)

            if self.p['save_h5']:
                if img_h5 is None:
                    img_h5, pos_h5 = self._create_h5(img, n)
                img_h5[i, :, :] = img
                pos_h5[i] = cur * 1000.0            # um
                self._h5.flush()
            count = i + 1

        self._trim_h5(count)
        self.finished_scan.emit(self._saved_path if self.p['save_h5'] else '')

    # ---- edge trigger: PI position trigger + n-frame sequence --------- #
    def _run_edge(self):
        start = self.p['start_pos']
        step_mm = self.p['step_um'] * 1e-3
        n = int(self.p['step_num'])
        if self.p.get('positions') is not None:
            self.status.emit("Loaded position list is ignored with Edge Trigger (uniform grid used).")

        # Final position is included: n triggers from start to start+(n-1)*step.
        target = start + (n - 1) * step_mm
        correction = _TRIG_CORRECTION_MM if target >= start else -_TRIG_CORRECTION_MM
        ch, ch_tot = _TRIG_CH, _TRIG_CH_TOT

        self.camera.set_trigger_mode('Edge Trigger')

        # Move a bit before the starting position at high velocity.
        self.scan.set_velocity(self.p.get('fast_velocity', 5.0))
        self.status.emit(f"Moving to {start - correction:.4f} mm (start − correction)")
        self.scan.move_absolute(start - correction, wait=True)

        vel = self._scan_velocity()
        self.scan.set_velocity(vel)
        self.status.emit(f"Scan velocity {vel:.4f} mm/s")

        # Sequential acquisition of a fixed number of frames, one per trigger.
        self.camera.start_seq(n)
        t0 = time.time()
        self.status.emit("Acquisition started (Edge Trigger)")
        self.scan.trigger(step_mm, start, target, ch, ch_tot)
        self.scan.move_absolute(target + correction, wait=False)

        positions = np.linspace(start, target, n)       # mm
        exp_s = self.camera.get_exposure_s()
        # Generous per-frame timeout: travel to the next trigger at `vel` (+ the
        # initial overshoot) plus exposure, with margin.
        timeout_ms = int(1000.0 * (5.0 * (abs(correction) + step_mm) / vel + exp_s + 2.0))

        img_h5 = pos_h5 = None
        count = 0
        misses = 0
        try:
            for i in range(n):
                if self._abort:
                    break
                try:
                    img = self.camera.poll_frame(timeout_ms=timeout_ms)
                except Exception as e:
                    if self._abort:
                        break
                    misses += 1
                    on_target = self.scan.is_on_target()
                    self.status.emit(f"Waiting for frame {i+1}/{n}… ({e})")
                    if on_target or misses >= 3:
                        self.status.emit(
                            f"Stage on target after {count} frame(s); no more triggers expected.")
                        break
                    continue
                misses = 0
                if img is None:
                    break

                self.new_image.emit(img)
                self.point.emit(float(positions[i]), self._intensity(img))
                self.progress.emit(i + 1, n)

                if self.p['save_h5']:
                    if img_h5 is None:
                        img_h5, pos_h5 = self._create_h5(img, n)
                    img_h5[i, :, :] = img
                    pos_h5[i] = positions[i] * 1000.0          # um
                    self._h5.flush()
                count = i + 1
        finally:
            if self._abort:
                self.scan.halt()
                self.status.emit("Interrupted.")
            self.scan.wait_on_target()
            self.camera.stop_acquisition()
            self.status.emit(f"Acquisition stopped: {count} frame(s) in {time.time()-t0:.2f} s")
            # Restore safe state (mirrors measure()).
            self.scan.trigger_disable(ch_tot)
            self.camera.set_trigger_mode('Internal Trigger')
            self.scan.set_velocity(self.p.get('fast_velocity', 5.0))

        self._trim_h5(count)
        self.finished_scan.emit(self._saved_path if self.p['save_h5'] else '')

    # ---- HDF5 helpers ------------------------------------------------- #
    def _create_h5(self, sample_img: np.ndarray, length: int):
        """Same layout as the ScopeFoundry h5_io file: measurement/hyper/t0/c0/{image,position_mm}."""
        if h5py is None:
            raise RuntimeError("h5py is not installed — cannot save .h5")
        save_dir = self.p.get('save_dir') or os.path.expanduser('~/hyper_data')
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.p.get('sample', '') or ''
        parts = [timestamp, 'hyper'] + ([sample] if sample else [])
        fname = os.path.join(save_dir, '_'.join(parts) + '.h5')

        self._h5 = h5py.File(fname, 'w')
        self._saved_path = fname
        grp = self._h5.create_group('measurement/hyper')
        h, w = sample_img.shape
        img_h5 = grp.create_dataset('t0/c0/image', shape=(length, h, w),
                                    maxshape=(None, h, w), dtype=sample_img.dtype)
        img_h5.attrs['element_size_um'] = [self.p.get('zsampling', 1.0),
                                           self.p.get('ysampling', 1.0),
                                           self.p.get('xsampling', 1.0)]
        pos_h5 = grp.create_dataset('t0/c0/position_mm', shape=(length,),
                                    maxshape=(None,), dtype=np.float32)
        # Stamp the scan settings + camera settings for provenance.
        for k in ('start_pos', 'step_um', 'step_num', 'trigger', 'motor_velocity',
                  'auto_velocity', 'sample'):
            v = self.p.get(k)
            if v is not None:
                grp.attrs[k] = v
        for k, v in (self.p.get('camera_settings') or {}).items():
            if v is not None:
                grp.attrs[f'camera_{k}'] = v
        self._img_h5, self._pos_h5 = img_h5, pos_h5
        return img_h5, pos_h5

    def _trim_h5(self, count: int):
        """Shrink the datasets to the frames actually acquired (no trailing zeros)."""
        if self._h5 is None:
            return
        try:
            if count > 0 and self._img_h5.shape[0] != count:
                self._img_h5.resize(count, axis=0)
                self._pos_h5.resize(count, axis=0)
            self._h5.attrs['frames_acquired'] = count
            self._h5.flush()
        except Exception as e:
            self.status.emit(f"Could not trim HDF5 datasets: {e}")

    def _close_h5(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
