"""
PI Stage wrapper — forwards to PI_VC_Device exactly as in your existing code.

PI_VC_Device drives ONE axis per instance. Create one PIStageWrapper per axis
and group them in a StageManager. The wrapper exposes the SAME functions as
PI_VC_Device (move, position, velocity, home, servo, and the trigger functions
used by the external-trigger hyperspectral scan), plus a non-blocking position
read used to update the UI while a move is in progress.

Conventions (identical to PI_VC_Device):
    move_absolute  -> mm   (absolute target)
    move_relative  -> um   (relative displacement)
    get_position   -> mm   (waits on target, like the device)
"""

import sys
import os
import time
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, '..')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def add_path(path):
    """Add a folder under the 'ScopeFoundry Projects' root to sys.path.

    This file is at  <ScopeFoundry Projects>/HyperMeasurementApp/hardware/stage.py
    so the projects root is three levels up; `path` (e.g. 'PI_ScopeFoundry') is a
    sibling of HyperMeasurementApp.
    """
    here = os.path.abspath(__file__)
    projects_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    target = os.path.join(projects_root, path)
    if not os.path.isdir(target):
        print(f"[Stage] WARNING: expected device folder not found: {target}")
    if target not in sys.path:
        sys.path.append(target)


class PIStageWrapper:
    """Thin, thread-safe wrapper around a single-axis PI_VC_Device."""

    def __init__(self, serial: str, axis: str = '1'):
        self.serial = serial
        self.axis = axis
        self.motor = None          # PI_VC_Device instance, set after connect()
        self._connected = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        try:
            add_path('PI_ScopeFoundry')
            from PI_VC_device import PI_VC_Device
            self.motor = PI_VC_Device(serial=self.serial, axis=self.axis)
            self._connected = True
            print(f"[Stage] Connected axis {self.axis} (serial {self.serial})")
            print(f"[Stage] Device: {self.motor.get_info()}")
            return True
        except Exception as e:
            print(f"[Stage] Connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        if self.motor and self._connected:
            try:
                self.motor.close()
            except Exception as e:
                print(f"[Stage] Close error: {e}")
        self._connected = False
        self.motor = None
        print(f"[Stage] Axis {self.axis} disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_info(self) -> str:
        if self._connected:
            with self._lock:
                return self.motor.get_info()
        return f"(axis {self.axis} not connected)"

    # ------------------------------------------------------------------ #
    #  Position / motion
    # ------------------------------------------------------------------ #

    def get_position(self) -> float:
        """Position in mm (waits on target, exactly like PI_VC_Device)."""
        if not self._connected:
            return 0.0
        with self._lock:
            return self.motor.get_position()

    def get_position_no_wait(self) -> float:
        """
        Position in mm WITHOUT waiting on target — for live UI updates while a
        move is running. Reads qPOS directly (PI_VC_Device.get_position waits).

        Locked: the GCS serial link is not thread-safe, so this must be
        serialised against moves/reads issued by the scan worker, otherwise
        interleaved commands corrupt the reply (qPOS throws -> returns 0).
        """
        if not self._connected:
            return 0.0
        try:
            with self._lock:
                pos = self.motor.pi_device.qPOS(self.motor.axis)[self.motor.axis]
            return pos - self.motor.home
        except Exception:
            return 0.0

    def move_absolute(self, pos_mm: float, wait: bool = True, correct_backslash: bool = False):
        if not self._connected:
            return
        with self._lock:
            self.motor.move_absolute(pos_mm, correct_backslash=correct_backslash)
            if wait:
                self.motor.wait_on_target()

    def move_relative(self, delta_um: float, wait: bool = True, correct_backslash: bool = False):
        if not self._connected:
            return
        with self._lock:
            self.motor.move_relative(delta_um, correct_backslash=correct_backslash)
            if wait:
                self.motor.wait_on_target()

    def wait_on_target(self):
        if self._connected:
            with self._lock:
                self.motor.wait_on_target()

    def halt(self):
        if self._connected:
            with self._lock:
                self.motor.stop()

    # ------------------------------------------------------------------ #
    #  Home / reference / servo
    # ------------------------------------------------------------------ #

    def set_home(self):
        if self._connected:
            with self._lock:
                self.motor.set_home()

    def go_home(self):
        if self._connected:
            with self._lock:
                self.motor.go_home()

    def goto_ref_switch(self):
        if self._connected:
            with self._lock:
                self.motor.gotoRefSwitch()

    def set_servo(self, mode: bool):
        if self._connected:
            with self._lock:
                self.motor.set_servo(mode)

    def get_servo(self) -> bool:
        if self._connected:
            with self._lock:
                return bool(self.motor.get_servo())
        return False

    def get_mode(self):
        if self._connected:
            with self._lock:
                return self.motor.get_mode()
        return None

    # ------------------------------------------------------------------ #
    #  Velocity
    # ------------------------------------------------------------------ #

    def get_velocity(self) -> float:
        if self._connected:
            with self._lock:
                return self.motor.get_velocity()
        return 0.0

    def set_velocity(self, v: float):
        if self._connected:
            with self._lock:
                self.motor.set_velocity(v)

    # ------------------------------------------------------------------ #
    #  Trigger (used by the external-trigger hyperspectral scan)
    # ------------------------------------------------------------------ #

    def before_trigger(self):
        if self._connected:
            with self._lock:
                self.motor.before_trigger()

    def trigger(self, step, start, stop, ch, ch_tot):
        if self._connected:
            with self._lock:
                self.motor.trigger(step, start, stop, ch, ch_tot)

    def trigger_start(self, start, stop, ch, ch_tot):
        if self._connected:
            with self._lock:
                self.motor.trigger_start(start, stop, ch, ch_tot)

    def trigger_disable(self, ch_tot):
        if self._connected:
            with self._lock:
                self.motor.trigger_disable(ch_tot)

    def correct_backslash(self, displacement):
        if self._connected:
            with self._lock:
                self.motor.correct_backslash(displacement)


# ------------------------------------------------------------------ #
#  Multi-axis stage manager
# ------------------------------------------------------------------ #

class StageManager:
    """
    Groups one or more PIStageWrapper axes so the rest of the app can treat
    them uniformly with a dict-based API ({axis_label: value}).

    For the hyperspectral scan, the measurement panel picks ONE axis and uses
    its PIStageWrapper directly via get_wrapper(axis).
    """

    def __init__(self, wrappers: list):
        self._wrappers: dict[str, PIStageWrapper] = {w.axis: w for w in wrappers}

    @property
    def axes(self) -> list[str]:
        return list(self._wrappers.keys())

    @property
    def is_connected(self) -> bool:
        return any(w.is_connected for w in self._wrappers.values())

    def connect(self) -> bool:
        results = [w.connect() for w in self._wrappers.values()]
        return all(results)

    def disconnect(self):
        for w in self._wrappers.values():
            w.disconnect()

    def get_position(self) -> dict[str, float]:
        return {ax: w.get_position() for ax, w in self._wrappers.items()}

    def get_position_no_wait(self) -> dict[str, float]:
        return {ax: w.get_position_no_wait() for ax, w in self._wrappers.items()}

    def move_absolute(self, positions: dict[str, float], wait: bool = True):
        for ax, pos in positions.items():
            if ax in self._wrappers:
                self._wrappers[ax].move_absolute(pos, wait=wait)

    def move_relative(self, deltas: dict[str, float], wait: bool = True):
        for ax, delta in deltas.items():
            if ax in self._wrappers:
                self._wrappers[ax].move_relative(delta, wait=wait)

    def halt(self):
        for w in self._wrappers.values():
            w.halt()

    def go_home(self):
        for w in self._wrappers.values():
            w.go_home()

    def get_wrapper(self, axis: str) -> PIStageWrapper:
        return self._wrappers[axis]


# ------------------------------------------------------------------ #
#  Mock stage (no hardware needed — for UI development)
# ------------------------------------------------------------------ #

class MockAxis:
    """Single-axis mock mirroring PIStageWrapper's float (mm/um) API."""

    def __init__(self, axis: str):
        self.axis = axis
        self._pos = 0.0          # mm
        self._home = 0.0
        self._velocity = 5.0
        self._connected = True
        self.motor = self        # so worker code that touches .motor still works

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_info(self) -> str:
        return f"MockAxis {self.axis}"

    def get_position(self) -> float:
        return self._pos - self._home

    def get_position_no_wait(self) -> float:
        return self._pos - self._home

    def move_absolute(self, pos_mm, wait=True, correct_backslash=False):
        self._pos = self._home + pos_mm
        if wait:
            time.sleep(0.02)

    def move_relative(self, delta_um, wait=True, correct_backslash=False):
        self._pos += delta_um * 0.001
        if wait:
            time.sleep(0.02)

    def wait_on_target(self):
        time.sleep(0.01)

    def halt(self):
        pass

    def set_home(self):
        self._home = self._pos

    def go_home(self):
        self._pos = self._home
        time.sleep(0.02)

    def goto_ref_switch(self):
        pass

    def set_servo(self, mode):
        pass

    def get_servo(self) -> bool:
        return True

    def get_mode(self):
        return True

    def get_velocity(self) -> float:
        return self._velocity

    def set_velocity(self, v):
        self._velocity = v

    def before_trigger(self):
        pass

    def trigger(self, step, start, stop, ch, ch_tot):
        pass

    def trigger_start(self, start, stop, ch, ch_tot):
        pass

    def trigger_disable(self, ch_tot):
        pass

    def correct_backslash(self, displacement):
        pass


class MockPIStage:
    """Simulates a multi-axis PI stage for UI work without hardware."""

    def __init__(self, axes=('1', '2', '3')):
        self._axis_objs = {ax: MockAxis(ax) for ax in axes}
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        print("[MockStage] Connected (simulation)")
        return True

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def axes(self) -> list[str]:
        return list(self._axis_objs.keys())

    def get_position(self) -> dict[str, float]:
        return {ax: w.get_position() for ax, w in self._axis_objs.items()}

    def get_position_no_wait(self) -> dict[str, float]:
        return {ax: w.get_position_no_wait() for ax, w in self._axis_objs.items()}

    def move_absolute(self, positions: dict, wait=True):
        for ax, pos in positions.items():
            if ax in self._axis_objs:
                self._axis_objs[ax].move_absolute(pos, wait=wait)

    def move_relative(self, deltas: dict, wait=True):
        for ax, d in deltas.items():
            if ax in self._axis_objs:
                self._axis_objs[ax].move_relative(d, wait=wait)

    def halt(self):
        pass

    def go_home(self):
        for w in self._axis_objs.values():
            w.go_home()

    def get_wrapper(self, axis: str) -> MockAxis:
        return self._axis_objs[axis]
