"""
PI Stage wrapper — uses PI_VC_Device exactly as written in your existing code.

Copy PI_VC_device.py into the hardware/ folder (or anywhere on sys.path),
then set the serial number below.
"""

import sys
import os
import time
import threading
import numpy as np

# Allow importing PI_VC_device from the project root or hardware folder
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


class PIStageWrapper:
    """
    Thin wrapper around PI_VC_Device that adds:
      - a connect() / disconnect() pattern for the UI
      - thread-safe position polling
      - the same dict-based API the rest of the app uses

    Your PI_VC_Device takes one axis at construction time.
    If you have multiple axes, instantiate one PIStageWrapper per axis
    and pass a list to the StageManager below.
    """

    def __init__(self, serial: str, axis: str = '1'):
        self.serial = serial
        self.axis = axis
        self.motor = None          # PI_VC_Device instance, set after connect()
        self._connected = False
        self._lock = threading.Lock()

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

    # ------------------------------------------------------------------ #
    #  Position
    # ------------------------------------------------------------------ #

    def get_position(self) -> float:
        """Returns position in mm (relative to home, as PI_VC_Device does)."""
        if not self._connected:
            return 0.0
        with self._lock:
            return self.motor.get_position()

    def move_absolute(self, pos_mm: float, wait: bool = True):
        """Move to absolute position in mm."""
        if not self._connected:
            return
        with self._lock:
            self.motor.move_absolute(pos_mm)
            if wait:
                self.motor.wait_on_target()

    def move_relative(self, delta_um: float, wait: bool = True):
        """Move relative by delta_um (micrometres, matching your existing move_relative signature)."""
        if not self._connected:
            return
        with self._lock:
            self.motor.move_relative(delta_um)   # PI_VC_Device.move_relative expects µm
            if wait:
                self.motor.wait_on_target()

    def halt(self):
        if self._connected and self.motor:
            self.motor.stop()

    def set_home(self):
        if self._connected and self.motor:
            self.motor.set_home()

    def go_home(self):
        if self._connected and self.motor:
            self.motor.go_home()

    def get_velocity(self) -> float:
        if self._connected and self.motor:
            return self.motor.get_velocity()
        return 0.0

    def set_velocity(self, v: float):
        if self._connected and self.motor:
            self.motor.set_velocity(v)


# ------------------------------------------------------------------ #
#  Multi-axis stage manager
# ------------------------------------------------------------------ #

class StageManager:
    """
    Groups one or more PIStageWrapper axes so the rest of the app can
    treat them uniformly (dict of {axis_label: value}).

    Example for a 3-axis system:
        stage = StageManager([
            PIStageWrapper('0185500001', axis='1'),
            PIStageWrapper('0185500002', axis='2'),
            PIStageWrapper('0185500003', axis='3'),
        ])
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

    def move_absolute(self, positions: dict[str, float], wait: bool = True):
        """positions: {axis: mm_value}"""
        for ax, pos in positions.items():
            if ax in self._wrappers:
                self._wrappers[ax].move_absolute(pos, wait=wait)

    def move_relative(self, deltas: dict[str, float], wait: bool = True):
        """deltas: {axis: µm_value}  — matching PI_VC_Device.move_relative convention"""
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

class MockPIStage:
    """Simulates a multi-axis PI stage for UI work without hardware."""

    def __init__(self, axes=('1', '2', '3')):
        self._axes = list(axes)
        self._pos = {ax: 0.0 for ax in axes}
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
        return self._axes

    def get_position(self) -> dict[str, float]:
        return dict(self._pos)

    def move_absolute(self, positions: dict, wait=True):
        self._pos.update(positions)
        if wait:
            time.sleep(0.05)

    def move_relative(self, deltas: dict, wait=True):
        for ax, d in deltas.items():
            # deltas are in µm, convert to mm for internal pos tracking
            self._pos[ax] = self._pos.get(ax, 0.0) + d * 0.001
        if wait:
            time.sleep(0.05)

    def halt(self):
        pass

    def go_home(self):
        for ax in self._axes:
            self._pos[ax] = 0.0
        time.sleep(0.1)

    def get_wrapper(self, axis):
        return None   # not available in mock
