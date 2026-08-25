from __future__ import annotations

from .camera_interface import CameraInterface
from .mock_camera import MockCamera


def create_camera(mode: str = "forge") -> CameraInterface:
    """Create a camera backend.

    mode: "forge" (Teledyne FLIR Forge 1GigE SWIR via Spinnaker/PySpin), "mock"
    (synthetic beam, no hardware), or "auto" (try the Forge, fall back to mock).
    """
    normalized = (mode or "forge").lower()

    if normalized == "mock":
        return MockCamera()

    if normalized in ("forge", "swir", "spinnaker", "pyspin", "gige", "auto"):
        try:
            from .forge_camera import ForgeSwirCamera
            return ForgeSwirCamera()
        except Exception:  # noqa: BLE001  (PySpin not installed -> mock)
            return MockCamera()

    return MockCamera()
