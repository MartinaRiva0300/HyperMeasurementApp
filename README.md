# Hyperspectral Control App

Standalone PyQt5 desktop application for coordinated control of:
- **Hamamatsu Orca Flash 4V3** camera (via DCAM4 / `CameraDevice.py` → `HamamatsuDevice`)
- **PI motorised stage** (via `pipython` / `PI_VC_device.py` → `PI_VC_Device`)

It reuses the *exact* device functions from your existing
`Hyperspectral_System/hyperspectral_measure.py` system — no device behaviour is
changed; the wrappers forward 1:1 to `PI_VC_Device` and `HamamatsuDevice`.

---

## Project structure

```
HyperMeasurementApp/
├── main.py                  ← entry point (--real for hardware, default = mock)
├── requirements.txt
├── hardware/
│   ├── camera.py            ← HamamatsuCamera wrapper + MockCamera
│   └── stage.py             ← PIStageWrapper + StageManager + MockPIStage
├── core/
│   └── acquisition.py       ← QThread workers (live, poller, jog, HyperSpectralWorker)
└── ui/
    ├── widgets.py           ← shared pyqtgraph ImageView helper
    ├── camera_panel.py      ← exposure, binning, ROI, trigger, live view
    ├── stage_panel.py       ← encoder read-back, to-go position, jog (µm), velocity
    ├── measurement_panel.py ← hyperspectral scan + live image + interferogram plot
    └── main_window.py       ← assembles & coordinates the three panels
```

## The three panels

1. **Camera** — exposure (s), binning (1/2/4), ROI/subarray (H/V size + position),
   trigger source, live view and snap. ROI/binning are locked while live (the
   camera must be idle to change them). Read-back shows model, internal frame
   rate, and sensor temperature.
2. **Stage** — per-axis current position (encoder read-back), an independent
   "Go to" target (to-go position), relative jog in µm, velocity, set-home,
   home, and HALT.
3. **Measurement** — every parameter from `hyperspectral_measure.py`: scan axis,
   `start_pos` (mm), `step` (µm), `step_num`, camera trigger (internal/external),
   motor velocity (auto from exposure/ROI, or manual), refresh period, the
   interferogram probe pixel, and HDF5 saving (sample name + directory). Shows
   the live measurement image and the interferogram plot in tabs.

## Multithreading

All hardware I/O runs in background `QThread`s (`core/acquisition.py`) so the Qt
event loop and the live image stay responsive. The hot paths are DCAM camera
reads and PI GCS moves — both spend their time inside C-extension calls that
release the Python GIL, so these threads run concurrently across cores. (Pure
Python is still GIL-bound, but none of the hot paths here are pure Python.)

## Setup

Use your existing `scopefoundry` conda environment (it already has PyQt5, numpy,
pyqtgraph and h5py), or:

```
pip install -r requirements.txt
```

Hardware drivers (Windows): the Hamamatsu **DCAM4 SDK** and the **PI GCS /
pipython** USB drivers must be installed. `PI_VC_device.py` and
`CameraDevice.py` are imported from their sibling `ScopeFoundry Projects`
folders automatically.

## Running

```bash
python main.py            # simulation mode — no hardware needed
python main.py --real     # real Hamamatsu + PI hardware
```

For real hardware, edit `make_real_hardware()` in `main.py` (PI serial / axis
and camera defaults).

## How the code maps to your existing scripts

| Your code | This app |
|---|---|
| `PI_VC_Device(serial, axis)` | `PIStageWrapper.connect()` |
| `motor.move_absolute(mm)` / `move_relative(µm)` | stage jog / "Go", scan worker |
| `motor.set_velocity(v)` / `wait_on_target()` | velocity "Apply", inside workers |
| `motor.trigger(...)` / `trigger_disable(...)` | external-trigger scan branch |
| `hamamatsu.setExposure/setBinning/setSubarray*` | camera panel controls |
| `startAcquisition → getLastFrame → stopAcquisition` | `LiveViewWorker`, internal scan |
| `startAcquisition → getFrames → stopAcquisitionNotReleasing` | external-trigger scan |
| `hyperspectral_measure.measure()` (internal + external) | `HyperSpectralWorker` |
| `h5_io` image/position datasets | `HyperSpectralWorker._create_h5` (h5py) |
