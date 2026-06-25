# Microscope Control App

PyQt5 desktop application for coordinated control of:
- **Hamamatsu Orca Flash 4V3** camera (via DCAM4 / `CameraDevice.py`)
- **PI motorised stage** (via `pipython` / `PI_VC_device.py`)

---

## Project structure

```
microscope_app/
├── main.py                  ← entry point
├── requirements.txt
├── hardware/
│   ├── camera.py            ← HamamatsuCamera wrapper (uses CameraDevice.py)
│   └── stage.py             ← PIStageWrapper + StageManager (uses PI_VC_device.py)
├── core/
│   └── acquisition.py       ← QThread workers: live view, polling, jog, sequence
└── ui/
    ├── camera_panel.py      ← live feed, exposure, trigger
    ├── stage_panel.py       ← per-axis readout, jog (µm), go-to (mm), velocity
    ├── acquisition_panel.py ← position list, run sequence, save .npy
    └── main_window.py       ← assembles all panels
```

## Setup

1. Copy `PI_VC_device.py` and `CameraDevice.py` into `microscope_app/`
   (or any folder on `sys.path`).

2. Install dependencies:
   ```
   pip install PyQt5 numpy pipython
   ```
   The Hamamatsu camera requires the DCAM4 SDK installed on Windows.

3. Edit `main.py` → `make_real_hardware()`:
   - Set the USB serial number(s) for your PI stage axis/axes.
   - Adjust camera parameters (frame size, binning, exposure) if needed.

## Running

```bash
# Simulation mode (no hardware needed)
python main.py

# Real hardware
python main.py --real
```

## How the code maps to your existing scripts

| Your code | This app |
|---|---|
| `PI_VC_Device(serial, axis)` | `PIStageWrapper.connect()` |
| `motor.move_absolute(mm)` | stage jog "Go" button / acquisition worker |
| `motor.move_relative(µm)` | stage jog ±step buttons |
| `motor.wait_on_target()` | inside `JogWorker.run()` and `AcquisitionWorker.run()` |
| `motor.set_velocity(v)` | "Apply" button in Stage panel |
| `motor.set_home()` | "Set Home" button |
| `hamamatsu.startAcquisition()` → `getLastFrame()` | `LiveViewWorker` (live preview) |
| `hamamatsu.startAcquisition()` → `getLastFrame()` → `stopAcquisition()` | `camera.snap()` (per position) |
| `hyperspectral_measure.measure()` internal-trigger loop | `AcquisitionWorker.run()` |
