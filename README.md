# Hyperspectral Control App — Retiga E7 + PI

Standalone PyQt5 desktop application for coordinated control of:
- **Teledyne Retiga E7** camera (via PVCAM / PyVCAM, `Teledyne_ScopeFoundry/CameraDevice.py` → `PVcamDevice`)
- **PI motorised stage** (via `pipython`, `PI_ScopeFoundry/PI_VC_device.py` → `PI_VC_Device`)

It is the non-ScopeFoundry version of
[Hyperspectral_microscope (branch RetigaE7_PI)](https://github.com/PolimiScopeFoundry/Hyperspectral_microscope/tree/RetigaE7_PI):
the device classes are reused untouched and the measurement logic of
`hyperspectral_measure.py` is ported 1:1 into a background worker; only the
user interface is new.

---

## Project structure

```
HyperMeasurementApp/
├── main.py                  ← entry point (--real for hardware, default = mock)
├── requirements.txt
├── hardware/
│   ├── camera.py            ← RetigaCamera wrapper (PVcamDevice) + MockCamera
│   └── stage.py             ← PIStageWrapper + StageManager + MockPIStage
├── core/
│   └── acquisition.py       ← QThread workers (live, poller, jog, HyperSpectralWorker)
└── ui/
    ├── widgets.py           ← shared pyqtgraph ImageView helper
    ├── camera_panel.py      ← exposure, binning, readout, gain, trigger, ROI
    ├── stage_panel.py       ← encoder read-back, to-go position, jog (µm), velocity
    ├── measurement_panel.py ← hyperspectral scan parameters + positions file + HDF5
    ├── live_view_panel.py   ← the single live image window (+ cursor profiles)
    ├── interferogram_panel.py ← intensity at the probe pixel vs position
    └── main_window.py       ← assembles & coordinates the panels
```

The app folder must sit **next to** `Teledyne_ScopeFoundry` and `PI_ScopeFoundry`
(the "ScopeFoundry Projects" root), exactly like the ScopeFoundry app: the
device classes are imported from those sibling folders.

## The three panels

1. **Camera** — exposure (integer ms), binning (1/2/4), readout port
   (0 Speed / 1 Long Exposure / 2 Long Exposure EDR), gain (1 Full Well /
   2 Sensitivity, readout 1–2 only), trigger mode (the five PVCAM exposure
   modes), ROI (h0, v0, width, height in sensor px), live view and snap.
   Binning/ROI/readout/gain/trigger are locked while live (the camera must be
   idle to change them); changing the exposure during live restarts the live
   acquisition. Read-back shows the camera name, sensor temperature and the
   estimated maximum frame rate for the current ROI/exposure.
2. **Stage** — per-axis current position (encoder read-back), an independent
   "Go to" target, relative jog in µm, velocity, set-home, home, and HALT.
3. **Measurement** — every parameter from `hyperspectral_measure.py`: scan axis,
   `start_pos` (mm), `step` (µm), `step_num`, camera trigger (Internal / Edge),
   positions loaded from a text file (first column, mm; Internal trigger only),
   motor velocity (auto from exposure/ROI, or manual), refresh period, the
   interferogram probe pixel, and HDF5 saving (sample name + directory).

## How the scan works (same as `measure()`)

| Trigger | Sequence |
|---|---|
| **Internal Trigger** | for every position: `move_absolute` → `wait_on_target` → `get_position` → `acq_start_seq(1)` → `get_nparray` → `acq_stop`. Positions are `start + i·step` (i = 0 … n−1) or the loaded list. |
| **Edge Trigger** | camera in `Edge Trigger`, `acq_start_seq(n)`; the stage moves to `start − 0.05 mm`, the scan velocity is set (`step/(4·exposure)`, capped at 75 % of the max frame rate from the Retiga readout-time model), `motor.trigger(step, start, stop, 1, 4)` and a single move to `stop + 0.05 mm`; frames are polled as the n triggers fire (positions = `linspace(start, stop, n)`). Afterwards the trigger is disabled, `Internal Trigger` and the fast velocity (5 mm/s) are restored. |

HDF5 layout matches the ScopeFoundry `h5_io` file: `measurement/hyper/t0/c0/image`
(`[frames, rows, cols]`, `element_size_um` attribute) and
`measurement/hyper/t0/c0/position_mm` (positions in µm, as in the original).
Scan and camera settings are stored as attributes of `measurement/hyper`.
Datasets are trimmed to the frames actually acquired.

## Multithreading

All hardware I/O runs in background `QThread`s (`core/acquisition.py`) so the
Qt event loop and the live image stay responsive. The hot paths are PVCAM frame
polls and PI GCS moves, both inside C-extension calls that release the GIL.

## Setup

Use the conda environment where PyVCAM and pipython are installed (e.g.
`retigaE7`), or:

```
pip install -r requirements.txt
```

Hardware drivers (Windows): **PVCAM + PVCAM SDK** with the **PyVCAM** wrapper
(`pip install .` inside the PyVCAM folder), and the **PI GCS / pipython** USB
drivers. `CameraDevice.py` and `PI_VC_device.py` are imported from their sibling
`ScopeFoundry Projects` folders automatically.

## Running

```bash
python main.py            # simulation mode — no hardware needed
python main.py --real     # real Retiga E7 + PI hardware
```

For real hardware, edit `make_real_hardware()` in `main.py` (PI serial / axis
and camera defaults; the default serial is the one of `hyperspectral_app.py`).

## How the code maps to the ScopeFoundry scripts

| ScopeFoundry code | This app |
|---|---|
| `PVcamHW` settings (exposure_time, binning, readout, gain, trigger_mode, subarray_*) | `RetigaCamera.set_*` ← Camera panel |
| `cam.acq_start` / `get_nparray` / `acq_stop` | `LiveViewWorker` (live view) |
| `cam.acq_start_seq(1)` → `get_nparray` → `acq_stop` | `RetigaCamera.snap`, internal scan step |
| `cam.acq_start_seq(n)` + `poll_frame` (Edge Trigger) | external-trigger scan branch |
| `PI_VC_Device(serial, axis)` | `PIStageWrapper.connect()` |
| `motor.move_absolute(mm)` / `move_relative(µm)` | stage jog / "Go", scan worker |
| `motor.set_velocity(v)` / `wait_on_target()` | velocity "Apply", inside workers |
| `motor.trigger(...)` / `trigger_disable(...)` | external-trigger scan branch |
| `hyperMeasure.measure()` (Internal + Edge) | `HyperSpectralWorker` |
| `hyperMeasure.set_motor_velocity()` | `HyperSpectralWorker._scan_velocity` + `hardware.camera.max_frame_rate_hz` |
| `hyperMeasure.load_positions()` | "Load positions…" in the Measurement panel |
| `h5_io` image/position datasets | `HyperSpectralWorker._create_h5` (h5py) |
