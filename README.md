# SWIR Hyperspectral Camera

A PyQt6 acquisition + analysis application for **static SWIR hyperspectral
imaging**: a Teledyne FLIR **Forge 1GigE SWIR** camera (1.3 MP, C-mount, Sony
IMX990 SenSWIR InGaAs, 1280×1024 @ 5 µm) combined with a **NIREOS TWINS**
common-path birefringent interferometer. The TWINS wedge is stepped by a SmarAct
**SLC-1750** closed-loop linear piezo stage, N frames are grabbed per step, and a
per-pixel interferogram is built and Fourier-transformed into a spectral cube.
Band of interest **~0.9–1.7 µm** (the IMX990 cut-off).

The app synchronises exactly two instruments — **the camera and the wedge
stage** — and nothing else. One `Acquire` = one wedge sweep = one hyperspectral
cube.

> The camera talks GigE Vision through Teledyne's **Spinnaker SDK** Python
> bindings (`PySpin`). The stage talks to a SmarAct **MCS2** controller through
> `smaract.ctl`. A synthetic **mock** camera plus the stage panel's "Simulate"
> checkbox let the whole app run with no hardware — `--mode mock` needs only the
> pip packages.

## Features

**Live camera**
- Live image with Inferno / Viridis / Magma / Grey / Turbo / Coolwarm colormaps.
- Auto or fixed Min/Max colorbar; click a pixel for horizontal/vertical profiles.
- Exposure 0.001–1000 ms, software frame averaging, background capture /
  subtraction, snapshot, camera temperature readout.
- Forge GenICam options exposed directly: `ExposureAuto`, `GainAuto`,
  `PixelFormat`, `AdcBitDepth`, `AcquisitionFrameRate`.
- On-image draggable **ROI** + binning, shared by the measurement panels.
- Software **bad-pixel mask** (auto-detect, paint, save/load).

**TWINS**
- Connect / go-to / jog for the SLC-1750 wedge stage, with a no-hardware
  "Simulate" mode.
- Live 1-D interferogram scan (single-pixel / ROI-average) with its FFT.

**Hyperspectral measurement (Measure tab)**
- Steps the TWINS wedge, grabs frame stacks, computes a **per-pixel DFT** →
  spectral cube, motor-nonlinearity calibrated, auto-saved on completion.
- Apodization, FT region/width, ZPD centring, walk-off correction, saturation
  masking, SVD denoise, and a **Recompute** button that re-runs the DFT on the
  stored raw interferogram with new settings — no re-scan.
- **Save format** selectable per run: NumPy `.npz` (default) or **HDF5** in the
  ScopeFoundry layout — see below.
- Built-in **HyperViewer**: λ-scrub / peak-λ / peak-intensity / SAM /
  **continuum-line** maps, per-pixel spectra, colormaps.

**Analysis**
- A standalone analyzer (`analysis_app.py`), a Stokes-polarimetry app
  (`stokes_app.py`, `stokes_maps_app.py`) and a lightweight cube viewer
  (`view_hyperspectral.py`) for the saved `.npz` hypercubes.

## Save formats

The Measure tab's **Format** box picks how the full cube is written. The
ROI-average CSV is unaffected by it.

**NumPy `.npz` (default)** — `<run-stamp>.<filename>.npz` in the run folder.
This is what `analysis_app.py`, `view_hyperspectral.py` and the Measure tab's own
Load button read, so leave it selected unless you specifically need HDF5.

**HDF5 `.h5`** — `<yymmdd_HHMMSS>_hyperspectral[_<filename>].h5`, written in the
ScopeFoundry `h5_io` layout so the lab's other tooling can read it. The Filename
box supplies the **sample** name; the timestamp is taken from the run stamp so
the file and its run folder always agree.

```
/                                    attrs: created, time_id, sample, measurement
/app/settings                        attrs: save_dir, sample
/measurement/hyperspectral/
    settings/                        every scan setting, as attributes
    t0/c0/image                      (n_pos, h, w)  the raw interferogram
                                     attrs: element_size_um = [z, y, x]
    t0/c0/position_mm                (n_pos,) float32  measured wedge axis
    t0/c0/position_mm_calibrated     (n_pos,) the axis the DFT actually used
    spectrum/wavelengths             (n_freq,) µm
    spectrum/cube                    (1, n_freq, h, w) float32
    masks/saturation                 (1, h, w) bool          [if enabled]
    background/map                   (h, w)   attrs: subtracted   [if captured]
    acquisition                      attrs: roi, binning, n_positions
    metadata_json                    the complete metadata, verbatim
```

`t0/c0/image` is this app's raw interferogram cube and `t0/c0/position_mm` its
raw positions, exactly as in the ScopeFoundry structure. Two notes:

- `element_size_um` defaults to **[1, 1, 1]**. Set `DEFAULT_ELEMENT_SIZE_UM` in
  `instruments/h5_writer.py` if you want a physical voxel size in Fiji (z = wedge
  step µm, y/x = 5 µm pitch × binning ÷ magnification). The true step size and
  binning are in the metadata regardless.
- `t0/c0/image` is **float32**, not the camera's uint16: frames are averaged and
  the background subtracted before the cube is stored, so it is the processed
  interferogram rather than raw counts.

HDF5 needs `h5py`; without it the format simply reports an error and `.npz`
keeps working. Because the raw interferogram *is* the primary dataset, a cube
loaded from an old file (which carries no raw data) cannot be re-saved as HDF5.

## Layout

```
main.py               entry point  (--mode forge|mock|auto, --fps N)
worker_camera.py      camera worker process (frames -> shared memory + queue)
camera/               camera abstraction + backends
  camera_interface.py   CameraInterface ABC + CameraStatus/MeasurementResult
  forge_camera.py       Forge 1GigE SWIR backend (Spinnaker / PySpin)
  mock_camera.py        synthetic drifting-beam camera (no hardware)
  factory.py            create_camera(mode)
ui/
  main_window.py        orchestrator: live view, controls, background, ROI, save
  stages.py             TWINS wedge-stage control panel
  twins_scan.py         live 1-D TWINS interferogram scan
  measure_kspace.py     the hyperspectral experiment + HyperViewer
instruments/          drivers + shared DSP
  twins_stage.py        SmarAct MCS2 driver for the SLC-1750 wedge stage
  subtwinslv.py         step-scan engine (scan / scan_cube)
  hyperspectral.py      2-D per-pixel DFT (compute_hyperspectral)
  spectrum_processor.py 1-D interferogram -> spectrum
  h5_writer.py          ScopeFoundry-layout HDF5 save/load
  calibration.py dsp.py analysis.py walkoff.py   shared processing
Twins/calibration/      parameters_{cal,int}.txt  spectral + motor calibration
selftest_acquisition.py  headless mock-camera + simulated-stage acquisition test
docs/                 ACQUISITION_APP.md (architecture) + CONTINUUM_SUBTRACTION.md
```

See **[docs/ACQUISITION_APP.md](docs/ACQUISITION_APP.md)** for a full technical
walk-through of the architecture, and
**[docs/CONTINUUM_SUBTRACTION.md](docs/CONTINUUM_SUBTRACTION.md)** for the
per-pixel continuum-subtraction method used to isolate the resonant line image.

## Setup (one time)

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

64-bit Python **3.10 or 3.12** — those are the versions Teledyne ships PySpin
wheels for. `mock` mode needs only the pip packages. The real hardware
additionally needs:

- **Spinnaker Python bindings** (the camera) — not on PyPI, install the wheel
  matching your Python:
  `pip install spinnaker_python-4.4.0.246-cp312-cp312-win_amd64.whl`
  The wheel bundles the Spinnaker runtime DLLs, so `import PySpin` works on its
  own. Also run the matching **Spinnaker SDK installer** (drivers + VS
  redistributables) so the GigE filter driver and jumbo frames are available.
- **SmarAct MCS2 SDK** (the stage) — not on PyPI:
  `pip install "C:\Program Files\SmarAct\MCS2\SDK\Python\packages\smaract_ctl-1.6.2.zip"`

Do **not** pip-install PyQt6 on a conda Python — conda's Qt6 shadows pip's and
`import PyQt6` fails with "the specified procedure could not be found". Use a
python.org venv, or install Qt from conda too.

## Run

```bat
run.bat                                       REM main.py --mode forge --fps 60
run_mock.bat                                  REM no camera/stage needed
view.bat  path\to\cube.npz                    REM standalone cube viewer (.npz)
analyze.bat                                   REM standalone analyzer
.venv\Scripts\python selftest_acquisition.py  REM headless end-to-end check
```

## Before the first real measurement

Three things in this repo are carried over from the MWIR rig and **must be set
for this instrument** — the app will run without them, but the wavelength axis
will be wrong:

1. **`Twins/calibration/parameters_cal.txt`** covers **1.50–15.79 µm**, i.e.
   it is the MWIR/LWIR TWINS calibration and only overlaps the SWIR band above
   1.5 µm. Replace it with the calibration for the TWINS unit used here.
2. **`DEFAULT_ZPD_MM = 24.33`** (`instruments/hyperspectral.py`) is the MWIR
   rig's zero-path-difference wedge position. Find the ZPD for this
   interferometer and set it, along with the default scan Start/Stop.
3. **`HOME_POSITION_MM` / `SAFE_POSITION_MM` / `TRAVEL_MM`**
   (`instruments/twins_stage.py`) are placeholders — `TRAVEL_MM = 50.0` assumes a
   50 mm SLC-1750. Check the label on the positioner and set the park positions
   to match your wedge mount.

Nyquist also bites harder in SWIR than MWIR: resolving 0.9 µm needs a far finer
wedge step than 4 µm did. The Measure tab shows the maximum permitted step and
turns it **red** when the chosen step under-samples the shortest wavelength.

## Hardware notes

- The camera is **single-access**: close SpinView (or any other Spinnaker
  client) before connecting, and close the app with its window **X**
  (force-killing can orphan the worker process).
- Frames are pulled as **Mono16**. The IMX990 ADC is 12-bit, so counts normally
  occupy 0–4095 — check the pixel format / ADC depth shown in the camera status
  line and set the Measure tab's saturation level to match.
- **1 GigE caps the full-frame rate at roughly 42 Hz** (1280×1024, Mono16).
  Enable **jumbo frames (9000 B MTU)** on the camera's NIC — the backend asks
  for a 9000-byte packet size and silently falls back if the adapter refuses,
  which shows up as dropped/incomplete frames rather than an error.
- Spinnaker has no on-camera frame averaging, so the **Averaging** control
  averages N frames in software; the effective rate drops by N.
- The stage **references itself on connect** and then parks at
  `HOME_POSITION_MM`, so connecting moves the wedge. Disconnecting leaves it
  where it is.
- Camera IP is auto-discovered; no IP needs to be set. If enumeration finds
  nothing, check the NIC is on the same subnet and that the Spinnaker GigE
  filter driver is installed.
