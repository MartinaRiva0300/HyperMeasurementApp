# SWIR Hyperspectral Camera

A PyQt6 acquisition + analysis application for **VIS-SWIR hyperspectral
imaging**: a Teledyne FLIR **Forge 1GigE SWIR** camera (1.3 MP, C-mount, Sony
IMX990 SenSWIR InGaAs, 1280×1024 @ 5 µm) combined with a **NIREOS TWINS**
common-path birefringent interferometer. The TWINS wedge is stepped by a SmarAct
**SLC-1750** closed-loop linear piezo stage, N frames are grabbed per step, and a
per-pixel interferogram is built and Fourier-transformed into a spectral cube.
Band of interest **~0.4–1.7 µm** (the IMX990 cut-off).

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

**TWINS**
- Connect / go-to / jog for the SLC-1750 wedge stage, with a no-hardware
  "Simulate" mode.
- Live 1-D interferogram scan (single-pixel / ROI-average) with its FFT.

**Hyperspectral measurement (Measure tab)**
- Steps the TWINS wedge, grabs frame stacks, computes a **per-pixel DFT** →
  spectral cube, motor-nonlinearity calibrated, auto-saved on completion.
- The **whole acquired interferogram** is always transformed. Apodization type
  and width, apodization-centre method, walk-off correction, saturation masking,
  and a **Recompute** button that re-runs the DFT on the stored raw interferogram
  with new settings — no re-scan.
- **Apod centre** picks where the apodization window sits:
  - `barycentre (per-pixel)` — each pixel's own I² centroid (default), so a ZPD
    that drifts across the field is followed pixel by pixel;
  - `envelope (field)` — one Hilbert-envelope centre-burst for the whole frame;
  - `geometric centre` — the midpoint sample of the scan, ignoring the signal
    entirely (use when the scan is already deliberately centred on ZPD).

  The first two are found from the acquired data — no expected ZPD position is
  assumed.
- **Save complex spectrum (keep phase)** — writes the cube as `complex64`
  (**float32 real + float32 imag**), keeping the interferometric phase alongside
  the amplitude instead of the `float32` magnitude alone. The precision is
  float32 either way; the cube grows from 4 to 8 bytes per element only because
  two numbers are stored per element instead of one — that is the smallest form
  that can carry phase. Both save paths pin the dtype to `complex64`, so nothing
  can upcast to `complex128`. The viewer, the maps and the ROI-average CSV always
  display `|spectrum|`, so nothing changes on screen. Off by default.
- **Save format** is HDF5 only: every run writes the two MATLAB-compatible
  hypercube files — see below.
- Built-in **HyperViewer**: λ-scrub / peak-λ / peak-intensity / SAM /
  **continuum-line** maps, per-pixel spectra, colormaps.

**Analysis**
- Hypercubes are analysed in external, pre-existing tools that read the saved
  HDF5 files (see below). A lightweight in-repo cube viewer
  (`view_hyperspectral.py`) can also open past scans.

## Save formats

Every run writes **two HDF5 files** into its run folder, in the layout the lab's
pre-existing MATLAB analysis codes expect. Spatial axes come first in both cubes.
HDF5 is the only format (it needs `h5py`; saving reports a clear error without
it). The ROI-average CSV is unaffected. The Measure tab's Load button reads the
spectral file back for viewing.

**`<run-stamp>.<filename>_hyp.h5`** — temporal hypercube, in the `HyperMatrix` /
DelayCorrection layout the pre-existing MATLAB code reads:

```
/measurement/hyper/settings        this app's measurement settings (attrs)
/measurement/hyper/t0/c0/image        interferogram cube, stored (n_pos, y, x) so
                                      MATLAB's h5read returns (x, y, motor position);
                                      attr element_size_um = [z, y, x]
/measurement/hyper/t0/c0/position_mm  the wedge axis actually used: the motor-
                                      corrected positions when the correction file
                                      (parameters_int.txt) is loaded, else the raw
                                      measured positions. Stored in MICROMETRES
                                      (name kept; units attr = "um"); attrs axis,
                                      calibration_file
/measurement/hyper/t0/c0/position_mm_raw  the raw non-corrected positions (µm),
                                      written ONLY when the correction file is
                                      loaded (else position_mm already IS the raw axis)
```

**`<run-stamp>.<filename>_SpectralHypercube.h5`** — spectral hypercube, in the
pre-existing MATLAB `SpectralHypercube` layout:

```
/SpectralHypercube/Hyperspectrum_cube  spectra, float32, stored (slices, y, x) so
                                       MATLAB's h5read returns (x, y, slices):
                                       (n_freq, y, x) for the magnitude, or
                                       (2*n_freq, y, x) when "Save complex spectrum"
                                       is on = real slices then imaginary slices
/SpectralHypercube/fr_real             optical frequency c/λ in THz, (n_freq, 1) float64
/SpectralHypercube/f                   stage pseudo-frequency axis (after the FT
                                       over motor positions), (n_freq, 1) float64
/SpectralHypercube/saturationMap       (y, x) float64; 1 = valid pixel, 0 = saturated
/file_totCal                           spectral calibration file path (at the ROOT)
```

The optical frequency is `fr_real[THz] = 299.792458 / λ[µm]`; the wavelength axis
is recoverable as `λ = 299.792458 / fr_real`. The FT of a real interferogram is
complex; "Save complex spectrum" chooses whether to keep the phase. When on, the
cube is a **real float32 stack** — the `n_freq` real slices followed by the
`n_freq` imaginary slices (so `2*n_freq` slices, while `f`/`fr_real` keep `n_freq`)
— matching the lab's MATLAB reader; when off it is the float32 `|spectrum|`. The
scan/spectrum settings are recorded once, in the temporal file's
`/measurement/hyper/settings`.

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
  measure_panel.py     the hyperspectral experiment + HyperViewer
instruments/          drivers + shared DSP
  twins_stage.py        SmarAct MCS2 driver for the SLC-1750 wedge stage
  subtwinslv.py         step-scan engine (scan / scan_cube)
  hyperspectral.py      2-D per-pixel DFT (compute_hyperspectral)
  spectrum_processor.py 1-D interferogram -> spectrum
  h5_writer.py          ScopeFoundry-layout HDF5 save/load
  calibration.py dsp.py analysis.py walkoff.py   shared processing
Twins/calibration/      parameters_{cal,int}.txt  spectral + motor calibration
selftest_acquisition.py  headless mock-camera + simulated-stage acquisition test
dump_h5_layout.py     print the exact HDF5 layout a measurement produces
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
ALternatively with conda
```bat
create -n env_name python=3.12
conda activate env_name
```
Installing dependencies
'''bat
pip install PyQt6, loguru
conda install -c conda-forge pyqtgraph
pip install "C:\yourDirectory\spinnaker_python-4.4.0.246-cp312-cp312-win_amd64\spinnaker_python-4.4.0.246-cp312-cp312-win_amd64.whl“
pip install wheels/smaract_ctl-1.5.3-py3-none-any.whl 
'''
Alternatively for dependencies
'''bat
pip install -r requirements.txt
pip install "C:\yourDirectory\spinnaker_python-4.4.0.246-cp312-cp312-win_amd64\spinnaker_python-4.4.0.246-cp312-cp312-win_amd64.whl“
'''

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
python main.py --mode mock #mock version for camera
python main.py # real camera connected
```

## Before the first real measurement

Two things in this repo are carried over from the MWIR rig and **must be set for
this instrument**:

1. **`Twins/calibration/parameters_cal.txt`** covers **1.50–15.79 µm**, i.e.
   it is the MWIR/LWIR TWINS calibration and only overlaps the SWIR band above
   1.5 µm. Replace it with the calibration for the TWINS unit used here, or the
   wavelength axis will be wrong.
2. **`HOME_POSITION_MM` / `SAFE_POSITION_MM` / `TRAVEL_MM`**
   (`instruments/twins_stage.py`) are placeholders — `TRAVEL_MM = 50.0` assumes a
   50 mm SLC-1750. Check the label on the positioner and set the park positions
   to match your wedge mount.

You still need to know roughly where ZPD is in order to choose the scan
Start/Stop, but the app no longer assumes a value: the apodization centre is
located in whatever you acquired.

Nyquist also bites harder in SWIR than MWIR: resolving 0.9 µm needs a far finer
wedge step than 4 µm did. The Measure tab shows the maximum permitted step and
turns it **red** when the chosen step under-samples the shortest wavelength.

## Hardware notes

### Camera

- The camera is **single-access**: close SpinView (or any other Spinnaker
  client) before connecting, and close the app with its window **X**
  (force-killing can orphan the worker process).
- Frames are pulled as **Mono16**. The IMX990 ADC is 12-bit, so counts normally
  occupy 0–4095. Check the **PixelFormat** and **ADC Bit Depth** shown in the
  camera status line and set the Measure tab's saturation level accordingly.
- **ADC Bit Depth** determines the number of digitization levels provided by
  the sensor. It is distinct from the pixel format size. For example, a 12-bit
  ADC provides 4096 possible values. When stored in a 16-bit format, the image
  data is left-aligned, so a 12-bit value may occupy the most significant bits
  of the 16-bit word.
- The camera supports different **PixelFormat** options depending on the
  sensor. Pixel format determines how pixel data is provided by the camera,
  including its bit depth and, for color cameras, the Bayer color filter
  information. The application's acquisition uses **Mono16**.
- **1 GigE caps the full-frame rate at roughly 42 Hz** (1280×1024, Mono16).
  Enable **jumbo frames (9000 B MTU)** on the camera's NIC — the backend asks
  for a 9000-byte packet size and silently falls back if the adapter refuses,
  which shows up as dropped/incomplete frames rather than an error.
- **Region of Interest (ROI)** can be used to restrict acquisition to a
  selected portion of the sensor. Only the pixels within the specified ROI are
  then processed, reducing the amount of image data that needs to be transferred
  and processed.
- **Binning** combines the signal from groups of neighboring photosensitive
  cells into a larger logical pixel. Depending on the selected binning mode,
  the signals can be **summed** (additive) or **averaged**. Summation increases
  sensitivity, while averaging can improve the signal-to-noise ratio.
  Binning settings can only be changed while the camera is not streaming. For this camera, only the software binning is available.
  **Binning and decimation cannot be active simultaneously.**
- Spinnaker has no on-camera frame averaging, so the **Averaging** control
  averages N frames in software; the effective rate drops by N.
  - **Gain** controls the amplification applied to the pixel signal during
  analog-to-digital conversion. Increasing gain produces a brighter image but
  also increases noise. The `Gain` control sets the amplification in dB. For this camera, the gain will be used in the Manual Mode. 
- **Black Level** controls the offset applied to the video signal and determines
  the image baseline in the absence of illumination. The total black level
  includes both analog and digital contributions. For reproducible
  measurements, keep the black level fixed between acquisitions; the user
  controls the total offset through the `All` Black Level selector.
Details available at this page: [FG-PGE-13S3S-U Technical Reference](https://www.teledynevisionsolutions.com/en-hk/products/forge-1gige-swir/?model=FG-PGE-13S3S-U-C&vertical=machine%20vision&segment=iis)

### Stage

- The stage **references itself on connect** and then parks at
  `HOME_POSITION_MM`, so connecting moves the wedge. Disconnecting leaves it
  where it is.

### Network connection

- Camera IP is auto-discovered; no IP needs to be set. If enumeration finds
  nothing, check the NIC is on the same subnet and that the Spinnaker GigE
  filter driver is installed.
