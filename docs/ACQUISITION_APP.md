# SWIR Acquisition App: technical handoff

Audience: an AI agent (or developer) taking over the **live acquisition app**. This
explains the architecture and the *logic*, not every line. The analyzer is
documented separately (`analysis_app.py`).

> Scope: **static hyperspectral imaging** — a Teledyne FLIR **Forge 1GigE SWIR**
> camera (Sony IMX990 SenSWIR, 1280×1024 @ 5 µm) + a **NIREOS TWINS** common-path
> birefringent interferometer whose wedge is driven by a SmarAct **SLC-1750**
> closed-loop linear piezo stage. Step the wedge, grab N frames per step, build an
> interferogram per pixel, per-pixel DFT → spectral cube. SWIR band **~0.9–1.7 µm**.
> NOT pump-probe (no lock-in).
>
> **The app synchronises exactly two instruments: the camera and the wedge stage.**
> There is no Z stage and no rotator, and therefore no Z-scan / angle-scan grid —
> one Acquire = one wedge sweep = one cube.
>
> ⚠️ **Carried over from the MWIR rig and not yet re-measured for this instrument:**
> the ZPD default (`DEFAULT_ZPD_MM = 24.33`), the spectral calibration table
> (`parameters_cal.txt`, which covers 1.50–15.79 µm), and the stage park positions
> in `instruments/twins_stage.py`. See the README's "Before the first real
> measurement".

Line numbers below are anchors at time of writing; trust **names** over numbers.

---

## 1. Big picture

```
main.py  ──spawns──►  camera_worker (separate PROCESS)  ──► ForgeSwirCamera (PySpin) or MockCamera
   │                        │  frames → shared memory + frame_queue;  status → frame_queue
   │ builds                 │  commands ◄── control_queue
   ▼                        ▼
MainWindow (ui/main_window.py)  ── QTimer 8 ms poll → self.latest_frame (fresh .copy each frame)
   ├── Camera tab      (exposure, GenICam opts, bad-pixel mask, ROI, colormap, background, save)
   ├── TWINS tab       (StagesPanel.twins_group  +  TwinsScanPanel: live 1-D scan)
   └── Measure tab     (MeasurePanel: the K-space HYPERSPECTRAL experiment)
                          frame_source = lambda: self.latest_frame
                          bg_provider, save_dir_provider, meta_provider, roi_provider
```

Three subsystems:
1. **Camera** — its own OS process; frames via shared memory + a queue; commands via a queue.
2. **Main window** — polls frames, displays them, owns `latest_frame`, exposes camera controls, background, ROI, save dir/filename, and the camera-metadata provider.
3. **Stage + acquisition** — the TWINS wedge driver, the live 1-D TWINS scan, and the **MeasurePanel** hyperspectral experiment (the main scientific output).

Two processing depths share the same calibration + math:
- **1-D** (`instruments/spectrum_processor.py`) — one ROI-mean value per wedge step → one spectrum (the TWINS tab live scan).
- **2-D** (`instruments/hyperspectral.py`) — full ROI image per wedge step → per-pixel spectral cube (the Measure tab).

---

## 2. Camera subsystem

Files: `worker_camera.py`, `camera/camera_interface.py`, `camera/factory.py`,
`camera/forge_camera.py`, `camera/mock_camera.py`.

### Process & transport
- The camera runs in a **separate `multiprocessing.Process`** (`camera_worker()` in
  `worker_camera.py`), spawned by `main.py`. Rationale: GenICam/Spinnaker grabbing must not
  stall the Qt event loop.
- `main.py` creates: `frame_queue = mp.Queue(maxsize=4)`, `control_queue = mp.Queue()`,
  and a **`SharedMemory`** block sized `(2048, 2048)` uint16 (over-allocated well past the
  Forge's 1280×1024). Config dict carries `mode`, `target_fps`, and the shm name/shape.
- **Frames → UI:** worker copies the frame into shared memory and pushes a packet
  `{"type":"frame","shape":(h,w),"shared":True,"measurement":{...}}` onto `frame_queue`.
  If shared memory is absent it embeds `"frame": ndarray` instead. Queue-full → drop oldest.
- **Status → UI:** same `frame_queue`, `{"type":"status","status":{...}}` (connected,
  acquiring, backend, serial, width/height, exposure_ms, average_count, board_temp_c,
  fpa_temp_k, message). Pushed when idle / after commands / on temp reads.
- **Commands UI → worker:** `control_queue`, dicts `{"type":..., "value":...}`:
  `start`, `pause`, `stop`, `set_exposure`, `set_average`, `set_option` (any GenICam node),
  `connect`, `disconnect`, `read_temp`, `snapshot`.
- **Auto-heal:** if no frame for ~4 s while streaming, worker `camera.reconnect()`
  (throttled ~6 s). Temps refreshed every ~5 min.

### Abstraction & backends
- `camera/camera_interface.py` — `CameraInterface` ABC: `connect/disconnect/
  start_acquisition/stop_acquisition/get_frame/set_exposure/get_status` (+ optional
  `set_average`). Dataclasses `CameraStatus`, `MeasurementResult`.
- `camera/factory.py` — `create_camera(mode)`: `forge`/`auto` → try `ForgeSwirCamera`,
  fall back to `MockCamera` if `PySpin` is missing; `mock` → `MockCamera`.
- `camera/forge_camera.py` — the real backend:
  - **Spinnaker SDK 4.4** GenICam via **PySpin** (the wheel bundles the runtime DLLs, so
    `import PySpin` needs no separate install; the SDK installer is still wanted for the
    GigE filter driver). `System.GetInstance()` -> `GetCameras()` -> pick the device whose
    model/vendor matches `MODEL_HINTS` -> `Init()`.
  - Frames: `get_frame()` = `GetNextImage(timeout)` -> reject `IsIncomplete()` ->
    `GetNDArray()` **copied** (it is a view into Spinnaker's buffer) -> **uint16**.
    `Release()` in a `finally`. Stream buffer mode is **`NewestOnly`** so the live view
    and the scan always get "now", never a backlog.
  - `_configure()` (all best-effort, node availability varies with firmware):
    `TriggerMode=Off`, `AcquisitionMode=Continuous`, `ExposureAuto=Off`,
    `ExposureMode=Timed`, `GainAuto=Off`, `PixelFormat` = the first of
    `Mono16 / Mono12p / Mono12Packed / Mono8` that takes, `GevSCPSPacketSize=9000`
    (jumbo), `DeviceLinkThroughputLimitMode=Off`.
  - **Exposure**: standard GenICam `ExposureTime` in **microseconds**, clamped to the
    node's own `GetMin()/GetMax()`, and read back so the status shows what was applied.
    `ExposureAuto` is forced Off first or the node is read-only.
  - **Averaging is software** — Spinnaker has no on-camera frame averaging, so
    `set_average(N)` makes `get_frame()` pull and mean N frames (rate drops by N).
  - `set_option(name, value)` dispatches on the node's own interface type
    (enum/float/int/bool/command), so any GenICam node is reachable from the UI.
  - `refresh_temperatures()` reads `DeviceTemperature` -> `board_temp_c`; there is no
    cooled FPA, so `fpa_temp_k` stays NaN. `reconnect()` heals dropped GigE links.
  - **Teardown order matters**: the camera pointer must be dropped *before*
    `System.ReleaseInstance()`, else Spinnaker raises "reference count is not zero".
- `camera/mock_camera.py` — synthetic drifting Gaussian beam (320×256, clipped 12-bit),
  used when the camera or PySpin is unavailable. Lets the whole app run with no camera.

### Key camera facts for downstream code
- Frame dtype **uint16**. The IMX990 ADC is **12-bit**, so counts normally run 0–4095 —
  confirm against the pixel format / `AdcBitDepth` shown in the status line and set the
  Measure tab's saturation level to match (it is a UI setting, not a constant).
- Sensor size is read at runtime (Forge 1280×1024; mock 320×256). Don't hard-code it.
- **1 GigE caps the full-frame rate near 42 Hz** (1280×1024, Mono16).

---

## 3. Main window (`ui/main_window.py`) — the orchestrator

`MainWindow.__init__` attaches to the shared frame + queues, builds the UI, and starts
a **`QTimer` every ~8 ms** → `update_from_worker()`.

### Frame pipeline (the most important invariant)
- `update_from_worker()` drains `frame_queue`: status packets → `_apply_status`, frame
  packets → keep the latest. For a shared frame it slices `self.shared_frame[:h,:w].copy()`
  — **a fresh numpy array every poll**.
- `_apply_frame(frame, measurement)` sets **`self.latest_frame = frame`** (the single
  source of truth), handles background capture/subtraction for *display*, pushes the image
  to pyqtgraph, and every 3rd frame updates colorbar levels, X/Y profiles, crosshair,
  beam measurements, FPS.
- **Object-identity freshness:** because each new frame is a brand-new `.copy()`, scanners
  can detect "a new frame arrived" by checking `frame is last_frame`. The MeasurePanel /
  TwinsScanner averaging relies on this (see §6).
- **`frame_source = lambda: self.latest_frame`** is handed to both TwinsScanPanel and
  MeasurePanel. They call it whenever they need the current frame.

### Control surface → `control_queue`
Exposure slider/spin → `set_exposure`; averaging → `set_average`; Start/Pause/Snapshot;
Connect → `connect`; NUC checkbox → `set_correction`; BPR → `set_bpr`; NUC slot →
`set_nuc_slot`; Load state → `load_state`; Poll temp → `read_temp`. Colormap / auto-scale /
display range are display-only (don't touch the camera).

### ROI, background, save, metadata, status (the callbacks MeasurePanel depends on)
- **ROI**: a draggable cyan box on the image; `get_roi_bounds()` → `(r0,r1,c0,c1)` clipped,
  or `None` (full frame). Passed as `roi_provider`.
- **Background**: "Capture" averages `bg_average_frames` (16) into `self.background_frame`
  (float32). `bg_provider = lambda: (self.background_frame, self.use_bg_subtraction)`.
- **Save**: `save_dir_edit` (default `D:\CAMERA`) + `filename_edit`. `save_dir_provider`
  and the camera's "Save image" (TIFF uint16 + NPY + colormapped PNG, stamped
  `YYYYMMDD_HHMMSS.<name>`).
- **Metadata**: `meta_provider = self._kspace_metadata` returns
  `{camera_serial, exposure_ms, averaging, fpa_temp_k, board_temp_c, nuc_corrected,
  save_filename_camera}` from `self.latest_status` — embedded in saved hypercubes.
- **Status**: `_apply_status` stores `self.latest_status` and updates labels (temps shown
  as °C / K, NaN-guarded).

---

## 4. Stage (`ui/stages.py`, `instruments/twins_stage.py`)

`StagesPanel` owns the one driver and exposes a single QGroupBox (`twins_group`, placed in
the TWINS tab). It runs a 500 ms position-poll timer and wraps blocking moves in a
`StageController` (background thread + Qt signals) so the UI never freezes.

- **`self.twins` — `TwinsStage`** (NIREOS wedge, SmarAct **MCS2** controller driving an
  **SLC-1750** closed-loop linear positioner, via the `smaract.ctl` Python SDK).
  `connect(home=True)` opens the first discovered MCS2, configures the channel
  (`MOVE_MODE=CL_ABSOLUTE`, velocity, acceleration, hold time), **references** the
  positioner if `CHANNEL_STATE` lacks `IS_REFERENCED`, then parks at `HOME_POSITION_MM`.
  `disconnect(safe=False)` leaves the wedge in place.
  `move_to(mm)`, `move_by(mm)`, `wait_for_stop()` (waits for the move to *begin* then to
  settle — guards the async status lag), `get_position()`, `is_moving()`.
  - Positions are **mm** at this class's API; the MCS2 works in **picometres**
    (`PM_PER_MM = 1e9`). `get_position()` reads the real sensor, not a cached value.
  - `is_moving()` tests the `ACTIVELY_MOVING` bit of `CHANNEL_STATE`.
  - `simulate=True` gives a full software stage, so the whole scan/DSP path runs with no
    controller present (this is what `selftest_acquisition.py` drives).
  - **Placeholders to set per rig:** `TRAVEL_MM` (assumes a 50 mm SLC-1750),
    `HOME_POSITION_MM`, `SAFE_POSITION_MM`, `DEFAULT_CHANNEL`.
- **`StagesPanel.freeze(True/False)`** — pauses the poll timer and disables manual controls
  during a scan (avoids DLL contention). MeasurePanel/Twins scan call this around runs.

---

## 5. Live 1-D TWINS scan (`ui/twins_scan.py` + `instruments/subtwinslv.py` + `spectrum_processor.py`)

The **TWINS tab** does a quick scalar interferogram for alignment / single-spectrum.
- `TwinsScanPanel`: params (start/stop mm, steps, frames/point), live interferogram +
  spectrum plots, persisted via `QSettings("ts_*")`. `Scan` → `_start_scan()` builds a
  `TwinsScanner(twins, frame_source)` and runs it on the controller thread.
- `TwinsScanner.scan(...)` (`subtwinslv.py`): for each `linspace(start,stop,n)` target →
  `move_to` → `wait_for_stop` → settle → `_read_scalar(roi, frames_avg)` (averages **fresh**
  frames, drops the first in-flight one) → progress callback. Returns `(positions, ifg_1d)`.
- `compute_spectrum()` → `SpectrumProcessor.compute_spectrum()`: baseline removal →
  `calibrate_position_axis` (motor nonlinearity) → `find_centerburst` (Hilbert envelope ZPD,
  searched ±`search_mm` around 24.33) → apodization → explicit DFT (matrix multiply) →
  `_freq_to_wavelength` (via `parameters_cal.txt`). Same math as the 2-D path.

---

## 6. The hyperspectral experiment — `MeasurePanel` (`ui/measure_kspace.py`)

This is the scientific core. It runs a worker thread that drives the TWINS wedge, grabs
frame stacks, computes per-pixel spectra, and auto-saves the result. Key classes in the
file: `MeasurePanel`, `HyperViewer`, `LiveInterferogram`; helpers `load_kspace_npz`,
`kspace_metadata`.

### Acquisition flow (two-phase worker `_worker`)
A single **Acquire** = one *run* = **one wedge sweep = one cube**. At `_start`, a per-run
folder `<camera_folder>/<run_stamp>.<filename>/` is created (`run_stamp = YYYYMMDD_HHMMSS`
captured once; `_run_target()` returns `(folder, stamp, fname)`).

- **Phase 1 — ACQUIRE (no FFT while the stage is moving).**
  `TwinsScanner.scan_cube(...)` returns `(positions, datacube)` where `datacube` is
  `(n_pos, h, w)` over the **binned ROI**. `progress(i, tot, pos, value)` emits
  `"pt i/tot: wedge step"`, which is what drives the progress bar (its maximum is set to
  the step count) and the live interferogram preview.
- **Phase 2 — TRANSFORM the acquired cube.** Optional saturation mask, `resolve_n_points`
  (Auto = 1.5× steps, clamped [512,4096]), `compute_hyperspectral(...)`, optional SVD
  denoise. `_on_done` then calls `_autosave_cube()` **once**, so no run is ever lost.

The two phases are still separate (rather than transforming inside the sweep) because the
per-pixel DFT is slow and must not sit between wedge steps. The `cubes` / `z_values` lists
they fill are kept **single-entry with `z = None`**, purely so the saved `.npz` keeps the
stacked layout the viewer and `analysis_app.py` already read.

**Disk guards.** `_start` estimates the run size (`(n_pos + n_freq) × h × w × 4` bytes) and
refuses to start — with a confirm dialog — if the save volume is short. Mid-run the worker
only *warns*: by then the sweep is already spent, and aborting would throw the cube away.

`scan_cube` (`subtwinslv.py`): per step, `wait_for_stop` + settle, then `_read_roi_slice`
averages **distinct** frames (object-identity, drop first in-flight), subtracts the
captured **background** (full-frame, before ROI crop+bin) if enabled, crops to ROI, bins by
`bin_factor`. Records the **real measured** wedge position (`stage.get_position()`). On a
camera freeze it `_wait_for_stream` then re-acquires that step. `progress(i,tot,pos,value)`
drives the live interferogram preview.

### Calibration of the wedge axis (important)
- `compute_hyperspectral` **always** applies the motor-nonlinearity correction
  `calibrate_position_axis(positions)` (from `parameters_int.txt`) before the DFT, unless
  told the axis is already calibrated (`positions_calibrated=True`). This linearizes the
  reproducible wedge motor error (same method as the user's `Desktop/calibrate_piezo.py`).
- **Saved axes:** every file stores BOTH `twins_positions_mm` (raw measured) and
  `twins_positions_calibrated_mm` (the axis the DFT used). Metadata records
  `motor_calibration_applied`, `motor_calibration_file`, `motor_calibration_max_dev_um`,
  `position_axis_used` ("calibrated"/"raw_measured"). On reload the analyzer reads these to
  avoid **double-calibration** (feeds the calibrated axis with `positions_calibrated=True`).

### The per-pixel transform (`instruments/hyperspectral.py::compute_hyperspectral`)
- Operates on the **ROI (binned) cube only** — the full frame is never transformed.
- Preprocess: moving-average baseline removal; `find_centerburst` (signed spatial sum →
  Hilbert envelope, searched ±`search_mm` around `expected_zero_mm=24.33`); optional
  symmetrize. Apodization via `dsp.apodization_window` — now **asymmetric-aware** (per-wing
  taper to each edge, keeps the full long tail; `gaussian` is a separate NIREOS two-sided
  branch). **FT-window** selection: `ft_region` full/center/tails + `ft_width_mm`, or explicit
  `ft_window_mm=(lo,hi)` (boxcar if the window excludes the ZPD, else taper×window).
- DFT: a single **vectorised matrix multiply (NUDFT)** —
  `phase_kernel = exp(-2πj·pos·freq)`; `spec_flat = phase_kernel.conj().T @ (weighted pixels)`
  → `(n_freq, h·w)` → magnitude (`np.abs`, correct for spatially-varying ZPD). Output cube is
  **float32**. Frequency grid → wavelength via `parameters_cal.txt`
  (`_get_frequency_limits` / `_freq_to_wavelength`).
- Returns `(wavelengths (n_freq,), spectrum_cube (n_freq,h,w))`.

### Saved file schema
The Measure tab's **Format** box selects the container; `_save_cube()` dispatches.

**`.npz` (default)** — one file per run, in the **stacked** form the viewer and analyzer
already read (the stack is simply length 1 here): `wavelengths`, `spectrum_cubes`
(1,n_freq,h,w float32), `z_values` (`[NaN]` — no Z axis), `z_unit`, `raw_interferograms`
(1,n_pos,h,w — the per-pixel interferogram, **always saved**), `raw_positions`,
`raw_positions_calibrated`, `roi`, `binning`, optional `saturation_masks`, optional
`background`+`background_subtracted`, and `metadata` (np object dict) + `metadata_json`.
Filename: `<stamp>.<name>.npz` inside the run folder.

**`.h5`** — `instruments/h5_writer.py`, the ScopeFoundry `h5_io` layout, so the lab's
other tooling reads it. `/measurement/hyperspectral/t0/c0/image` is the raw interferogram
cube (attrs `element_size_um`, default `[1,1,1]`) and `.../t0/c0/position_mm` the measured
wedge axis; `t<i>` indexes the cube, so multi-cube files would write `t0/`, `t1/`, … .
Everything the `.npz` carries is kept alongside under `spectrum/`, `masks/`, `background/`,
`acquisition` and `metadata_json`, so HDF5 loses nothing. Filename:
`<yymmdd_HHMMSS>_hyperspectral[_<sample>].h5`, sample = the Filename box, timestamp
sliced from the run stamp so file and run folder agree. `h5py` is imported lazily —
without it the app still runs and still saves `.npz`. Because the raw interferogram *is*
the primary dataset, a cube with no raw data (one loaded from an old file) cannot be
written as HDF5; `_save_cube_h5` raises and `_save` surfaces it.

### Viewer
`HyperViewer` (in-app "Open Viewer" + standalone `view_hyperspectral.py`) shows the cube:
λ slider (the Z slider stays hidden — there is no Z axis), maps
(λ-slice/Peak-λ/Peak-intensity/SAM), colormap/gamma, pixel spectra,
and a calibration badge via `set_calibration_note(meta)`. `set_result` (in-RAM) /
`set_result_lazy` (per-Z lazy load). The big **analysis** app is separate (`analysis_app.py`).

### Metadata captured at scan start (`_scan_meta` + `_cam_meta`)
start/stop/steps/step_um, frames/point, binning, ROI, apodization+width, wl range,
n_freq setting, ZPD, walk-off, background_subtracted, saturation level, svd,
filename — plus the camera dict from `meta_provider` (serial, exposure, averaging,
board temp, backend).

---

## 7. Shared processing modules (`instruments/`)

- `calibration.py` — loads `parameters_cal.txt` (spectral: row0 wavelength µm, row1
  reciprocal) and `parameters_int.txt` (motor: row0 position, row1 reference IFG), both under
  `Twins/calibration/`. `calibrate_position_axis` (motor nonlinearity via
  analytic-signal phase), `position_calibration_status`. **Frozen-aware** (finds data via
  `sys._MEIPASS` when packaged).
- `dsp.py` — apodization window library (Happ-Genzel, Blackman-Harris 3/4, triangular,
  super-gaussian) **asymmetric-aware**; FWHM-based resolution estimate.
- `analysis.py` — cube analysis: `saturation_mask`, `roi_average`, `svd_denoise`,
  `svd_explained_variance`, peak maps, `spectral_derivative`, `spectral_angle_map`.
- `walkoff.py` — TWINS wedge walk-off correction (per-frame parametric shift +
  phase-correlation registration). Wired into `compute_hyperspectral(walkoff=...)`.
- `spectrum_processor.py` / `hyperspectral.py` — the 1-D and 2-D processors (same math,
  different output dimensionality).

---

## 8. Conventions, gotchas, invariants

- **`self.latest_frame` is replaced by a fresh `.copy()` each poll** — averaging code relies
  on object identity to detect new frames and drop in-flight ones. Don't mutate it in place.
- **Saturation** is a UI setting, not a constant — the IMX990 ADC is 12-bit (0–4095) in
  the usual `Mono16`/`Bit12` configuration. The mask is computed on the RAW counts (add the
  background back if it was subtracted).
- **TWINS `wait_for_stop`** must really wait, or the scan records mid-motion.
- **Exposure** is clamped to the `ExposureTime` node's own min/max and read back; do not
  re-add a hard-coded clamp (that was an IRC806 firmware workaround).
- **Copy the PySpin frame before `Release()`** — `GetNDArray()` returns a view into a
  buffer Spinnaker immediately recycles.
- **The DFT is ROI+binned only** — to speed up: tighter ROI, higher bin, fewer `n_freq`, or a
  `complex64` kernel (~2×; currently complex128).
- **Calibrated axis is applied at compute time, never double-applied on reload** (metadata flag
  + `positions_calibrated`). Raw axis is preserved so files can be re-derived with a new cal.
- **Raw interferogram is always saved** → every file is reprocessable (don't re-add a gate).
- Persistence via `QSettings` org **"SWIR_CAMERA"** (`SETTINGS_ORG` in
  `ui/measure_kspace.py`): apps "KSpace"/"TwinsScan"/"HyperViewer". The org is
  deliberately *not* "MIR_CAMERA" — the MWIR app uses that, and sharing it makes the two
  apps silently overwrite each other's saved scan parameters.
- **Anything that drives the Measure spin boxes persists settings** (every `valueChanged`
  hits `_save_settings`). `selftest_acquisition.py` therefore redirects `QSettings` to a
  throwaway `.ini`; keep that if you add tests.

---

## 9. Run / entry points

- `run.bat` → `python main.py --mode forge --fps 60`; `run_mock.bat` → `--mode mock`
  (no camera and no stage needed — tick "Simulate" in the TWINS panel).
- `selftest_acquisition.py` runs a full headless acquisition (mock camera + simulated
  stage) and asserts one cube out plus a valid auto-saved `.npz`. Exits non-zero on failure.
- `.venv` holds the deps (PyQt6, pyqtgraph, numpy, scipy, pandas, tifffile, Pillow, loguru).
- Hardware SDKs, both pip-installed from local files, neither on PyPI:
  `spinnaker_python-*.whl` (camera) and `smaract_ctl-*.zip` from the MCS2 SDK (stage).

## 10. File map (acquisition app)

| File | Role |
|---|---|
| `main.py` | entry: spawns camera process, shared mem + queues, builds MainWindow |
| `worker_camera.py` | camera worker process: frame/status out, command in, auto-heal |
| `camera/{camera_interface,factory,forge_camera,mock_camera}.py` | camera abstraction + backends |
| `ui/main_window.py` | orchestrator: frame poll, latest_frame, controls, bad-pixel mask, save, metadata |
| `ui/stages.py` | StagesPanel: TWINS wedge UI, StageController threading, freeze() |
| `ui/twins_scan.py` | live 1-D TWINS scan UI |
| `ui/measure_kspace.py` | **MeasurePanel**: the hyperspectral experiment + HyperViewer |
| `instruments/twins_stage.py` | SmarAct MCS2 driver for the SLC-1750 wedge stage |
| `instruments/subtwinslv.py` | TwinsScanner: step-scan engine (`scan`, `scan_cube`) |
| `instruments/hyperspectral.py` | 2-D per-pixel DFT (`compute_hyperspectral`) |
| `instruments/spectrum_processor.py` | 1-D interferogram → spectrum |
| `instruments/h5_writer.py` | ScopeFoundry-layout HDF5 writer/reader |
| `instruments/{calibration,dsp,analysis,walkoff}.py` | shared processing |
| `selftest_acquisition.py` | headless end-to-end acquisition check (no hardware) |
| `Twins/calibration/parameters_{cal,int}.txt` | spectral + motor calibration data |
