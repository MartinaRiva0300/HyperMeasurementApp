# TWINS Hyperspectral Imaging App

A control and acquisition application for hyperspectral imaging based on a **TWINS** (Translating-Wedge-based Identical pulses eNcoding System) birefringent interferometer. The app synchronizes a **camera** and a **linear translation stage** to acquire interferogram stacks that are Fourier-transformed into spectral data cubes.

## Overview

TWINS is a common-path, birefringent interferometer that generates two collinear, delay-tunable replicas of the incoming light by translating a pair of birefringent wedges. By stepping the wedge/stage position and recording a camera frame at each step, the app builds a spatial interferogram stack (x, y, delay). A Fourier transform along the delay axis then reconstructs a full spectral cube (x, y, λ) for every pixel in the field of view.

Because the interferometer is common-path and compact, it offers high mechanical and thermal stability without the need for active phase stabilization, making it well suited for field and benchtop hyperspectral applications.

## Key Features

- **Synchronized acquisition** — camera exposure and stage/wedge position are triggered and logged in lock-step for each interferogram step.
- **Dual hardware configuration** — supports two interchangeable hardware branches (see below), selectable via configuration.
- **Automated delay scanning** — configurable step size, range, and dwell time for the translation stage.
- **Spectral reconstruction** — Fourier-transform pipeline to convert interferogram stacks into calibrated spectral cubes.
- **Data export** — raw interferogram stacks and reconstructed cubes saved in standard formats (e.g., HDF5/TIFF stack + metadata).
- **Live preview** — real-time monitoring of interferogram frames and (optionally) a live spectral preview during acquisition.

## Supported Hardware Configurations

The app is organized into two branches, each targeting a different spectral range and hardware set. Only one branch is active per session, selected in the configuration file.

### Branch 1 — Visible / NIR

| Component | Model |
|---|---|
| Camera | Hamamatsu ORCA-Flash4.0 LT3 Digital CMOS Camera (**C11440-42U40**) |
| Linear stage | **PI** (Physik Instrumente) linear translation stage |

### Branch 2 — SWIR

| Component | Model |
|---|---|
| Camera | Forge 1GigE SWIR Camera — 1.3 MP, C-mount |
| Linear stage | **SLC-1750** — SmarAct linear piezo stage |

## Requirements

- Python 3.9+
- Vendor drivers/SDKs, installed and licensed separately:
  - Hamamatsu **DCAM-API** (for the ORCA-Flash4.0 LT3, Branch 1)
  - **PI GCS2** / PIPython (for the PI stage, Branch 1)
  - GigE Vision driver for the Forge 1GigE SWIR camera (Branch 2)
  - SmarAct **MCS2** driver/SDK (for the SLC-1750 piezo stage, Branch 2)
- Python packages: `numpy`, `scipy`, `h5py`, `matplotlib` (or equivalent GUI framework), plus vendor Python bindings for the above SDKs


## Installation

```bash
# Clone the repository
git clone <repository-url>

# Create and activate an environment


# Install dependencies



## Usage

```bash
# Run an acquisition using the active branch from config.yaml
python run_acquisition.py --config config.yaml

# Reconstruct a spectral cube from a saved interferogram stack
python reconstruct.py --input ./data/scan_001.h5 --output ./data/scan_001_cube.h5
```

Typical workflow:

1. Select the hardware branch 
2. Connect and home the linear stage; connect to the camera.
3. Run the acquisition script — the app steps the stage across the configured delay range, capturing a synchronized camera frame at each position.
4. The raw interferogram stack is saved, and (optionally) reconstructed into a spectral cube via the Fourier-transform pipeline.
5. Inspect results with the live preview or the reconstruction/visualization tools.

## Project Structure

```
twins-hyperspectral-app/
├── hardware/
│   ├── branch1/
│   │   ├── camera_orca_flash4.py
│   │   └── stage_pi.py
│   └── branch2/
│       ├── camera_forge_swir.py
│       └── stage_smaract_slc1750.py

```
