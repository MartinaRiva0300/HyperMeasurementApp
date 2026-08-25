"""HDF5 writer/reader for hyperspectral measurements, in the ScopeFoundry layout.

The Measure tab saves NumPy `.npz` by default; picking **HDF5** in its Format box
routes the same data through here instead. The file follows the ScopeFoundry
`h5_io` convention the rest of the lab's tooling expects:

    /                                  attrs: created, time_id, sample, ...
    /app/settings                      attrs: save_dir, sample
    /measurement/hyper/                attrs: measurement name + scan settings
        settings                       group, one attr per scan setting
        t0/c0/image                    (n_pos, h, w)   the raw interferogram
                                       attrs: element_size_um = [z, y, x]
        t0/c0/position_mm              (n_pos,) float32  wedge positions, MICROMETRES
        ...

`t0/c0/image` is this app's `raw_interferograms[0]` and `t0/c0/position_mm` is
its `raw_positions[0]` -- i.e. the per-pixel interferogram and the real measured
wedge axis. NOTE the dataset keeps its historical `position_mm` NAME (that is
what the MATLAB reader opens) but the VALUES are in MICROMETRES, matching that
reader's expectation; the app works in mm internally and converts here, at the
write boundary only. Every position dataset carries an explicit `units` attr.

`t<i>` indexes the cube, so a file holding several cubes writes `t0/`, `t1/`, ...
(this app acquires exactly one per run, so it is always `t0`).

The MATLAB analysis reader opens `/measurement/hyper/t0/c0/`, so MEASUREMENT_NAME
below must stay `hyper` unless that reader changes too.

Everything the `.npz` carries beyond those two datasets is kept as further groups
under the same measurement group -- the computed spectrum, the calibrated
position axis, saturation mask, background and the full metadata JSON -- so an
HDF5 save loses nothing relative to an `.npz` save.

`h5py` is imported lazily: the app runs, and saves `.npz`, without it installed.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np

# ScopeFoundry's `element_size_um` is [z, y, x] and is what Fiji/ImageJ reads as
# the voxel size. Defaults to 1/1/1 -- set these only if you want the saved file
# to carry a physical voxel size (z = wedge step in um, y/x = pixel pitch x
# binning / magnification). The true values are recorded in the metadata either
# way, so nothing is lost by leaving them at 1.
DEFAULT_ELEMENT_SIZE_UM = (1.0, 1.0, 1.0)

# The app carries wedge positions in millimetres; the HDF5 position datasets are
# written in MICROMETRES because that is what the MATLAB analysis reader expects.
MM_TO_UM = 1000.0

# ScopeFoundry names the measurement group; the sample comes from the Measure
# tab's Filename box, giving <timestamp>_<name>[_<sample>].h5 and the group
# /measurement/<name>/. MUST match what the analysis reader opens -- the MATLAB
# reader uses '/measurement/hyper/t0/c0/' -- so do not rename this casually.
MEASUREMENT_NAME = "hyper"


def _require_h5py():
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Saving HDF5 needs the 'h5py' package (pip install h5py). "
            "Switch the Format box back to NumPy (.npz) to save without it."
        ) from exc
    return h5py


def h5_filename(timestamp: str, sample: str = "", name: str = MEASUREMENT_NAME) -> str:
    """`<timestamp>_<name>.h5`, or `<timestamp>_<name>_<sample>.h5` when a sample
    is given -- the ScopeFoundry naming rule. `timestamp` is `%y%m%d_%H%M%S`."""
    sample = (sample or "").strip()
    parts = [timestamp, name] + ([sample] if sample else [])
    return "_".join(parts) + ".h5"


def _set_attrs(node, mapping) -> None:
    """Write a flat dict as HDF5 attributes, coercing what h5py cannot store.

    Nested dicts / lists of dicts / None have no attribute representation, so
    they are JSON-encoded rather than dropped."""
    for key, value in (mapping or {}).items():
        if value is None:
            node.attrs[key] = "None"
            continue
        if isinstance(value, (str, bytes, bool, int, float, np.generic)):
            node.attrs[key] = value
            continue
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            # Object/ragged arrays (e.g. lists of dicts) can't be attributes.
            if arr.dtype == object:
                node.attrs[key] = json.dumps(value, default=str)
            else:
                node.attrs[key] = arr
            continue
        node.attrs[key] = json.dumps(value, default=str)


def save_measurement_h5(
    path: str,
    *,
    raw_cubes,
    raw_positions,
    raw_positions_calibrated=None,
    wavelengths=None,
    spectrum_cubes=None,
    sat_masks=None,
    background=None,
    background_subtracted: bool = False,
    roi=None,
    binning: int = 1,
    metadata: dict | None = None,
    sample: str = "",
    save_dir: str = "",
    name: str = MEASUREMENT_NAME,
    element_size_um=DEFAULT_ELEMENT_SIZE_UM,
) -> str:
    """Write one measurement to `path` in the ScopeFoundry HDF5 layout.

    `raw_cubes` is a sequence of (n_pos, h, w) interferogram cubes and
    `raw_positions` the matching (n_pos,) wedge axes -- cube *i* becomes
    `t<i>/c0/image` + `t<i>/c0/position_mm`. Returns `path`.
    """
    h5py = _require_h5py()
    meta = dict(metadata or {})

    if not raw_cubes:
        raise ValueError("HDF5 save needs the raw interferogram; none was captured. "
                         "Save as .npz instead, or re-run the acquisition.")

    with h5py.File(path, "w") as f:
        now = datetime.now()
        f.attrs["ScopeFoundry_type"] = "h5_base_file"
        f.attrs["created"] = now.isoformat(timespec="seconds")
        f.attrs["time_id"] = now.strftime("%y%m%d_%H%M%S")
        f.attrs["sample"] = sample or ""
        f.attrs["measurement"] = name

        app = f.create_group("app")
        app_settings = app.create_group("settings")
        app_settings.attrs["save_dir"] = save_dir or ""
        app_settings.attrs["sample"] = sample or ""

        m = f.create_group(f"measurement/{name}")
        m.attrs["name"] = name
        # Scan parameters as attributes; the camera dict / zone lists inside the
        # metadata are nested, so they go to their own subgroup + the JSON blob.
        flat = {k: v for k, v in meta.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in meta.items() if isinstance(v, dict)}
        settings = m.create_group("settings")
        _set_attrs(settings, flat)
        for key, sub in nested.items():
            _set_attrs(settings.create_group(key), sub)

        # -- the two datasets the ScopeFoundry layout is built around ---------
        for i, cube in enumerate(raw_cubes):
            cube = np.asarray(cube)
            pos = np.asarray(raw_positions[i], dtype=np.float32)
            c = m.create_group(f"t{i}/c0")
            img = c.create_dataset("image", data=cube, dtype=cube.dtype)
            img.attrs["element_size_um"] = np.asarray(element_size_um, dtype=np.float32)
            img.attrs["units"] = "counts"
            # Name says _mm (the reader opens that path) but the VALUES are um.
            ds = c.create_dataset("position_mm", data=pos * MM_TO_UM,
                                  dtype=np.float32)
            ds.attrs["units"] = "um"
            if raw_positions_calibrated is not None and i < len(raw_positions_calibrated):
                cal = raw_positions_calibrated[i]
                if cal is not None:
                    # The motor-nonlinearity-corrected axis the DFT actually used.
                    ds = c.create_dataset(
                        "position_mm_calibrated",
                        data=np.asarray(cal, dtype=np.float32) * MM_TO_UM)
                    ds.attrs["units"] = "um"

        # -- everything the .npz also carries ---------------------------------
        if wavelengths is not None and spectrum_cubes is not None:
            spec = m.create_group("spectrum")
            spec.create_dataset("wavelengths", data=np.asarray(wavelengths))
            spec["wavelengths"].attrs["units"] = "um"
            cubes = np.asarray(spectrum_cubes)
            # complex64 (float32 real + float32 imag) when the phase was kept,
            # else float32. Pinned rather than echoing cubes.dtype so a
            # complex128 input can never silently double the file size.
            spec.create_dataset("cube", data=cubes,
                                dtype=np.complex64 if np.iscomplexobj(cubes)
                                else np.float32)
            spec["cube"].attrs["dims"] = "n_maps, n_freq, height, width"

        if sat_masks and any(mk is not None for mk in sat_masks):
            h, w = np.asarray(raw_cubes[0]).shape[1:]
            stack = np.array([(np.zeros((h, w), bool) if mk is None
                               else np.asarray(mk, bool)) for mk in sat_masks])
            m.create_dataset("masks/saturation", data=stack)

        if background is not None:
            bg = m.create_group("background")
            bg.create_dataset("map", data=np.asarray(background))
            bg.attrs["subtracted"] = bool(background_subtracted)

        acq = m.create_group("acquisition")
        acq.attrs["binning"] = int(binning)
        if roi is not None and len(roi):
            acq.attrs["roi"] = np.asarray(roi, dtype=np.int64)
        acq.attrs["n_positions"] = int(np.asarray(raw_positions[0]).size)

        # Full metadata verbatim, so nothing is lost to attribute coercion.
        m.create_dataset("metadata_json",
                         data=json.dumps(meta, default=str, indent=2))
    return path


def load_measurement_h5(path: str):
    """Read a file written by `save_measurement_h5`.

    Returns `(wavelengths, cubes, z_values, sat_masks)` -- the same tuple
    `load_kspace_npz` returns -- or None when the file holds no spectrum."""
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        group = f.get("measurement")
        if group is None:
            return None
        m = group[next(iter(group.keys()))]
        if "spectrum" not in m:
            return None
        wl = np.asarray(m["spectrum/wavelengths"])
        # A cube saved in complex form is handed back as its magnitude, so the
        # viewer never has to care which way it was written.
        cubes = [np.abs(c) if np.iscomplexobj(c) else np.asarray(c)
                 for c in np.asarray(m["spectrum/cube"])]
        masks = None
        if "masks/saturation" in m:
            masks = [np.asarray(mk, bool) for mk in np.asarray(m["masks/saturation"])]
        # This app has no Z axis; keep the slot so callers stay uniform.
        return wl, cubes, [None] * len(cubes), masks


def metadata_h5(path: str) -> dict:
    """The embedded metadata dict of an HDF5 measurement (or {})."""
    try:
        h5py = _require_h5py()
        with h5py.File(path, "r") as f:
            group = f.get("measurement")
            if group is None:
                return {}
            m = group[next(iter(group.keys()))]
            if "metadata_json" not in m:
                return {}
            raw = m["metadata_json"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}
