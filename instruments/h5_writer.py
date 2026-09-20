"""HDF5 helpers shared by the measurement save path.

The Measure tab writes its results as the two MATLAB-compatible HDF5 files
directly (see ui/measure_panel.py); this module just provides two small helpers
those writers reuse:

  - ``_require_h5py`` -- lazy h5py import with a clear error if it's missing;
  - ``_set_attrs``   -- coerce a flat dict into HDF5 attributes.

``h5py`` is imported lazily so the rest of the app still runs without it.
"""
from __future__ import annotations

import json

import numpy as np


def _require_h5py():
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Reading/writing HDF5 needs the 'h5py' package (pip install h5py)."
        ) from exc
    return h5py


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
