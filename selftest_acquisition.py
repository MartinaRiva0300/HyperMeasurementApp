"""Headless end-to-end self-test of the acquisition path -- no hardware needed.

Drives a real Measure run against the simulated SLC-1750 wedge stage and a
synthetic interferogram frame source, and checks that it produces exactly ONE
hyperspectral cube (no Z / angle axis) and auto-saves it -- in BOTH save
formats, including the exact ScopeFoundry dataset layout of the HDF5 one.

    python selftest_acquisition.py

Exits non-zero on failure.
"""
import os, sys, glob, json, tempfile
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from ui.stages import StagesPanel
import ui.measure_kspace as mk
from ui.measure_kspace import MeasurePanel

# The app's pre-flight disk guard refuses to start a scan when the save volume
# is nearly full (correct for real GB-scale acquisitions). This test's cube is
# ~2 MB, so stub the check out rather than have the self-test depend on how much
# free space the machine happens to have.
mk._free_gb = lambda *_a, **_k: 999.0

H, W = 24, 32
ZPD_MM = 24.33          # matches DEFAULT_ZPD_MM in the app
LAMBDA_UM = 1.55        # a SWIR line, well inside the Forge band


def check_npz(path):
    d = np.load(path, allow_pickle=True)
    print(f"npz keys        : {sorted(d.files)}")
    for key in ("spectrum_cubes", "raw_interferograms", "raw_positions",
                "wavelengths", "z_values", "metadata_json"):
        assert key in d.files, f"missing {key} in saved npz"
    assert d["spectrum_cubes"].shape[0] == 1, "expected a single cube in the npz"
    assert np.isnan(d["z_values"]).all(), "z axis should be empty"


def check_h5(path, n_pos):
    """The layout the Measure tab's HDF5 format must produce."""
    import h5py
    name = os.path.basename(path)
    print(f"h5 filename     : {name}")
    # <yymmdd_HHMMSS>_hyperspectral_<sample>.h5
    stem = name[:-3].split("_")
    assert len(stem) >= 3, f"unexpected h5 filename {name}"
    assert len(stem[0]) == 6 and stem[0].isdigit(), f"bad date field in {name}"
    assert len(stem[1]) == 6 and stem[1].isdigit(), f"bad time field in {name}"
    assert stem[2] == "hyperspectral", f"bad measurement name in {name}"

    with h5py.File(path, "r") as f:
        m = f["measurement/hyperspectral"]
        found = []
        m.visit(found.append)
        print(f"h5 datasets     : {[k for k in found if isinstance(m.get(k), h5py.Dataset)]}")

        img = m["t0/c0/image"]
        pos = m["t0/c0/position_mm"]
        assert img.shape == (n_pos, H, W), f"t0/c0/image is {img.shape}, want {(n_pos, H, W)}"
        assert pos.shape == (n_pos,), f"t0/c0/position_mm is {pos.shape}"
        assert pos.dtype == np.float32, f"position_mm dtype {pos.dtype}, want float32"
        esu = img.attrs["element_size_um"]
        assert list(esu) == [1.0, 1.0, 1.0], f"element_size_um is {list(esu)}"
        print(f"element_size_um : {list(esu)}   image dtype: {img.dtype}")

        # the wedge axis really is the measured one, not an index
        assert abs(float(pos[0]) - (ZPD_MM - 0.10)) < 1e-3, f"first position {pos[0]}"
        assert abs(float(pos[-1]) - (ZPD_MM + 0.10)) < 1e-3, f"last position {pos[-1]}"

        # everything the .npz carries must survive here too
        for key in ("spectrum/wavelengths", "spectrum/cube",
                    "t0/c0/position_mm_calibrated", "metadata_json"):
            assert key in m, f"missing {key} in saved h5"
        assert f.attrs["sample"] == "swirtest", f"sample attr {f.attrs['sample']}"
        meta = json.loads(m["metadata_json"][()])
        assert meta.get("n_steps") == n_pos, "metadata lost the scan settings"
        print(f"h5 sample attr  : {f.attrs['sample']}   metadata keys: {len(meta)}")


def isolate_settings(outdir):
    """Point every QSettings the panels build at a throwaway .ini.

    Without this the test writes the REAL saved scan parameters (it drives the
    spin boxes, and every valueChanged persists), silently replacing whatever the
    operator last set up."""
    from PyQt6.QtCore import QSettings
    ini = os.path.join(outdir, "selftest_settings.ini")
    real = QSettings
    mk.QtCore.QSettings = lambda *a, **k: real(ini, real.Format.IniFormat)


def main():
    app = QApplication(sys.argv)
    outdir = tempfile.mkdtemp(prefix="e2e_")
    isolate_settings(outdir)

    sp = StagesPanel()
    assert sp.twins.connect(simulate=True, home=True), "sim stage failed to connect"

    # Frame source: a two-beam interferogram whose fringe phase follows the
    # CURRENT wedge position, so the scan produces a real centerburst.
    def frames():
        pos = sp.twins.get_position()
        delay_mm = pos - ZPD_MM
        # 1 mm of wedge travel -> a few hundred fringes at 1.55 um.
        phase = 2 * np.pi * delay_mm * 1000.0 / LAMBDA_UM
        env = np.exp(-(delay_mm / 0.05) ** 2)
        val = 1000.0 * (1.0 + env * np.cos(phase))
        return np.full((H, W), val, dtype=np.uint16)

    mp_ = MeasurePanel(sp, frame_source=frames, roi_provider=lambda: None,
                       roi_show=lambda *_: None,
                       bg_provider=lambda: (None, False),
                       save_dir_provider=lambda: outdir,
                       meta_provider=lambda: {"camera_serial": "E2E"},
                       save_dir=outdir)
    mp_.live_monitor = None

    # A short but real scan across ZPD.
    mp_.spin_start.setValue(ZPD_MM - 0.10)
    mp_.spin_stop.setValue(ZPD_MM + 0.10)
    mp_.spin_steps.setValue(61)
    mp_.spin_frames.setValue(1)
    mp_.spin_bin.setValue(1)
    mp_.spin_wl0.setValue(1.2)
    mp_.spin_wl1.setValue(2.0)
    mp_.edit_filename.setText("swirtest")
    mp_.combo_format.setCurrentText(mk.FORMAT_NPZ)   # the default path first

    mp_.sig_status.connect(lambda m: print("  [status]", m, flush=True))
    mp_.sig_warn.connect(lambda t, m: print("  [WARN]", t, m, flush=True))

    done = {}
    real_on_done = mp_._on_done

    def on_done(wl, cubes, zvals, masks):
        real_on_done(wl, cubes, zvals, masks)
        done["wl"], done["cubes"], done["z"] = wl, cubes, zvals
        app.quit()

    mp_.sig_done.disconnect()
    mp_.sig_done.connect(on_done)
    mp_._open_viewer = lambda *a, **k: None   # no viewer window in a headless run

    QTimer.singleShot(0, mp_._start)
    QTimer.singleShot(60000, app.quit)        # hard timeout
    app.exec()

    assert done, "acquisition never completed"
    wl, cubes, z = done["wl"], done["cubes"], done["z"]
    assert cubes and len(cubes) == 1, f"expected exactly 1 cube, got {cubes and len(cubes)}"
    cube = cubes[0]
    print(f"cube shape      : {cube.shape}  (lambda x h x w)")
    print(f"wavelength range: {wl.min():.3f} - {wl.max():.3f} um")
    print(f"z values        : {z}")
    assert cube.shape[1:] == (H, W), f"bad spatial shape {cube.shape}"
    assert z == [None], f"expected no z axis, got {z}"

    # NB: the recovered wavelength is NOT asserted -- mapping wedge mm to optical
    # delay goes through the TWINS calibration tables, so a synthetic fringe rate
    # in "mm of travel" does not land at LAMBDA_UM. What this test checks is the
    # acquisition loop, the single-cube collapse and the save path; the DSP is
    # unchanged from the reference repo. The ZPD the app prints (~24.325 mm vs the
    # 24.33 injected here) is the evidence the interferogram arrived intact.
    peak_um = float(wl[np.argmax(cube[:, H // 2, W // 2])])
    print(f"peak bin        : {peak_um:.3f} um (informational)")

    saved = sorted(glob.glob(os.path.join(outdir, "**", "*.npz"), recursive=True))
    print(f"auto-saved      : {[os.path.basename(f) for f in saved]}")
    assert saved, "nothing was auto-saved"
    check_npz(saved[0])

    # --- now the HDF5 format, from the SAME in-memory result -----------------
    print("\n--- HDF5 format ---")
    mp_.combo_format.setCurrentText(mk.FORMAT_H5)
    h5path = mp_._autosave_cube()
    assert h5path.endswith(".h5"), f"expected an .h5, got {h5path}"
    check_h5(h5path, n_pos=len(mp_.raw_positions[0]))

    # and it round-trips back through the Load path
    res = mk.load_measurement_h5(h5path)
    assert res is not None, "load_measurement_h5 returned None"
    wl2, cubes2, z2, _ = res
    assert len(cubes2) == 1 and cubes2[0].shape == cube.shape, "h5 round-trip shape"
    assert np.allclose(wl2, wl), "h5 round-trip wavelengths"
    assert z2 == [None], "h5 round-trip z axis"
    print(f"h5 round-trip   : cube {cubes2[0].shape}, wl {wl2.min():.3f}-{wl2.max():.3f} um")

    sp.shutdown()
    print("\nE2E OK  (npz + h5)")


if __name__ == "__main__":
    main()
