"""
hyperspectral.py -- per-pixel TWINS DFT (Measurement hyperspectral).
Takes a 3-D datacube
(n_positions, h, w) of ROI frames acquired while scanning the TWINS wedge and
runs an independent DFT for every pixel, returning a spectrum cube
(n_freq, h, w) plus the wavelength axis.

Heavy step is `phase_kernel.conj().T @ flat`; keep the ROI modest (a spectrum
cube is n_freq * h * w complex128 during compute).
"""
import numpy as np
from pathlib import Path


DEFAULT_START_MM = 23.8
DEFAULT_STOP_MM = 24.8
DEFAULT_N_STEPS = 100
# Spectral window for the Forge 1GigE SWIR (Sony IMX990 SenSWIR, ~0.4-1.7 µm).
# The useful upper edge is the sensor cut-off at 1.7 µm.
DEFAULT_WL_START = 0.4       # µm
DEFAULT_WL_STOP = 1.7        # µm

DEFAULT_CALIBRATION_FILE = r".\Twins\calibration\parameters_cal.txt"

# Spectral output sampling ("Auto"), imported from the reference repo. Zero-
# filling does NOT add true spectral resolution (that is fixed by the scan
# length / max OPD) -- it only interpolates the spectrum so the curve looks
# smooth. Auto n_freq = ZEROFILL_FACTOR x scan steps, clamped to a sane band.
ZEROFILL_FACTOR = 1.5
ZEROFILL_MIN = 512
ZEROFILL_MAX = 4096

def resolve_n_points(n_steps, manual=None):
    """Number of spectral output bins (interpolation only).

    A manual value > 0 wins; otherwise "Auto" = ZEROFILL_FACTOR x n_steps,
    clamped to [ZEROFILL_MIN, ZEROFILL_MAX]. Does NOT change the true spectral
    resolution -- that is set by the scan length.
    """
    if manual and manual > 0:
        return int(manual)
    return int(np.clip(ZEROFILL_FACTOR * int(n_steps), ZEROFILL_MIN, ZEROFILL_MAX))


def barycenter_map(sig):
    """Per-pixel ZPD index via the I^2 barycentre (MATLAB
    center = round((1:N)*(I.^2)/(sum(I.^2)+1))), computed INDEPENDENTLY for every
    pixel so a ZPD that shifts across the field of view is followed per pixel.

    `sig` is the (n_pos, h, w) baseline-removed interferogram; returns an (h, w)
    integer index map (0-based, clipped to the valid range). The `+1` in the
    denominator regularises signal-free pixels (they collapse toward index 0).
    """
    w2 = np.asarray(sig, dtype=float) ** 2
    n = w2.shape[0]
    k = np.arange(n, dtype=float)[:, None, None]
    c = np.round(np.sum(k * w2, axis=0) / (np.sum(w2, axis=0) + 1.0)).astype(int)
    return np.clip(c, 0, n - 1)


class HyperspectralProcessor:
    """
    Process a 3D datacube (n_positions, h, w) into a spectrum cube.
    Each pixel gets its own DFT independently.
    """

    def __init__(self, calibration_file=None):
        self.calibration_file = calibration_file or DEFAULT_CALIBRATION_FILE
        self.wavelength_cal = None
        self.reciprocal_cal = None
        self._load_calibration()

    def _load_calibration(self):
        try:
            import pandas as pd
            cal_path = Path(self.calibration_file)
            if not cal_path.exists():
                import sys
                rel = Path("Twins") / "calibration" / "parameters_cal.txt"
                # _MEIPASS = PyInstaller bundle dir when frozen; else the package root.
                base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
                alt_paths = [
                    base / rel,                                  # bundled / package-relative
                    Path(__file__).resolve().parent.parent / rel,
                    Path(".") / rel,                             # CWD-relative
                ]
                for alt in alt_paths:
                    if alt.exists():
                        cal_path = alt
                        break
            if cal_path.exists():
                ref = pd.read_csv(cal_path, sep="\t", header=None)
                self.wavelength_cal = ref.iloc[0].to_numpy(dtype='float64')
                self.reciprocal_cal = ref.iloc[1].to_numpy(dtype='float64')
                print(f"[OK] Measurement: Loaded calibration: {cal_path.name}")
        except Exception as e:
            print(f"[WARN] Measurement calibration: {e}")

    def _get_frequency_limits(self, wl_start, wl_stop):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            return float(fn(1.0 / wl_stop)), float(fn(1.0 / wl_start))
        return 1.0 / wl_stop, 1.0 / wl_start

    def _freq_to_wavelength(self, frequencies):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(self.reciprocal_cal, 1.0 / self.wavelength_cal,
                          kind="linear", fill_value="extrapolate")
            return 1.0 / fn(frequencies)
        return 1.0 / frequencies

    def pseudo_frequencies(self, wavelengths_um):
        """Stage pseudo-frequency axis (1/mm reciprocal, the DFT frequency axis
        after the FT over motor positions) for each wavelength. Inverse of
        _freq_to_wavelength -- lets callers recover the `f` axis from the saved
        wavelength axis."""
        wl = np.asarray(wavelengths_um, dtype=float)
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            return fn(1.0 / wl)
        return 1.0 / wl

    # -- scan-parameter estimation (imported from sub_twins_lw) --------------
    def max_step_um(self, wl_short_um, samples_per_cycle=5):
        """Maximum stage step (µm) for `samples_per_cycle` points per optical
        cycle at the shortest wavelength (2 = Nyquist, 5 = safe oversampling)."""
        if not wl_short_um or wl_short_um <= 0:
            return None
        if not samples_per_cycle or samples_per_cycle <= 0:
            return None
        if self.wavelength_cal is None or self.reciprocal_cal is None:
            return None
        try:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            k_max = float(fn(1.0 / wl_short_um))
            if k_max <= 0:
                return None
            return (1.0 / (float(samples_per_cycle) * k_max)) * 1000.0
        except Exception:  # noqa: BLE001
            return None

    def estimate_resolution(self, scan_range_mm, wl_center_um, apod_type="happ-genzel"):
        """Spectral resolution from the stage scan range, as ``(value, unit)``.

        The scan sets the pseudo-frequency FWHM ``δk`` (the apodization-broadened
        FWHM of the chosen FTIR window, or ``1/scan_range`` as a fallback):
          - no calibration  -> ``(δk, "1/mm")`` -- the pseudo-frequency resolution
            itself, since there is nothing to convert it with;
          - calibration (optical frequency in THz vs pseudo-frequency k) ->
            ``(|dν/dk|·δk, "THz")`` -- the optical-frequency resolution at the band
            centre, by the chain rule.
        Returns None if the inputs are invalid."""
        if not scan_range_mm or scan_range_mm <= 0:
            return None
        if not wl_center_um or wl_center_um <= 0:
            return None
        delta_recip = 1.0 / scan_range_mm
        try:
            from instruments.dsp import apodization_fwhm
            fwhm = apodization_fwhm(apod_type, scan_range_mm)
            if fwhm:
                delta_recip = fwhm
        except Exception:  # noqa: BLE001
            pass
        if self.wavelength_cal is None or self.reciprocal_cal is None:
            return (delta_recip, "1/mm")            # reciprocal (pseudo-frequency) units
        try:
            from scipy.interpolate import interp1d
            # Calibration as the physics uses it: optical frequency ν (THz) as a
            # function of the stage pseudo-frequency k (reciprocal units).
            fr_cal = 299.792458 / self.wavelength_cal          # THz  (c / λ)
            k_of_wl = interp1d(self.wavelength_cal, self.reciprocal_cal,
                               kind="linear", fill_value="extrapolate")
            fr_of_k = interp1d(self.reciprocal_cal, fr_cal,
                               kind="linear", fill_value="extrapolate")
            k_c = float(k_of_wl(wl_center_um))
            eps = max(abs(k_c) * 1e-3, 1e-6)
            dnu_dk = float((fr_of_k(k_c + eps) - fr_of_k(k_c - eps)) / (2 * eps))
            return (abs(dnu_dk) * delta_recip, "THz")          # optical-frequency resolution
        except Exception:  # noqa: BLE001
            return None

    def compute_hyperspectral(self, positions, datacube,
                               wl_start=8.0, wl_stop=14.0,
                               n_freq=200,
                               apod_type="happ-genzel",
                               positions_calibrated=False, center_method="barycenter",
                               complex_output=False):
        """
        Compute per-pixel DFT on a (n_pos, h, w) datacube.

        `center_method`: where the apodization window is centred --
        "barycenter" (default) = an independent I^2 barycentre per pixel, so a ZPD
        that varies across the field is followed per pixel; "geometric" = the
        midpoint sample of the acquired scan, ignoring the signal entirely (use
        when the scan is deliberately centred on ZPD).

        `complex_output`: keep the COMPLEX DFT instead of its magnitude, so the
        interferometric phase survives into the saved cube (complex64). Default
        False = |spectrum| as float32, the historical behaviour.

        `positions_calibrated`: set True when `positions` is ALREADY the
        motor-corrected axis (e.g. reloaded from a saved file's
        twins_positions_calibrated_mm), so the nonlinearity correction is NOT
        applied a second time. Default False = raw measured axis -> calibrate.

        Returns:
            wavelengths : 1D array (n_freq,)
            spectrum_cube : 3D array (n_freq, h, w)
        """
        positions = np.asarray(positions, dtype=float)
        datacube = np.asarray(datacube, dtype=float)
        n_pos, h, w = datacube.shape
        if n_pos < 3:
            return None, None

        # Delay axis correction
        # Remove the TWINS wedge motor's reproducible nonlinearity: replace the
        # nominal stage positions with the calibrated axis (parameters_int.txt).
        # No-op if the position calibration file isn't present, or if the caller
        # says the axis is already calibrated (avoids double-correction).
        if not positions_calibrated:
            try:
                from instruments.calibration import calibrate_position_axis
                positions = np.asarray(calibrate_position_axis(positions), dtype=float)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Measurement: motor calibration skipped: {e}")

        _method = str(center_method).lower()

        # Dataset preprocessing: baseline & apodization
        def preprocess(cube, c_pos=None):
            if c_pos is None:
                c_pos = positions

            window = max(1, len(c_pos) // 5)
            from scipy.ndimage import uniform_filter1d
            # 'nearest' (not constant/0): keeps the baseline sane at the scan
            # ends instead of inflating the edge samples.
            baseline = uniform_filter1d(cube, size=window, axis=0, mode='nearest')
            sig = cube - baseline

            # ZPD centre: the geometric midpoint, or an independent per-pixel
            # I^2 barycentre map (default).
            if _method.startswith("geom"):
                # Geometrical centre of the acquired interferogram: the midpoint
                # sample. Derived from the scan geometry alone -- no burst search,
                # no dependence on signal quality.
                center = len(c_pos) // 2
            else:
                center = barycenter_map(sig)                 # (h, w) per-pixel index map

            scalar = np.ndim(center) == 0 # scalar = True only if the geometric midpoint is used, else False for a per-pixel barycentre.

            try:
                cpos_c = c_pos[center]              # scalar, or (h, w) per-pixel
                _c = float(cpos_c) if scalar else float(np.median(cpos_c))
                print(f"[Measurement PP] ZPD centre ({center_method}): {_c:.4f} mm"
                      + ("" if scalar else " (per-pixel median)"))
            except Exception:
                pass

            if scalar:
                from instruments.dsp import apodization_window
                apod = apodization_window(apod_type, len(c_pos), center)
            else:
                from instruments.dsp import apodization_window_map
                apod = apodization_window_map(apod_type, len(c_pos), center)
            apod3 = apod[:, np.newaxis, np.newaxis] if apod.ndim == 1 else apod
            return sig * apod3, center, c_pos

        signal, _, final_pos = preprocess(datacube)

        # Frequency grid
        start_freq, end_freq = self._get_frequency_limits(wl_start, wl_stop)
        # Generate frequencies in descending order so that wavelengths (which are inversely proportional) are ascending
        frequencies = np.linspace(end_freq, start_freq, n_freq)
        wavelengths = self._freq_to_wavelength(frequencies)

        # DFT: for each frequency, sum over positions
        # phase: (n_pos, n_freq)
        pos = final_pos.reshape(-1, 1)
        dpos = np.diff(final_pos)
        dpos = np.append(dpos, dpos[-1] if len(dpos) > 0 else 0)

        phase_kernel = np.exp(-2j * np.pi * pos * frequencies)  # (n_pos, n_freq)

        n_pos_final = len(final_pos)
        # DFT of signal
        weighted = signal * dpos[:, np.newaxis, np.newaxis]  # (n_pos, h, w)
        flat = weighted.reshape(n_pos_final, -1)     # (n_pos, h*w)
        spec_flat = phase_kernel.conj().T @ flat  # (n_freq, h*w)

        flat_out = spec_flat if complex_output else np.abs(spec_flat)
        spectrum_cube = flat_out.reshape(n_freq, h, w)

        # Magnitude spectra don't need float64 -- float32 halves the
        # cube size in RAM and on disk with no meaningful precision loss. The
        # complex form keeps the phase, at 2x the size of the float32 magnitude.
        return wavelengths, spectrum_cube.astype(
            np.complex64 if complex_output else np.float32)
