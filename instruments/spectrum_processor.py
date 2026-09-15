"""
spectrum_processor.py -- interferogram -> spectrum DFT.

Copied verbatim from the pump-probe repo (gmike92/Labview-pumprobepython,
sub_twins_lw.SpectrumProcessor) so the TWINS spectrum math here is identical to
the reference setup: moving-average baseline removal, NIREOS Gaussian
apodization, explicit DFT, and a calibration file mapping pseudo-frequency to
real wavelength (µm).

Defaults and the calibration-file path match the repo. If the calibration file
is absent it falls back to a plain 1/frequency conversion.
"""
import numpy as np
from pathlib import Path


# ----------------------------------------------------------------------------
# Defaults (from the repo)
# ----------------------------------------------------------------------------
DEFAULT_START_MM = 23.8     # Default start position (mm)
DEFAULT_STOP_MM = 24.8      # Default stop position (mm)
DEFAULT_N_STEPS = 120       # Default number of steps
DEFAULT_APODIZATION = 0.2   # Apodization width
DEFAULT_WL_START = 0.9      # Spectrum display start (µm) -- Forge SWIR band
DEFAULT_WL_STOP = 1.7       # Spectrum display stop (µm)  -- IMX990 cut-off

# Calibration file path
DEFAULT_CALIBRATION_FILE = r".\Twins\calibration\parameters_cal.txt"

# Spectral output sampling ("Auto"), imported from the reference repo. Zero-
# filling does NOT add true spectral resolution (fixed by scan length / max OPD);
# it only interpolates the curve. Auto n_points = ZEROFILL_FACTOR x scan steps,
# clamped to [ZEROFILL_MIN, ZEROFILL_MAX].
ZEROFILL_FACTOR = 1.5
ZEROFILL_MIN = 512
ZEROFILL_MAX = 4096

# Centerburst (ZPD) search defaults (NIREOS TWINS wedge).
DEFAULT_ZPD_MM = 24.33
DEFAULT_ZPD_WINDOW_MM = 0.1


def resolve_n_points(n_steps, manual=None):
    """Spectral output bins (interpolation only). manual>0 wins; else "Auto" =
    ZEROFILL_FACTOR x n_steps clamped to [ZEROFILL_MIN, ZEROFILL_MAX]."""
    if manual and manual > 0:
        return int(manual)
    return int(np.clip(ZEROFILL_FACTOR * int(n_steps), ZEROFILL_MIN, ZEROFILL_MAX))


def find_centerburst(signal_1d, positions, expected_zero_mm=None, search_mm=None):
    """Locate the ZPD (center burst) index from a 1-D interferogram.

    Uses the analytic-signal (Hilbert) envelope rather than argmax(|signal|):
    the envelope is smooth, so it picks the true burst instead of jumping to the
    tallest individual fringe or to a baseline edge artifact. If
    ``expected_zero_mm`` is given the search is limited to +/- ``search_mm``
    around it (default 5% of the scan span); otherwise the outer ~3% of points
    are excluded so baseline roll-off at the ends can't win.
    """
    s = np.asarray(signal_1d, dtype=float).ravel()
    n = s.size
    if n < 4:
        return int(np.argmax(np.abs(s))) if n else 0
    try:
        from scipy.signal import hilbert
        env = np.abs(hilbert(s - s.mean()))
    except Exception:  # noqa: BLE001
        env = np.abs(s - s.mean())

    pos = np.asarray(positions, dtype=float).ravel()
    span = abs(pos[-1] - pos[0]) if n > 1 else 0.0

    mask = np.ones(n, dtype=bool)
    if expected_zero_mm is not None and span > 0:
        hw = search_mm if search_mm is not None else max(0.05 * span, 3.0 * span / n)
        mask = np.abs(pos - float(expected_zero_mm)) <= hw
        if not mask.any():
            mask = np.ones(n, dtype=bool)
    else:
        guard = max(1, int(0.03 * n))
        mask[:guard] = False
        mask[-guard:] = False
    return int(np.argmax(np.where(mask, env, -np.inf)))


class SpectrumProcessor:
    """
    Process interferogram to spectrum using DFT.
    Uses calibration file to convert pseudo-frequency to real wavelength.
    """

    def __init__(self, calibration_file=None):
        self.interferogram = None
        self.positions = None
        self.spectrum = None
        self.wavelengths = None
        self.freq = None

        self.calibration_file = calibration_file or DEFAULT_CALIBRATION_FILE
        self.wavelength_cal = None
        self.reciprocal_cal = None
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration file for wavelength conversion."""
        try:
            import pandas as pd

            cal_path = Path(self.calibration_file)
            if not cal_path.exists():
                alt_paths = [
                    Path(r"C:\Users\mguizzardi\Desktop\Camera python\TWINS FILE\Twins\calibration\parameters_cal.txt"),
                    Path(r".\Twins\calibration\parameters_cal.txt"),
                ]
                for alt in alt_paths:
                    if alt.exists():
                        cal_path = alt
                        break

            if cal_path.exists():
                ref = pd.read_csv(cal_path, sep="\t", header=None)
                self.wavelength_cal = ref.iloc[0].to_numpy(dtype='float64')
                self.reciprocal_cal = ref.iloc[1].to_numpy(dtype='float64')
                print(f"[OK] Loaded calibration: {cal_path.name}")
                print(f"     Wavelength range: {self.wavelength_cal.min():.2f} - {self.wavelength_cal.max():.2f} µm")
            else:
                print(f"[WARN] Calibration file not found: {self.calibration_file}")
                print("       Using simple 1/frequency conversion")

        except Exception as e:
            print(f"[WARN] Error loading calibration: {e}")

    def set_data(self, positions, interferogram):
        self.positions = positions
        self.interferogram = interferogram

    def moving_average(self, data, window):
        if window < 2:
            return np.zeros_like(data)
        import pandas as pd
        ser = pd.Series(data)
        return ser.rolling(window=window, min_periods=1, center=True).mean().to_numpy()

    def _get_frequency_limits(self, wl_start, wl_stop):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            start_freq = fn(1.0 / wl_stop)
            end_freq = fn(1.0 / wl_start)
            return float(start_freq), float(end_freq)
        else:
            return 1.0 / wl_stop, 1.0 / wl_start

    def _freq_to_wavelength(self, frequencies):
        if self.wavelength_cal is not None and self.reciprocal_cal is not None:
            from scipy.interpolate import interp1d
            fn = interp1d(self.reciprocal_cal, 1.0 / self.wavelength_cal,
                          kind="linear", fill_value="extrapolate")
            inv_wavelength = fn(frequencies)
            return 1.0 / inv_wavelength
        else:
            return 1.0 / frequencies

    # -- scan-parameter estimation (imported from sub_twins_lw) --------------
    def max_step_um(self, wl_short_um, samples_per_cycle=5):
        """Maximum stage step (µm) for `samples_per_cycle` points per optical
        cycle at the shortest wavelength. samples_per_cycle=2 is the Nyquist
        limit; 5 is a safe oversampling default.

            k_max  = reciprocal_cal(1/λ_short)        [1/mm-stage]
            Δx_max = 1 / (samples_per_cycle · k_max)  [mm]
        """
        if not wl_short_um or wl_short_um <= 0:
            return None
        if not samples_per_cycle or samples_per_cycle <= 0:
            return None
        if self.wavelength_cal is None or self.reciprocal_cal is None:
            return wl_short_um * 1000.0 / float(samples_per_cycle)
        try:
            from scipy.interpolate import interp1d
            fn = interp1d(1.0 / self.wavelength_cal, self.reciprocal_cal,
                          kind="linear", fill_value="extrapolate")
            k_max = float(fn(1.0 / wl_short_um))   # 1/mm-stage
            if k_max <= 0:
                return None
            return (1.0 / (float(samples_per_cycle) * k_max)) * 1000.0  # µm
        except Exception:  # noqa: BLE001
            return None

    def nyquist_step_um(self, wl_short_um):
        """Nyquist (2 samples/cycle) stage step at the shortest wavelength."""
        return self.max_step_um(wl_short_um, samples_per_cycle=2)

    def estimate_resolution(self, scan_range_mm, wl_center_um, apod_type="happ-genzel"):
        """Spectral resolution from the stage scan range, as ``(value, unit)``.

        The scan sets the pseudo-frequency FWHM ``δk`` (apodization-broadened FWHM
        of the chosen FTIR window, or ``1/scan_range``):
          - no calibration  -> ``(δk, "1/mm")`` -- pseudo-frequency resolution;
          - calibration (optical frequency THz vs pseudo-frequency k) ->
            ``(|dν/dk|·δk, "THz")`` -- optical-frequency resolution at the band
            centre.
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

    def compute_spectrum(self, wl_start=8.0, wl_stop=14.0,
                         n_points=10000, invert=False,
                         expected_zero_mm=None, search_mm=None, apod_type="happ-genzel"):
        """Compute spectrum from interferogram using DFT."""
        if self.interferogram is None or self.positions is None:
            return None, None

        window_size = max(1, len(self.interferogram) // 5)
        baseline = self.moving_average(self.interferogram, window_size)
        signal = self.interferogram - baseline

        if invert:
            signal = -signal

        # Remove the TWINS wedge motor's reproducible nonlinearity (no-op if the
        # parameters_int.txt position calibration isn't present).
        try:
            from instruments.calibration import calibrate_position_axis
            c_positions = np.asarray(calibrate_position_axis(self.positions), dtype=float)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] motor calibration skipped: {e}")
            c_positions = np.asarray(self.positions, dtype=float)

        center_idx = find_centerburst(signal, c_positions, expected_zero_mm, search_mm)
        try:
            print(f"[SpectrumProcessor] ZPD (burst center): "
                  f"{c_positions[center_idx]:.4f} mm (index {center_idx})")
        except Exception:  # noqa: BLE001
            pass

        from instruments.dsp import apodization_window
        window = apodization_window(apod_type, len(signal), center_idx)
        apodized = signal * window

        start_freq, end_freq = self._get_frequency_limits(wl_start, wl_stop)
        frequencies = np.linspace(end_freq, start_freq, n_points)

        pos = c_positions.reshape(-1, 1)
        dpos = np.diff(c_positions)
        dpos = np.append(dpos, dpos[-1] if len(dpos) > 0 else 0)

        phase = -2j * np.pi * pos * frequencies
        spectrum = (dpos * apodized).dot(np.exp(phase))
        spectrum = np.abs(spectrum)

        wavelengths = self._freq_to_wavelength(frequencies)

        self.freq = frequencies
        self.wavelengths = wavelengths
        self.spectrum = spectrum
        self.apodized_signal = apodized
        self.apodized_positions = c_positions

        return wavelengths, spectrum
