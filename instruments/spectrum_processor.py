"""
spectrum_processor.py -- interferogram -> spectrum DFT.

"""
import numpy as np
from pathlib import Path


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_START_MM = 23.8     # Default start position (mm)
DEFAULT_STOP_MM = 24.8      # Default stop position (mm)
DEFAULT_N_STEPS = 120       # Default number of steps
DEFAULT_WL_START = 0.4      # Spectrum display start (µm) -- Forge SWIR band
DEFAULT_WL_STOP = 1.7       # Spectrum display stop (µm)  -- IMX990 cut-off

# Calibration file path
DEFAULT_CALIBRATION_FILE = r".\Twins\calibration\parameters_cal.txt"


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

    def compute_spectrum(self, wl_start=8.0, wl_stop=14.0,
                         n_points=10000,
                         apod_type="happ-genzel", center_method="barycenter"):
        """Compute spectrum from interferogram using DFT.

        `center_method` sets where the apodization window is centred:
        "barycenter" (default) = the I^2 barycentre of the interferogram (same
        formula as the 2-D per-pixel barycentre); "geometric" = the midpoint
        sample of the scan, ignoring the signal.
        """
        if self.interferogram is None or self.positions is None:
            return None, None

        window_size = max(1, len(self.interferogram) // 5)
        baseline = self.moving_average(self.interferogram, window_size)
        signal = self.interferogram - baseline

        # Remove the TWINS wedge motor's reproducible nonlinearity (no-op if the
        # parameters_int.txt position calibration isn't present).
        try:
            from instruments.calibration import calibrate_position_axis
            c_positions = np.asarray(calibrate_position_axis(self.positions), dtype=float)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] motor calibration skipped: {e}")
            c_positions = np.asarray(self.positions, dtype=float)

        # Apodization centre (ZPD) from the chosen method.
        sig = np.asarray(signal, dtype=float)
        if str(center_method).lower().startswith("geom"):
            center_idx = len(sig) // 2
        else:
            w2 = sig ** 2                            # I^2 barycentre
            k = np.arange(len(sig), dtype=float)
            center_idx = int(np.round(np.sum(k * w2) / (np.sum(w2) + 1.0)))
        center_idx = int(np.clip(center_idx, 0, max(0, len(sig) - 1)))
        try:
            print(f"[SpectrumProcessor] ZPD centre ({center_method}): "
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

        self.wavelengths = wavelengths
        self.spectrum = spectrum

        return wavelengths, spectrum
