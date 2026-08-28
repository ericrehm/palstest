import numpy as np
import pandas as pd

# from r9880U import R9880UResponsivity, R9880UGain


class R9880UResponsivity:
    """
    Interpolate R9880U cathode responsivity
    from digitized WebPlotDigitizer data.

    The original datasheet plot is log-linear:
      - x-axis: wavelength (nm), linear
      - y-axis: responsivity (mA/W), logarithmic

    To respect that, this class interpolates in log(responsivity)
    as a function of wavelength.

    Parameters
    ----------
    csv_path : str
        Path to the CSV exported from WebPlotDigitizer.
    wavelength_col : str, optional
        Name of the wavelength column in the CSV (nm).
        Default is 'wavelength_nm'.
    responsivity_col : str, optional
        Name of the responsivity column in the CSV (mA/W).
        These should be *linear* values, not log10.
        Default is 'responsivity_mA_per_W'.
    rel_digitization_sigma : float, optional
        Baseline relative 1-sigma uncertainty due to reading error,
        grid resolution, etc. e.g. 0.05 = 5%. Default: 0.07 (7%).
    rel_interpolation_sigma : float, optional
        Extra relative 1-sigma uncertainty added when interpolating
        between points. This scales with the fractional distance
        between neighboring sample points. Default: 0.05 (5%).

    Notes
    -----
    - If your WebPlotDigitizer export gave you log10(y) values instead
      of linear responsivity, set `responsivity_is_log10=True` (see
      alternate constructor `from_arrays` below) or pre-transform them
      before constructing this class.
    """

    def __init__(self,
                 csv_path,
                 wavelength_col='wavelength_nm',
                 responsivity_col='responsivity_mA_per_W',
                 rel_digitization_sigma=0.07,
                 rel_interpolation_sigma=0.05):
        # Load CSV
        df = pd.read_csv(csv_path)

        if wavelength_col not in df.columns:
            raise KeyError(f"Column '{wavelength_col}' not found in CSV "
                           f"columns={list(df.columns)}")
        if responsivity_col not in df.columns:
            raise KeyError(f"Column '{responsivity_col}' not found in CSV "
                           f"columns={list(df.columns)}")

        wl = df[wavelength_col].to_numpy(dtype=float)
        resp = df[responsivity_col].to_numpy(dtype=float)

        self._init_from_arrays(
            wl, resp,
            rel_digitization_sigma=rel_digitization_sigma,
            rel_interpolation_sigma=rel_interpolation_sigma
        )

    @classmethod
    def from_arrays(cls,
                    wavelength_nm,
                    responsivity,
                    rel_digitization_sigma=0.07,
                    rel_interpolation_sigma=0.05,
                    responsivity_is_log10=False):
        """
        Alternate constructor from in-memory arrays instead of CSV.

        Parameters
        ----------
        wavelength_nm : array-like
            Wavelength samples in nm.
        responsivity : array-like
            Responsivity samples (mA/W if responsivity_is_log10=False).
            If responsivity_is_log10=True, these are log10(mA/W) values.
        rel_digitization_sigma : float
            Baseline relative digitization 1-sigma uncertainty.
        rel_interpolation_sigma : float
            Relative interpolation 1-sigma component.
        responsivity_is_log10 : bool
            If True, `responsivity` is log10(mA/W); will be exponentiated.
        """
        obj = cls.__new__(cls)
        if responsivity_is_log10:
            resp_linear = 10 ** np.asarray(responsivity, dtype=float)
        else:
            resp_linear = np.asarray(responsivity, dtype=float)

        obj._init_from_arrays(
            np.asarray(wavelength_nm, dtype=float),
            resp_linear,
            rel_digitization_sigma=rel_digitization_sigma,
            rel_interpolation_sigma=rel_interpolation_sigma
        )
        return obj

    def _init_from_arrays(self,
                          wl,
                          resp,
                          rel_digitization_sigma=0.07,
                          rel_interpolation_sigma=0.05):
        # Filter NaNs and non-positive values
        mask = np.isfinite(wl) & np.isfinite(resp) & (resp > 0)
        if np.count_nonzero(mask) < 2:
            raise ValueError("Not enough valid data points after filtering "
                             "NaNs and non-positive responsivity values.")

        wl = wl[mask]
        resp = resp[mask]

        # Sort by wavelength
        order = np.argsort(wl)
        self.wavelength_nm = wl[order]
        self.resp_mA_per_W = resp[order]

        # Precompute log-space values for interpolation
        self.log_resp = np.log(self.resp_mA_per_W)  # natural log

        # Store uncertainty model parameters
        self.rel_digitization_sigma = float(rel_digitization_sigma)
        self.rel_interpolation_sigma = float(rel_interpolation_sigma)

        # Precompute typical spacing for interpolation-uncertainty scaling
        diffs = np.diff(self.wavelength_nm)
        self.mean_spacing = np.mean(diffs)

    @property
    def wavelength_range(self):
        return float(self.wavelength_nm[0]), float(self.wavelength_nm[-1])

    def _fractional_distance_to_nearest_points(self, wl):
        """
        For each query wavelength, estimate fractional distance between
        neighboring sample wavelengths. 0 = at a data point, 1 = halfway
        between neighbors (approx). Used to scale interpolation uncertainty.
        """
        wl_samples = self.wavelength_nm
        idx = np.searchsorted(wl_samples, wl)

        # Left and right neighbors
        idx_left = np.clip(idx - 1, 0, len(wl_samples) - 1)
        idx_right = np.clip(idx, 0, len(wl_samples) - 1)

        wl_left = wl_samples[idx_left]
        wl_right = wl_samples[idx_right]

        span = np.maximum(wl_right - wl_left, 1e-12)
        frac = np.abs(wl - wl_left) / span
        frac = np.clip(frac, 0.0, 1.0)
        return frac

    def __call__(self, wavelength_nm, extrapolate=False):
        """
        Interpolate responsivity at the given wavelength(s).

        Parameters
        ----------
        wavelength_nm : float or array-like
            Wavelength(s) in nm at which to interpolate the responsivity.
        extrapolate : bool, optional
            If False (default), raise ValueError when wavelength is
            outside the digitized range. If True, allow log-linear
            extrapolation, which may be unreliable.

        Returns
        -------
        resp : float or np.ndarray
            Interpolated responsivity in mA/W.
        sigma : float or np.ndarray
            Estimated 1-sigma uncertainty in mA/W.

        Uncertainty model
        -----------------
        sigma_rel^2 = sigma_digitization^2 + (f * sigma_interp)^2

        where:
            - sigma_digitization = rel_digitization_sigma
            - sigma_interp = rel_interpolation_sigma
            - f = fractional distance between nearest sample points
                  (0 at a sample, ~1 midway between them)
        """
        wl_query = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))

        wl_min, wl_max = self.wavelength_range

        if not extrapolate:
            if np.any((wl_query < wl_min) | (wl_query > wl_max)):
                raise ValueError(
                    f"Query wavelength outside digitized range "
                    f"[{wl_min:.1f}, {wl_max:.1f}] nm. "
                    f"Got min={wl_query.min():.1f}, max={wl_query.max():.1f}."
                )

        # Log-space interpolation (consistent with log-linear original graph)
        log_resp_interp = np.interp(
            wl_query,
            self.wavelength_nm,
            self.log_resp,
            left=np.nan,
            right=np.nan
        )
        resp_interp = np.exp(log_resp_interp)

        # Relative uncertainty model
        frac = self._fractional_distance_to_nearest_points(wl_query)
        sigma_rel = np.sqrt(
            self.rel_digitization_sigma ** 2 +
            (frac * self.rel_interpolation_sigma) ** 2
        )
        sigma = sigma_rel * resp_interp

        if np.isscalar(wavelength_nm):
            return float(resp_interp[0]), float(sigma[0])
        return resp_interp, sigma


class R9880UGain:
    """
    Power-law gain model for the R9880U PMT, based on digitized
    gain vs. supply voltage data from the log-log curve.

    Model:
        G(V) = A * V**k

    The fit is performed in log-space:
        log(G) = k * log(V) + log(A)

    Parameters
    ----------
    csv_path : str
        Path to CSV file with at least two columns: voltage (V) and gain.
    voltage_col : str, optional
        Name of the voltage column in the CSV. Default: 'voltage_V'.
    gain_col : str, optional
        Name of the gain column in the CSV. Default: 'gain'.
    rel_digitization_sigma : float, optional
        Baseline relative 1-sigma uncertainty to account for
        digitization error from the datasheet curve (default: 0.07 = 7%).
    rel_model_sigma : float, optional
        Additional relative 1-sigma term to represent model / fit
        imperfections, scaled by how far you are (in log V) from the
        center of the fitted range (default: 0.05 = 5%).

    Notes
    -----
    - With only two points, the power-law fit is exact in log-space,
      so residual-based uncertainty is ~0. We therefore adopt a
      phenomenological uncertainty model with:
        * A fixed digitization term (rel_digitization_sigma)
        * A model term that grows slowly away from the center of the
          fitted log(V) range.
    """

    def __init__(self,
                 csv_path,
                 voltage_col='voltage_V',
                 gain_col='gain',
                 rel_digitization_sigma=0.07,
                 rel_model_sigma=0.05):
        df = pd.read_csv(csv_path)

        if voltage_col not in df.columns:
            raise KeyError(f"Column '{voltage_col}' not found in CSV "
                           f"columns={list(df.columns)}")
        if gain_col not in df.columns:
            raise KeyError(f"Column '{gain_col}' not found in CSV "
                           f"columns={list(df.columns)}")

        voltage = df[voltage_col].to_numpy(dtype=float)
        gain = df[gain_col].to_numpy(dtype=float)

        self._init_from_arrays(voltage,
                               gain,
                               rel_digitization_sigma=rel_digitization_sigma,
                               rel_model_sigma=rel_model_sigma)

    @classmethod
    def from_arrays(cls,
                    voltage_V,
                    gain,
                    rel_digitization_sigma=0.07,
                    rel_model_sigma=0.05):
        """
        Alternate constructor from in-memory arrays.

        Parameters
        ----------
        voltage_V : array-like
            Supply voltages in volts.
        gain : array-like
            Dimensionless gain values.
        rel_digitization_sigma : float
            Baseline relative digitization 1-sigma uncertainty.
        rel_model_sigma : float
            Relative model 1-sigma uncertainty scale.
        """
        obj = cls.__new__(cls)
        obj._init_from_arrays(np.asarray(voltage_V, dtype=float),
                              np.asarray(gain, dtype=float),
                              rel_digitization_sigma=rel_digitization_sigma,
                              rel_model_sigma=rel_model_sigma)
        return obj

    def _init_from_arrays(self,
                          voltage,
                          gain,
                          rel_digitization_sigma=0.07,
                          rel_model_sigma=0.05):
        # Filter bad values
        mask = np.isfinite(voltage) & np.isfinite(gain) & (voltage > 0) & (gain > 0)
        if np.count_nonzero(mask) < 2:
            raise ValueError("Need at least two valid (V, G) points with V>0 and G>0.")

        v = voltage[mask]
        g = gain[mask]

        # Sort by voltage
        order = np.argsort(v)
        self.voltage_V = v[order]
        self.gain = g[order]

        # Log-transform for fitting
        logV = np.log(self.voltage_V)
        logG = np.log(self.gain)

        # Linear fit in log-log space: logG = k*logV + b
        # np.polyfit returns [k, b]
        k, b = np.polyfit(logV, logG, 1)
        self.k = float(k)
        self.logA = float(b)
        self.A = float(np.exp(b))

        # Store range and center in log(V) for uncertainty scaling
        self.logV_min = float(logV.min())
        self.logV_max = float(logV.max())
        self.logV_center = 0.5 * (self.logV_min + self.logV_max)
        self.logV_halfspan = max(self.logV_center - self.logV_min, 1e-12)

        # Uncertainty model parameters
        self.rel_digitization_sigma = float(rel_digitization_sigma)
        self.rel_model_sigma = float(rel_model_sigma)

    @property
    def voltage_range(self):
        """Return the min and max voltages covered by the fit, in volts."""
        return float(self.voltage_V[0]), float(self.voltage_V[-1])

    def _relative_model_uncertainty(self, logV):
        """
        Model term grows (slowly) with distance from the center of the
        fitted log(V) range:

            sigma_model_rel = rel_model_sigma * (1 + d)

        where d is the absolute distance (in units of half-span) from
        the center in log-space.
        """
        d = np.abs(logV - self.logV_center) / self.logV_halfspan
        return self.rel_model_sigma * (1.0 + d)

    def __call__(self, voltage_V, extrapolate=False):
        """
        Evaluate gain at the given voltage(s), with uncertainty.

        Parameters
        ----------
        voltage_V : float or array-like
            Supply voltage(s) in volts.
        extrapolate : bool, optional
            If False (default), raise ValueError when voltage is
            outside the fitted range. If True, allow power-law
            extrapolation, but uncertainty will grow.

        Returns
        -------
        gain : float or np.ndarray
            Modeled gain G(V) = A * V**k.
        sigma_gain : float or np.ndarray
            Estimated 1-sigma uncertainty in gain.

        Uncertainty model
        -----------------
        The relative 1-sigma uncertainty is:

            sigma_rel^2 = sigma_digitization^2 + sigma_model^2

        where:
            sigma_digitization = rel_digitization_sigma
            sigma_model = rel_model_sigma * (1 + d)

        and d is the distance in log(V) from the center of the
        fitted range, measured in units of the half-span.
        """
        V = np.atleast_1d(np.asarray(voltage_V, dtype=float))

        if np.any(V <= 0):
            raise ValueError("Voltage must be positive.")

        vmin, vmax = self.voltage_range
        if not extrapolate:
            if np.any((V < vmin) | (V > vmax)):
                raise ValueError(
                    f"Query voltage outside fitted range "
                    f"[{vmin:.1f}, {vmax:.1f}] V. "
                    f"Got min={V.min():.1f}, max={V.max():.1f}."
                )

        logV = np.log(V)

        # Power-law model in log-space
        logG = self.k * logV + self.logA
        G = np.exp(logG)

        # Relative uncertainty components
        sigma_digit_rel = self.rel_digitization_sigma
        sigma_model_rel = self._relative_model_uncertainty(logV)

        sigma_rel = np.sqrt(sigma_digit_rel**2 + sigma_model_rel**2)
        sigma_G = sigma_rel * G

        if np.isscalar(voltage_V):
            return float(G[0]), float(sigma_G[0])
        return G, sigma_G

def main():
    
    from pathlib import Path

    rootDir = Path(__file__).parent.parent.parent.parent
    dataDir = rootDir / Path("data") / Path("PMT")


    # Initialize from your WebPlotDigitizer CSV
    R_interp = R9880UResponsivity(dataDir / "R9880U210_responsivity_digitized.csv")

    # Check the domain
    print("Valid wavelength range (nm):", R_interp.wavelength_range)

    # Get the responsivity at 532 nm
    wl = 532.0
    # wl = 355.0
    resp_532, sigma_532 = R_interp(wl)

    print(f"R( {wl:.1f} nm ) = {resp_532/1000:.3e} A/W ± {sigma_532/1000:.3e} A/W (1σ)")


    gain_model = R9880UGain(dataDir / "R9880U20_gain_digitized.csv")

    print("Valid voltage range (V):", gain_model.voltage_range)

    V = 3 * 250  # 750V
    gain, sigma = gain_model(V)
    print(f"G({V:.1f} V) = {gain:.3e} ± {sigma:.3e} (1σ)")

if __name__ == "__main__":
    main()