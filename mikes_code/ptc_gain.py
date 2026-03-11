"""
Photon Transfer Curve (PTC) analysis for up-the-ramp detector data.

Given a 3-D array of images with shape (n_exposures, n_reads, n_pixels),
this module fits a linear PTC model to extract:

    - Detector gain  [e⁻/DN]
    - Read noise     [DN  and  e⁻]

PTC model
---------
For a shot-noise-limited detector the variance scales linearly with signal:

    Var(signal)  =  (1 / gain) × Mean(signal)  +  σ_read²

Fitting slope = 1/gain and intercept = σ_read² (in DN²).

Linearity correction
--------------------
Real detectors are non-linear.  Before computing the PTC the code fits a
polynomial to each pixel's (expected_linear_signal, observed_signal) curve
and inverts it.  This maps each observed DN value back onto a linearised
signal axis so the subsequent PTC fit is unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PTCResult:
    """All outputs from a PTC fit."""

    gain: float
    """Detector gain in e⁻/DN."""

    read_noise_dn: float
    """Read noise in DN (= sqrt of the PTC intercept)."""

    read_noise_e: float
    """Read noise in electrons (= read_noise_dn × gain)."""

    all_means: np.ndarray
    """Linearised mean signal for every (pixel, read) pair, in DN."""

    all_vars: np.ndarray
    """Temporal variance for every (pixel, read) pair, in DN²."""

    fit_params: np.ndarray
    """Optimised [intercept, slope] of the linear PTC model."""

    # Linearity diagnostics — populated during compute_ptc
    # Each array has shape (n_sample_pixels, n_reads)
    linear_expectation: np.ndarray | None = None
    """Ideal linear signal at each read for a sample of pixels [DN]."""

    observed_mean_bias_sub: np.ndarray | None = None
    """Bias-subtracted observed mean signal at each read, same sample [DN]."""

    corrected_mean: np.ndarray | None = None
    """Linearised (corrected) mean signal at each read, same sample [DN]."""

    def __str__(self) -> str:
        return (
            f"PTC fit results\n"
            f"  Gain       : {self.gain:.3f} e⁻/DN\n"
            f"  Read noise : {self.read_noise_dn:.3f} DN  "
            f"({self.read_noise_e:.3f} e⁻)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_linearity_corrector(
    expected_signal: np.ndarray,
    observed_mean: np.ndarray,
    observed_std: np.ndarray,
    fit_max_mean_dn: np.float,
    poly_degree: int,
) -> tuple[interp1d, Polynomial]:
    """
    Fit a polynomial mapping (expected linear signal) → (observed DN) for
    one pixel, then return an interpolating function for the *inverse*:
    observed DN → linearised signal.

    Parameters
    ----------
    expected_signal : 1-D array, shape (n_reads,)
        Ideal linear ramp values (read index × mean flux rate).
    observed_mean : 1-D array, shape (n_reads,)
        Bias-subtracted mean DN per read.
    observed_std : 1-D array, shape (n_reads,)
        Standard deviation per read (used as inverse weights).
    fit_max_mean_dn: float
        Maximum dn to include in fit (to avoid saturated regions)
    poly_degree : int
        Degree of the polynomial non-linearity model.

    Returns
    -------
    f_inv : callable
        Interpolating inverse: f_inv(observed_DN) → linearised_DN.
    poly : Polynomial
        Forward polynomial: poly(linear_signal) → observed_DN.
    """
    in_range = np.where(observed_mean < fit_max_mean_dn)[0]
    weights = 1.0 / np.clip(observed_std[in_range], 1e-6, None)  # avoid /0
    poly = Polynomial.fit(expected_signal[in_range], observed_mean[in_range], deg=poly_degree, w=weights)

    # Sample the polynomial on a fine grid to build the inverse interpolant
    x_fine = np.linspace(expected_signal[0], expected_signal[in_range][-1], 200)
    y_fine = poly(x_fine)

    f_inv = interp1d(y_fine, x_fine, kind="linear", bounds_error=False,
                    fill_value="extrapolate")
    return f_inv, poly


def _linearise_pixel(
    raw_stack: np.ndarray,
    bias_level: float,
    f_inv: interp1d,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply linearity correction to all exposures of one pixel.

    Parameters
    ----------
    raw_stack : 2-D array, shape (n_exposures, n_reads)
        Raw DN values for one pixel across all exposures and reads.
    bias_level : float
        Mean bias (zeroth-read level) to subtract before correction.
    f_inv : callable
        Inverse linearity map from `_build_linearity_corrector`.

    Returns
    -------
    mean_per_read : 1-D array, shape (n_reads,)
    var_per_read  : 1-D array, shape (n_reads,)
    """
    bias_corrected = raw_stack - bias_level          # subtract bias
    linearised = f_inv(bias_corrected)               # undo non-linearity
    mean_per_read = linearised.mean(axis=0)
    var_per_read = linearised.std(axis=0) ** 2
    return mean_per_read, var_per_read


def _fit_ptc_line(
    mean_signal: np.ndarray,
    variance: np.ndarray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> np.ndarray:
    """
    Fit the linear PTC model  Var = slope × Mean + intercept  using a
    robust (Cauchy) least-squares estimator.

    Only data points with  0 ≤ mean ≤ x_max  and  0 ≤ var ≤ y_max  are
    included in the fit.

    Returns
    -------
    params : ndarray, shape (2,)
        [intercept, slope]  →  [σ_read², 1/gain]
    """
    x = mean_signal.ravel()
    y = variance.ravel()

    in_range = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= x_min) & (x <= x_max)
        & (y >= y_min) & (y <= y_max)
    )
    x_fit, y_fit = x[in_range], y[in_range]

    def residuals(params: np.ndarray) -> np.ndarray:
        intercept, slope = params
        return y_fit - (slope * x_fit + intercept)

    initial_params = np.array([10.0, 0.02])   # [intercept_DN², slope~1/gain]
    result = least_squares(residuals, initial_params, loss="soft_l1")

    print(f"Optimizer status {result.status}: {result.message}")
    return result.x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ptc(
    images: np.ndarray,
    poly_degree: int = 4,
    fit_min_mean_dn: float=10_000,
    fit_min_var_dn2: float=100**2,
    fit_max_mean_dn: float = 25_000,
    fit_max_var_dn2: float = 350 ** 2,
    plot_max_mean_dn: float = 25_000,
    plot_max_var_dn2: float | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> PTCResult:
    """
    Compute the Photon Transfer Curve from an up-the-ramp image stack
    and return gain and read noise.

    Parameters
    ----------
    images : ndarray, shape (n_exposures, n_reads, n_pixels)
        Raw detector data.  n_exposures >= 2 (temporal variance requires
        at least two independent exposures per read level).

    poly_degree : int, default 4
        Degree of the polynomial used to model and correct pixel
        non-linearity before PTC fitting.

    fit_max_mean_dn : float, default 25 000
        Upper signal limit [DN] included in the PTC linear fit.
        Points above this value are excluded to avoid saturation effects.

    fit_max_var_dn2 : float, default 122 500  (= 350^2)
        Upper variance limit [DN^2] included in the PTC linear fit.

    plot_max_mean_dn : float, default 25 000
        Upper signal limit [DN] shown in the 2-D histogram plot.

    plot_max_var_dn2 : float or None
        Upper variance limit [DN^2] shown in the histogram.
        Defaults to ``fit_max_var_dn2`` when None.

    title : str or None
        Optional title for the output figure.

    save_path : str or None
        If given, the figure is saved to this path (e.g. ``"ptc.png"``).

    Returns
    -------
    PTCResult
        Dataclass with ``.gain``, ``.read_noise_dn``, ``.read_noise_e``,
        ``.mean_signal``, ``.variance``, and ``.fit_params``.

    Raises
    ------
    ValueError
        If ``images`` does not have exactly three dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # 10 exposures, 20 reads, 512 pixels; true gain = 2 e-/DN
    >>> images = rng.poisson(np.arange(20)[None,:,None] * 500, (10, 20, 512)).astype(float)
    >>> result = compute_ptc(images)
    >>> print(result)
    """
    if images.ndim != 3:
        raise ValueError(
            f"images must be 3-D (n_exposures, n_reads, n_pixels), "
            f"got shape {images.shape}"
        )

    if plot_max_var_dn2 is None:
        plot_max_var_dn2 = fit_max_var_dn2

    n_exposures, n_reads, n_pixels = images.shape

    # ---- Per-pixel summary statistics across exposures -------------------
    # mean[read, pixel]  and  std[read, pixel]
    per_read_mean = images.mean(axis=0)   # (n_reads, n_pixels)
    per_read_std  = images.std(axis=0)    # (n_reads, n_pixels)

    # ---- Build the expected-linear-signal model for each pixel -----------
    #   bias  : average DN at the zeroth read (one value per pixel)
    #   flux  : mean DN deposited between read 1 and read 2 (proxy for rate)
    #   linear_model[read, pixel] = read_index x flux_rate
    bias         = 0*per_read_mean[0, :]                      # (n_pixels,)
    flux_rate    = per_read_mean[2, :] - per_read_mean[1, :]  # (n_pixels,)
    read_index   = np.arange(n_reads)                         # (n_reads,)
    linear_model = read_index[:, np.newaxis] * flux_rate[np.newaxis, :] + per_read_mean[0, :]
    # (n_reads, n_pixels)

    # Bias-subtracted observed mean for fitting the non-linearity polynomial
    bias_sub_mean = per_read_mean - bias[np.newaxis, :]  # (n_reads, n_pixels)

    # ---- Linearity correction and PTC point collection -------------------
    all_means = np.empty((n_pixels, n_reads))
    all_vars  = np.empty((n_pixels, n_reads))

    # Store linearity diagnostics for a representative sample of pixels
    # (up to 9, evenly spaced) so we can plot them without overwhelming memory
    n_sample = min(9, n_pixels)
    sample_idx = np.round(np.linspace(0, n_pixels - 1, n_sample)).astype(int)
    lin_expected  = np.empty((n_sample, n_reads))  # ideal linear signal
    lin_observed  = np.empty((n_sample, n_reads))  # bias-sub observed mean
    lin_corrected = np.empty((n_sample, n_reads))  # corrected mean

    for pix in range(n_pixels):
        f_inv, poly = _build_linearity_corrector(
            expected_signal = linear_model[:, pix],
            observed_mean   = bias_sub_mean[:, pix],
            observed_std    = per_read_std[:, pix],
            fit_max_mean_dn = fit_max_mean_dn,
            poly_degree     = poly_degree,
        )
        raw_stack = images[:, :, pix]          # (n_exposures, n_reads)
        mean_lin, var_lin = _linearise_pixel(raw_stack, bias[pix], f_inv)
        all_means[pix] = mean_lin
        all_vars[pix]  = var_lin

        # Save diagnostics for sampled pixels
        sample_pos = np.where(sample_idx == pix)[0]
        if sample_pos.size:
            s = sample_pos[0]
            lin_expected[s]  = linear_model[:, pix]
            lin_observed[s]  = bias_sub_mean[:, pix]
            lin_corrected[s] = mean_lin

    # ---- Fit the linear PTC model ----------------------------------------
    fit_params = _fit_ptc_line(all_means, all_vars, fit_min_mean_dn, fit_min_var_dn2, fit_max_mean_dn, fit_max_var_dn2)
    intercept, slope = fit_params

    gain          = 1.0 / slope
    read_noise_dn = np.sqrt(max(intercept, 0.0))
    read_noise_e  = read_noise_dn * gain

    result = PTCResult(
        gain                  = gain,
        read_noise_dn         = read_noise_dn,
        read_noise_e          = read_noise_e,
        all_means             = all_means,
        all_vars              = all_vars,
        fit_params            = fit_params,
        linear_expectation    = lin_expected,
        observed_mean_bias_sub= lin_observed,
        corrected_mean        = lin_corrected,
    )
    print(result)

    # ---- Plots -----------------------------------------------------------
    _plot_ptc_histogram(
        result,
        plot_max_mean_dn = plot_max_mean_dn,
        plot_max_var_dn2 = plot_max_var_dn2,
        fit_min_mean_dn  = fit_min_mean_dn,
        fit_min_var_dn2  = fit_min_var_dn2,
        fit_max_mean_dn  = fit_max_mean_dn,
        fit_max_var_dn2  = fit_max_var_dn2,
        title            = title,
        save_path        = save_path,
    )
    _plot_raw_ptc(bias_sub_mean, per_read_std, all_means, all_vars, fit_params, plot_max_mean_dn)

    _plot_linearisation(result,f_inv,fit_max_mean_dn)

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ptc_model(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate  y = slope * x + intercept."""
    intercept, slope = params
    return slope * x + intercept


def _plot_linearisation(result: PTCResult, f_inv, fit_max_mean_dn) -> None:
    """
    Visualise the linearity correction for a sample of pixels.

    Two panels are shown for each sampled pixel:

    Top panel — raw observed signal vs. linear expectation
        • Data points  : bias-subtracted observed mean at each read
        • Dashed line  : ideal linear expectation (y = x, perfect linearity)
        Deviation of the data from the dashed line reveals the raw
        non-linearity of the detector.

    Bottom panel — residuals before and after correction
        • Orange dots  : (observed − expected) / expected  × 100  [%]
          i.e. the raw non-linearity error
        • Blue dots    : (corrected − expected) / expected  × 100  [%]
          i.e. the residual error after the polynomial correction
        A flat line at 0 % in the bottom panel means perfect linearisation.
    """
    if result.linear_expectation is None:
        return

    n_sample = result.linear_expectation.shape[0]
    ncols = min(3, n_sample)
    nrows = int(np.ceil(n_sample / ncols))

    fig, axes = plt.subplots(
        nrows * 2, ncols,
        figsize=(4.5 * ncols, 3.5 * nrows * 2),
        sharex="col",
    )
    # Normalise axes shape to (2*nrows, ncols) for uniform indexing
    if n_sample == 1:
        axes = np.array(axes).reshape(2, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Linearity correction — sample pixels", fontsize=13, y=1.01)

    for s in range(n_sample):
        row_top = (s // ncols) * 2          # top panel row index
        row_bot = row_top + 1               # residual panel row index
        col     = s % ncols

        ax_top = axes[row_top, col]
        ax_bot = axes[row_bot, col]

        expected  = result.linear_expectation[s]      # ideal linear [DN]
        observed  = result.observed_mean_bias_sub[s]  # raw observed [DN]
        corrected = result.corrected_mean[s]           # after correction [DN]

        # --- Top panel: observed vs. expected --------------------------------
        ax_top.plot(expected, expected,
                    color="gray", ls="--", lw=1.5, label="Ideal (y = x)")
        ax_top.scatter(expected, observed,
                       s=18, color="tomato", zorder=3, label="Observed (raw)")
        ax_top.scatter(expected, corrected,
                       s=18, color="steelblue", zorder=3, marker="^",
                       label="Corrected")
        y_fine = np.arange(np.min(observed), np.max(observed),100)
        ax_top.plot(f_inv(y_fine), y_fine, color='gray', alpha=0.5, label='Inverse Function')

        ax_top.set_ylabel("Signal [DN]")
        ax_top.set_title(f"Pixel sample {s + 1}", fontsize=9)
        if s == 0:
            ax_top.legend(fontsize=8, loc="upper left")

        # --- Bottom panel: residuals [%] -------------------------------------
        # Guard against zeros in expected (first read is 0 DN)
        safe_exp = np.where(np.abs(expected) > 1, expected, np.nan)

        raw_resid  = (observed  - expected) / safe_exp * 100
        corr_resid = (corrected - expected) / safe_exp * 100

        ax_bot.axhline(0, color="gray", ls="--", lw=1)
        ax_bot.scatter(expected, raw_resid,
                       s=18, color="tomato",    zorder=3, label="Raw residual")
        ax_bot.scatter(expected, corr_resid,
                       s=18, color="steelblue", zorder=3, marker="^",
                       label="Corrected residual")
        ax_bot.axvline(fit_max_mean_dn,color='gray',label='Max DN to Fit')
        ax_bot.set_xlabel("Linear expectation [DN]")
        ax_bot.set_ylabel("Residual [%]")
        if s == 0:
            ax_bot.legend(fontsize=8, loc="lower left")

    # Hide any unused subplots
    for s in range(n_sample, nrows * ncols):
        row_top = (s // ncols) * 2
        axes[row_top,     s % ncols].set_visible(False)
        axes[row_top + 1, s % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()


def _plot_ptc_histogram(
    result: PTCResult,
    plot_max_mean_dn: float,
    plot_max_var_dn2: float,
    fit_min_mean_dn: float,
    fit_min_var_dn2: float,
    fit_max_mean_dn: float,
    fit_max_var_dn2: float,
    title: str | None,
    save_path: str | None,
) -> None:
    """
    Display a 2-D histogram of (mean signal, variance) with the fitted
    PTC line overlaid.
    """
    x = result.all_means.ravel()
    y = result.all_vars.ravel()

    in_plot = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= 0) & (x <= plot_max_mean_dn)
        & (y >= 0) & (y <= plot_max_var_dn2)
    )

    # 2-D histogram, column-normalised so each signal column has peak = 1
    hist, x_edges, y_edges = np.histogram2d(
        x[in_plot], y[in_plot],
        bins=(200, 100),
        range=((0, plot_max_mean_dn), (0, plot_max_var_dn2)),
    )
    hist = hist.T                                  # (y-bins, x-bins)
    col_max = hist.max(axis=0)
    col_max[col_max == 0] = 1                      # avoid divide-by-zero
    hist = hist / col_max[np.newaxis, :]           # normalise per column

    # Fitted model line
    x_line = np.linspace(0, plot_max_mean_dn, 400)
    y_line = _ptc_model(result.fit_params, x_line)

    # Annotation text
    ann = (
        f"Gain       = {result.gain:.2f} e\u207b/DN\n"
        f"\u03c3_read     = {result.read_noise_dn:.2f} DN\n"
        f"\u03c3_read     = {result.read_noise_e:.2f} e\u207b"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(
        hist,
        interpolation="nearest",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        origin="lower",
        cmap=mpl.cm.inferno,
    )
    ax.plot(x_line, y_line, color="deepskyblue", lw=2, label="Linear PTC fit")
    # plot fitting limits
    ax.axvline(fit_max_mean_dn, color="g", ls="--", lw=1,
               label=f"Fit ceiling ({fit_max_mean_dn:.0f} DN)")
    ax.axvline(fit_min_mean_dn, color="m", ls="--", lw=1,
               label=f"Fit floor ({fit_min_mean_dn:.0f} DN)")
    ax.axhline(fit_max_var_dn2, color="g", ls="--", lw=1,
               label=f"Fit y ceiling ({fit_max_var_dn2:.0f} DN)")
    ax.axhline(fit_min_var_dn2, color="m", ls="--", lw=1,
               label=f"Fit y floor ({fit_min_var_dn2:.0f} DN)")

    ax.set_xlabel("Mean signal [DN]")
    ax.set_ylabel("Variance [DN\u00b2]")
    ax.set_title(title or "Photon Transfer Curve")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.text(
        0.97, 0.05, ann,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")


def _plot_raw_ptc(
    bias_sub_mean: np.ndarray,
    per_read_std: np.ndarray,
    all_means: np.ndarray,
    all_vars: np.ndarray,
    fit_params: np.ndarray,
    x_max: float,
) -> None:
    """
    Simple scatter plot of (mean signal, variance) before linearity
    correction, for comparison with the corrected PTC.
    """
    x_line = np.linspace(0, x_max, 400)
    y_line = _ptc_model(fit_params, x_line)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(bias_sub_mean.ravel(), per_read_std.ravel() ** 2,
               s=4, alpha=0.3, color="steelblue", label="Raw data")
    ax.scatter(all_means.ravel(), all_vars.ravel(),
               s=2, alpha=0.2, color="m", label="Linearized data")
    ax.plot(x_line, y_line, color="tomato", lw=2, label="PTC fit (from corrected data)")
    ax.set_xlabel("Mean signal [DN]")
    ax.set_ylabel("Variance [DN\u00b2]")
    ax.set_title("Raw and Corrected PTC Data")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
