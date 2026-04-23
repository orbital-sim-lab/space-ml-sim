"""Weibull curve fitting for cross-section vs LET data.

The standard Weibull function for SEE cross-section is:
    sigma(L) = sigma_sat * (1 - exp(-((L - L0) / W)^s))  for L > L0
    sigma(L) = 0                                           for L <= L0

Where:
    sigma_sat = saturation cross-section (cm²/bit)
    L0 = LET threshold (MeV·cm²/mg)
    W = width parameter
    s = shape parameter (Weibull exponent)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeibullFitResult:
    """Parameters of a fitted Weibull cross-section curve."""

    threshold_let: float
    saturation_xsec: float
    width: float
    shape: float

    def predict(self, let: float) -> float:
        """Predict cross-section at a given LET value.

        Args:
            let: Linear Energy Transfer in MeV·cm²/mg.

        Returns:
            Cross-section in cm²/bit.
        """
        if let <= self.threshold_let:
            return 0.0
        x = (let - self.threshold_let) / self.width
        return self.saturation_xsec * (1.0 - math.exp(-(x ** self.shape)))


def fit_weibull(
    let_values: list[float],
    xsec_values: list[float],
) -> WeibullFitResult:
    """Fit a Weibull function to cross-section vs LET data.

    Uses a grid search + least-squares approach since the Weibull SEE
    function is not easily linearizable. This is a simplified fitter
    suitable for 4-10 data points typical of beam test campaigns.

    Args:
        let_values: LET values in MeV·cm²/mg.
        xsec_values: Corresponding cross-sections in cm²/bit.

    Returns:
        WeibullFitResult with fitted parameters.
    """
    lets = np.array(let_values, dtype=np.float64)
    xsecs = np.array(xsec_values, dtype=np.float64)

    # Estimate initial parameters from data
    max_xsec = float(np.max(xsecs))
    if max_xsec == 0:
        max_xsec = 1e-13  # Fallback

    # Threshold: largest LET with zero cross-section
    threshold_candidates = lets[xsecs <= 0]
    threshold_est = float(np.max(threshold_candidates)) if len(threshold_candidates) > 0 else 0.0

    # Grid search over parameters
    best_error = float("inf")
    best_params = (threshold_est, max_xsec, 10.0, 2.0)

    # Search ranges
    thresholds = np.linspace(max(0.1, threshold_est - 3), threshold_est + 1, 8)
    saturations = np.linspace(max_xsec * 0.8, max_xsec * 1.5, 5)
    widths = np.logspace(0, 2.5, 10)
    shapes = np.linspace(0.5, 5.0, 8)

    for l0 in thresholds:
        for s_sat in saturations:
            for w in widths:
                for s in shapes:
                    predicted = _weibull_predict(lets, l0, s_sat, w, s)
                    # Normalized error to avoid scale issues
                    err = np.sum(((predicted - xsecs) / max(max_xsec, 1e-20)) ** 2)
                    if err < best_error:
                        best_error = err
                        best_params = (l0, s_sat, w, s)

    return WeibullFitResult(
        threshold_let=best_params[0],
        saturation_xsec=best_params[1],
        width=best_params[2],
        shape=best_params[3],
    )


def _weibull_predict(
    lets: np.ndarray,
    threshold: float,
    saturation: float,
    width: float,
    shape: float,
) -> np.ndarray:
    """Vectorized Weibull prediction."""
    result = np.zeros_like(lets)
    mask = lets > threshold
    x = (lets[mask] - threshold) / width
    result[mask] = saturation * (1.0 - np.exp(-(x ** shape)))
    return result
