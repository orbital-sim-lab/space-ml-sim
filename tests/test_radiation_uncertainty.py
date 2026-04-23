"""TDD tests for radiation model uncertainty quantification.

Written FIRST before implementation (RED phase).

Uncertainty quantification adds confidence intervals to SEU and TID
rate estimates, accounting for:
- Model parameter uncertainty (parametric fits have error bands)
- Environmental variability (solar cycle, SAA crossing frequency)
- Shielding effectiveness uncertainty
"""

from __future__ import annotations

import pytest

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.environment.radiation import RadiationEnvironment


class TestRadiationUncertainty:
    """Radiation model must provide uncertainty bounds."""

    def test_seu_confidence_interval(self) -> None:
        from space_ml_sim.environment.radiation_uncertainty import (
            seu_rate_with_uncertainty,
        )

        rad_env = RadiationEnvironment.leo_500km()
        result = seu_rate_with_uncertainty(rad_env)

        assert result.nominal > 0
        assert result.lower_bound > 0
        assert result.upper_bound > result.nominal
        assert result.lower_bound < result.nominal

    def test_tid_confidence_interval(self) -> None:
        from space_ml_sim.environment.radiation_uncertainty import (
            tid_rate_with_uncertainty,
        )

        rad_env = RadiationEnvironment.leo_500km()
        result = tid_rate_with_uncertainty(rad_env)

        assert result.nominal > 0
        assert result.lower_bound < result.nominal
        assert result.upper_bound > result.nominal

    def test_higher_altitude_wider_uncertainty(self) -> None:
        """Uncertainty bands should widen at higher altitudes (less well-characterized)."""
        from space_ml_sim.environment.radiation_uncertainty import (
            seu_rate_with_uncertainty,
        )

        low = RadiationEnvironment.leo_500km()
        high = RadiationEnvironment.leo_2000km()

        result_low = seu_rate_with_uncertainty(low)
        result_high = seu_rate_with_uncertainty(high)

        spread_low = (result_low.upper_bound - result_low.lower_bound) / result_low.nominal
        spread_high = (result_high.upper_bound - result_high.lower_bound) / result_high.nominal

        assert spread_high >= spread_low

    def test_confidence_level_parameter(self) -> None:
        from space_ml_sim.environment.radiation_uncertainty import (
            seu_rate_with_uncertainty,
        )

        rad_env = RadiationEnvironment.leo_500km()
        r90 = seu_rate_with_uncertainty(rad_env, confidence=0.90)
        r99 = seu_rate_with_uncertainty(rad_env, confidence=0.99)

        # 99% CI should be wider than 90% CI
        spread_90 = r90.upper_bound - r90.lower_bound
        spread_99 = r99.upper_bound - r99.lower_bound
        assert spread_99 > spread_90


class TestMissionUncertainty:
    """Mission-level dose and SEU count should have uncertainty."""

    def test_mission_tid_with_uncertainty(self) -> None:
        from space_ml_sim.environment.radiation_uncertainty import (
            mission_tid_with_uncertainty,
        )

        rad_env = RadiationEnvironment.leo_500km()
        result = mission_tid_with_uncertainty(rad_env, mission_years=5.0)

        assert result.nominal > 0
        assert result.lower_bound < result.nominal
        assert result.upper_bound > result.nominal
        # 5-year mission at 500km should be in single-digit krad range
        assert result.nominal < 50  # krad

    def test_mission_seus_with_uncertainty(self) -> None:
        from space_ml_sim.environment.radiation_uncertainty import (
            mission_seus_with_uncertainty,
        )

        rad_env = RadiationEnvironment.leo_500km()
        result = mission_seus_with_uncertainty(
            rad_env, mission_years=5.0, total_bits=10_000_000
        )

        assert result.nominal >= 0
        assert result.lower_bound <= result.nominal
        assert result.upper_bound >= result.nominal
