"""TDD tests for mission trade-study comparison API.

Enables side-by-side comparison of multiple mission configurations:
- Different orbits, chips, TMR strategies, shielding levels
- Outputs structured comparison with cost/benefit/risk per option
"""

from __future__ import annotations

import torch.nn as nn

from space_ml_sim.core.orbit import OrbitConfig
from space_ml_sim.models.chip_profiles import RAD5500, TRILLIUM_V6E, XQRKU060


def _model_factory():
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


class TestTradeStudyConfig:
    """Trade study must accept multiple mission configurations."""

    def test_create_study(self) -> None:
        from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig

        configs = [
            MissionConfig(
                name="Option A",
                orbit=OrbitConfig(
                    altitude_km=500, inclination_deg=53, raan_deg=0, true_anomaly_deg=0
                ),
                chip=RAD5500,
                tmr_strategy="full_tmr",
                shielding_mm_al=3.0,
                mission_years=5.0,
            ),
            MissionConfig(
                name="Option B",
                orbit=OrbitConfig(
                    altitude_km=500, inclination_deg=53, raan_deg=0, true_anomaly_deg=0
                ),
                chip=TRILLIUM_V6E,
                tmr_strategy="selective_tmr",
                shielding_mm_al=5.0,
                mission_years=5.0,
            ),
        ]
        study = TradeStudy(configs=configs)
        assert len(study.configs) == 2

    def test_run_returns_results(self) -> None:
        from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig

        configs = [
            MissionConfig(
                name="Baseline",
                orbit=OrbitConfig(
                    altitude_km=550, inclination_deg=97.6, raan_deg=0, true_anomaly_deg=0
                ),
                chip=RAD5500,
                tmr_strategy="none",
                shielding_mm_al=2.0,
                mission_years=3.0,
            ),
        ]
        study = TradeStudy(configs=configs)
        results = study.run()
        assert len(results) == 1


class TestTradeStudyResults:
    """Results must include all comparison metrics."""

    def test_result_has_radiation_metrics(self) -> None:
        from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig

        configs = [
            MissionConfig(
                name="Test",
                orbit=OrbitConfig(
                    altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0
                ),
                chip=TRILLIUM_V6E,
                tmr_strategy="full_tmr",
                shielding_mm_al=3.0,
                mission_years=5.0,
            ),
        ]
        study = TradeStudy(configs=configs)
        results = study.run()
        r = results[0]

        assert r.seu_rate_per_day > 0
        assert r.tid_over_mission_krad > 0
        assert r.expected_seus_per_orbit > 0

    def test_result_has_cost_metrics(self) -> None:
        from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig

        configs = [
            MissionConfig(
                name="Full TMR",
                orbit=OrbitConfig(
                    altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0
                ),
                chip=RAD5500,
                tmr_strategy="full_tmr",
                shielding_mm_al=2.0,
                mission_years=5.0,
            ),
        ]
        study = TradeStudy(configs=configs)
        results = study.run()
        r = results[0]

        assert r.compute_multiplier >= 1.0
        assert r.power_watts > 0

    def test_compare_multiple_options(self) -> None:
        from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        configs = [
            MissionConfig(
                name="Rad-hard",
                orbit=orbit,
                chip=RAD5500,
                tmr_strategy="none",
                shielding_mm_al=2.0,
                mission_years=5.0,
            ),
            MissionConfig(
                name="COTS+TMR",
                orbit=orbit,
                chip=TRILLIUM_V6E,
                tmr_strategy="full_tmr",
                shielding_mm_al=5.0,
                mission_years=5.0,
            ),
            MissionConfig(
                name="FPGA",
                orbit=orbit,
                chip=XQRKU060,
                tmr_strategy="selective_tmr",
                shielding_mm_al=3.0,
                mission_years=5.0,
            ),
        ]
        study = TradeStudy(configs=configs)
        results = study.run()

        assert len(results) == 3
        # Rad-hard chip should have lower SEU rate
        assert results[0].seu_rate_per_day < results[1].seu_rate_per_day

    def test_to_dataframe(self) -> None:
        from space_ml_sim.analysis.trade_study import TradeStudy, MissionConfig

        orbit = OrbitConfig(altitude_km=550, inclination_deg=53, raan_deg=0, true_anomaly_deg=0)
        configs = [
            MissionConfig(
                name="A",
                orbit=orbit,
                chip=RAD5500,
                tmr_strategy="none",
                shielding_mm_al=2.0,
                mission_years=5.0,
            ),
            MissionConfig(
                name="B",
                orbit=orbit,
                chip=TRILLIUM_V6E,
                tmr_strategy="full_tmr",
                shielding_mm_al=3.0,
                mission_years=5.0,
            ),
        ]
        study = TradeStudy(configs=configs)
        df = study.to_dataframe()

        assert len(df) == 2
        assert "name" in df.columns
        assert "seu_rate_per_day" in df.columns
        assert "tid_over_mission_krad" in df.columns
