"""TDD tests for satellite link budget calculator.

Covers RF downlink/uplink and optical ISL link budgets with:
- EIRP computation
- Free-space path loss
- Atmospheric attenuation
- G/T (figure of merit)
- Link margin
- Data rate from Shannon capacity
"""

from __future__ import annotations


class TestFrequencyBands:
    """Standard frequency band presets."""

    def test_s_band_preset(self) -> None:
        from space_ml_sim.comms.link_budget import FREQUENCY_BANDS

        s_band = FREQUENCY_BANDS["S"]
        assert 2.0e9 <= s_band.center_freq_hz <= 4.0e9

    def test_x_band_preset(self) -> None:
        from space_ml_sim.comms.link_budget import FREQUENCY_BANDS

        x_band = FREQUENCY_BANDS["X"]
        assert 8.0e9 <= x_band.center_freq_hz <= 12.0e9

    def test_ka_band_preset(self) -> None:
        from space_ml_sim.comms.link_budget import FREQUENCY_BANDS

        ka = FREQUENCY_BANDS["Ka"]
        assert 26.0e9 <= ka.center_freq_hz <= 40.0e9


class TestFreeSpacePathLoss:
    """FSPL must follow 20*log10(4*pi*d*f/c)."""

    def test_fspl_500km_s_band(self) -> None:
        from space_ml_sim.comms.link_budget import free_space_path_loss_db

        # 500 km slant range, 2.2 GHz S-band
        fspl = free_space_path_loss_db(distance_km=500.0, frequency_hz=2.2e9)
        # Expected ~153 dB for 500km slant range at S-band
        assert 150 < fspl < 158

    def test_fspl_increases_with_distance(self) -> None:
        from space_ml_sim.comms.link_budget import free_space_path_loss_db

        fspl_500 = free_space_path_loss_db(500.0, 2.2e9)
        fspl_2000 = free_space_path_loss_db(2000.0, 2.2e9)
        assert fspl_2000 > fspl_500

    def test_fspl_increases_with_frequency(self) -> None:
        from space_ml_sim.comms.link_budget import free_space_path_loss_db

        fspl_s = free_space_path_loss_db(500.0, 2.2e9)
        fspl_ka = free_space_path_loss_db(500.0, 26.0e9)
        assert fspl_ka > fspl_s


class TestLinkBudget:
    """Full link budget computation."""

    def test_compute_returns_result(self) -> None:
        from space_ml_sim.comms.link_budget import compute_link_budget, LinkBudgetResult

        result = compute_link_budget(
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=6.0,
            frequency_hz=2.2e9,
            distance_km=600.0,
            rx_antenna_gain_dbi=30.0,
            system_noise_temp_k=300.0,
            bandwidth_hz=1e6,
            atmospheric_loss_db=1.0,
            pointing_loss_db=0.5,
        )
        assert isinstance(result, LinkBudgetResult)

    def test_eirp_computed(self) -> None:
        from space_ml_sim.comms.link_budget import compute_link_budget

        result = compute_link_budget(
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=6.0,
            frequency_hz=2.2e9,
            distance_km=600.0,
            rx_antenna_gain_dbi=30.0,
            system_noise_temp_k=300.0,
            bandwidth_hz=1e6,
        )
        assert abs(result.eirp_dbw - 16.0) < 0.01  # 10 + 6

    def test_positive_margin_is_good(self) -> None:
        from space_ml_sim.comms.link_budget import compute_link_budget

        result = compute_link_budget(
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=6.0,
            frequency_hz=2.2e9,
            distance_km=600.0,
            rx_antenna_gain_dbi=35.0,
            system_noise_temp_k=200.0,
            bandwidth_hz=1e6,
            required_eb_no_db=10.0,
        )
        # With a good antenna and low noise, should have positive margin
        assert result.link_margin_db > 0

    def test_data_rate_reported(self) -> None:
        from space_ml_sim.comms.link_budget import compute_link_budget

        result = compute_link_budget(
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=6.0,
            frequency_hz=2.2e9,
            distance_km=600.0,
            rx_antenna_gain_dbi=30.0,
            system_noise_temp_k=300.0,
            bandwidth_hz=5e6,
        )
        assert result.max_data_rate_bps > 0

    def test_higher_distance_worse_margin(self) -> None:
        from space_ml_sim.comms.link_budget import compute_link_budget

        kwargs = dict(
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=6.0,
            frequency_hz=2.2e9,
            rx_antenna_gain_dbi=30.0,
            system_noise_temp_k=300.0,
            bandwidth_hz=1e6,
        )
        r_close = compute_link_budget(distance_km=500.0, **kwargs)
        r_far = compute_link_budget(distance_km=2000.0, **kwargs)
        assert r_far.link_margin_db < r_close.link_margin_db
