"""TDD tests for Weibull curve fitting of cross-section vs LET data."""

from __future__ import annotations


class TestWeibullFit:
    """Fit Weibull function to cross-section vs LET test data."""

    def test_fit_returns_parameters(self) -> None:
        from space_ml_sim.data.weibull_fit import fit_weibull

        let_values = [5.0, 10.0, 20.0, 30.0, 50.0, 80.0]
        xsec_values = [0.0, 1e-15, 5e-14, 8e-14, 9.5e-14, 1e-13]

        result = fit_weibull(let_values, xsec_values)
        assert result.threshold_let > 0
        assert result.saturation_xsec > 0
        assert result.width > 0
        assert result.shape > 0

    def test_predict_cross_section(self) -> None:
        from space_ml_sim.data.weibull_fit import fit_weibull

        let_values = [5.0, 10.0, 20.0, 40.0, 80.0]
        xsec_values = [0.0, 1e-15, 5e-14, 9e-14, 1e-13]

        result = fit_weibull(let_values, xsec_values)

        # Below threshold: should be ~0
        assert result.predict(1.0) < 1e-16

        # At high LET: should approach saturation
        assert result.predict(200.0) > result.saturation_xsec * 0.8

    def test_monotonically_increasing(self) -> None:
        from space_ml_sim.data.weibull_fit import fit_weibull

        let_values = [3.0, 8.0, 15.0, 30.0, 60.0]
        xsec_values = [0.0, 5e-15, 3e-14, 8e-14, 1e-13]

        result = fit_weibull(let_values, xsec_values)

        predictions = [result.predict(let) for let in [5, 10, 20, 40, 80]]
        for i in range(len(predictions) - 1):
            assert predictions[i + 1] >= predictions[i]
