"""Tests for expanded CLI commands."""

from __future__ import annotations

from click.testing import CliRunner


class TestCLILinkBudget:
    def test_link_budget_s_band(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["link-budget", "--orbit", "550/53", "--freq", "S"])
        assert result.exit_code == 0
        assert "EIRP" in result.output
        assert "Margin" in result.output

    def test_link_budget_ka_band(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["link-budget", "--orbit", "550/53", "--freq", "Ka"])
        assert result.exit_code == 0


class TestCLIConstellations:
    def test_list_constellations(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["constellations"])
        assert result.exit_code == 0
        assert "Starlink" in result.output
        assert "OneWeb" in result.output


class TestCLIAnalyze:
    def test_full_analysis(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--orbit",
                "550/53",
                "--chip",
                "RAD5500",
                "--mission-years",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert "Radiation" in result.output
        assert "Thermal" in result.output
        assert "Link" in result.output
        assert "Overall" in result.output

    def test_analysis_with_solar_cycle(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--orbit",
                "550/53",
                "--chip",
                "TRILLIUM_V6E",
                "--solar",
                "solar_max",
            ],
        )
        assert result.exit_code == 0

    def test_analysis_with_report_output(self, tmp_path) -> None:
        from space_ml_sim.cli import cli

        output = tmp_path / "analysis.html"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--orbit",
                "550/97.6",
                "--chip",
                "RAD5500",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
