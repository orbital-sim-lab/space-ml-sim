"""TDD tests for CLI tool.

The CLI enables quick analysis without writing Python:
  space-ml-sim trade-study --orbit 550/53 --chip RAD5500 --tmr full
  space-ml-sim report --type ecss --orbit 550/97.6 --chip TRILLIUM_V6E
  space-ml-sim chips --list
"""

from __future__ import annotations

from click.testing import CliRunner


class TestCLIChips:
    """CLI must list available chip profiles."""

    def test_list_chips(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["chips"])
        assert result.exit_code == 0
        assert "RAD5500" in result.output
        assert "SAMRH71" in result.output

    def test_chip_detail(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["chips", "--name", "RAD5500"])
        assert result.exit_code == 0
        assert "RAD5500" in result.output
        assert "TID" in result.output or "krad" in result.output


class TestCLITradeStudy:
    """CLI must run trade studies from command line."""

    def test_single_config(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "trade-study",
                "--orbit",
                "550/53",
                "--chip",
                "RAD5500",
                "--tmr",
                "none",
            ],
        )
        assert result.exit_code == 0
        assert "SEU" in result.output or "seu" in result.output

    def test_multiple_configs(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "trade-study",
                "--orbit",
                "550/53",
                "--chip",
                "RAD5500",
                "--chip",
                "TRILLIUM_V6E",
                "--tmr",
                "full_tmr",
            ],
        )
        assert result.exit_code == 0

    def test_with_shielding(self) -> None:
        from space_ml_sim.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "trade-study",
                "--orbit",
                "550/53",
                "--chip",
                "RAD5500",
                "--shielding",
                "5.0",
            ],
        )
        assert result.exit_code == 0


class TestCLIReport:
    """CLI must generate compliance reports."""

    def test_ecss_report(self, tmp_path) -> None:
        from space_ml_sim.cli import cli

        output = tmp_path / "report.html"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "report",
                "--type",
                "ecss",
                "--orbit",
                "550/97.6",
                "--chip",
                "TRILLIUM_V6E",
                "--mission-years",
                "5",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "ECSS" in content or "Mission Overview" in content
