"""Tests for the ``space-ml-sim doctor`` CLI command."""

from __future__ import annotations

from click.testing import CliRunner

from space_ml_sim.cli import cli


def test_doctor_runs_clean_in_dev_environment() -> None:
    """In the dev environment all required deps are installed, so exit code is 0."""
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "Install is healthy" in result.output


def test_doctor_reports_version() -> None:
    """The doctor output must surface the installed package version."""
    import space_ml_sim

    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert space_ml_sim.__version__ in result.output


def test_doctor_lists_python_and_required_deps() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    for required in ["Python", "numpy", "sgp4", "torch", "pydantic", "rich", "click"]:
        assert required in result.output, f"doctor output missing {required}"


def test_doctor_lists_optional_extras() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    for extra in ["onnx", "poliastro", "torchvision"]:
        assert extra in result.output
