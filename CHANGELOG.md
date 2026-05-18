# Changelog

All notable changes to space-ml-sim are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) starting
at version 1.0.0.

## [1.0.0rc1] - 2026-05-18

First release candidate for 1.0. The codebase had already accumulated the
features originally scoped for v0.6–v0.9; this RC formalises that surface area
and commits to API stability.

### Added
- Public API stability contract documented in `ROADMAP.md`.
- `Development Status :: 5 - Production/Stable` classifier in `pyproject.toml`.
- Heliocentric / interplanetary radiation environment (`HeliocentricEnvironment`).
- Statistical Solar Particle Event model (ESP–PSYCHIC tail, Xapsos 2000).
- AURIX TC4x chip profile and invariant tests.
- Cubesat-to-Venus mission notebook (`notebooks/04_cubesat_to_venus_mission.ipynb`).
- Chip selection guide (`docs/chip_selection_guide.md`).

### Changed
- Version bumped from `0.5.0` to `1.0.0rc1`.

### Notes
- IARC copyright waiver still pending; required before the final 1.0.0 tag is announced publicly.

## [0.9.x] - earlier 2026

MEO/GEO radiation, power budget, end-to-end mission pipeline, SEL modeling,
expanded CLI, GPS/GEO factory presets. (Shipped as part of the 0.5.0 PyPI
release; promoted to its own changelog entry retroactively.)

## [0.8.x] - earlier 2026

RF/optical link budget, requirements traceability matrix, orbital thermal
cycling, frequency band presets.

## [0.7.x] - earlier 2026

CLI tool, constellation presets, shielding optimization recommender, solar
cycle presets, additional rad-hard chip profiles (SAMRH71, GR740, XQRKU060),
trade-study comparison API, Weibull cross-section fitting, radiation
uncertainty quantification.

## [0.6.x] - earlier 2026

ECSS-Q-ST-60-15C compliance report export, MIL-STD-883 TM 1019 methodology,
radiation test facility CSV import, automated TMR recommendation engine.

## [0.5.0] - earlier 2026

Distributed inference across ISL links, ground station visibility, model-parallel
inference, bandwidth-constrained federated learning.

## [0.4.0] - earlier 2026

SPENVIS AP-9/AE-9 validation, Monte Carlo reliability estimation, ground track
visualization, poliastro import, mission budget calculator, tutorial notebooks.

## [0.3.0] - earlier 2026

Radiation timeline with SAA detection, quantization-aware fault comparison
(FP32/FP16/INT8), per-layer sensitivity heatmap, ONNX model import, FP16/BF16
bit-flip support, 37-test deterministic accuracy suite, first PyPI release.

## [0.2.0] - earlier 2026

J2 perturbations, selective TMR, transformer-aware fault injection, TLE/SGP4
ingestion, additional chip profiles (Jetson Orin, Zynq, Versal AI Core), GitHub
Actions CI.

## [0.1.0] - earlier 2026

Initial release. Keplerian propagation, Walker-Delta and SSO constellations,
parametric SEU/TID/SAA radiation, ML weight/activation fault injection, full
TMR, four chip profiles (TERAFAB D3, Trillium, RAD5500, NOEL-V).
