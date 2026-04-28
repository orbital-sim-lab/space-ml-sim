# space-ml-sim

[![PyPI version](https://img.shields.io/pypi/v/space-ml-sim.svg)](https://pypi.org/project/space-ml-sim/)
[![Python versions](https://img.shields.io/pypi/pyversions/space-ml-sim.svg)](https://pypi.org/project/space-ml-sim/)
[![Downloads](https://static.pepy.tech/badge/space-ml-sim/month)](https://pepy.tech/project/space-ml-sim)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![CI](https://github.com/orbital-sim-lab/space-ml-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/orbital-sim-lab/space-ml-sim/actions/workflows/ci.yml)
[![SPENVIS validated](https://github.com/orbital-sim-lab/space-ml-sim/actions/workflows/spenvis-validation.yml/badge.svg)](https://github.com/orbital-sim-lab/space-ml-sim/actions/workflows/spenvis-validation.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen.svg)](#quality--security)

**Simulate AI inference on orbital satellite constellations under realistic space radiation.**

SpaceX is building TERAFAB with 200 TOPS rad-hardened chips for AI Sat Mini. Cloud-grade TPUs are being tested for on-orbit inference. But what happens to a ResNet or a transformer when a galactic cosmic ray flips a bit in a weight tensor 550 km above Earth?

`space-ml-sim` answers that question.

---

## Features

**Orbital mechanics** -- Walker-Delta and sun-synchronous constellation generation, Keplerian propagation with J2 secular perturbations, eclipse detection, real TLE ingestion via SGP4

**Radiation environment** -- Parametric SEU and TID models for LEO (500 km to 2000 km), SAA enhancement, shielding attenuation, altitude/inclination-dependent rates

**Heliocentric / interplanetary radiation** -- GCR-only background model for missions outside Earth's magnetosphere (lunar transfer, cislunar, Mars transit, Venus flyby), with solar-cycle modulation and heliocentric-distance scaling. Calibrated against CRaTER and Voyager-class measurements, drop-in replacement for `RadiationEnvironment`

**Solar Particle Events** -- Statistical SPE model with ESP–PSYCHIC tail (Xapsos 2000) for episodic high-energy proton bursts. Annual frequency by magnitude (small/medium/large/extreme), Monte-Carlo mission sampling, 95th-percentile worst-case dose budgeting

**ML fault injection** -- Flip bits in PyTorch model weights and activations using radiation-derived Poisson rates. Sweep fault counts and measure accuracy degradation. Transformer-aware targeting for attention, LayerNorm, and embedding layers

**Fault tolerance** -- Full TMR, selective TMR (per-layer vulnerability ranking), and checkpoint rollback with majority voting and anomaly detection

**Radiation timeline** -- Generate time-series radiation exposure from real TLEs with SAA crossing detection and visualization

**Quantization comparison** -- Compare FP32/FP16/INT8 fault resilience curves for the same model in one call

**Sensitivity heatmap** -- Visual per-layer vulnerability ranking showing which layers need protection

**ONNX import** -- Load `.onnx` models for fault injection without writing PyTorch code (`pip install space-ml-sim[onnx]`)

**Mission budget** -- Deterministic SEU/TID projections over mission lifetime with shielding recommendations

**Monte Carlo reliability** -- Statistical mission survival estimation with confidence intervals (`pip install space-ml-sim`)

**Ground track visualization** -- World map with satellite ground track, radiation color overlay, and SAA boundary

**poliastro import** -- Convert poliastro Orbit objects to space-ml-sim (`pip install space-ml-sim[poliastro]`)

**Hardware profiles** -- TERAFAB D3, Trillium TPU v6e, BAE RAD5500, NOEL-V RISC-V, Jetson Orin, Zynq, Versal AI Core

---

## Install

```bash
pip install space-ml-sim
```

From source:

```bash
git clone https://github.com/orbital-sim-lab/space-ml-sim.git
cd space-ml-sim
pip install -e ".[dev]"
```

---

## Quickstart

### Fault sweep in 10 lines

```python
import torch, torchvision, copy
from space_ml_sim.compute.fault_injector import FaultInjector
from space_ml_sim.environment.radiation import RadiationEnvironment
from space_ml_sim.models.chip_profiles import TRILLIUM_V6E

model = torchvision.models.resnet18(weights="DEFAULT").eval()
injector = FaultInjector(RadiationEnvironment.leo_500km(), TRILLIUM_V6E)

for n_faults in [0, 10, 50, 100, 500]:
    test = copy.deepcopy(model)
    report = injector.inject_weight_faults(test, num_faults=n_faults)
    out = test(torch.randn(1, 3, 224, 224))
    print(f"{n_faults:>4d} faults -> argmax={out.argmax().item()}, layers_hit={len(report.layers_affected)}")
```

### Build a constellation and simulate

```python
from space_ml_sim.core import Constellation
from space_ml_sim.models.chip_profiles import TERAFAB_D3

constellation = Constellation.walker_delta(
    num_planes=10, sats_per_plane=10,
    altitude_km=550, inclination_deg=53,
    chip_profile=TERAFAB_D3,
)

for _ in range(95):  # ~1 orbit
    metrics = constellation.step(dt_seconds=60.0)

print(f"Active: {metrics['active_count']}, SEUs: {metrics['total_seus']}")
```

### Load real satellites from TLE

```python
from space_ml_sim.core import parse_tle, Constellation
from space_ml_sim.models.chip_profiles import TERAFAB_D3

tle_line1 = "1 25544U 98067A   24045.54783565  .00016717  00000+0  30057-3 0  9993"
tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 229.6116 15.49815508441075"

orbit = parse_tle(tle_line1, tle_line2)
print(f"ISS: {orbit.altitude_km:.0f} km, {orbit.inclination_deg:.1f} deg")
```

---

## Examples

```bash
python examples/01_basic_constellation.py          # Propagate 100 sats for 1 orbit
python examples/02_radiation_fault_sweep.py        # Accuracy vs bit flips (all 4 chips)
python examples/03_tmr_comparison.py               # TMR vs unprotected under faults
python examples/04_reproduce_published_seu.py      # Reproduce ISS/SSO/high-LEO published SEU rates
```

## Notebooks

Interactive tutorials under `notebooks/`:

- `01_orbital_fault_injection.ipynb` — orbit setup, fault injection, per-layer sensitivity
- `02_tmr_fault_tolerance.ipynb` — full vs selective TMR, checkpoint rollback
- `03_constellation_distributed_inference.ipynb` — distributed inference across ISL links
- `04_cubesat_to_venus_mission.ipynb` — end-to-end mission design: "will your CubeSat's AI survive a Venus flyby?"

---

## Architecture

```
space_ml_sim/
├── core/                  # Orbital mechanics and satellite state
│   ├── orbit.py           # Keplerian propagation, J2 drift, Walker-Delta, SSO
│   ├── satellite.py       # Satellite with power/thermal/radiation tracking
│   ├── constellation.py   # Bulk operations, ISL link detection
│   ├── tle.py             # TLE parsing and SGP4 propagation
│   └── clock.py           # Simulation time management
├── environment/           # Space environment models
│   ├── radiation.py       # SEU rates, TID accumulation, SAA
│   ├── thermal.py         # Steady-state thermal balance
│   ├── power.py           # Solar/battery power model
│   └── comms.py           # Inter-satellite link latency
├── compute/               # ML inference and fault tolerance
│   ├── fault_injector.py  # Bit-flip injection into PyTorch models
│   ├── transformer_fault.py # Attention/LayerNorm/embedding targeting
│   ├── tmr.py             # Full TMR, selective TMR, checkpoint rollback
│   ├── checkpoint.py      # Model checkpointing for fault recovery
│   └── scheduler.py       # Power/thermal-aware inference scheduling
├── models/                # Hardware profiles
│   ├── chip_profiles.py   # TERAFAB D3, Trillium, RAD5500, NOEL-V
│   └── rad_profiles.py    # Radiation environment presets
├── metrics/               # Reliability and performance tracking
└── viz/                   # Plotly visualization
```

---

## Chip selection

Need help picking a chip for your mission? See
[`docs/chip_selection_guide.md`](docs/chip_selection_guide.md) for a quick
decision tree by mission profile (LEO, SSO, MEO, GEO, lunar transfer,
Mars transit, Venus flyby) and by compute requirement.

## Chip Profiles

| Chip | Constant | Node | TDP | INT8 TOPS | TID Tolerance | Notes |
|------|----------|------|-----|-----------|---------------|-------|
| TERAFAB D3 (projected) | `TERAFAB_D3` | 2 nm | 300 W | 200 | 100 krad | SpaceX rad-hardened, AI Sat Mini |
| Trillium TPU v6e | `TRILLIUM_V6E` | 4 nm | 200 W | 450 | 15 krad | COTS TPU with shielding |
| Jetson AGX Orin | `JETSON_AGX_ORIN` | 8 nm | 60 W | 275 | 10 krad | Flying on Planet Labs |
| Versal AI Core XQRVC1902 | `VERSAL_AI_CORE` | 7 nm | 75 W | 130 | 100 krad | Space-grade, 15-year missions |
| Zynq UltraScale+ (Xiphos Q8S) | `ZYNQ_ULTRASCALE` | 16 nm | 10 W | 0.5 | 30 krad | Rad-tolerant FPGA SoC OBC |
| BAE RAD5500 | `RAD5500` | 45 nm | 15 W | 0.001 | 1000 krad | Space-grade baseline |
| NOEL-V Fault-Tolerant | `NOEL_V_FT` | 28 nm | 5 W | 0.01 | 50 krad | Open RISC-V (TRISAT-R) |
| Microchip SAMRH71F20C | `SAMRH71` | 65 nm | 1.5 W | 0.0005 | 100 krad | Rad-hard Cortex-M7, ESA JUICE |
| Cobham GR740 | `GR740` | 65 nm | 3 W | 0.002 | 300 krad | Rad-hard LEON4 quad, PLATO/FLEX |
| AMD XQRKU060 | `XQRKU060` | 20 nm | 12 W | 1.5 | 100 krad | Most-flown space-grade FPGA |
| Infineon AURIX TC4x ⚠ | `AURIX_TC4X` | 28 nm | 6 W | 0.05 | 5 krad | **Automotive ASIL-D, NOT space-qualified** |

⚠ AURIX values are derived from generic 28 nm CMOS literature, not direct beam testing.
Use only for relative trade-study comparison.

```python
from space_ml_sim.models import ALL_CHIPS, TERAFAB_D3
for chip in ALL_CHIPS:
    print(chip.name, chip.compute_tops, chip.tid_tolerance_krad)
```

---

## Quality & Security

Every PR is automatically checked by CI before merge:

| Check | What it does |
|-------|-------------|
| **Tests + Coverage** | 497 tests, 80% minimum coverage enforced |
| **Published-measurement reproduction** | SEU predictions validated against ISS, sun-sync EO, and high-LEO published ranges (see `examples/04_reproduce_published_seu.py`) |
| **Lint & Format** | `ruff check` + `ruff format` |
| **Security Scan** | `pip-audit` (dependency CVEs) + `bandit` (code security) |
| **License Compliance** | Verifies all dependencies are AGPL-compatible |
| **Performance Benchmarks** | Fault injection, constellation step, and orbit propagation speed gates |
| **Branch Protection** | PRs require passing CI + 1 review before merge |
| **Dependabot** | Weekly automated dependency updates |
| **Pre-commit Hooks** | Local checks: ruff, bandit, secret detection, conventional commits |

```bash
# Run all checks locally
pytest tests/ -v --cov=space_ml_sim --cov-fail-under=80
ruff check src/ tests/ && ruff format --check src/ tests/
bandit -r src/ -c pyproject.toml -ll
```

---

## Traction monitor

`scripts/traction_monitor.py` collects public signals about the project
— PyPI downloads, GitHub stars/forks/traffic, Hacker News and Reddit
mentions, and any external repos referencing the package — and prints a
concise markdown summary with week-over-week deltas and actionable
recommendations. It requires only the Python standard library.

```bash
# Print to stdout (no files written)
python scripts/traction_monitor.py --print

# Archive a dated report (default: ~/.space-ml-sim/traction/)
python scripts/traction_monitor.py
```

For richer GitHub data (traffic, clones, referrers), set `GITHUB_TOKEN`
with repo-scoped access before running.

---

## Roadmap

- [x] **v0.1** -- Keplerian orbits, parametric radiation, fault injection, full TMR
- [x] **v0.2** -- J2 perturbations, selective TMR, transformer faults, TLE/SGP4 ingestion, CI
- [x] **v0.3** -- Radiation timeline with SAA detection, quantization-aware fault comparison, sensitivity heatmap, ONNX model import
- [x] **v0.4** -- SPENVIS validation, Monte Carlo reliability, mission budget calculator, ground track viz, poliastro import
- [x] **v0.5** (current) -- Distributed inference across constellation, ISL communication delays, ground station scheduling, link budget, ECSS/MIL-STD reports, CLI
- [ ] **v0.6** -- Hardware-in-the-loop validation, downlink-aware task placement, additional chip profiles

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow, standards, and CLA.

**Focus areas:**
- Distributed inference across ISL links
- Ground station downlink scheduling
- ECSS compliance report export
- More chip profiles and radiation model refinements

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

---

## License

This project is dual-licensed:

- **AGPL-3.0** for open-source use -- see [LICENSE](LICENSE)
- **Commercial license** for proprietary use -- see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)

If you are building proprietary software or a SaaS product with `space-ml-sim`, you need a commercial license. [Learn more](COMMERCIAL_LICENSE.md).
