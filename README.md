# space-ml-sim

**Simulate AI inference on orbital satellite constellations under realistic space radiation.**

SpaceX is building TERAFAB with 200 TOPS rad-hardened chips for AI Sat Mini. Google is testing Trillium TPUs aboard Suncatcher for on-orbit inference. But what happens to a ResNet or a transformer when a galactic cosmic ray flips a bit in a weight tensor 550 km above Earth?

`space-ml-sim` answers that question.

---

## Features

**Orbital mechanics** -- Walker-Delta and sun-synchronous constellation generation, Keplerian propagation with J2 secular perturbations, eclipse detection, real TLE ingestion via SGP4

**Radiation environment** -- Parametric SEU and TID models for LEO (500 km to 2000 km), SAA enhancement, shielding attenuation, altitude/inclination-dependent rates

**ML fault injection** -- Flip bits in PyTorch model weights and activations using radiation-derived Poisson rates. Sweep fault counts and measure accuracy degradation. Transformer-aware targeting for attention, LayerNorm, and embedding layers

**Fault tolerance** -- Full TMR, selective TMR (per-layer vulnerability ranking), and checkpoint rollback with majority voting and anomaly detection

**Hardware profiles** -- TERAFAB D3, Google Trillium TPU v6e, BAE RAD5500, NOEL-V RISC-V

---

## Install

```bash
pip install space-ml-sim
```

From source:

```bash
git clone https://github.com/yaitsmesj/space-ml-sim.git
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
from space_ml_sim.models.chip_profiles import GOOGLE_TRILLIUM_V6E

model = torchvision.models.resnet18(weights="DEFAULT").eval()
injector = FaultInjector(RadiationEnvironment.leo_500km(), GOOGLE_TRILLIUM_V6E)

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
python examples/01_basic_constellation.py    # Propagate 100 sats for 1 orbit
python examples/02_radiation_fault_sweep.py  # Accuracy vs bit flips (all 4 chips)
python examples/03_tmr_comparison.py         # TMR vs unprotected under faults
```

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

## Chip Profiles

| Chip | Node | TDP | INT8 TOPS | TID Tolerance | SEU Cross-Section | Notes |
|------|------|-----|-----------|---------------|-------------------|-------|
| TERAFAB D3 | 2 nm | 300 W | 200 | 100 krad | 1e-14 cm^2/bit | SpaceX rad-hardened |
| Trillium TPU v6e | 4 nm | 200 W | 450 | 15 krad | 5e-13 cm^2/bit | COTS with shielding |
| BAE RAD5500 | 45 nm | 15 W | 0.001 | 1000 krad | 1e-15 cm^2/bit | Space-grade baseline |
| NOEL-V FT | 28 nm | 5 W | 0.01 | 50 krad | 1e-14 cm^2/bit | Open RISC-V in orbit |

---

## Quality & Security

Every PR is automatically checked by CI before merge:

| Check | What it does |
|-------|-------------|
| **Tests + Coverage** | 395 tests, 80% minimum coverage enforced |
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

## Roadmap

- [x] **v0.1** -- Keplerian orbits, parametric radiation, fault injection, full TMR
- [x] **v0.2** (current) -- J2 perturbations, selective TMR, transformer faults, TLE/SGP4 ingestion, CI
- [ ] **v0.3** -- Distributed inference across constellation, ISL communication delays
- [ ] **v0.4** -- Quantization-aware fault injection, mixed-precision vulnerability analysis
- [ ] **v0.5** -- Ground station scheduling, downlink-aware task placement

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow, standards, and CLA.

**Focus areas:**
- More realistic radiation models (AP-9/AE-9 integration)
- Additional chip profiles (Jetson Orin, Versal AI Edge)
- Ground station downlink scheduling
- Distributed inference simulation across ISL links

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

---

## License

This project is dual-licensed:

- **AGPL-3.0** for open-source use -- see [LICENSE](LICENSE)
- **Commercial license** for proprietary use -- see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md)

If you are building proprietary software or a SaaS product with `space-ml-sim`, you need a commercial license. [Learn more](COMMERCIAL_LICENSE.md).
