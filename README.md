# space-ml-sim

**Simulate AI inference on orbital satellite constellations under realistic space radiation.**

The orbital compute race is here. SpaceX is building TERAFAB with 200 TOPS rad-hardened chips for AI Sat Mini. Google is testing Trillium TPUs aboard Suncatcher for on-orbit inference. But what happens to a ResNet or a transformer when a galactic cosmic ray flips a bit in the exponent of a weight tensor 550 km above Earth?

`space-ml-sim` answers that question.

## What it does

- **Constellation builder** — Walker-Delta and sun-synchronous orbit generation with Keplerian propagation
- **Radiation environment** — Parametric SEU and TID models for LEO (500km to 2000km), including SAA enhancement and shielding effects
- **ML fault injection** — Flip bits in PyTorch model weights and activations using radiation-derived Poisson rates. Sweep fault counts, measure accuracy degradation
- **Fault tolerance** — Full TMR, selective TMR, and checkpoint rollback strategies with majority voting
- **Chip profiles** — TERAFAB D3, Google Trillium TPU v6e, BAE RAD5500, NOEL-V RISC-V

## Quick install

```bash
pip install space-ml-sim
```

Or from source:

```bash
git clone https://github.com/yourorg/space-ml-sim.git
cd space-ml-sim
pip install -e ".[dev]"
```

## Quickstart: Fault sweep in 10 lines

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

## Architecture

```
space_ml_sim/
├── core/               # Orbital mechanics, satellite state, constellation
│   ├── orbit.py        # Keplerian propagation, Walker-Delta, SSO
│   ├── satellite.py    # Satellite with power/thermal/radiation state
│   ├── constellation.py# Bulk operations on satellite groups
│   └── clock.py        # Simulation time management
├── environment/        # Space environment models
│   ├── radiation.py    # SEU rates, TID accumulation, SAA
│   ├── thermal.py      # Steady-state thermal model
│   ├── power.py        # Solar/battery power model
│   └── comms.py        # Inter-satellite link model
├── compute/            # ML inference and fault tolerance
│   ├── fault_injector.py  # *** Core: bit-flip injection into PyTorch models
│   ├── tmr.py             # *** Core: Triple Modular Redundancy
│   ├── checkpoint.py   # Model checkpointing for rollback
│   └── scheduler.py    # Inference scheduling across constellation
├── models/             # Hardware profiles
│   ├── chip_profiles.py   # TERAFAB D3, Trillium, RAD5500, NOEL-V
│   └── rad_profiles.py    # Radiation environment presets
├── metrics/            # Reliability and performance tracking
└── viz/                # Plotly visualization
```

## Chip profiles

| Chip | Node | TDP | TOPS | TID Tolerance | SEU Cross-Section | Notes |
|------|------|-----|------|---------------|-------------------|-------|
| TERAFAB D3 | 2nm | 300W | 200 | 100 krad | 1e-14 cm²/bit | SpaceX rad-hardened |
| Trillium TPU v6e | 4nm | 200W | 450 | 15 krad | 5e-13 cm²/bit | COTS with shielding |
| BAE RAD5500 | 45nm | 15W | 0.001 | 1000 krad | 1e-15 cm²/bit | Space-grade baseline |
| NOEL-V FT | 28nm | 5W | 0.01 | 50 krad | 1e-14 cm²/bit | Open RISC-V in orbit |

## Examples

```bash
# Basic constellation propagation
python examples/01_basic_constellation.py

# Fault sweep across all chip profiles (the killer demo)
python examples/02_radiation_fault_sweep.py

# TMR comparison: unprotected vs TMR vs checkpoint
python examples/03_tmr_comparison.py
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Roadmap

- **v0.1** (current): Keplerian orbits, parametric radiation, fault injection, TMR
- **v0.2**: Selective TMR with per-layer vulnerability ranking, J2 perturbations
- **v0.3**: Distributed inference across constellation, ISL communication delays
- **v0.4**: Transformer model support, quantization-aware fault injection
- **v0.5**: Integration with real TLE data, ground station scheduling

## Contributing

Contributions welcome. Please:
1. Fork the repo
2. Create a feature branch
3. Write tests (pytest, 80%+ coverage)
4. Submit a PR

Focus areas:
- More realistic radiation models (AP-9/AE-9 integration)
- Additional chip profiles (Jetson Orin, Versal AI Edge)
- Transformer-specific fault injection patterns
- Ground station downlink scheduling

## License

Apache 2.0. See [LICENSE](LICENSE).
