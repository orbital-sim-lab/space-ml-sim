# Public API reference

This is the **1.0 stability surface**. Every symbol listed here is importable
from `space_ml_sim` and is covered by the semantic-versioning contract: it will
not be removed or have its signature changed without a major version bump and
at least one minor-release deprecation cycle.

Symbols are imported lazily — `import space_ml_sim` is cheap; heavy submodules
load only when a public name from that submodule is first accessed.

```python
import space_ml_sim
print(space_ml_sim.__version__)            # cheap, no torch import

from space_ml_sim import FaultInjector     # lazy: now torch loads
```

For symbols not listed here, prefer the explicit submodule path (e.g.
`from space_ml_sim.compute.fault_injector import FaultReport`). Anything not
listed and not in the lazy registry is **not** part of the 1.0 contract.

---

## Core orbital mechanics

Source: `space_ml_sim.core`

| Symbol | Kind | Summary |
|---|---|---|
| `OrbitConfig` | dataclass | Keplerian orbital elements for a single satellite. |
| `Satellite` | class | A single satellite with state, chip, and environment. |
| `SatelliteState` | dataclass | Snapshot of a satellite at a given time (position, status, temperature). |
| `Constellation` | class | Collection of satellites with shared environment and clock. |
| `SimClock` | class | Discrete-time simulation clock with configurable step. |
| `propagate(orbit, t)` | function | Propagate an `OrbitConfig` to time `t`. |
| `position_at(orbit, t)` | function | ECI position (km) for an orbit at time `t`. |
| `walker_delta_orbits(...)` | function | Generate Walker-Delta constellation orbits. |
| `sun_synchronous_orbits(...)` | function | Generate sun-synchronous constellation orbits. |
| `is_in_eclipse(position, sun)` | function | Boolean eclipse check for a satellite position. |
| `parse_tle(line1, line2)` | function | Parse a single two-line element set. |
| `load_tle_file(path)` | function | Bulk-load TLEs from a Celestrak-style file. |
| `propagate_sgp4(tle, t)` | function | SGP4-propagate a parsed TLE to time `t`. |
| `ConstellationPreset` | dataclass | Spec for a named real-world constellation. |
| `CONSTELLATION_PRESETS` | mapping | Starlink, OneWeb, Kuiper, Iridium, Planet, GPS, GEO presets. |
| `generate_from_preset(name)` | function | Materialise a `Constellation` from a preset name. |

## Radiation environments

Source: `space_ml_sim.environment`

| Symbol | Kind | Summary |
|---|---|---|
| `RadiationEnvironment` | class | Parametric LEO/MEO/GEO SEU + TID model with SAA enhancement. |
| `HeliocentricEnvironment` | class | Interplanetary GCR-only background for cislunar/Mars/Venus missions. |
| `SPEStatisticalModel` | class | Statistical Solar Particle Event model (ESP–PSYCHIC tail, Xapsos 2000). |
| `SolarParticleEvent` | dataclass | A single sampled SPE with fluence and dose. |
| `mission_spe_dose(...)` | function | Worst-case SPE dose budgeting over a mission lifetime. |
| `RadiationTimeline` | class | Time-series radiation exposure with SAA crossing detection. |
| `radiation_timeline(...)` | function | Build a `RadiationTimeline` from a TLE and a mission duration. |
| `PowerModel` | class | Solar array + battery + eclipse power budget. |
| `ThermalModel` | class | Component temperature model with derating curves. |
| `GroundStation` | dataclass | Ground station with location, elevation mask, and band. |
| `ContactWindow` | dataclass | A scheduled satellite ↔ ground contact. |
| `find_contact_windows(...)` | function | Enumerate contact windows over a time range. |
| `GROUND_STATION_PRESETS` | mapping | Common ground station sites (e.g. KSAT, AWS). |
| `ISLNetwork` | class | Inter-satellite link graph with latency and bandwidth. |

## Compute, fault injection, and TMR

Source: `space_ml_sim.compute`

| Symbol | Kind | Summary |
|---|---|---|
| `FaultInjector` | class | Inject Poisson-distributed bit flips into PyTorch weights/activations. |
| `FaultReport` | dataclass | Per-call report of fault counts, locations, and bits affected. |
| `TransformerFaultInjector` | class | Attention / LayerNorm / embedding-aware fault targeting. |
| `TMRWrapper` | class | Full or selective Triple Modular Redundancy with majority voting. |
| `CheckpointManager` | class | Periodic state snapshots with rollback on detected fault. |
| `InferenceScheduler` | class | Schedule inference across a constellation with ISL routing. |
| `quantize_model(model, dtype)` | function | Quantise a PyTorch model to FP16, BF16, or INT8. |
| `compare_quantization_resilience(...)` | function | Sweep fault counts across FP32/FP16/INT8 in one call. |

## Hardware profiles

Source: `space_ml_sim.models`

| Symbol | Kind | Summary |
|---|---|---|
| `ChipProfile` | dataclass | Chip spec: TOPS, TID tolerance, SEU cross-section, power. |
| `ALL_CHIPS` | sequence | The 11 built-in chip profiles (TERAFAB D3, Trillium v6e, Jetson Orin, Versal AI Core, BAE RAD5500, NOEL-V FT, Zynq UltraScale, SAMRH71, GR740, XQRKU060, AURIX TC4x). |

## Mission metrics

Source: `space_ml_sim.metrics`

| Symbol | Kind | Summary |
|---|---|---|
| `MissionBudget` | dataclass | Deterministic SEU/TID projections over a mission lifetime. |
| `compute_mission_budget(...)` | function | Build a `MissionBudget` from orbit + chip + duration. |
| `MonteCarloResult` | dataclass | Statistical mission survival distribution with CIs. |
| `estimate_mission_reliability(...)` | function | Monte-Carlo mission survival simulation. |

## Analysis pipelines

Source: `space_ml_sim.analysis`

| Symbol | Kind | Summary |
|---|---|---|
| `MissionConfig` | dataclass | Inputs to the end-to-end mission analysis pipeline. |
| `TradeStudy` | class | Multi-mission comparison engine with DataFrame export. |
| `TradeStudyResult` | dataclass | Per-mission row of a trade study. |
| `run_mission_analysis(config)` | function | Run the full radiation + thermal + link + risk pipeline. |
| `MissionAnalysisResult` | dataclass | Output of `run_mission_analysis`. |

## Link budget

Source: `space_ml_sim.comms`

| Symbol | Kind | Summary |
|---|---|---|
| `FrequencyBand` | dataclass | A named RF/optical band with frequency range and typical loss. |
| `FREQUENCY_BANDS` | mapping | UHF, S, X, Ku, Ka, V, optical presets. |
| `LinkBudgetResult` | dataclass | EIRP, G/T, free-space path loss, margin, Shannon capacity. |
| `compute_link_budget(...)` | function | Full link budget calculation given a band and geometry. |

---

## What is *not* in the 1.0 stability contract

The following remain importable from submodules but are **not** part of the
versioning guarantee. Treat them as internal — they can change in any release.

- Anything starting with a leading underscore.
- `space_ml_sim.reports.*` — output file formats are stable, but the Python
  API is still considered experimental.
- `space_ml_sim.viz.*` — plot styling and matplotlib/plotly figure structure.
- `space_ml_sim.data.*` — the on-disk CSV schemas are stable, but the loader
  signatures may evolve.
- `space_ml_sim.cli` — the CLI flags are stable, but the Python-level
  argparse plumbing is internal.

If you depend on any of these and want them in the 1.0 contract, open an issue.
