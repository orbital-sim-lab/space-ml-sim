# Development Roadmap

## Completed

### v0.1 (Initial Release)
- [x] Keplerian orbit propagation
- [x] Walker-Delta and SSO constellation generation
- [x] Parametric radiation environment (SEU, TID, SAA)
- [x] ML fault injection (weight SEU, activation SET)
- [x] Full TMR with majority voting
- [x] 4 chip profiles (TERAFAB D3, Trillium, RAD5500, NOEL-V)

### v0.2
- [x] J2 secular perturbations (RAAN drift)
- [x] Selective TMR with per-layer vulnerability ranking
- [x] Transformer-aware fault injection (attention, LayerNorm, embedding)
- [x] TLE/SGP4 ingestion from real orbital data
- [x] 3 additional chip profiles (Jetson Orin, Zynq, Versal AI Core)
- [x] GitHub Actions CI (6 jobs: tests, coverage, lint, security, license, benchmarks)

### v0.3
- [x] Radiation timeline with SAA crossing detection from TLE
- [x] Quantization-aware fault comparison (FP32/FP16/INT8)
- [x] Per-layer sensitivity heatmap visualization
- [x] ONNX model import for fault injection
- [x] FP16/BF16 bit-flip support
- [x] Deterministic accuracy validation suite (37 tests against analytical solutions)
- [x] Published to PyPI

### v0.4 (Accuracy & Integration)
Focus: Make the simulation results trustworthy enough for flight qualification support.

- [x] Validate SEU rates against SPENVIS AP-9/AE-9 reference data
- [x] Validate TID rates against ESA SPENVIS outputs for standard orbits
- [x] Add ground track visualization with radiation overlay on world map
- [x] Integrate with poliastro for orbit objects import
- [x] Add mission lifetime reliability estimation (Monte Carlo)
- [x] Add per-orbit SEU/TID budget calculator
- [x] Create Jupyter notebook tutorial series (3 exercises for university courses)

### v0.5 (Distributed & Communication)
Focus: Simulate realistic multi-satellite compute scenarios.

- [x] Distributed inference across ISL links with latency modeling
- [x] Ground station visibility and downlink scheduling
- [x] Model-parallel inference across constellation members
- [x] Bandwidth-constrained gradient aggregation for federated learning

### v0.6 (Compliance & Enterprise)
Focus: Make the tool usable in flight qualification workflows.

- [x] ECSS-Q-ST-60-15C compliance report export (HTML, print-to-PDF)
- [x] MIL-STD-883 TM 1019 test methodology documentation
- [x] Integration with radiation test facility data formats (CSV import)
- [x] Automated TMR recommendation engine with cost/benefit analysis

### v0.7 (Individual User & Adoption)
Focus: Features that drive individual user adoption and licensing.

- [x] CLI tool (`space-ml-sim trade-study`, `report`, `chips`)
- [x] Constellation presets (Starlink, OneWeb, Kuiper, Iridium, Planet)
- [x] Shielding optimization recommender with mass penalty analysis
- [x] Solar cycle radiation presets (solar min/max/ascending/descending)
- [x] 3 additional rad-hard chip profiles (SAMRH71, GR740, XQRKU060) — 10 total
- [x] Mission trade-study comparison API with DataFrame export
- [x] Weibull cross-section curve fitting for test data
- [x] Radiation uncertainty quantification (confidence intervals)

### v0.8 (Revenue Features)
Focus: Features that directly generate individual user revenue.

- [x] RF/optical link budget calculator (FSPL, EIRP, G/T, margin, Shannon capacity)
- [x] Requirements traceability matrix (RTM) generator with auto-generation from simulation
- [x] Orbital thermal cycling model (eclipse/sunlit transitions, component derating)
- [x] Frequency band presets (UHF, S, X, Ku, Ka, V, optical)

### v0.9 (Market Expansion)
Focus: Expand addressable market beyond LEO, add mission-critical analysis tools.

- [x] MEO/GEO radiation environment (Van Allen belt parametric model)
- [x] Power budget calculator (solar/battery/eclipse analysis)
- [x] Dose-depth curve analysis with shielding optimization
- [x] End-to-end mission analysis pipeline (radiation + thermal + link + risk)
- [x] SEL (Single Event Latchup) modeling with mission probability
- [x] Expanded CLI (analyze, link-budget, constellations commands)
- [x] MEO/GEO factory presets (GPS, GEO orbits)

## Accuracy Priorities

Every release must pass the deterministic accuracy validation suite. New features
must include analytical validation tests where possible. The following accuracy
improvements are tracked independently of feature releases:

- [x] Cross-validate orbit propagation against SGP4 for standard TLEs
- [x] Cross-validate radiation rates against published SPENVIS reference orbits
- [x] Add uncertainty quantification to radiation model outputs
- [x] Validate fault injection statistics against analytical Poisson expectations
- [x] Benchmark TMR correction rate against theoretical 2-of-3 voting probability
