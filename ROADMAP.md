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

### v0.3 (Current)
- [x] Radiation timeline with SAA crossing detection from TLE
- [x] Quantization-aware fault comparison (FP32/FP16/INT8)
- [x] Per-layer sensitivity heatmap visualization
- [x] ONNX model import for fault injection
- [x] FP16/BF16 bit-flip support
- [x] Deterministic accuracy validation suite (37 tests against analytical solutions)
- [x] Published to PyPI

## In Progress

### v0.4 (Accuracy & Integration) — Target: May 2026
Focus: Make the simulation results trustworthy enough for flight qualification support.

- [ ] Validate SEU rates against SPENVIS AP-9/AE-9 reference data
- [ ] Validate TID rates against ESA SPENVIS outputs for standard orbits
- [ ] Add ground track visualization with radiation overlay on world map
- [ ] Integrate with poliastro for orbit objects import
- [ ] Add mission lifetime reliability estimation (Monte Carlo)
- [ ] Add per-orbit SEU/TID budget calculator
- [ ] Create Jupyter notebook tutorial series (3 exercises for university courses)

### v0.5 (Distributed & Communication) — Target: Jul 2026
Focus: Simulate realistic multi-satellite compute scenarios.

- [ ] Distributed inference across ISL links with latency modeling
- [ ] Ground station visibility and downlink scheduling
- [ ] Model-parallel inference across constellation members
- [ ] Bandwidth-constrained gradient aggregation for federated learning

### v0.6 (Compliance & Enterprise) — Target: Sep 2026
Focus: Make the tool usable in flight qualification workflows.

- [ ] ECSS-Q-ST-60-15C compliance report export (PDF)
- [ ] MIL-STD-883 TM 1019 test methodology documentation
- [ ] Integration with radiation test facility data formats
- [ ] Automated TMR recommendation engine with cost/benefit analysis

## Accuracy Priorities

Every release must pass the deterministic accuracy validation suite. New features
must include analytical validation tests where possible. The following accuracy
improvements are tracked independently of feature releases:

- [ ] Cross-validate orbit propagation against SGP4 for standard TLEs
- [ ] Cross-validate radiation rates against published SPENVIS reference orbits
- [ ] Add uncertainty quantification to radiation model outputs
- [ ] Validate fault injection statistics against analytical Poisson expectations
- [ ] Benchmark TMR correction rate against theoretical 2-of-3 voting probability
