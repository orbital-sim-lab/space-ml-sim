# Chip selection guide

This guide maps mission profiles to the built-in `ChipProfile` constants.
It is intended as a starting point for `space_ml_sim.analysis.trade_study`,
not a flight-qualification recommendation.

> **Status.** Parametric guidance. Numbers are first-order; for a flight
> mission run a full trade study using `analysis.trade_study` and the SPE
> Monte Carlo (`environment.SPEStatisticalModel.sample_mission`).

---

## Quick decision tree

```
Need ≥10 TOPS for AI?
 ├─ Yes →  TID budget over 50 krad?
 │        ├─ Yes → VERSAL_AI_CORE  (130 TOPS, 100 krad, 7 nm, space-grade)
 │        └─ No  → JETSON_AGX_ORIN with shielding   (275 TOPS, 10 krad, COTS)
 │                 or TRILLIUM_V6E with heavy shielding (450 TOPS, 15 krad)
 │
 └─ No  →  Reconfigurable / FPGA?
          ├─ Yes → XQRKU060 (1.5 TOPS, 100 krad, most-flown space FPGA)
          │        or ZYNQ_ULTRASCALE (0.5 TOPS, 30 krad, Q8S OBC class)
          │
          └─ No  →  Need >100 krad?
                   ├─ Yes → SAMRH71 (Cortex-M7, 100 krad)
                   │        or GR740 (LEON4 quad, 300 krad)
                   │        or RAD5500 (1 Mrad, glacial)
                   └─ No  → NOEL_V_FT (open RISC-V, 50 krad)
                            or AURIX_TC4X (auto-grade, ⚠ not space-qualified)
```

## By mission profile

### LEO (500 km, 53° / SSO 650 km, 98°)

- **Background TID over 5 yr:** 0.5–10 krad
- **SEU regime:** sparse, manageable with selective TMR
- **Dominant risk:** Single Event Latch-up in the SAA — pick chips with
  SEL immunity ≥40 MeV·cm²/mg

**Recommended:** `JETSON_AGX_ORIN` with 5 mm Al shielding, or
`VERSAL_AI_CORE` for longer missions / heavier ML workloads.

**Avoid:** `RAD5500` (overkill, sacrifices compute for no benefit at LEO).

### Sun-synchronous EO (700–900 km, 98°)

- Higher trapped proton dose than ISS-class
- 5–50 krad over 5 years behind 2 mm Al
- **Recommended:** `VERSAL_AI_CORE` for AI workloads, `XQRKU060` for FPGA
  pipelines, `GR740` for control-plane only.

### MEO / GPS-class (20 200 km, 55°)

- Outer-belt protons + electrons
- 50–500 krad over 10 yr
- **Recommended:** `RAD5500` or `GR740` for OBC; `XQRKU060` only with
  active scrubbing (uses built-in FRAME_ECC).

### GEO (35 786 km, 0°)

- Electron-dominated environment
- **Recommended:** `VERSAL_AI_CORE`, `RAD5500`, or `XQRKU060`. Do not fly
  bare COTS chips here.

### Lunar transfer / cislunar

- Outside the magnetosphere; **GCR background is small** but **SPE risk is
  the dominant TID source.**
- Use `HeliocentricEnvironment.lunar_transfer()` for background and
  `SPEStatisticalModel(solar_phase="max")` for the worst-case event budget.
- For a 6-month CubeSat mission at solar max with 2 mm Al, expect ~5–15
  krad of SPE-driven TID at the 95th percentile.

**Recommended:** `VERSAL_AI_CORE` (100 krad budget covers SPE p95
comfortably). For lower-power / lower-compute payloads: `SAMRH71`,
`GR740`, or `NOEL_V_FT`.

**Don't fly bare:** `JETSON_AGX_ORIN` (10 krad budget can be exhausted by
a single major SPE).

### Mars transit (1.0 → 1.5 AU, 7–9 months)

- Higher GCR than Earth orbit, plus full SPE statistics.
- 20–50 krad SPE-driven worst case at solar max.
- **Recommended:** `RAD5500` for the OBC, `VERSAL_AI_CORE` for AI payload,
  or `XQRKU060` if FPGA-based.

### Venus flyby (~0.72 AU, short)

- GCR is suppressed by stronger solar modulation
- SPE risk is still real (events arrive radially from the Sun)
- For a 1–2 month flyby window: `VERSAL_AI_CORE` is the cheapest sufficient
  option; `JETSON_AGX_ORIN` works only with thick shielding (>5 mm Al)
  during solar minimum.

### Outer planets (Jupiter, Saturn)

- Out of scope for v0.5. The Jovian magnetosphere dominates and the
  built-in models do not include trapped-electron belts beyond Earth.

---

## By compute requirement

| If you need …                           | Pick                                         |
|-----------------------------------------|----------------------------------------------|
| ≥200 TOPS for transformer inference     | `TERAFAB_D3` (projected) or `TRILLIUM_V6E`   |
| 100–200 TOPS, mission-critical          | `VERSAL_AI_CORE` (qualified for 15 yr)       |
| 200+ TOPS, COTS, tolerable risk         | `JETSON_AGX_ORIN` + 5 mm Al + active TMR     |
| 1–10 TOPS, FPGA pipelines               | `XQRKU060` or `ZYNQ_ULTRASCALE`              |
| <1 TOPS, control-plane only             | `GR740`, `SAMRH71`, `NOEL_V_FT`              |
| Reliability above all (1 Mrad budget)   | `RAD5500`                                    |
| Lab / cost-down trade-study placeholder | `AURIX_TC4X` (⚠ not space-qualified)         |

---

## Working through a trade study

The chip table is data; the decision is a function. Use:

```python
from space_ml_sim.analysis.trade_study import compare_chips
from space_ml_sim.models import ALL_CHIPS
from space_ml_sim.environment import HeliocentricEnvironment, SPEStatisticalModel

env_background = HeliocentricEnvironment.cruise_1au_solar_min()
spe = SPEStatisticalModel(solar_phase="max", shielding_mm_al=2.0)

study = compare_chips(
    chips=ALL_CHIPS,
    background_env=env_background,
    spe_model=spe,
    mission_days=210,
)
print(study.ranked_by_margin())
```

The trade study folds in TID margin, SEU rate at the chip's published
cross-section, and compute headroom for the workload. Use it as a first
filter, not the final answer.
