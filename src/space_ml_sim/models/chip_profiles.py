"""Hardware chip profiles for space-grade and COTS processors.

Each profile captures the key radiation sensitivity and compute
characteristics needed for fault injection simulation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChipProfile(BaseModel):
    """Hardware chip profile for radiation simulation."""

    name: str = Field(description="Human-readable chip name")
    node_nm: int = Field(gt=0, description="Process node in nanometers")
    tdp_watts: float = Field(gt=0, description="Thermal design power in watts")
    max_temp_c: float = Field(description="Maximum operating temperature in Celsius")
    seu_cross_section_cm2: float = Field(
        gt=0, description="SEU cross-section per bit in cm^2"
    )
    tid_tolerance_krad: float = Field(
        gt=0, description="Total ionizing dose tolerance in krad(Si)"
    )
    compute_tops: float = Field(ge=0, description="INT8 TOPS (tera operations per second)")
    memory_bits: int = Field(gt=0, description="Total bits in on-chip memory")
    notes: str = Field(default="", description="Additional notes about the chip")


# --- Pre-defined profiles ---

TERAFAB_D3 = ChipProfile(
    name="TERAFAB D3 (projected)",
    node_nm=2,
    tdp_watts=300,
    max_temp_c=125,
    seu_cross_section_cm2=1e-14,
    tid_tolerance_krad=100,
    compute_tops=200,
    memory_bits=32 * 8 * 1024**3,  # 32 GB
    notes="SpaceX D3 — radiation-hardened for orbital AI Sat Mini",
)

GOOGLE_TRILLIUM_V6E = ChipProfile(
    name="Google Trillium TPU v6e",
    node_nm=4,
    tdp_watts=200,
    max_temp_c=85,
    seu_cross_section_cm2=5e-13,
    tid_tolerance_krad=15,
    compute_tops=450,
    memory_bits=32 * 8 * 1024**3,  # 32 GB
    notes="COTS TPU with shielding — survived 67 MeV proton beam test",
)

RAD5500 = ChipProfile(
    name="BAE RAD5500",
    node_nm=45,
    tdp_watts=15,
    max_temp_c=125,
    seu_cross_section_cm2=1e-15,
    tid_tolerance_krad=1000,
    compute_tops=0.001,
    memory_bits=256 * 8 * 1024**2,  # 256 MB
    notes="Current space-grade baseline — 0.9 GFLOPS",
)

NOEL_V_FT = ChipProfile(
    name="NOEL-V Fault-Tolerant (PolarFire FPGA)",
    node_nm=28,
    tdp_watts=5,
    max_temp_c=105,
    seu_cross_section_cm2=1e-14,
    tid_tolerance_krad=50,
    compute_tops=0.01,
    memory_bits=512 * 8 * 1024**2,  # 512 MB
    notes="Open-source RISC-V — first fault-tolerant RISC-V in orbit (TRISAT-R)",
)

ALL_CHIPS: list[ChipProfile] = [TERAFAB_D3, GOOGLE_TRILLIUM_V6E, RAD5500, NOEL_V_FT]
