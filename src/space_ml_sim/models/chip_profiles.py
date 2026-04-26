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
    seu_cross_section_cm2: float = Field(gt=0, description="SEU cross-section per bit in cm^2")
    tid_tolerance_krad: float = Field(gt=0, description="Total ionizing dose tolerance in krad(Si)")
    compute_tops: float = Field(ge=0, description="INT8 TOPS (tera operations per second)")
    memory_bits: int = Field(gt=0, description="Total bits in on-chip memory")
    notes: str = Field(default="", description="Additional notes about the chip")

    
        """Auto-generated docstring."""
        return f"ChipProfile({self.name!r}, {self.compute_tops} TOPS, TID={self.tid_tolerance_krad} krad)"


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

TRILLIUM_V6E = ChipProfile(
    name="Trillium TPU v6e",
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

JETSON_AGX_ORIN = ChipProfile(
    name="NVIDIA Jetson AGX Orin",
    node_nm=8,
    tdp_watts=60,
    max_temp_c=85,
    seu_cross_section_cm2=3e-13,
    tid_tolerance_krad=10,
    compute_tops=275,
    memory_bits=64 * 8 * 1024**3,  # 64 GB
    notes="COTS — flying on Planet Labs sats, requires radiation shielding",
)

ZYNQ_ULTRASCALE = ChipProfile(
    name="AMD Zynq UltraScale+ (Xiphos Q8S)",
    node_nm=16,
    tdp_watts=10,
    max_temp_c=100,
    seu_cross_section_cm2=5e-14,
    tid_tolerance_krad=30,
    compute_tops=0.5,
    memory_bits=4 * 8 * 1024**3,  # 4 GB
    notes="Rad-tolerant FPGA SoC — widely used OBC platform",
)

VERSAL_AI_CORE = ChipProfile(
    name="AMD Versal AI Core XQRVC1902",
    node_nm=7,
    tdp_watts=75,
    max_temp_c=100,
    seu_cross_section_cm2=2e-14,
    tid_tolerance_krad=100,
    compute_tops=130,
    memory_bits=16 * 8 * 1024**3,  # 16 GB
    notes="Space-grade — qualified for 15-year missions, sampling 2026",
)

SAMRH71 = ChipProfile(
    name="Microchip SAMRH71F20C (Arm Cortex-M7)",
    node_nm=65,
    tdp_watts=1.5,
    max_temp_c=125,
    seu_cross_section_cm2=8e-15,
    tid_tolerance_krad=100,
    compute_tops=0.0005,
    memory_bits=2 * 8 * 1024**2,  # 2 MB SRAM
    notes="Rad-hard MCU — 300 MHz Cortex-M7, ESA-qualified for JUICE mission, "
    "SEL immune up to 62 MeV·cm²/mg",
)

GR740 = ChipProfile(
    name="Cobham Gaisler GR740 (LEON4 quad-core)",
    node_nm=65,
    tdp_watts=3.0,
    max_temp_c=125,
    seu_cross_section_cm2=5e-15,
    tid_tolerance_krad=300,
    compute_tops=0.002,
    memory_bits=4 * 8 * 1024**2,  # 4 MB on-chip
    notes="Rad-hard SPARC V8 quad-core — ESA's next-gen OBC processor, "
    "EDAC on all memories, flying on PLATO and FLEX",
)

XQRKU060 = ChipProfile(
    name="AMD Xilinx XQRKU060 (Kintex UltraScale)",
    node_nm=20,
    tdp_watts=12,
    max_temp_c=100,
    seu_cross_section_cm2=3e-14,
    tid_tolerance_krad=100,
    compute_tops=1.5,
    memory_bits=8 * 8 * 1024**2,  # 8 MB BRAM
    notes="Space-grade FPGA — most widely used reconfigurable space processor, "
    "built-in FRAME_ECC for scrubbing, qualified for 15-year missions",
)

ALL_CHIPS: list[ChipProfile] = [
    TERAFAB_D3,
    TRILLIUM_V6E,
    RAD5500,
    NOEL_V_FT,
    JETSON_AGX_ORIN,
    ZYNQ_ULTRASCALE,
    VERSAL_AI_CORE,
    SAMRH71,
    GR740,
    XQRKU060,
]
