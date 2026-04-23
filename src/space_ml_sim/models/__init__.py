"""Hardware and radiation profile models."""

from space_ml_sim.models.chip_profiles import (
    ChipProfile,
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
    ALL_CHIPS,
)
from space_ml_sim.models.rad_profiles import RadPreset

__all__ = [
    "ChipProfile",
    "TERAFAB_D3",
    "TRILLIUM_V6E",
    "RAD5500",
    "NOEL_V_FT",
    "JETSON_AGX_ORIN",
    "ZYNQ_ULTRASCALE",
    "VERSAL_AI_CORE",
    "SAMRH71",
    "GR740",
    "XQRKU060",
    "ALL_CHIPS",
    "RadPreset",
]
