"""Pre-defined radiation environment presets."""

from __future__ import annotations

from enum import Enum


class RadPreset(str, Enum):
    """Common LEO radiation environment presets."""

    LEO_500KM = "leo_500km"
    SSO_650KM = "sso_650km"
    LEO_2000KM = "leo_2000km"
