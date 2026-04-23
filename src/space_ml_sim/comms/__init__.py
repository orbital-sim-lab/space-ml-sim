"""Communication systems: link budget analysis and RF/optical modeling."""

from space_ml_sim.comms.link_budget import (
    FrequencyBand,
    LinkBudgetResult,
    FREQUENCY_BANDS,
    free_space_path_loss_db,
    compute_link_budget,
)

__all__ = [
    "FrequencyBand",
    "LinkBudgetResult",
    "FREQUENCY_BANDS",
    "free_space_path_loss_db",
    "compute_link_budget",
]
