"""Model checkpointing for fault recovery.

Periodically saves model state so it can be restored after
radiation-induced corruption is detected.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import torch


class CheckpointManager:
    """Manages model checkpoints for radiation fault recovery.

    Keeps a sliding window of recent checkpoints so the model
    can be rolled back to a known-good state.
    """

    def __init__(self, max_checkpoints: int = 3) -> None:
        """Initialize checkpoint manager.

        Args:
            max_checkpoints: Maximum number of checkpoints to retain.
        """
        self._checkpoints: deque[dict[str, Any]] = deque(maxlen=max_checkpoints)
        self._max = max_checkpoints

    def save(self, model: torch.nn.Module, metadata: dict[str, Any] | None = None) -> int:
        """Save a model checkpoint.

        Args:
            model: Model to checkpoint.
            metadata: Optional metadata (e.g., step count, accuracy).

        Returns:
            Index of the saved checkpoint.
        """
        entry = {
            "state_dict": copy.deepcopy(model.state_dict()),
            "metadata": metadata or {},
        }
        self._checkpoints.append(entry)
        return len(self._checkpoints) - 1

    def restore(self, model: torch.nn.Module, index: int = -1) -> dict[str, Any]:
        """Restore a model from a checkpoint.

        Args:
            model: Model to restore into.
            index: Checkpoint index (-1 for most recent).

        Returns:
            Metadata associated with the restored checkpoint.

        Raises:
            IndexError: If no checkpoints are available.
        """
        if not self._checkpoints:
            raise IndexError("No checkpoints available")

        entry = self._checkpoints[index]
        # No deepcopy needed — load_state_dict copies tensor data into
        # the model's existing buffers without mutating the source dict.
        model.load_state_dict(entry["state_dict"])
        return entry["metadata"]

    @property
    def count(self) -> int:
        """Number of stored checkpoints."""
        return len(self._checkpoints)

    def clear(self) -> None:
        """Remove all checkpoints."""
        self._checkpoints.clear()
