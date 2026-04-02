"""Simulation clock for managing time steps."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, Field


class SimClock(BaseModel):
    """Discrete simulation clock.

    Tracks elapsed time and provides iteration over time steps.
    """

    current_time: float = Field(default=0.0, description="Current simulation time in seconds")
    dt_seconds: float = Field(default=1.0, gt=0, description="Time step in seconds")
    elapsed_steps: int = Field(default=0, ge=0, description="Number of elapsed steps")

    def tick(self) -> "SimClock":
        """Advance the clock by one time step. Returns a new SimClock (immutable)."""
        return SimClock(
            current_time=self.current_time + self.dt_seconds,
            dt_seconds=self.dt_seconds,
            elapsed_steps=self.elapsed_steps + 1,
        )

    def steps_until(self, end_time: float) -> int:
        """Number of steps remaining until end_time."""
        remaining = end_time - self.current_time
        if remaining <= 0:
            return 0
        return int(remaining / self.dt_seconds)

    def iterate(self, duration_seconds: float) -> Iterator["SimClock"]:
        """Yield successive clock states for the given duration.

        Uses integer step counting to avoid floating-point drift
        in the loop termination condition.

        Yields:
            SimClock at each time step (new object each iteration).
        """
        num_steps = int(duration_seconds / self.dt_seconds)
        for i in range(num_steps):
            yield SimClock(
                current_time=self.current_time + i * self.dt_seconds,
                dt_seconds=self.dt_seconds,
                elapsed_steps=self.elapsed_steps + i,
            )
