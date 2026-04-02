"""Tests for clock.py — SimClock discrete simulation clock."""

from __future__ import annotations

import pytest

from space_ml_sim.core.clock import SimClock


# ---------------------------------------------------------------------------
# SimClock.tick
# ---------------------------------------------------------------------------


class TestTick:
    def test_tick_advances_time(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        next_clock = clock.tick()
        assert next_clock.current_time == pytest.approx(1.0)

    def test_tick_advances_by_dt(self):
        clock = SimClock(current_time=0.0, dt_seconds=30.0)
        next_clock = clock.tick()
        assert next_clock.current_time == pytest.approx(30.0)

    def test_tick_increments_elapsed_steps(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0, elapsed_steps=0)
        next_clock = clock.tick()
        assert next_clock.elapsed_steps == 1

    def test_tick_preserves_dt_seconds(self):
        clock = SimClock(current_time=0.0, dt_seconds=60.0)
        next_clock = clock.tick()
        assert next_clock.dt_seconds == pytest.approx(60.0)

    def test_tick_returns_new_instance(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        next_clock = clock.tick()
        assert next_clock is not clock

    def test_original_clock_unchanged_after_tick(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0, elapsed_steps=0)
        clock.tick()
        assert clock.current_time == pytest.approx(0.0)
        assert clock.elapsed_steps == 0

    def test_multiple_ticks_accumulate_correctly(self):
        clock = SimClock(current_time=0.0, dt_seconds=10.0)
        for _ in range(5):
            clock = clock.tick()
        assert clock.current_time == pytest.approx(50.0)
        assert clock.elapsed_steps == 5

    def test_tick_from_nonzero_start(self):
        clock = SimClock(current_time=100.0, dt_seconds=5.0, elapsed_steps=20)
        next_clock = clock.tick()
        assert next_clock.current_time == pytest.approx(105.0)
        assert next_clock.elapsed_steps == 21

    def test_tick_with_fractional_dt(self):
        clock = SimClock(current_time=0.0, dt_seconds=0.5)
        next_clock = clock.tick()
        assert next_clock.current_time == pytest.approx(0.5)

    def test_tick_is_immutable_chain(self):
        """Verify original is never mutated across a chain of ticks."""
        original = SimClock(current_time=0.0, dt_seconds=1.0)
        original_time = original.current_time
        original_steps = original.elapsed_steps

        c1 = original.tick()
        c2 = c1.tick()
        _c3 = c2.tick()

        assert original.current_time == pytest.approx(original_time)
        assert original.elapsed_steps == original_steps


# ---------------------------------------------------------------------------
# SimClock.steps_until
# ---------------------------------------------------------------------------


class TestStepsUntil:
    def test_steps_until_future_time(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        assert clock.steps_until(10.0) == 10

    def test_steps_until_with_larger_dt(self):
        clock = SimClock(current_time=0.0, dt_seconds=60.0)
        # 3600 / 60 = 60 steps
        assert clock.steps_until(3600.0) == 60

    def test_steps_until_past_time_returns_zero(self):
        clock = SimClock(current_time=100.0, dt_seconds=1.0)
        assert clock.steps_until(50.0) == 0

    def test_steps_until_same_time_returns_zero(self):
        clock = SimClock(current_time=100.0, dt_seconds=1.0)
        assert clock.steps_until(100.0) == 0

    def test_steps_until_one_step_ahead(self):
        clock = SimClock(current_time=0.0, dt_seconds=5.0)
        assert clock.steps_until(5.0) == 1

    def test_steps_until_returns_int(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        result = clock.steps_until(10.0)
        assert isinstance(result, int)

    def test_steps_until_from_nonzero_start(self):
        clock = SimClock(current_time=50.0, dt_seconds=10.0)
        # (100 - 50) / 10 = 5 steps
        assert clock.steps_until(100.0) == 5

    def test_steps_until_large_duration(self):
        # One year in seconds with 1-second steps
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        one_year = 365 * 24 * 3600
        assert clock.steps_until(float(one_year)) == one_year

    def test_steps_until_does_not_mutate_clock(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        clock.steps_until(100.0)
        assert clock.current_time == pytest.approx(0.0)
        assert clock.elapsed_steps == 0


# ---------------------------------------------------------------------------
# SimClock.iterate
# ---------------------------------------------------------------------------


class TestIterate:
    def test_iterate_yields_correct_count(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        clocks = list(clock.iterate(duration_seconds=5.0))
        assert len(clocks) == 5

    def test_iterate_with_larger_dt(self):
        clock = SimClock(current_time=0.0, dt_seconds=60.0)
        # 300 / 60 = 5 clocks
        clocks = list(clock.iterate(duration_seconds=300.0))
        assert len(clocks) == 5

    def test_iterate_yields_correct_times(self):
        clock = SimClock(current_time=0.0, dt_seconds=10.0)
        clocks = list(clock.iterate(duration_seconds=30.0))
        expected_times = [0.0, 10.0, 20.0]
        for yielded, expected in zip(clocks, expected_times):
            assert yielded.current_time == pytest.approx(expected)

    def test_iterate_starts_at_current_time(self):
        clock = SimClock(current_time=100.0, dt_seconds=5.0)
        clocks = list(clock.iterate(duration_seconds=10.0))
        assert clocks[0].current_time == pytest.approx(100.0)

    def test_iterate_each_clock_is_new_object(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        clocks = list(clock.iterate(duration_seconds=3.0))
        # All yielded clocks should be distinct objects
        ids = [id(c) for c in clocks]
        assert len(set(ids)) == len(ids)

    def test_iterate_elapsed_steps_increase(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0, elapsed_steps=0)
        clocks = list(clock.iterate(duration_seconds=4.0))
        for i, c in enumerate(clocks):
            assert c.elapsed_steps == i

    def test_iterate_zero_duration_yields_nothing(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        clocks = list(clock.iterate(duration_seconds=0.0))
        assert clocks == []

    def test_iterate_duration_less_than_dt_yields_nothing(self):
        """Duration smaller than one step should yield nothing — zero complete steps fit."""
        clock = SimClock(current_time=0.0, dt_seconds=2.0)
        clocks = list(clock.iterate(duration_seconds=1.0))
        assert len(clocks) == 0

    def test_iterate_does_not_mutate_original_clock(self):
        clock = SimClock(current_time=0.0, dt_seconds=1.0, elapsed_steps=0)
        list(clock.iterate(duration_seconds=10.0))
        assert clock.current_time == pytest.approx(0.0)
        assert clock.elapsed_steps == 0

    def test_iterate_preserves_dt_across_all_yielded_clocks(self):
        dt = 15.0
        clock = SimClock(current_time=0.0, dt_seconds=dt)
        clocks = list(clock.iterate(duration_seconds=60.0))
        for c in clocks:
            assert c.dt_seconds == pytest.approx(dt)

    def test_iterate_one_orbit_period(self):
        """Simulate one 90-minute LEO orbit at 60-second steps -> 90 ticks."""
        clock = SimClock(current_time=0.0, dt_seconds=60.0)
        clocks = list(clock.iterate(duration_seconds=90 * 60.0))
        assert len(clocks) == 90
        assert clocks[-1].current_time == pytest.approx(89 * 60.0)

    def test_iterate_returns_generator(self):
        """iterate() should be a generator (lazy evaluation)."""
        import types

        clock = SimClock(current_time=0.0, dt_seconds=1.0)
        result = clock.iterate(duration_seconds=10.0)
        assert isinstance(result, types.GeneratorType)


# ---------------------------------------------------------------------------
# SimClock construction and field validation
# ---------------------------------------------------------------------------


class TestSimClockConstruction:
    def test_default_values(self):
        clock = SimClock()
        assert clock.current_time == pytest.approx(0.0)
        assert clock.dt_seconds == pytest.approx(1.0)
        assert clock.elapsed_steps == 0

    def test_custom_values(self):
        clock = SimClock(current_time=500.0, dt_seconds=10.0, elapsed_steps=50)
        assert clock.current_time == pytest.approx(500.0)
        assert clock.dt_seconds == pytest.approx(10.0)
        assert clock.elapsed_steps == 50

    def test_dt_must_be_positive(self):
        """dt_seconds=0 should be rejected (gt=0 constraint)."""
        with pytest.raises(Exception):
            SimClock(dt_seconds=0.0)

    def test_negative_dt_rejected(self):
        with pytest.raises(Exception):
            SimClock(dt_seconds=-1.0)

    def test_negative_elapsed_steps_rejected(self):
        """elapsed_steps has ge=0 constraint."""
        with pytest.raises(Exception):
            SimClock(elapsed_steps=-1)
