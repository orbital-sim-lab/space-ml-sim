"""Tests for CheckpointManager: save, restore, sliding window, clear, count."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from space_ml_sim.compute.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed: int = 0) -> nn.Linear:
    """Small deterministic linear model for fast testing."""
    torch.manual_seed(seed)
    return nn.Linear(4, 2)


def _model_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a model in eval mode with no_grad and return output."""
    model.eval()
    with torch.no_grad():
        return model(x)


def _weights_equal(a: nn.Module, b: nn.Module) -> bool:
    """Return True if all parameter tensors are identical between two models."""
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if not torch.equal(pa, pb):
            return False
    return True


# ---------------------------------------------------------------------------
# save: return value
# ---------------------------------------------------------------------------

class TestSaveReturnValue:
    def test_first_save_returns_zero(self):
        mgr = CheckpointManager()
        model = _make_model()
        idx = mgr.save(model)
        assert idx == 0

    def test_second_save_returns_one(self):
        mgr = CheckpointManager()
        model = _make_model()
        mgr.save(model)
        idx = mgr.save(model)
        assert idx == 1

    def test_third_save_returns_two(self):
        mgr = CheckpointManager(max_checkpoints=5)
        model = _make_model()
        for _ in range(2):
            mgr.save(model)
        idx = mgr.save(model)
        assert idx == 2

    def test_save_stores_metadata(self):
        mgr = CheckpointManager()
        model = _make_model()
        mgr.save(model, metadata={"step": 42, "accuracy": 0.95})
        meta = mgr.restore(model, index=0)
        assert meta == {"step": 42, "accuracy": 0.95}

    def test_save_with_no_metadata_returns_empty_dict_on_restore(self):
        mgr = CheckpointManager()
        model = _make_model()
        mgr.save(model)
        meta = mgr.restore(model)
        assert meta == {}


# ---------------------------------------------------------------------------
# restore: correct weights
# ---------------------------------------------------------------------------

class TestRestoreWeights:
    def test_restore_loads_saved_weights(self):
        mgr = CheckpointManager()

        # Save model A
        model_a = _make_model(seed=0)
        mgr.save(model_a)

        # Corrupt weights in-place (mutate deliberately for test setup only)
        model_b = _make_model(seed=99)
        assert not _weights_equal(model_a, model_b), "Seeds should produce different weights"

        # Restore into model_b — should now match model_a
        mgr.restore(model_b, index=0)
        assert _weights_equal(model_a, model_b)

    def test_restore_does_not_mutate_checkpoint(self):
        """Restoring into a corrupted model must not corrupt the stored checkpoint."""
        mgr = CheckpointManager()
        model_a = _make_model(seed=1)
        mgr.save(model_a)

        # Corrupt model_a's weights after saving
        with torch.no_grad():
            for p in model_a.parameters():
                p.fill_(0.0)

        # Restore and verify it matches original (not the zeroed weights)
        model_c = _make_model(seed=1)  # fresh copy with same original weights
        mgr.restore(model_a, index=0)
        assert _weights_equal(model_a, model_c)

    def test_restore_negative_one_returns_most_recent(self):
        mgr = CheckpointManager(max_checkpoints=5)

        model_early = _make_model(seed=10)
        mgr.save(model_early)

        model_late = _make_model(seed=20)
        mgr.save(model_late)

        # Restore -1 should give us model_late's weights
        target = _make_model(seed=99)
        mgr.restore(target, index=-1)
        assert _weights_equal(target, model_late)

    def test_restore_first_index_after_two_saves(self):
        mgr = CheckpointManager(max_checkpoints=5)

        model_first = _make_model(seed=10)
        mgr.save(model_first)

        model_second = _make_model(seed=20)
        mgr.save(model_second)

        target = _make_model(seed=99)
        mgr.restore(target, index=0)
        assert _weights_equal(target, model_first)


# ---------------------------------------------------------------------------
# restore with no checkpoints raises IndexError
# ---------------------------------------------------------------------------

class TestRestoreNoCheckpoints:
    def test_raises_index_error_when_empty(self):
        mgr = CheckpointManager()
        model = _make_model()
        with pytest.raises(IndexError, match="No checkpoints available"):
            mgr.restore(model)

    def test_raises_after_clear(self):
        mgr = CheckpointManager()
        model = _make_model()
        mgr.save(model)
        mgr.clear()
        with pytest.raises(IndexError):
            mgr.restore(model)


# ---------------------------------------------------------------------------
# Sliding window: max_checkpoints=2 evicts oldest
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_count_does_not_exceed_max(self):
        mgr = CheckpointManager(max_checkpoints=2)
        model = _make_model()
        for _ in range(5):
            mgr.save(model)
        assert mgr.count == 2

    def test_oldest_is_evicted_when_window_full(self):
        """With max_checkpoints=2, the third save evicts the first checkpoint."""
        mgr = CheckpointManager(max_checkpoints=2)

        model_a = _make_model(seed=1)
        mgr.save(model_a)  # slot 0 — will be evicted

        model_b = _make_model(seed=2)
        mgr.save(model_b)  # slot 1

        model_c = _make_model(seed=3)
        mgr.save(model_c)  # slot 2 — evicts model_a, keeps model_b and model_c

        # After eviction the deque has [model_b, model_c].
        # Index 0 should be model_b, index -1 should be model_c.
        target = _make_model(seed=99)

        mgr.restore(target, index=0)
        assert _weights_equal(target, model_b)

        mgr.restore(target, index=-1)
        assert _weights_equal(target, model_c)

    def test_third_save_index_reflects_current_deque_size(self):
        """save() returns len(deque) - 1 which is max - 1 once the window is full."""
        mgr = CheckpointManager(max_checkpoints=2)
        model = _make_model()
        mgr.save(model)
        mgr.save(model)
        idx = mgr.save(model)  # deque is now full at 2
        assert idx == 1  # len(deque) - 1 = 2 - 1

    def test_max_checkpoints_one_keeps_only_last(self):
        mgr = CheckpointManager(max_checkpoints=1)
        model_a = _make_model(seed=5)
        mgr.save(model_a)

        model_b = _make_model(seed=6)
        mgr.save(model_b)  # evicts model_a

        assert mgr.count == 1

        target = _make_model(seed=99)
        mgr.restore(target)
        assert _weights_equal(target, model_b)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_all_checkpoints(self):
        mgr = CheckpointManager()
        model = _make_model()
        mgr.save(model)
        mgr.save(model)
        mgr.clear()
        assert mgr.count == 0

    def test_clear_on_empty_does_not_raise(self):
        mgr = CheckpointManager()
        mgr.clear()  # should be a no-op
        assert mgr.count == 0

    def test_can_save_again_after_clear(self):
        mgr = CheckpointManager()
        model = _make_model()
        mgr.save(model)
        mgr.clear()
        idx = mgr.save(model)
        assert idx == 0
        assert mgr.count == 1


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_count_starts_at_zero(self):
        mgr = CheckpointManager()
        assert mgr.count == 0

    def test_count_increments_with_each_save(self):
        mgr = CheckpointManager(max_checkpoints=10)
        model = _make_model()
        for expected in range(1, 6):
            mgr.save(model)
            assert mgr.count == expected

    def test_count_capped_at_max_checkpoints(self):
        mgr = CheckpointManager(max_checkpoints=3)
        model = _make_model()
        for _ in range(10):
            mgr.save(model)
        assert mgr.count == 3


# ---------------------------------------------------------------------------
# Restored model produces same output as when saved
# ---------------------------------------------------------------------------

class TestRestoredModelOutput:
    def test_restored_model_matches_saved_output(self):
        mgr = CheckpointManager()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        model_orig = _make_model(seed=7)
        original_output = _model_output(model_orig, x).clone()
        mgr.save(model_orig)

        # Corrupt the model
        with torch.no_grad():
            for p in model_orig.parameters():
                p.fill_(float("nan"))

        # Restore and run inference again
        mgr.restore(model_orig)
        restored_output = _model_output(model_orig, x)

        assert torch.allclose(original_output, restored_output, atol=1e-6)

    def test_sequential_model_output_survives_restore(self):
        mgr = CheckpointManager()
        x = torch.randn(4, 4)

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        torch.manual_seed(0)
        for p in model.parameters():
            nn.init.normal_(p)

        model.eval()
        with torch.no_grad():
            expected = model(x).clone()

        mgr.save(model)

        # Scramble
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()

        mgr.restore(model)
        model.eval()
        with torch.no_grad():
            actual = model(x)

        assert torch.allclose(expected, actual, atol=1e-6)
