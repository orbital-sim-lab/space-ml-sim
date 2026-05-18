"""Tests for the post-1.0 deprecation utility."""

from __future__ import annotations

import warnings

import pytest

from space_ml_sim._deprecation import deprecated


def test_function_emits_deprecation_warning() -> None:
    @deprecated(since="1.1", replacement="new_func", removal_in="2.0")
    def old_func(x: int) -> int:
        return x + 1

    with pytest.warns(DeprecationWarning, match=r"old_func is deprecated since space-ml-sim 1\.1"):
        result = old_func(41)
    assert result == 42


def test_function_warning_message_includes_replacement_and_removal() -> None:
    @deprecated(since="1.1", replacement="NewClass", removal_in="2.0")
    def old_func() -> None:
        return None

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        old_func()
    assert len(record) == 1
    msg = str(record[0].message)
    assert "NewClass" in msg
    assert "2.0" in msg
    assert "1.1" in msg


def test_function_warning_works_without_replacement_or_removal() -> None:
    @deprecated(since="1.1")
    def old_func() -> int:
        return 7

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = old_func()
    assert result == 7
    assert len(record) == 1
    assert "1.1" in str(record[0].message)


def test_class_emits_deprecation_warning_on_instantiation() -> None:
    @deprecated(since="1.2", replacement="NewModel")
    class OldModel:
        def __init__(self, value: int) -> None:
            self.value = value

    with pytest.warns(DeprecationWarning, match="OldModel is deprecated"):
        instance = OldModel(value=10)
    assert instance.value == 10


def test_function_metadata_is_preserved() -> None:
    @deprecated(since="1.1")
    def documented(x: int) -> int:
        """The original docstring."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ is not None
    assert "DEPRECATED" in documented.__doc__
    assert "original docstring" in documented.__doc__


def test_class_docstring_is_marked_deprecated() -> None:
    @deprecated(since="1.2")
    class Marked:
        """Original class doc."""

        def __init__(self) -> None:
            pass

    assert Marked.__doc__ is not None
    assert "DEPRECATED" in Marked.__doc__
    assert "Original class doc" in Marked.__doc__
