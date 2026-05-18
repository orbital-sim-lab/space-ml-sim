"""Deprecation helpers for the post-1.0 evolution of space-ml-sim.

Use :func:`deprecated` to mark a function or class as deprecated. The
warning is emitted exactly once per call site (via ``warnings.warn``) so
downstream notebooks do not get spammed.

Example::

    from space_ml_sim._deprecation import deprecated

    @deprecated(since="1.1", removal_in="2.0", replacement="FaultInjector.inject_v2")
    def inject_v1(...):
        ...
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar, cast

__all__ = ["deprecated"]

F = TypeVar("F", bound=Callable[..., object])


def _build_message(
    qualname: str,
    *,
    since: str,
    replacement: str | None,
    removal_in: str | None,
) -> str:
    parts: list[str] = [f"{qualname} is deprecated since space-ml-sim {since}"]
    if removal_in is not None:
        parts.append(f"and will be removed in {removal_in}")
    if replacement is not None:
        parts.append(f"; use {replacement} instead")
    return " ".join(parts).rstrip(";") + "."


def deprecated(
    *,
    since: str,
    replacement: str | None = None,
    removal_in: str | None = None,
) -> Callable[[F], F]:
    """Mark a function or class as deprecated.

    :param since: The release in which the deprecation took effect (e.g. ``"1.1"``).
    :param replacement: Optional fully-qualified replacement for the deprecated API.
    :param removal_in: Optional release at which the symbol will be removed (e.g. ``"2.0"``).
    :returns: A decorator that emits a :class:`DeprecationWarning` on call.

    For classes, the warning fires on instantiation. For functions, it fires on call.
    The decorator preserves the wrapped object's ``__doc__`` and ``__qualname__``
    and is safe to compose with ``@staticmethod`` and ``@classmethod`` *after*
    they are applied.
    """

    def decorator(obj: F) -> F:
        message = _build_message(
            obj.__qualname__,
            since=since,
            replacement=replacement,
            removal_in=removal_in,
        )

        if isinstance(obj, type):
            original_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def __init__(self: object, *args: object, **kwargs: object) -> None:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = __init__  # type: ignore[misc]
            obj.__doc__ = f"[DEPRECATED] {obj.__doc__ or ''}".rstrip()
            return cast(F, obj)

        @functools.wraps(obj)
        def wrapper(*args: object, **kwargs: object) -> object:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapper.__doc__ = f"[DEPRECATED] {obj.__doc__ or ''}".rstrip()
        return cast(F, wrapper)

    return decorator
