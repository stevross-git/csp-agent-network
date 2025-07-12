"""Cryptographically secure random utilities."""
from __future__ import annotations

import os
import secrets
import warnings
import random
from typing import Sequence, TypeVar

_T = TypeVar("_T")


_system_random = random.SystemRandom()


def _entropy_available() -> bool:
    """Return True if the system reports sufficient entropy."""
    entropy_file = "/proc/sys/kernel/random/entropy_avail"
    try:
        with open(entropy_file, "r", encoding="utf-8") as fh:
            value = int(fh.read().strip())
        if value < 128:
            warnings.warn(
                f"Low system entropy detected: {value}",
                RuntimeWarning,
            )
        return True
    except Exception:
        # File may not exist on non-Linux systems; assume OK
        return True


def _get_urandom(length: int) -> bytes:
    try:
        return os.urandom(length)
    except NotImplementedError:
        # Fallback when os.urandom is unavailable
        warnings.warn("os.urandom unavailable, using SystemRandom", RuntimeWarning)
        return bytes(_system_random.getrandbits(8) for _ in range(length))


def secure_randint(min_val: int, max_val: int) -> int:
    """Return a random integer securely between min_val and max_val inclusive."""
    if min_val > max_val:
        raise ValueError("min_val must be <= max_val")
    range_size = max_val - min_val + 1
    # Use secrets.randbelow for uniform distribution
    return min_val + secrets.randbelow(range_size)


def secure_choice(seq: Sequence[_T]) -> _T:
    """Select a random element from a non-empty sequence securely."""
    if not seq:
        raise IndexError("cannot choose from an empty sequence")
    idx = secrets.randbelow(len(seq))
    return seq[idx]


def secure_bytes(length: int) -> bytes:
    """Return cryptographically secure random bytes."""
    if length <= 0:
        raise ValueError("length must be positive")
    _entropy_available()
    return _get_urandom(length)


def secure_token(length: int) -> str:
    """Return a URL-safe random text token with the given number of bytes."""
    if length <= 0:
        raise ValueError("length must be positive")
    _entropy_available()
    return secrets.token_urlsafe(length)


__all__ = [
    "secure_randint",
    "secure_choice",
    "secure_bytes",
    "secure_token",
]
