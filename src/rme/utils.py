"""Small shared helpers."""

from __future__ import annotations

_SEP = "␟"  # unit separator; never appears in ordinary text


def item_key(prompt: str, response: str) -> str:
    """Stable id for a (prompt, response) pair, used as a matrix column key."""
    return f"{prompt}{_SEP}{response}"


def triplet_key(prompt: str, response_1: str, response_2: str) -> str:
    """Stable id for a (prompt, response_1, response_2) comparison, used as a row key."""
    return f"{prompt}{_SEP}{response_1}{_SEP}{response_2}"