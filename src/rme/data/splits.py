"""Train/test and out-of-distribution splitting (Section 3.2).

`train_test_split` groups by prompt before splitting so the same prompt never
leaks across train and test (otherwise pairwise accuracy is inflated by the
model having memorized that exact prompt's response set).

`ood_split` implements the three OOD conditions named in the proposal:
different dataset, different response length/style, or -- if the caller
supplies a `domain_fn` -- a different topical domain.
"""

from __future__ import annotations

import random
from typing import Callable

from rme.types import Comparison


def train_test_split(
    comparisons: list[Comparison], test_frac: float = 0.2, seed: int = 0
) -> tuple[list[Comparison], list[Comparison]]:
    prompts = sorted({c.prompt for c in comparisons})
    rng = random.Random(seed)
    rng.shuffle(prompts)
    n_test = max(1, int(len(prompts) * test_frac))
    test_prompts = set(prompts[:n_test])
    train = [c for c in comparisons if c.prompt not in test_prompts]
    test = [c for c in comparisons if c.prompt in test_prompts]
    return train, test


def ood_split_by_source(
    comparisons: list[Comparison], train_sources: list[str], test_sources: list[str]
) -> tuple[list[Comparison], list[Comparison]]:
    """Train on one dataset, test on another entirely -- the strongest OOD test."""
    train = [c for c in comparisons if c.source in train_sources]
    test = [c for c in comparisons if c.source in test_sources]
    return train, test


def ood_split_by_length(
    comparisons: list[Comparison], percentile: float = 50.0
) -> tuple[list[Comparison], list[Comparison]]:
    """Train on shorter responses, test on longer ones (a style/length OOD split)."""
    lengths = sorted(len(c.response_1) + len(c.response_2) for c in comparisons)
    if not lengths:
        return [], []
    cut = lengths[int(len(lengths) * percentile / 100.0)]
    train = [c for c in comparisons if len(c.response_1) + len(c.response_2) <= cut]
    test = [c for c in comparisons if len(c.response_1) + len(c.response_2) > cut]
    return train, test


def ood_split_by_domain(
    comparisons: list[Comparison], domain_fn: Callable[[Comparison], str], train_domain: str
) -> tuple[list[Comparison], list[Comparison]]:
    """Train on one domain (e.g. topic bucket from `domain_fn`), test on the rest."""
    train = [c for c in comparisons if domain_fn(c) == train_domain]
    test = [c for c in comparisons if domain_fn(c) != train_domain]
    return train, test