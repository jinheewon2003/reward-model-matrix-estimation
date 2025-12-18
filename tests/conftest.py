"""Shared synthetic-data fixtures.

All models are exercised against a deterministic, network-free synthetic
preference task: the *only* signal distinguishing a preferred response from
a rejected one is the presence of the token "excellent" vs. "terrible"; each
comparison also carries an item-unique, uninformative token ("item42") that
should wash out under regularization. This gives every model something
genuinely learnable without needing real embeddings or real datasets, and
`HashingEmbedder` (no model download, no network) is a sufficient encoder
for it.
"""

from __future__ import annotations

import pytest

from rme.embeddings import HashingEmbedder
from rme.types import POPULATION_USER, Comparison

PROMPTS = ["Summarize the article.", "Answer the question.", "Write a short story."]


def make_comparisons(
    n: int, seed: int = 0, prompts: list[str] | None = None, item_offset: int = 0
) -> list[Comparison]:
    prompts = prompts or PROMPTS
    comparisons = []
    for i in range(item_offset, item_offset + n):
        prompt = prompts[i % len(prompts)]
        good = f"This is an excellent response to item{i}."
        bad = f"This is a terrible response to item{i}."
        # alternate which side the good response is on, so the model can't
        # cheat by learning "response_1 is always preferred"
        if i % 2 == 0:
            comparisons.append(Comparison(POPULATION_USER, prompt, good, bad, 1.0, source="synthetic"))
        else:
            comparisons.append(Comparison(POPULATION_USER, prompt, bad, good, 0.0, source="synthetic"))
    return comparisons


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64, seed=0)


@pytest.fixture
def train_comparisons() -> list[Comparison]:
    return make_comparisons(120, seed=0)


@pytest.fixture
def held_out_comparisons() -> list[Comparison]:
    # disjoint item ids from train_comparisons so this genuinely tests
    # generalization to responses never seen during fit, not memorization.
    return make_comparisons(30, seed=1, prompts=PROMPTS, item_offset=10_000)
