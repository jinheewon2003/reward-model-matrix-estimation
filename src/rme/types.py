"""Shared data types.

The proposal defines every training sample as a tuple ``(u, P, r1, r2, y)``:
a user/entity id, a prompt, two candidate responses, and a label indicating
which response was preferred (``y=1`` means ``r1`` was preferred, ``y=0``
means ``r2`` was preferred; soft labels in ``[0, 1]`` are allowed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Sentinel user id for datasets with no per-annotator identifiers (Section 3.1:
# Anthropic_HH_Golden and HelpSteer2 are treated as one aggregate "population user").
POPULATION_USER = "population"


@dataclass(frozen=True)
class Comparison:
    user: str
    prompt: str
    response_1: str
    response_2: str
    label: float  # P(response_1 preferred); 1.0 / 0.0 for hard labels
    source: str = ""  # originating dataset name, for bookkeeping/OOD splits

    def __post_init__(self) -> None:
        if not 0.0 <= self.label <= 1.0:
            raise ValueError(f"label must be in [0, 1], got {self.label}")


class Embedder(ABC):
    """Maps free text to a fixed-size vector. See rme.embeddings for implementations."""

    @abstractmethod
    def embed(self, texts: list[str]) -> "np.ndarray":  # noqa: F821 - see embeddings.py
        ...

    def embed_pair(self, prompt: str, response: str) -> "np.ndarray":  # noqa: F821
        return self.embed([f"{prompt}\n\n{response}"])[0]


class RewardModel(ABC):
    """Common interface for the baseline and the four proposed reward models."""

    @abstractmethod
    def fit(self, comparisons: list[Comparison], embedder: Embedder) -> "RewardModel":
        ...

    @abstractmethod
    def score(self, prompt: str, response: str, user: str = POPULATION_USER) -> float:
        """Return a scalar quality score r(prompt, response) for the given user."""
        ...

    def predict_proba(
        self, prompt: str, response_1: str, response_2: str, user: str = POPULATION_USER
    ) -> float:
        """P(response_1 preferred over response_2), via a BT-style softmax over scores."""
        import numpy as np

        s1 = self.score(prompt, response_1, user)
        s2 = self.score(prompt, response_2, user)
        return float(1.0 / (1.0 + np.exp(-(s1 - s2))))