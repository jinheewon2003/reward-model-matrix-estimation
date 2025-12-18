"""The literature-standard baseline (Section 2.1): score(x, y) = w . embed(x, y),
trained with the Bradley-Terry logistic loss

    P(y_w > y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))

Every other model in this package is compared against this one.
"""

from __future__ import annotations

import numpy as np

from rme.embeddings import CachingEmbedder
from rme.linear import fit_logistic
from rme.types import POPULATION_USER, Comparison, Embedder, RewardModel


class BradleyTerryModel(RewardModel):
    def __init__(self, reg: float = 1e-3):
        self.reg = reg
        self.w: np.ndarray | None = None
        self.embedder: CachingEmbedder | None = None

    def fit(self, comparisons: list[Comparison], embedder: Embedder) -> "BradleyTerryModel":
        self.embedder = embedder if isinstance(embedder, CachingEmbedder) else CachingEmbedder(embedder)

        diffs = np.stack(
            [
                self.embedder.embed_pair(c.prompt, c.response_1)
                - self.embedder.embed_pair(c.prompt, c.response_2)
                for c in comparisons
            ]
        )
        labels = np.array([c.label for c in comparisons], dtype=np.float64)
        self.w = fit_logistic(diffs, labels, reg=self.reg)
        return self

    def score(self, prompt: str, response: str, user: str = POPULATION_USER) -> float:
        e = self.embedder.embed_pair(prompt, response)
        return float(self.w @ e)