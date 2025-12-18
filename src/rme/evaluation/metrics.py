"""Evaluation metrics (Section 3.2). Pairwise ranking accuracy is the
headline metric; `bootstrap_pairwise_accuracy` resamples the held-out set to
get a mean +/- std across "a series of 5 runs" so single-run noise doesn't
get mistaken for a real difference between models.
"""

from __future__ import annotations

import random

import numpy as np

from rme.types import RewardModel


def pairwise_accuracy(model: RewardModel, comparisons: list) -> float:
    if not comparisons:
        return float("nan")
    correct = 0
    for c in comparisons:
        p = model.predict_proba(c.prompt, c.response_1, c.response_2, c.user)
        predicted_1_preferred = p >= 0.5
        actual_1_preferred = c.label >= 0.5
        correct += int(predicted_1_preferred == actual_1_preferred)
    return correct / len(comparisons)


def bootstrap_pairwise_accuracy(
    model: RewardModel, comparisons: list, n_runs: int = 5, seed: int = 0
) -> dict[str, float]:
    """Resample `comparisons` with replacement `n_runs` times; report mean/std.

    This measures the *evaluation* set's sampling noise, not training-seed
    variance -- cheap to compute (no re-fitting) and enough to tell "0.62 vs
    0.60" apart from "0.62 vs 0.51".
    """
    if not comparisons:
        return {"mean": float("nan"), "std": float("nan"), "values": []}
    rng = random.Random(seed)
    per_example = np.array(
        [
            int(
                (model.predict_proba(c.prompt, c.response_1, c.response_2, c.user) >= 0.5)
                == (c.label >= 0.5)
            )
            for c in comparisons
        ]
    )
    values = []
    n = len(per_example)
    for _ in range(n_runs):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(float(per_example[idx].mean()))
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}