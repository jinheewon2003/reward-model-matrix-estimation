from rme.evaluation import bootstrap_pairwise_accuracy, pairwise_accuracy
from rme.types import POPULATION_USER, Comparison


class _StubModel:
    """Always says response_1 is preferred."""

    def predict_proba(self, prompt, r1, r2, user=POPULATION_USER):
        return 0.9


def test_pairwise_accuracy_matches_hand_count():
    comparisons = [
        Comparison(POPULATION_USER, "p", "r1", "r2", 1.0),  # correct
        Comparison(POPULATION_USER, "p", "r1", "r2", 1.0),  # correct
        Comparison(POPULATION_USER, "p", "r1", "r2", 0.0),  # wrong
        Comparison(POPULATION_USER, "p", "r1", "r2", 0.0),  # wrong
    ]
    assert pairwise_accuracy(_StubModel(), comparisons) == 0.5


def test_pairwise_accuracy_empty_is_nan():
    result = pairwise_accuracy(_StubModel(), [])
    assert result != result  # NaN != NaN


def test_bootstrap_pairwise_accuracy_matches_point_estimate_in_expectation():
    comparisons = [Comparison(POPULATION_USER, "p", "r1", "r2", 1.0) for _ in range(50)]
    result = bootstrap_pairwise_accuracy(_StubModel(), comparisons, n_runs=5, seed=0)
    assert result["mean"] == 1.0
    assert result["std"] == 0.0
    assert len(result["values"]) == 5