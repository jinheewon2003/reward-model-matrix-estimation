import pytest

from rme.evaluation import pairwise_accuracy
from rme.models.pairwise_to_response import PairwiseToResponseModel, _update_prompt_scores


def test_pairwise_to_response_fits_train_and_generalizes(embedder, train_comparisons, held_out_comparisons):
    model = PairwiseToResponseModel(n_factors=8, n_epochs=100).fit(train_comparisons, embedder)
    assert pairwise_accuracy(model, train_comparisons) > 0.7
    assert pairwise_accuracy(model, held_out_comparisons) > 0.55


def test_update_prompt_scores_first_observation_splits_p_and_1_minus_p():
    scores: dict[str, float] = {}
    _update_prompt_scores(scores, "a", "b", 0.7)
    assert scores == pytest.approx({"a": 0.7, "b": 0.3})


def test_update_prompt_scores_infers_unknown_side_from_odds():
    scores = {"a": 0.7, "b": 0.3}
    _update_prompt_scores(scores, "b", "c", 0.5)  # p(b > c) = 0.5 -> c should equal b
    assert abs(scores["c"] - scores["b"]) < 1e-9


def test_update_prompt_scores_resets_to_uniform_on_direction_conflict():
    scores = {"a": 0.9, "b": 0.1}  # a strongly preferred over b
    _update_prompt_scores(scores, "a", "b", 0.05)  # new evidence says b strongly preferred
    assert scores["a"] == scores["b"] == 1.0