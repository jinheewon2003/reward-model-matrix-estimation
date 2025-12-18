from rme.evaluation import pairwise_accuracy
from rme.models.score_diff_mf import ScoreDiffMFModel


def test_score_diff_mf_fits_train_and_generalizes(embedder, train_comparisons, held_out_comparisons):
    model = ScoreDiffMFModel(n_factors=8, n_epochs=150).fit(train_comparisons, embedder)
    assert pairwise_accuracy(model, train_comparisons) > 0.75
    assert pairwise_accuracy(model, held_out_comparisons) > 0.6