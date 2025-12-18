from rme.evaluation import pairwise_accuracy
from rme.models.diff_pairwise_baseline import DiffPairwiseBaselineModel


def test_diff_pairwise_baseline_fits_train_and_generalizes(embedder, train_comparisons, held_out_comparisons):
    model = DiffPairwiseBaselineModel(n_baselines=30).fit(train_comparisons, embedder)
    assert pairwise_accuracy(model, train_comparisons) > 0.8
    assert pairwise_accuracy(model, held_out_comparisons) > 0.7