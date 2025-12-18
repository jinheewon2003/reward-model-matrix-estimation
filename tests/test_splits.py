from rme.data.splits import ood_split_by_length, ood_split_by_source, train_test_split
from rme.types import POPULATION_USER, Comparison


def _comparisons_with_prompts(n_prompts=10, per_prompt=3):
    out = []
    for p in range(n_prompts):
        for i in range(per_prompt):
            out.append(
                Comparison(POPULATION_USER, f"prompt{p}", f"r1_{i}", f"r2_{i}", 1.0, source="a")
            )
    return out


def test_train_test_split_has_no_prompt_leakage():
    comparisons = _comparisons_with_prompts()
    train, test = train_test_split(comparisons, test_frac=0.3, seed=0)
    train_prompts = {c.prompt for c in train}
    test_prompts = {c.prompt for c in test}
    assert train_prompts.isdisjoint(test_prompts)
    assert len(train) + len(test) == len(comparisons)


def test_ood_split_by_source_separates_datasets():
    a = [Comparison(POPULATION_USER, "p", "r1", "r2", 1.0, source="a")]
    b = [Comparison(POPULATION_USER, "p", "r1", "r2", 1.0, source="b")]
    train, test = ood_split_by_source(a + b, train_sources=["a"], test_sources=["b"])
    assert train == a
    assert test == b


def test_ood_split_by_length_puts_longer_responses_in_test():
    shorts = [Comparison(POPULATION_USER, "p", "hi", "yo", 1.0) for _ in range(9)]
    long = Comparison(POPULATION_USER, "p", "x" * 500, "y" * 500, 1.0)
    train, test = ood_split_by_length(shorts + [long], percentile=50.0)
    assert all(s in train for s in shorts)
    assert long in test