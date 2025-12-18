#!/usr/bin/env python
"""Reproduce Figure 1: pairwise val_accuracy for (train_dataset, test_dataset,
embedder_type) x model_kind, over a grid, with bootstrapped mean/std.

Examples:
    # Fast, network-free smoke test (hashing embedder, small samples):
    python scripts/run_experiments.py --embedders hashing --train-sample 500 --test-sample 200

    # Reproduce the real figure (needs `pip install rme[data,embeddings]` and
    # network access / an OPENAI_API_KEY for the llm embedder):
    python scripts/run_experiments.py \\
        --train-datasets anthropic_hh_golden summarize_from_feedback \\
        --test-datasets anthropic_hh_golden summarize_from_feedback \\
        --embedders sentence_transformer llm \\
        --train-sample 10000 --test-sample 2000 --plot
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rme.data import LOADERS, train_test_split  # noqa: E402
from rme.embeddings import CachingEmbedder, HashingEmbedder  # noqa: E402
from rme.evaluation import bootstrap_pairwise_accuracy  # noqa: E402
from rme.models import MODEL_REGISTRY  # noqa: E402

EMBEDDER_FACTORIES = {
    "hashing": lambda: HashingEmbedder(),
    "sentence_transformer": lambda: __import__(
        "rme.embeddings", fromlist=["SentenceTransformerEmbedder"]
    ).SentenceTransformerEmbedder(),
    "llm": lambda: __import__("rme.embeddings", fromlist=["OpenAIEmbedder"]).OpenAIEmbedder(),
}


def _subsample(comparisons: list, n: int, seed: int) -> list:
    if len(comparisons) <= n:
        return comparisons
    return random.Random(seed).sample(comparisons, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-datasets", nargs="+", default=list(LOADERS.keys()))
    parser.add_argument("--test-datasets", nargs="+", default=list(LOADERS.keys()))
    parser.add_argument("--embedders", nargs="+", default=["hashing"], choices=list(EMBEDDER_FACTORIES))
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--train-sample", type=int, default=2000)
    parser.add_argument("--test-sample", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/experiment_results.csv")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    dataset_cache: dict[str, list] = {}

    def get_dataset(name: str) -> list:
        if name not in dataset_cache:
            print(f"loading {name} ...", file=sys.stderr)
            dataset_cache[name] = LOADERS[name]()
        return dataset_cache[name]

    rows = []
    for train_ds in args.train_datasets:
        for test_ds in args.test_datasets:
            if train_ds == test_ds:
                train_all, test_all = train_test_split(get_dataset(train_ds), seed=args.seed)
            else:
                train_all, test_all = get_dataset(train_ds), get_dataset(test_ds)
            train_c = _subsample(train_all, args.train_sample, args.seed)
            test_c = _subsample(test_all, args.test_sample, args.seed)
            if not train_c or not test_c:
                print(f"skipping ({train_ds}, {test_ds}): empty split", file=sys.stderr)
                continue

            for embedder_name in args.embedders:
                embedder = CachingEmbedder(EMBEDDER_FACTORIES[embedder_name]())
                for model_name in args.models:
                    print(
                        f"tr={train_ds} ts={test_ds} emb={embedder_name} model={model_name} "
                        f"(n_train={len(train_c)}, n_test={len(test_c)})",
                        file=sys.stderr,
                    )
                    model = MODEL_REGISTRY[model_name]().fit(train_c, embedder)
                    result = bootstrap_pairwise_accuracy(model, test_c, n_runs=args.n_runs, seed=args.seed)
                    rows.append(
                        {
                            "train_dataset": train_ds,
                            "test_dataset": test_ds,
                            "embedder_type": embedder_name,
                            "model_kind": model_name,
                            "train_sample": len(train_c),
                            "test_sample": len(test_c),
                            "val_accuracy": result["mean"],
                            "val_accuracy_std": result["std"],
                        }
                    )

    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")

    if args.plot:
        _plot(df, out_path.with_suffix(".png"))
        print(f"wrote plot to {out_path.with_suffix('.png')}")


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    df = df.copy()
    df["config"] = df["train_dataset"] + " -> " + df["test_dataset"] + " (" + df["embedder_type"] + ")"
    pivot = df.pivot_table(index="config", columns="model_kind", values="val_accuracy")
    ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 1.2), 6))
    ax.set_ylabel("val_accuracy")
    ax.set_title("Pairwise accuracy by (train, test, embedder) x model")
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    main()