#!/usr/bin/env python
"""Print the actual column names/types and a sample row of a HF dataset.

Run this before trusting a loader in `rme.data.datasets` -- dataset schemas
drift, and `collective_alignment_1` in particular is built from an
unverified schema (see that loader's docstring).

Usage:
    python scripts/inspect_dataset.py openai/collective-alignment-1
    python scripts/inspect_dataset.py nvidia/HelpSteer2 --split train
    python scripts/inspect_dataset.py openai/summarize_from_feedback --config comparisons
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.repo_id, args.config, split=args.split)
    print(f"repo_id={args.repo_id} config={args.config} split={args.split}")
    print(f"num_rows={len(ds)}")
    print("features:")
    print(ds.features)
    print("\nfirst row:")
    print(json.dumps(ds[0], indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()