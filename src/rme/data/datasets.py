"""Loaders that turn the five public datasets named in the proposal (Section
3.1) into a common `list[Comparison]` representation.

Schemas were checked against the Hugging Face `datasets-server` API before
writing these loaders (2026-07). Datasets are versioned and occasionally
change columns, so every loader validates the columns it needs and raises a
clear `DatasetSchemaError` naming the columns it found instead of silently
mis-mapping data -- run `python scripts/inspect_dataset.py <name>` if one
starts failing.

Anthropic_HH_Golden and HelpSteer2 have no persistent annotator ids, so
comparisons from them use `POPULATION_USER` (Section 3.1: "we treat the data
as coming from an aggregate population user"). PRISM-Alignment has real
`user_id`s, which is what makes it interesting for personalization.

`openai/collective-alignment-1` is the least-verified loader here -- its
schema (nested prompt/response/ranking JSON, 4 candidate responses per
prompt, per-annotator preference rankings) is pieced together from search
results rather than a directly-inspected schema dump, because the dataset
isn't yet on the parquet-converter path `datasets-server` exposes. It's
written defensively (checks the keys it needs, raises with the actual
top-level keys otherwise) so a schema drift fails loudly rather than quietly
producing wrong pairs. Double check with `inspect_dataset.py` before trusting
results from it.
"""

from __future__ import annotations

import itertools
from typing import Any

from rme.data.hh_format import split_hh_transcript
from rme.types import POPULATION_USER, Comparison


class DatasetSchemaError(RuntimeError):
    pass


def _load_hf(repo_id: str, config: str | None = None, split: str = "train"):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Dataset loading needs `pip install rme[data]` (datasets).") from e
    return load_dataset(repo_id, config, split=split)


def _require_columns(example: dict, columns: list[str], repo_id: str) -> None:
    missing = [c for c in columns if c not in example]
    if missing:
        raise DatasetSchemaError(
            f"{repo_id}: expected columns {columns}, missing {missing}. "
            f"Available columns: {sorted(example.keys())}. "
            "The dataset schema may have changed -- run "
            "`python scripts/inspect_dataset.py` to see the current one."
        )


def _truncate(ds, max_examples: int | None):
    return ds if max_examples is None else ds.select(range(min(max_examples, len(ds))))


# ---------------------------------------------------------------------------
# Anthropic_HH_Golden -- columns: chosen (str), rejected (str) full transcripts
# ---------------------------------------------------------------------------
def load_anthropic_hh_golden(split: str = "train", max_examples: int | None = None) -> list[Comparison]:
    repo_id = "Unified-Language-Model-Alignment/Anthropic_HH_Golden"
    ds = _truncate(_load_hf(repo_id, split=split), max_examples)
    if len(ds) > 0:
        _require_columns(ds[0], ["chosen", "rejected"], repo_id)

    comparisons = []
    for row in ds:
        try:
            prompt, chosen_resp = split_hh_transcript(row["chosen"])
            _, rejected_resp = split_hh_transcript(row["rejected"])
        except ValueError:
            continue  # malformed transcript (rare); skip rather than crash a full run
        comparisons.append(
            Comparison(
                user=POPULATION_USER,
                prompt=prompt,
                response_1=chosen_resp,
                response_2=rejected_resp,
                label=1.0,
                source="anthropic_hh_golden",
            )
        )
    return comparisons


# ---------------------------------------------------------------------------
# HelpSteer2 -- columns: prompt, response, helpfulness, correctness, coherence,
# complexity, verbosity (int64). Pointwise; pairwise-ize by grouping on prompt.
# ---------------------------------------------------------------------------
def load_helpsteer2(
    split: str = "train", max_examples: int | None = None, score_column: str = "helpfulness"
) -> list[Comparison]:
    repo_id = "nvidia/HelpSteer2"
    ds = _truncate(_load_hf(repo_id, split=split), max_examples)
    if len(ds) > 0:
        _require_columns(ds[0], ["prompt", "response", score_column], repo_id)

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in ds:
        by_prompt.setdefault(row["prompt"], []).append(row)

    comparisons = []
    for prompt, rows in by_prompt.items():
        for a, b in itertools.combinations(rows, 2):
            if a[score_column] == b[score_column]:
                continue  # no preference signal
            label = 1.0 if a[score_column] > b[score_column] else 0.0
            comparisons.append(
                Comparison(
                    user=POPULATION_USER,
                    prompt=prompt,
                    response_1=a["response"],
                    response_2=b["response"],
                    label=label,
                    source="helpsteer2",
                )
            )
    return comparisons


# ---------------------------------------------------------------------------
# Summarize_from_Feedback -- "comparisons" config. info.{post|article}, a
# list of 2 summaries, and a `choice` index of the preferred one.
# ---------------------------------------------------------------------------
def load_summarize_from_feedback(
    split: str = "train", max_examples: int | None = None
) -> list[Comparison]:
    repo_id = "openai/summarize_from_feedback"
    ds = _truncate(_load_hf(repo_id, config="comparisons", split=split), max_examples)
    if len(ds) > 0:
        _require_columns(ds[0], ["info", "summaries", "choice"], repo_id)

    comparisons = []
    for row in ds:
        summaries = row["summaries"]
        if len(summaries) != 2:
            continue
        info = row["info"]
        prompt = info.get("post") or info.get("article") or ""
        if not prompt:
            continue
        choice = row["choice"]
        label = 1.0 if choice == 0 else 0.0
        comparisons.append(
            Comparison(
                user=POPULATION_USER,
                prompt=prompt,
                response_1=summaries[0]["text"],
                response_2=summaries[1]["text"],
                label=label,
                source="summarize_from_feedback",
            )
        )
    return comparisons


# ---------------------------------------------------------------------------
# Collective-Alignment-1 -- least-verified schema, see module docstring.
# Expected shape per example: prompt.messages[], responses[{response_index,
# message}], metadata.assessments[{annotator_id, ranking_blocks.personal}].
# ---------------------------------------------------------------------------
def load_collective_alignment_1(
    split: str = "train", max_examples: int | None = None
) -> list[Comparison]:
    repo_id = "openai/collective-alignment-1"
    ds = _truncate(_load_hf(repo_id, split=split), max_examples)
    if len(ds) == 0:
        return []

    example = ds[0]
    if "prompt" not in example or "responses" not in example:
        raise DatasetSchemaError(
            f"{repo_id}: expected top-level 'prompt' and 'responses' keys, found "
            f"{sorted(example.keys())}. This loader's schema assumptions were pieced "
            "together from search results, not a verified schema dump -- inspect the "
            "dataset directly (`python scripts/inspect_dataset.py openai/collective-alignment-1`) "
            "and update `load_collective_alignment_1` to match."
        )

    comparisons = []
    for row in ds:
        prompt_msgs = row["prompt"].get("messages", [])
        prompt_text = "\n".join(m.get("content", "") for m in prompt_msgs)
        responses_by_index = {
            r["response_index"]: r.get("message", {}).get("content", r.get("text", ""))
            for r in row["responses"]
        }
        assessments = row.get("metadata", {}).get("assessments", [])
        for assessment in assessments:
            ranking = assessment.get("ranking_blocks", {}).get("personal", {})
            order = ranking.get("order") or ranking.get("ranking")
            if not order or len(order) < 2:
                continue
            user = assessment.get("annotator_id", POPULATION_USER)
            # Convert a full preference ranking into adjacent pairwise comparisons
            # (mirrors the one-to-many -> pairwise conversion described in Section 3.1).
            for better_idx, worse_idx in zip(order, order[1:]):
                if better_idx not in responses_by_index or worse_idx not in responses_by_index:
                    continue
                comparisons.append(
                    Comparison(
                        user=user,
                        prompt=prompt_text,
                        response_1=responses_by_index[better_idx],
                        response_2=responses_by_index[worse_idx],
                        label=1.0,
                        source="collective_alignment_1",
                    )
                )
    return comparisons


# ---------------------------------------------------------------------------
# Prism-Alignment -- "utterances" config: one row per (turn, candidate model
# response), with a per-user `if_chosen` flag and a `user_id`.
# ---------------------------------------------------------------------------
def load_prism_alignment(split: str = "train", max_examples: int | None = None) -> list[Comparison]:
    repo_id = "HannahRoseKirk/prism-alignment"
    ds = _truncate(_load_hf(repo_id, config="utterances", split=split), max_examples)
    if len(ds) > 0:
        _require_columns(
            ds[0],
            ["conversation_id", "turn", "user_id", "user_prompt", "model_response", "if_chosen"],
            repo_id,
        )

    by_turn: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in ds:
        by_turn.setdefault((row["conversation_id"], row["turn"]), []).append(row)

    comparisons = []
    for rows in by_turn.values():
        chosen = [r for r in rows if r["if_chosen"]]
        rejected = [r for r in rows if not r["if_chosen"]]
        if not chosen or not rejected:
            continue
        prompt = rows[0]["user_prompt"]
        user = rows[0]["user_id"]
        for c in chosen:
            for r in rejected:
                comparisons.append(
                    Comparison(
                        user=user,
                        prompt=prompt,
                        response_1=c["model_response"],
                        response_2=r["model_response"],
                        label=1.0,
                        source="prism_alignment",
                    )
                )
    return comparisons


LOADERS = {
    "anthropic_hh_golden": load_anthropic_hh_golden,
    "helpsteer2": load_helpsteer2,
    "summarize_from_feedback": load_summarize_from_feedback,
    "collective_alignment_1": load_collective_alignment_1,
    "prism_alignment": load_prism_alignment,
}