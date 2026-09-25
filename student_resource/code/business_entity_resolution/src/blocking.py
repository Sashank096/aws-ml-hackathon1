"""High-recall, multi-stage candidate generation for entity resolution.

The blocker indexes candidate-source records only.  It never creates labels,
uses external data, or compares every reference record with every candidate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from config import BlockingConfig, EntityResolutionConfig, SchemaConfig, default_config
from preprocessing import build_normalized_columns


CandidateKey = tuple[str, str]


@dataclass
class BlockIndexes:
    """Inverted indexes over candidate records, keyed by blocking method."""

    records: dict[CandidateKey, dict]
    postings: dict[str, dict[str, list[CandidateKey]]]
    total_candidate_records: int


@dataclass(frozen=True)
class BlockingResult:
    """Deterministic candidate pairs and their blocking provenance."""

    pairs: pd.DataFrame
    prepared_reference: pd.DataFrame
    prepared_candidates: dict[str, pd.DataFrame]


def _text(row: Mapping[str, object], column: str) -> str:
    value = row.get(column, "")
    return "" if value is None or pd.isna(value) else str(value)


def _tokens(value: object, minimum_length: int) -> set[str]:
    return {token for token in str(value or "").split() if len(token) >= minimum_length}


def _ngrams(value: str, size: int) -> set[str]:
    compact = "".join(value.split())
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[i:i + size] for i in range(len(compact) - size + 1)}


def _soundex(value: str) -> str:
    """Small deterministic phonetic key; used only when explicitly enabled."""
    letters = "".join(ch for ch in value.upper() if ch.isalpha())
    if not letters:
        return ""
    codes = {**dict.fromkeys("BFPV", "1"), **dict.fromkeys("CGJKQSXZ", "2"), **dict.fromkeys("DT", "3"), "L": "4", **dict.fromkeys("MN", "5"), "R": "6"}
    result = letters[0]
    previous = codes.get(letters[0], "")
    for char in letters[1:]:
        code = codes.get(char, "")
        if code and code != previous:
            result += code
        previous = code
    return (result + "000")[:4]


def _key_values(row: Mapping[str, object], schema: SchemaConfig, settings: BlockingConfig) -> dict[str, set[str]]:
    country = _text(row, "country_normalized")
    name = _text(row, "name_normalized")
    address = _text(row, "address_normalized")
    name_tokens = _tokens(row.get("name_tokens", name), settings.minimum_token_length)
    address_tokens = _tokens(row.get("address_tokens", address), settings.minimum_token_length)
    postal = _tokens(row.get("address_postal_tokens", ""), settings.minimum_token_length)
    locality = _tokens(row.get("address_locality_tokens", ""), settings.minimum_token_length)
    result: dict[str, set[str]] = defaultdict(set)
    if settings.use_country_as_blocking_key and country:
        result["country"].add(country)
    if settings.use_exact_normalized_name and name:
        result["normalized_name_exact"].add(name)
    if settings.use_name_tokens:
        result["name_token"].update(name_tokens)
        if name_tokens:
            result["name_prefix"].add(next(iter(sorted(name_tokens)))[: settings.ngram_size])
    if settings.use_address_tokens:
        result["address_token"].update(address_tokens)
    if postal:
        result["postal_token"].update(postal)
    if locality:
        result["locality_token"].update(locality)
    if settings.use_character_ngrams:
        result["name_ngram"].update(_ngrams(name, settings.ngram_size))
        result["address_ngram"].update(_ngrams(address, settings.ngram_size))
    if settings.use_phonetic_keys and name:
        result["name_phonetic"].add(_soundex(name))
    if settings.use_combined_keys and country and name:
        result["country_name"] .add(f"{country}|{name}")
    if settings.use_combined_keys and country and address:
        result["country_address"].add(f"{country}|{address}")
    return result


def build_block_indexes(
    candidate_sources: Mapping[str, pd.DataFrame],
    config: EntityResolutionConfig | None = None,
    *,
    schema: SchemaConfig | None = None,
    settings: BlockingConfig | None = None,
) -> BlockIndexes:
    """Build inverted indexes from candidate-source frames only."""
    cfg = config or default_config()
    schema = schema or cfg.schema
    settings = settings or cfg.blocking
    records: dict[CandidateKey, dict] = {}
    postings: dict[str, dict[str, list[CandidateKey]]] = defaultdict(lambda: defaultdict(list))
    total = 0
    for source, frame in candidate_sources.items():
        prepared = frame if "name_normalized" in frame.columns else build_normalized_columns(frame, schema, cfg.normalization)
        for row in prepared.to_dict(orient="records"):
            entity_id = _text(row, schema.entity_id_column)
            key = (str(source), entity_id)
            if not entity_id:
                continue
            records[key] = row
            total += 1
            for method, values in _key_values(row, schema, settings).items():
                for value in values:
                    postings[method][value].append(key)
    for method in postings:
        for value in postings[method]:
            postings[method][value].sort()
    return BlockIndexes(records=records, postings={m: dict(v) for m, v in postings.items()}, total_candidate_records=total)


def _block(row: Mapping[str, object], indexes: BlockIndexes, schema: SchemaConfig, settings: BlockingConfig, method: str) -> dict[CandidateKey, set[str]]:
    found: dict[CandidateKey, set[str]] = defaultdict(set)
    keys = _key_values(row, schema, settings).get(method, set())
    for key in sorted(keys):
        for candidate in indexes.postings.get(method, {}).get(key, []):
            if settings.use_country_as_blocking_key and method != "country":
                query_country = _text(row, "country_normalized")
                candidate_country = _text(indexes.records.get(candidate, {}), "country_normalized")
                if query_country and candidate_country and query_country != candidate_country:
                    continue
            found[candidate].add(method)
    return found


def block_by_exact_name(row, indexes, schema=None, settings=None):
    return _block(row, indexes, schema or default_config().schema, settings or default_config().blocking, "normalized_name_exact")


def block_by_name_tokens(row, indexes, schema=None, settings=None):
    settings = settings or default_config().blocking
    schema = schema or default_config().schema
    result = _block(row, indexes, schema, settings, "name_token")
    _merge(result, _block(row, indexes, schema, settings, "name_prefix"))
    return result


def block_by_address_tokens(row, indexes, schema=None, settings=None):
    return _block(row, indexes, schema or default_config().schema, settings or default_config().blocking, "address_token")


def block_by_ngrams(row, indexes, schema=None, settings=None):
    settings = settings or default_config().blocking
    schema = schema or default_config().schema
    result = _block(row, indexes, schema, settings, "name_ngram")
    for candidate, methods in _block(row, indexes, schema, settings, "address_ngram").items():
        result[candidate].update(methods)
    return result


def block_by_composite_keys(row, indexes, schema=None, settings=None):
    settings = settings or default_config().blocking
    schema = schema or default_config().schema
    result = _block(row, indexes, schema, settings, "country_name")
    for candidate, methods in _block(row, indexes, schema, settings, "country_address").items():
        result[candidate].update(methods)
    return result


def _merge(found: dict[CandidateKey, set[str]], new: dict[CandidateKey, set[str]]) -> None:
    for candidate, methods in new.items():
        found[candidate].update(methods)


def _rank_candidates(found: Mapping[CandidateKey, set[str]], limit: int) -> list[tuple[CandidateKey, set[str]]]:
    """Keep the best-supported candidates before deterministic ID tie-breaking."""
    ranked = sorted(found.items(), key=lambda item: (-len(item[1]), item[0]))
    return ranked[:limit]


def deduplicate_candidates(pairs: pd.DataFrame) -> pd.DataFrame:
    """Union duplicate candidate pairs and sort deterministically."""
    if pairs.empty:
        return pairs.copy()
    result = pairs.copy()
    result["blocking_methods"] = result["blocking_methods"].map(lambda value: ",".join(sorted(set(str(value).split(",")))))
    columns = ["reference_source", "reference_entity_id", "candidate_source", "candidate_entity_id"]
    result = result.groupby(columns, as_index=False, sort=True)["blocking_methods"].first()
    return result.sort_values(columns, kind="mergesort").reset_index(drop=True)


def generate_candidates(
    reference: pd.DataFrame,
    candidate_sources: Mapping[str, pd.DataFrame],
    config: EntityResolutionConfig | None = None,
    *,
    reference_source: str = "reference",
    schema: SchemaConfig | None = None,
    settings: BlockingConfig | None = None,
    indexes: BlockIndexes | None = None,
) -> BlockingResult:
    """Generate the final candidate set passed to downstream pair scoring."""
    cfg = config or default_config()
    schema = schema or cfg.schema
    settings = settings or cfg.blocking
    prepared_reference = reference if "name_normalized" in reference.columns else build_normalized_columns(reference, schema, cfg.normalization)
    prepared_candidates = {source: frame if "name_normalized" in frame.columns else build_normalized_columns(frame, schema, cfg.normalization) for source, frame in candidate_sources.items()}
    indexes = indexes or build_block_indexes(prepared_candidates, cfg, schema=schema, settings=settings)
    rows: list[dict[str, str]] = []
    # Country is used as a compatibility gate inside each substantive block,
    # rather than as a standalone all-records block that would explode the
    # candidate set for large countries.
    methods = ("normalized_name_exact", "name_token", "address_token", "postal_token", "locality_token", "name_prefix", "name_ngram", "address_ngram", "name_phonetic", "country_name", "country_address")
    for row in prepared_reference.to_dict(orient="records"):
        ref_id = _text(row, schema.entity_id_column)
        found: dict[CandidateKey, set[str]] = defaultdict(set)
        for method in methods:
            if method in settings.enabled_methods or method in {"name_prefix", "locality_token", "postal_token"}:
                _merge(found, _block(row, indexes, schema, settings, method))
        selected = _rank_candidates(found, settings.max_candidates_per_source1)
        for (candidate_source, candidate_id), method_names in selected:
            rows.append({"reference_source": str(reference_source), "reference_entity_id": ref_id, "candidate_source": candidate_source, "candidate_entity_id": candidate_id, "blocking_methods": ",".join(sorted(method_names))})
    columns = ["reference_source", "reference_entity_id", "candidate_source", "candidate_entity_id", "blocking_methods"]
    pairs = deduplicate_candidates(pd.DataFrame(rows, columns=columns))
    return BlockingResult(pairs=pairs, prepared_reference=prepared_reference, prepared_candidates=prepared_candidates)


def evaluate_blocking_recall(
    candidates: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    truth_reference_column: str = "source1_entity_id",
    truth_matches_column: str = "matched_entity_ids",
    candidate_universe_size: int | None = None,
) -> dict[str, float]:
    """Measure union recall, candidate volume, and blocking reduction metrics."""
    if isinstance(ground_truth, pd.DataFrame):
        truth = {str(row[truth_reference_column]): set(filter(None, str(row[truth_matches_column] or "").split(","))) for _, row in ground_truth.iterrows()}
    else:
        truth = {str(key): set(map(str, values)) for key, values in ground_truth.items()}
    actual = defaultdict(set)
    by_method = defaultdict(lambda: defaultdict(set))
    for row in candidates.to_dict(orient="records"):
        actual[str(row["reference_entity_id"])].add(str(row["candidate_entity_id"]))
        for method in str(row.get("blocking_methods", "")).split(","):
            if method:
                by_method[method][str(row["reference_entity_id"])].add(str(row["candidate_entity_id"]))
    denominators = sum(len(values) for values in truth.values())
    hits = sum(len(values & actual.get(ref, set())) for ref, values in truth.items())
    reference_count = max(1, len(truth))
    candidate_count = len(candidates)
    average = candidate_count / reference_count
    # Reduction ratio is reported against the reference x candidate universe
    # when the universe size is available from the candidate table.
    universe = reference_count * (candidate_universe_size if candidate_universe_size is not None else len(set(candidates.get("candidate_entity_id", []))))
    metrics = {
        "candidate_recall": hits / denominators if denominators else 1.0,
        "number_of_candidates": float(candidate_count),
        "average_candidates_per_reference": average,
        "candidate_reduction_ratio": 1.0 - candidate_count / universe if universe else 0.0,
    }
    metrics["union_recall"] = metrics["candidate_recall"]
    for method, method_rows in by_method.items():
        method_hits = sum(len(truth.get(ref, set()) & method_rows.get(ref, set())) for ref in truth)
        metrics[f"recall_by_method.{method}"] = method_hits / denominators if denominators else 1.0
    return metrics


__all__ = [
    "BlockIndexes", "BlockingResult", "block_by_address_tokens", "block_by_composite_keys",
    "block_by_exact_name", "block_by_name_tokens", "block_by_ngrams", "build_block_indexes",
    "deduplicate_candidates", "evaluate_blocking_recall", "generate_candidates",
]
