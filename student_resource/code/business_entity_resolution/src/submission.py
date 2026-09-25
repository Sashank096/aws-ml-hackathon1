"""Official TSV output generation and structural submission validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from config import EntityResolutionConfig, SchemaConfig, default_config


class SubmissionValidationError(ValueError):
    """Raised when output files violate challenge submission rules."""


def _ids(value: object) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [str(item) for item in str(value).split(",") if str(item)]


def _reference_ids(reference: pd.DataFrame, schema: SchemaConfig) -> list[str]:
    if schema.entity_id_column not in reference.columns:
        raise SubmissionValidationError(f"Reference data is missing '{schema.entity_id_column}'")
    values = reference[schema.entity_id_column].astype("string").fillna("").tolist()
    if any(not str(value) for value in values):
        raise SubmissionValidationError("Reference data contains an empty entity ID")
    if len(values) != len(set(values)):
        raise SubmissionValidationError("Reference data contains duplicate entity IDs")
    return [str(value) for value in values]


def _candidate_universe(candidate_sources: Mapping[str, pd.DataFrame], schema: SchemaConfig) -> set[str]:
    universe: set[str] = set()
    for source, frame in candidate_sources.items():
        if schema.entity_id_column not in frame.columns:
            raise SubmissionValidationError(f"Candidate source '{source}' is missing '{schema.entity_id_column}'")
        ids = [str(value) for value in frame[schema.entity_id_column].astype("string").fillna("")]
        if any(not value for value in ids):
            raise SubmissionValidationError(f"Candidate source '{source}' contains an empty entity ID")
        if len(ids) != len(set(ids)):
            raise SubmissionValidationError(f"Candidate source '{source}' contains duplicate entity IDs")
        universe.update(ids)
    return universe


def build_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    reference: pd.DataFrame,
    config: EntityResolutionConfig | None = None,
) -> pd.DataFrame:
    """Build one candidate-list row for every reference entity."""
    cfg = config or default_config()
    required = {"reference_entity_id", "candidate_entity_id"} - set(candidate_pairs.columns)
    if required:
        raise SubmissionValidationError(f"Candidate pairs are missing columns: {sorted(required)}")
    refs = _reference_ids(reference, cfg.schema)
    grouped: dict[str, set[str]] = {ref: set() for ref in refs}
    for row in candidate_pairs.to_dict(orient="records"):
        ref = str(row["reference_entity_id"])
        if ref in grouped:
            grouped[ref].add(str(row["candidate_entity_id"]))
    return pd.DataFrame({cfg.schema.output_source_id_column: refs, cfg.schema.candidate_matches_column: [",".join(sorted(grouped[ref])) for ref in refs]})


def build_matching_results(
    scored_candidates: pd.DataFrame,
    reference: pd.DataFrame,
    config: EntityResolutionConfig | None = None,
) -> pd.DataFrame:
    """Build one final-match row for every reference entity."""
    cfg = config or default_config()
    required = {"reference_entity_id", "candidate_entity_id", "predicted_match"} - set(scored_candidates.columns)
    if required:
        raise SubmissionValidationError(f"Scored candidates are missing columns: {sorted(required)}")
    refs = _reference_ids(reference, cfg.schema)
    grouped: dict[str, set[str]] = {ref: set() for ref in refs}
    selected = scored_candidates[scored_candidates["predicted_match"].astype(bool)]
    for row in selected.to_dict(orient="records"):
        ref = str(row["reference_entity_id"])
        if ref in grouped:
            grouped[ref].add(str(row["candidate_entity_id"]))
    return pd.DataFrame({cfg.schema.output_source_id_column: refs, cfg.schema.output_matches_column: [",".join(sorted(grouped[ref])) for ref in refs]})


def validate_output_schema(frame: pd.DataFrame, expected_columns: Iterable[str], table_name: str) -> None:
    expected = list(expected_columns)
    if list(frame.columns) != expected:
        raise SubmissionValidationError(f"{table_name} columns must be exactly {expected}; received {list(frame.columns)}")


def validate_reference_coverage(frame: pd.DataFrame, reference: pd.DataFrame, config: EntityResolutionConfig | None = None) -> None:
    cfg = config or default_config()
    expected = _reference_ids(reference, cfg.schema)
    actual = [str(value) for value in frame[cfg.schema.output_source_id_column].astype("string").fillna("")]
    if actual != expected:
        raise SubmissionValidationError("Output must contain exactly one row per reference entity in reference-file order")


def validate_duplicates(frame: pd.DataFrame, id_list_column: str, table_name: str) -> None:
    if frame[id_list_column].isna().any():
        raise SubmissionValidationError(f"{table_name} contains null ID-list values")
    for row_number, value in enumerate(frame[id_list_column], 2):
        ids = _ids(value)
        if len(ids) != len(set(ids)):
            raise SubmissionValidationError(f"{table_name} row {row_number} contains duplicate IDs")


def validate_candidate_ids(frame: pd.DataFrame, valid_candidate_ids: set[str], id_list_column: str, table_name: str) -> None:
    for row_number, value in enumerate(frame[id_list_column], 2):
        invalid = sorted(set(_ids(value)) - valid_candidate_ids)
        if invalid:
            raise SubmissionValidationError(f"{table_name} row {row_number} contains invalid candidate IDs: {invalid[:5]}")


def validate_match_subset(matching: pd.DataFrame, candidates: pd.DataFrame, config: EntityResolutionConfig | None = None) -> None:
    cfg = config or default_config()
    candidate_map = {str(row[cfg.schema.output_source_id_column]): set(_ids(row[cfg.schema.candidate_matches_column])) for _, row in candidates.iterrows()}
    for row_number, row in enumerate(matching.to_dict(orient="records"), 2):
        ref = str(row[cfg.schema.output_source_id_column])
        invalid = set(_ids(row[cfg.schema.output_matches_column])) - candidate_map.get(ref, set())
        if invalid:
            raise SubmissionValidationError(f"matching_results row {row_number} contains IDs absent from candidate_pairs: {sorted(invalid)[:5]}")


def validate_outputs(
    matching: pd.DataFrame,
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    candidate_sources: Mapping[str, pd.DataFrame],
    config: EntityResolutionConfig | None = None,
) -> None:
    """Run every structural check before any output is written."""
    cfg = config or default_config()
    match_columns = [cfg.schema.output_source_id_column, cfg.schema.output_matches_column]
    candidate_columns = [cfg.schema.output_source_id_column, cfg.schema.candidate_matches_column]
    validate_output_schema(matching, match_columns, "matching_results")
    validate_output_schema(candidates, candidate_columns, "candidate_pairs")
    validate_reference_coverage(matching, reference, cfg)
    validate_reference_coverage(candidates, reference, cfg)
    if matching[cfg.schema.output_source_id_column].duplicated().any() or candidates[cfg.schema.output_source_id_column].duplicated().any():
        raise SubmissionValidationError("Outputs contain duplicate reference rows")
    valid_ids = _candidate_universe(candidate_sources, cfg.schema)
    validate_duplicates(matching, cfg.schema.output_matches_column, "matching_results")
    validate_duplicates(candidates, cfg.schema.candidate_matches_column, "candidate_pairs")
    validate_candidate_ids(matching, valid_ids, cfg.schema.output_matches_column, "matching_results")
    validate_candidate_ids(candidates, valid_ids, cfg.schema.candidate_matches_column, "candidate_pairs")
    validate_match_subset(matching, candidates, cfg)


def write_tsv(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write a validated frame as UTF-8 tab-separated text."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    return output


def generate_outputs(
    reference: pd.DataFrame,
    candidate_sources: Mapping[str, pd.DataFrame],
    candidate_pairs: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    config: EntityResolutionConfig | None = None,
    *,
    output_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, validate, and write both official output tables."""
    cfg = config or default_config()
    candidates = build_candidate_pairs(candidate_pairs, reference, cfg)
    matching = build_matching_results(scored_candidates, reference, cfg)
    validate_outputs(matching, candidates, reference, candidate_sources, cfg)
    directory = Path(output_dir) if output_dir is not None else cfg.output.output_dir
    write_tsv(matching, directory / cfg.output.matching_results_file)
    write_tsv(candidates, directory / cfg.output.candidate_pairs_file)
    return matching, candidates


__all__ = ["SubmissionValidationError", "build_candidate_pairs", "build_matching_results", "generate_outputs", "validate_candidate_ids", "validate_duplicates", "validate_match_subset", "validate_output_schema", "validate_outputs", "validate_reference_coverage", "write_tsv"]
