"""End-to-end inference for unseen entity-resolution records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd

from blocking import BlockingResult, generate_candidates
from config import EntityResolutionConfig, default_config
from features import FeatureTransformers, transform_features
from model import TrainedModel, load_model, predict_proba
from preprocessing import build_normalized_columns


@dataclass(frozen=True)
class InferenceArtifacts:
    """Fitted artifacts required for test-time inference."""

    model: TrainedModel
    feature_transformers: FeatureTransformers


@dataclass(frozen=True)
class InferenceResult:
    """Complete inference output, including all candidates and decisions."""

    candidate_pairs: pd.DataFrame
    scored_candidates: pd.DataFrame
    matches: pd.DataFrame
    blocking_result: BlockingResult


def load_artifacts(
    model_path: Path | str,
    transformer_path: Path | str,
) -> InferenceArtifacts:
    """Load the persisted model and fitted feature transformers."""
    model = load_model(model_path)
    path = Path(transformer_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature-transformer artifact does not exist: {path}")
    transformers = joblib.load(path)
    if not isinstance(transformers, FeatureTransformers):
        raise TypeError(f"Transformer artifact {path} is not a FeatureTransformers object")
    return InferenceArtifacts(model=model, feature_transformers=transformers)


def prepare_inference_data(
    reference: pd.DataFrame,
    candidate_sources: Mapping[str, pd.DataFrame],
    config: EntityResolutionConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply the same deterministic preprocessing to reference and candidates."""
    cfg = config or default_config()
    prepared_reference = build_normalized_columns(reference, cfg.schema, cfg.normalization)
    prepared_candidates = {source: build_normalized_columns(frame, cfg.schema, cfg.normalization) for source, frame in candidate_sources.items()}
    return prepared_reference, prepared_candidates


def score_candidates(
    blocking_result: BlockingResult,
    artifacts: InferenceArtifacts,
    config: EntityResolutionConfig | None = None,
) -> pd.DataFrame:
    """Score exactly the final candidate set produced by blocking."""
    cfg = config or default_config()
    pairs = blocking_result.pairs.copy()
    if pairs.empty:
        pairs["score"] = pd.Series(dtype="float64")
        return pairs
    features = transform_features(
        pairs,
        blocking_result.prepared_reference,
        blocking_result.prepared_candidates,
        artifacts.feature_transformers,
        cfg,
    )
    result = pairs.copy()
    result["score"] = predict_proba(artifacts.model, features.matrix)
    return result


def apply_match_decision(
    scored_candidates: pd.DataFrame,
    config: EntityResolutionConfig | None = None,
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Apply the validated threshold and conservative support rules."""
    cfg = config or default_config()
    selected_threshold = cfg.inference.probability_threshold if threshold is None else threshold
    if not 0 <= selected_threshold <= 1:
        raise ValueError("Inference threshold must be between 0 and 1")
    result = scored_candidates.copy()
    result["predicted_match"] = result["score"].astype(float) >= selected_threshold
    if cfg.inference.require_independent_support and "blocking_methods" in result.columns:
        method_counts = result["blocking_methods"].fillna("").map(lambda value: len({part for part in str(value).split(",") if part}))
        result.loc[method_counts < cfg.inference.minimum_independent_signals, "predicted_match"] = False
    return result


def group_matches_by_reference(
    scored_candidates: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    reference_id_column: str = "entity_id",
    output_reference_column: str = "source1_entity_id",
    output_matches_column: str = "matched_entity_ids",
) -> pd.DataFrame:
    """Return exactly one deterministic row for every reference entity."""
    required = {"reference_entity_id", "candidate_entity_id", "predicted_match"} - set(scored_candidates.columns)
    if required:
        raise ValueError(f"Scored candidates are missing columns: {sorted(required)}")
    reference_ids = reference[reference_id_column].astype("string").fillna("").tolist()
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference entity IDs must be unique for inference")
    selected = scored_candidates[scored_candidates["predicted_match"].astype(bool)].copy()
    grouped: dict[str, list[str]] = {}
    for ref_id, group in selected.groupby("reference_entity_id", sort=False):
        ids = sorted({str(value) for value in group["candidate_entity_id"].tolist() if str(value)})
        grouped[str(ref_id)] = ids
    return pd.DataFrame({output_reference_column: [str(value) for value in reference_ids], output_matches_column: [",".join(grouped.get(str(value), [])) for value in reference_ids]})


def run_inference(
    reference: pd.DataFrame,
    candidate_sources: Mapping[str, pd.DataFrame],
    artifacts: InferenceArtifacts,
    config: EntityResolutionConfig | None = None,
    *,
    reference_source: str = "reference",
    threshold: float | None = None,
) -> InferenceResult:
    """Run preprocessing, blocking, scoring, decision logic, and grouping."""
    cfg = config or default_config()
    prepared_reference, prepared_candidates = prepare_inference_data(reference, candidate_sources, cfg)
    blocking_result = generate_candidates(prepared_reference, prepared_candidates, cfg, reference_source=reference_source)
    scored = score_candidates(blocking_result, artifacts, cfg)
    decided = apply_match_decision(scored, cfg, threshold=threshold)
    matches = group_matches_by_reference(decided, prepared_reference, reference_id_column=cfg.schema.entity_id_column, output_reference_column=cfg.schema.output_source_id_column, output_matches_column=cfg.schema.output_matches_column)
    return InferenceResult(candidate_pairs=blocking_result.pairs, scored_candidates=decided, matches=matches, blocking_result=blocking_result)


__all__ = ["InferenceArtifacts", "InferenceResult", "apply_match_decision", "group_matches_by_reference", "load_artifacts", "prepare_inference_data", "run_inference", "score_candidates"]
