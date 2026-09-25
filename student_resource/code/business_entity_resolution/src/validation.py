"""Leakage-safe validation and official-style macro F0.5 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from config import ValidationConfig, default_config


@dataclass(frozen=True)
class ValidationSplit:
    """Reference-entity split; all records for an entity stay in one partition."""

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_reference: pd.DataFrame
    validation_reference: pd.DataFrame


def _truth_mapping(ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame, reference_column: str, matches_column: str) -> dict[str, set[str]]:
    if isinstance(ground_truth, pd.DataFrame):
        required = {reference_column, matches_column} - set(ground_truth.columns)
        if required:
            raise ValueError(f"Ground truth is missing columns: {sorted(required)}")
        return {str(row[reference_column]): set(filter(None, str(row[matches_column] or "").split(","))) for _, row in ground_truth.iterrows()}
    return {str(key): set(map(str, values)) for key, values in ground_truth.items()}


def create_validation_split(
    reference: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    reference_id_column: str = "entity_id",
    truth_reference_column: str = "source1_entity_id",
    truth_matches_column: str = "matched_entity_ids",
    validation_fraction: float = 0.2,
    random_state: int = 42,
    stratify_by_match_presence: bool = True,
) -> ValidationSplit:
    """Split reference entities reproducibly, preventing entity leakage."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if reference_id_column not in reference.columns:
        raise ValueError(f"Reference data is missing ID column '{reference_id_column}'")
    ids = reference[reference_id_column].astype("string").fillna("").tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Reference IDs must be unique before validation splitting")
    truth = _truth_mapping(ground_truth, truth_reference_column, truth_matches_column)
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(ids))
    if stratify_by_match_presence:
        groups = {0: [i for i, eid in enumerate(ids) if not truth.get(str(eid), set())], 1: [i for i, eid in enumerate(ids) if truth.get(str(eid), set())]}
        validation_indices = []
        for group in groups.values():
            group = np.asarray(group, dtype=int)
            rng.shuffle(group)
            validation_indices.extend(group[: max(1, int(round(len(group) * validation_fraction)))] if len(group) > 1 else group.tolist())
        validation_indices = np.asarray(sorted(set(validation_indices)), dtype=int)
    else:
        rng.shuffle(indices)
        validation_indices = np.sort(indices[: max(1, int(round(len(ids) * validation_fraction)))])
    validation_set = set(validation_indices.tolist())
    train_indices = np.asarray([i for i in range(len(ids)) if i not in validation_set], dtype=int)
    train_ids = tuple(str(ids[i]) for i in train_indices)
    validation_ids = tuple(str(ids[i]) for i in validation_indices)
    return ValidationSplit(train_ids, validation_ids, reference.iloc[train_indices].copy(), reference.iloc[validation_indices].copy())


def build_training_pairs(
    candidate_pairs: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    reference_id_column: str = "reference_entity_id",
    candidate_id_column: str = "candidate_entity_id",
    truth_reference_column: str = "source1_entity_id",
    truth_matches_column: str = "matched_entity_ids",
) -> pd.DataFrame:
    """Label blocked pairs from supplied ground truth for model training only."""
    required = {reference_id_column, candidate_id_column} - set(candidate_pairs.columns)
    if required:
        raise ValueError(f"Candidate pairs are missing columns: {sorted(required)}")
    truth = _truth_mapping(ground_truth, truth_reference_column, truth_matches_column)
    result = candidate_pairs.copy()
    result["target"] = [int(str(candidate) in truth.get(str(reference), set())) for reference, candidate in zip(result[reference_id_column], result[candidate_id_column])]
    return result


def build_validation_pairs(*args, **kwargs) -> pd.DataFrame:
    """Build validation labels using the same candidate representation."""
    return build_training_pairs(*args, **kwargs)


def calculate_f05(precision: float, recall: float) -> float:
    """Calculate F0.5, weighting precision twice as strongly as recall."""
    denominator = 0.25 * precision + recall
    return float((1.25 * precision * recall) / denominator) if denominator else 0.0


def evaluate_per_entity(
    predictions: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    reference_id_column: str = "reference_entity_id",
    candidate_id_column: str = "candidate_entity_id",
    prediction_column: str = "predicted_match",
    truth_reference_column: str = "source1_entity_id",
    truth_matches_column: str = "matched_entity_ids",
    reference_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return per-reference diagnostics, including zero-match entities."""
    required = {reference_id_column, candidate_id_column} - set(predictions.columns)
    if required:
        raise ValueError(f"Predictions are missing columns: {sorted(required)}")
    truth = _truth_mapping(ground_truth, truth_reference_column, truth_matches_column)
    refs = list(map(str, reference_ids)) if reference_ids is not None else sorted(set(truth) | set(predictions[reference_id_column].astype(str)))
    grouped = predictions[predictions[prediction_column].astype(bool)] if prediction_column in predictions else predictions.iloc[0:0]
    rows = []
    for reference in refs:
        predicted = set(map(str, grouped.loc[grouped[reference_id_column].astype(str) == reference, candidate_id_column]))
        actual = truth.get(reference, set())
        true_positive = len(predicted & actual)
        false_positive = len(predicted - actual)
        false_negative = len(actual - predicted)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(actual) if actual else (1.0 if not predicted else 0.0)
        rows.append({"reference_entity_id": reference, "predicted_match_count": len(predicted), "true_match_count": len(actual), "true_positive_count": true_positive, "false_positive_count": false_positive, "false_negative_count": false_negative, "precision": precision, "recall": recall, "f0.5": calculate_f05(precision, recall) if actual or predicted else 1.0, "is_singleton": not actual, "singleton_correct": not actual and not predicted})
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    reference_ids: Iterable[str] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Evaluate predictions macro-averaged per reference entity."""
    diagnostics = evaluate_per_entity(predictions, ground_truth, reference_ids=reference_ids, **kwargs)
    if diagnostics.empty:
        return {"precision": 0.0, "recall": 0.0, "f0.5": 0.0, "false_merge_rate": 0.0, "predicted_match_count": 0.0, "true_match_count": 0.0, "candidate_count": float(len(predictions)), "singleton_accuracy": 0.0}
    predicted_total = diagnostics["predicted_match_count"].sum()
    false_positive_total = diagnostics["false_positive_count"].sum()
    singleton = diagnostics[diagnostics["is_singleton"]]
    return {"precision": float(diagnostics["precision"].mean()), "recall": float(diagnostics["recall"].mean()), "f0.5": float(diagnostics["f0.5"].mean()), "false_merge_rate": float(false_positive_total / predicted_total) if predicted_total else 0.0, "predicted_match_count": float(predicted_total), "true_match_count": float(diagnostics["true_match_count"].sum()), "candidate_count": float(len(predictions)), "singleton_accuracy": float(singleton["singleton_correct"].mean()) if not singleton.empty else 1.0}


def calculate_candidate_recall(
    candidate_pairs: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    reference_id_column: str = "reference_entity_id",
    candidate_id_column: str = "candidate_entity_id",
    truth_reference_column: str = "source1_entity_id",
    truth_matches_column: str = "matched_entity_ids",
) -> float:
    """Measure the fraction of true links present in the candidate set."""
    truth = _truth_mapping(ground_truth, truth_reference_column, truth_matches_column)
    found = {str(ref): set(map(str, group[candidate_id_column])) for ref, group in candidate_pairs.groupby(reference_id_column)} if not candidate_pairs.empty else {}
    total = sum(len(values) for values in truth.values())
    hits = sum(len(values & found.get(ref, set())) for ref, values in truth.items())
    return hits / total if total else 1.0


def calculate_false_merge_rate(metrics: Mapping[str, float]) -> float:
    """Return false-positive predictions divided by all predicted matches."""
    return float(metrics.get("false_merge_rate", 0.0))


def evaluate_singletons(diagnostics: pd.DataFrame) -> dict[str, float]:
    """Evaluate entities whose ground-truth match set is empty."""
    if diagnostics.empty or "is_singleton" not in diagnostics:
        return {"singleton_count": 0.0, "singleton_correct": 0.0, "singleton_accuracy": 1.0}
    subset = diagnostics[diagnostics["is_singleton"]]
    return {"singleton_count": float(len(subset)), "singleton_correct": float(subset["singleton_correct"].sum()), "singleton_accuracy": float(subset["singleton_correct"].mean()) if len(subset) else 1.0}


def optimize_threshold(
    scored_candidates: pd.DataFrame,
    ground_truth: Mapping[str, Iterable[str]] | pd.DataFrame,
    *,
    score_column: str = "score",
    minimum: float = 0.30,
    maximum: float = 0.90,
    step: float = 0.01,
    minimum_precision: float | None = None,
    maximum_false_merge_rate: float | None = None,
    reference_ids: Iterable[str] | None = None,
    **kwargs,
) -> tuple[float, pd.DataFrame]:
    """Search thresholds and select the highest-F0.5 safe operating point."""
    if not 0 <= minimum <= maximum <= 1 or step <= 0:
        raise ValueError("threshold range must satisfy 0 <= minimum <= maximum <= 1 and step > 0")
    if score_column not in scored_candidates.columns:
        raise ValueError(f"Scored candidates are missing '{score_column}'")
    rows = []
    threshold = minimum
    while threshold <= maximum + 1e-12:
        current = scored_candidates.copy()
        current["predicted_match"] = current[score_column].astype(float) >= threshold
        metrics = evaluate_predictions(current, ground_truth, reference_ids=reference_ids, **kwargs)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
        threshold += step
    table = pd.DataFrame(rows)
    safe = table
    if minimum_precision is not None:
        safe = safe[safe["precision"] >= minimum_precision]
    if maximum_false_merge_rate is not None:
        safe = safe[safe["false_merge_rate"] <= maximum_false_merge_rate]
    if safe.empty:
        safe = table
    best = safe.sort_values(["f0.5", "precision", "threshold"], ascending=[False, False, False], kind="mergesort").iloc[0]
    return float(best["threshold"]), table


def summarize_experiment(metrics: Mapping[str, float], *, experiment_name: str = "entity_resolution") -> dict[str, float | str]:
    """Return a compact serializable experiment record."""
    return {"experiment_name": experiment_name, **{str(key): float(value) if isinstance(value, (int, float, np.number)) else value for key, value in metrics.items()}}


__all__ = ["ValidationSplit", "build_training_pairs", "build_validation_pairs", "calculate_candidate_recall", "calculate_f05", "calculate_false_merge_rate", "create_validation_split", "evaluate_per_entity", "evaluate_predictions", "evaluate_singletons", "optimize_threshold", "summarize_experiment"]
