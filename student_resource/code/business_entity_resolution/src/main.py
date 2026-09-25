"""Single reproducible orchestration entry point for entity resolution."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from dataclasses import replace

from config import ConfigurationError, EntityResolutionConfig, config_from_mapping, default_config


# Compatibility names retained for older module adapters.  The actual
# implementations live in their respective modules and are not used by this
# orchestrator directly.
def candidates(*args, **kwargs):
    raise RuntimeError("Use blocking.generate_candidates; this compatibility symbol is not an implementation")


def init_db(*args, **kwargs):
    raise RuntimeError("SQLite indexing is no longer orchestrated by main")


def load_source(*args, **kwargs):
    raise RuntimeError("Use data_loader.load_source")


def char_sim(*args, **kwargs):
    raise RuntimeError("Use features feature functions")


def features(*args, **kwargs):
    raise RuntimeError("Use features.transform_features")


def jaccard(*args, **kwargs):
    raise RuntimeError("Use features feature functions")


def postal_tokens(*args, **kwargs):
    raise RuntimeError("Use preprocessing helpers")


def tokens(*args, **kwargs):
    raise RuntimeError("Use preprocessing helpers")


def calibrate_threshold(*args, **kwargs):
    raise RuntimeError("Use validation.optimize_threshold")


def f05(*args, **kwargs):
    raise RuntimeError("Use validation.calculate_f05")


def train_model(*args, **kwargs):
    raise RuntimeError("Use model.train_model")


def _imports():
    from blocking import generate_candidates
    from data_loader import load_test_data, load_training_data
    from features import fit_feature_transformers, transform_features
    from inference import run_inference
    from model import compare_models, predict_proba, save_model, train_model as fit_model
    from preprocessing import preprocess_sources
    from submission import generate_outputs
    from validation import build_training_pairs, create_validation_split, evaluate_predictions, optimize_threshold
    return locals()


def _load_config(path: Path | None, project_root: Path) -> EntityResolutionConfig:
    if path is None:
        return default_config().resolved(project_root)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not load JSON configuration {path}: {exc}") from exc
    return config_from_mapping(values).resolved(project_root)


def _logger(config: EntityResolutionConfig) -> logging.Logger:
    logging.basicConfig(level=getattr(logging, config.runtime.logging_level.upper()), format=config.runtime.logging_format)
    return logging.getLogger("entity_resolution")


def _reference_and_candidates(sources: dict[str, Any], config: EntityResolutionConfig):
    source_names = list(sources)
    if len(source_names) < 2:
        raise ConfigurationError("At least one reference and one candidate source are required")
    reference_source = source_names[0]
    return reference_source, sources[reference_source], {name: sources[name] for name in source_names[1:]}


def _artifact_paths(config: EntityResolutionConfig) -> tuple[Path, Path, Path]:
    model_path = config.model.model_artifact_path
    transformer_path = config.output.report_dir.parent / "models" / "feature_transformers.joblib"
    report_path = config.output.report_dir / "run_summary.json"
    return model_path, transformer_path, report_path


def _run_official_validator(config: EntityResolutionConfig, logger: logging.Logger) -> str:
    """Run the supplied validator when it is present in the project."""
    validator = config.dataset.data_root.parent / "utils" / "validate_submission.py"
    if not validator.exists():
        logger.warning("Official validator not found at %s", validator)
        return "not_available"
    command = [sys.executable, str(validator), "--matching", str(config.output.matching_results_path), "--candidate", str(config.output.candidate_pairs_path), "--test-dir", str(config.dataset.data_root / config.dataset.test_dir)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Official validator failed (exit {completed.returncode}): {completed.stdout}\n{completed.stderr}")
    logger.info("Official validator passed: %s", completed.stdout.strip())
    return "PASS"


def run_all(config: EntityResolutionConfig, logger: logging.Logger) -> dict[str, Any]:
    """Run training, validation, final inference, outputs, and checks."""
    modules = _imports()
    training = modules["load_training_data"](config)
    test = modules["load_test_data"](config)
    logger.info("Loaded training sources=%s and test sources=%s", list(training.sources), list(test.sources))
    reference_source, reference, candidates_train = _reference_and_candidates(training.sources, config)
    _, test_reference, candidates_test = _reference_and_candidates(test.sources, config)
    pre_train = modules["preprocess_sources"](training.sources, config.schema, config.normalization)
    pre_test = modules["preprocess_sources"](test.sources, config.schema, config.normalization)
    reference = pre_train[reference_source]
    candidates_train = {name: frame for name, frame in pre_train.items() if name != reference_source}
    test_reference = pre_test[reference_source]
    candidates_test = {name: frame for name, frame in pre_test.items() if name != reference_source}
    train_block = modules["generate_candidates"](reference, candidates_train, config, reference_source=reference_source)
    logger.info("Training candidate pairs: %d", len(train_block.pairs))
    split = modules["create_validation_split"](reference, training.ground_truth, validation_fraction=config.validation.validation_fraction, random_state=config.validation.random_state)
    train_ids, valid_ids = set(split.train_ids), set(split.validation_ids)
    train_pairs = train_block.pairs[train_block.pairs["reference_entity_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    valid_pairs = train_block.pairs[train_block.pairs["reference_entity_id"].astype(str).isin(valid_ids)].reset_index(drop=True)
    train_labeled = modules["build_training_pairs"](train_pairs, training.ground_truth)
    valid_labeled = modules["build_training_pairs"](valid_pairs, training.ground_truth)
    transformers = modules["fit_feature_transformers"](split.train_reference, candidates_train, config)
    X_train = modules["transform_features"](train_pairs, split.train_reference, candidates_train, transformers, config).matrix
    X_valid = modules["transform_features"](valid_pairs, split.validation_reference, candidates_train, transformers, config).matrix
    y_train, y_valid = train_labeled["target"].to_numpy(), valid_labeled["target"].to_numpy()
    train_features = modules["transform_features"](train_pairs, split.train_reference, candidates_train, transformers, config)
    comparison, trained = modules["compare_models"]([config.model.model_type, "extra_trees"], train_features.matrix, y_train, X_valid, y_valid, config.model, feature_names=train_features.feature_names, threshold=0.5, random_state=config.runtime.random_seed)
    if comparison.empty or comparison["status"].iloc[0] != "ok":
        raise RuntimeError("No configured candidate model trained successfully")
    selected_type = str(comparison.iloc[0]["model_type"])
    selected_model = trained[selected_type]
    valid_scores = valid_pairs.copy()
    valid_scores["score"] = modules["predict_proba"](selected_model, X_valid)
    threshold, threshold_table = modules["optimize_threshold"](valid_scores, training.ground_truth, score_column="score", minimum=config.threshold_search.minimum, maximum=config.threshold_search.maximum, step=config.threshold_search.step, reference_ids=split.validation_ids)
    valid_scores["predicted_match"] = valid_scores["score"] >= threshold
    validation_metrics = modules["evaluate_predictions"](valid_scores, training.ground_truth, reference_ids=split.validation_ids)
    logger.info("Selected model=%s threshold=%.3f validation_f0.5=%.4f", selected_type, threshold, validation_metrics["f0.5"])
    # After model/threshold selection, refit on all labeled training pairs.
    final_config = replace(config.model, model_type=selected_type)
    full_transformers = modules["fit_feature_transformers"](reference, candidates_train, config)
    full_labeled = modules["build_training_pairs"](train_block.pairs, training.ground_truth)
    full_features = modules["transform_features"](train_block.pairs, reference, candidates_train, full_transformers, config)
    final_model = modules["fit_model"](full_features.matrix, full_labeled["target"].to_numpy(), final_config, feature_names=full_features.feature_names, random_state=config.runtime.random_seed)
    model_path, transformer_path, report_path = _artifact_paths(config)
    model_path.parent.mkdir(parents=True, exist_ok=True); transformer_path.parent.mkdir(parents=True, exist_ok=True); report_path.parent.mkdir(parents=True, exist_ok=True)
    modules["save_model"](final_model, model_path)
    joblib.dump(full_transformers, transformer_path)
    inference_artifacts = modules["run_inference"]
    from inference import InferenceArtifacts
    result = inference_artifacts(test_reference, candidates_test, InferenceArtifacts(final_model, full_transformers), config, reference_source=reference_source, threshold=threshold)
    matching, candidate_output = modules["generate_outputs"](test_reference, candidates_test, result.candidate_pairs, result.scored_candidates, config)
    validator_status = _run_official_validator(config, logger)
    summary = {"model": selected_type, "threshold": threshold, "validation": validation_metrics, "training_candidate_count": len(train_block.pairs), "test_candidate_count": len(result.candidate_pairs), "matching_rows": len(matching), "candidate_output_rows": len(candidate_output), "model_artifact": str(model_path), "transformer_artifact": str(transformer_path), "threshold_search_rows": len(threshold_table), "official_validator": validator_status}
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote outputs to %s", config.output.output_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible entity-resolution pipeline")
    parser.add_argument("--mode", choices=("train", "validate", "infer", "all"), default="all")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON configuration")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = _load_config(args.config, args.project_root)
    if args.output_dir is not None:
        from dataclasses import replace
        config = replace(config, output=replace(config.output, output_dir=args.output_dir.resolve()))
    logger = _logger(config)
    if args.mode != "all":
        logger.warning("Mode '%s' uses the complete reproducible flow; partial artifact reuse is not enabled yet", args.mode)
    summary = run_all(config, logger)
    print(json.dumps({"status": "PASS", **summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
