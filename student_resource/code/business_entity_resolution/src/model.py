"""Generic binary classification models for entity-resolution candidate pairs."""

from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

from config import ModelConfig, default_config


class ModelConfigurationError(ValueError):
    """Raised when a requested model is unsupported or unsafe."""


@dataclass
class TrainedModel:
    """Estimator plus reproducibility and feature-compatibility metadata."""

    estimator: Any
    model_name: str
    feature_names: tuple[str, ...] = ()
    transformer_metadata: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    license_name: str = "BSD-3-Clause"
    license_url: str = "https://scikit-learn.org/stable/about.html#citing-scikit-learn"


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit-learn": _version("scikit-learn"), "numpy": _version("numpy"), "scipy": _version("scipy")}


def _dense(X: Any) -> Any:
    return X.toarray() if hasattr(X, "toarray") else X


def create_model(
    config: ModelConfig | None = None,
    *,
    model_type: str | None = None,
    random_state: int = 42,
    hyperparameters: Mapping[str, Any] | None = None,
) -> Any:
    """Create a configured estimator without fitting it."""
    cfg = config or default_config().model
    name = (model_type or cfg.model_type).casefold().replace("-", "_").replace(" ", "_")
    params = dict(cfg.hyperparameters)
    params.update(hyperparameters or {})
    if name in {"logistic", "logistic_regression"}:
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)
    if name in {"hist_gradient_boosting", "histgradientboosting"}:
        params.setdefault("random_state", random_state)
        return HistGradientBoostingClassifier(**params)
    if name in {"random_forest", "randomforest"}:
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)
    if name in {"extra_trees", "extratrees"}:
        params.setdefault("random_state", random_state)
        return ExtraTreesClassifier(**params)
    if name in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ModelConfigurationError("LightGBM was requested but is not installed; install it explicitly or choose a scikit-learn model") from exc
        params.setdefault("random_state", random_state)
        return LGBMClassifier(**params)
    if name in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ModelConfigurationError("XGBoost was requested but is not installed; install it explicitly or choose a scikit-learn model") from exc
        params.setdefault("random_state", random_state)
        return XGBClassifier(**params)
    raise ModelConfigurationError(f"Unsupported model_type '{model_type or cfg.model_type}'. Supported built-ins: logistic_regression, hist_gradient_boosting, random_forest, extra_trees; optional: lightgbm, xgboost")


def _license_for(estimator: Any) -> tuple[str, str]:
    module = estimator.__class__.__module__
    if module.startswith("sklearn."):
        return "BSD-3-Clause", "https://scikit-learn.org/stable/about.html#citing-scikit-learn"
    if module.startswith("lightgbm"):
        return "MIT", "https://github.com/microsoft/LightGBM/blob/master/LICENSE"
    if module.startswith("xgboost"):
        return "Apache-2.0", "https://github.com/dmlc/xgboost/blob/master/LICENSE"
    raise ModelConfigurationError(f"Cannot verify the license of model class {module}.{estimator.__class__.__name__}")


def train_model(
    X: Any,
    y: Iterable[int],
    config: ModelConfig | None = None,
    *,
    feature_names: Iterable[str] = (),
    transformer_metadata: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> TrainedModel:
    """Fit one estimator using only supplied training features and labels."""
    cfg = config or default_config().model
    estimator = create_model(cfg, random_state=random_state)
    y_array = np.asarray(list(y))
    if y_array.ndim != 1 or len(y_array) == 0:
        raise ValueError("y must be a non-empty one-dimensional binary target")
    if not set(np.unique(y_array)).issubset({0, 1, False, True}):
        raise ValueError("y must contain only binary labels 0 and 1")
    if len(X) != len(y_array):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y_array)}")
    estimator.fit(_dense(X) if isinstance(estimator, HistGradientBoostingClassifier) else X, y_array)
    license_name, license_url = _license_for(estimator)
    return TrainedModel(estimator=estimator, model_name=estimator.__class__.__name__, feature_names=tuple(feature_names), transformer_metadata=dict(transformer_metadata or {}), hyperparameters=dict(estimator.get_params(deep=False)), package_versions=_versions(), license_name=license_name, license_url=license_url)


def predict_proba(model: TrainedModel | Any, X: Any) -> np.ndarray:
    """Return positive-class probabilities for a trained estimator."""
    estimator = model.estimator if isinstance(model, TrainedModel) else model
    values = _dense(X) if isinstance(estimator, HistGradientBoostingClassifier) else X
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(values))
        return probabilities[:, 1] if probabilities.ndim == 2 else probabilities
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(values), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -60, 60)))
    raise TypeError(f"Estimator {type(estimator).__name__} cannot produce probabilities")


def predict(model: TrainedModel | Any, X: Any, *, threshold: float = 0.5) -> np.ndarray:
    """Predict binary labels using a configurable probability threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return (predict_proba(model, X) >= threshold).astype(np.int8)


def _f05(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return float((1.25 * precision * recall) / (0.25 * precision + recall)) if precision + recall else 0.0


def compare_models(
    model_types: Iterable[str],
    X_train: Any,
    y_train: Iterable[int],
    X_valid: Any,
    y_valid: Iterable[int],
    config: ModelConfig | None = None,
    *,
    feature_names: Iterable[str] = (),
    threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, TrainedModel]]:
    """Train candidate families and rank them by validation F0.5."""
    y_valid_array = np.asarray(list(y_valid))
    rows = []
    trained: dict[str, TrainedModel] = {}
    for model_type in model_types:
        local_config = config or default_config().model
        try:
            estimator_config = ModelConfig(model_type=model_type, hyperparameters=dict(local_config.hyperparameters), probability_output=local_config.probability_output, positive_label=local_config.positive_label, negative_sampling_ratio=local_config.negative_sampling_ratio, hard_negative_sampling=local_config.hard_negative_sampling, model_artifact_path=local_config.model_artifact_path)
            fitted = train_model(X_train, y_train, estimator_config, feature_names=feature_names, random_state=random_state)
            probabilities = predict_proba(fitted, X_valid)
            labels = (probabilities >= threshold).astype(np.int8)
            trained[model_type] = fitted
            rows.append({"model_type": model_type, "model_name": fitted.model_name, "threshold": threshold, "precision": precision_score(y_valid_array, labels, zero_division=0), "recall": recall_score(y_valid_array, labels, zero_division=0), "f0.5": _f05(y_valid_array, labels), "license": fitted.license_name, "status": "ok"})
        except (ImportError, ModelConfigurationError, ValueError) as exc:
            rows.append({"model_type": model_type, "model_name": "", "threshold": threshold, "precision": np.nan, "recall": np.nan, "f0.5": np.nan, "license": "unverified", "status": f"unavailable: {exc}"})
    result = pd.DataFrame(rows).sort_values(["f0.5", "precision"], ascending=False, na_position="last", kind="mergesort").reset_index(drop=True)
    return result, trained


def save_model(model: TrainedModel, path: Path | str) -> Path:
    """Persist estimator and all compatibility metadata to a joblib artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output


def load_model(path: Path | str) -> TrainedModel:
    """Load and validate a persisted model artifact."""
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {artifact}")
    loaded = joblib.load(artifact)
    if not isinstance(loaded, TrainedModel):
        raise ModelConfigurationError(f"Artifact {artifact} is not a compatible TrainedModel")
    return loaded


def model_summary(model: TrainedModel | Any) -> dict[str, Any]:
    """Return a serializable model summary for experiment tracking."""
    if not isinstance(model, TrainedModel):
        license_name, license_url = _license_for(model)
        return {"model_name": model.__class__.__name__, "hyperparameters": model.get_params(deep=False), "package_versions": _versions(), "license": license_name, "license_url": license_url}
    return {"model_name": model.model_name, "feature_count": len(model.feature_names), "feature_names": list(model.feature_names), "hyperparameters": model.hyperparameters, "package_versions": model.package_versions, "license": model.license_name, "license_url": model.license_url, "transformer_metadata": model.transformer_metadata}


__all__ = ["ModelConfigurationError", "TrainedModel", "compare_models", "create_model", "load_model", "model_summary", "predict", "predict_proba", "save_model", "train_model"]
