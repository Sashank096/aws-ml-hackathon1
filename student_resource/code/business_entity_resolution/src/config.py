"""Typed, reusable configuration for an entity-resolution pipeline.

This module is configuration-only: it performs no dataset/model I/O and does
not import heavy machine-learning libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when configuration values are invalid."""


@dataclass(frozen=True)
class DatasetConfig:
    data_root: Path = Path("dataset")
    train_dir: Path = Path("train")
    test_dir: Path = Path("test")
    train_source_files: dict[str, str] = field(default_factory=lambda: {"S1": "train_source1.tsv", "S2": "train_source2.tsv", "S3": "train_source3.tsv"})
    test_source_files: dict[str, str] = field(default_factory=lambda: {"S1": "test_source1.tsv", "S2": "test_source2.tsv", "S3": "test_source3.tsv"})
    ground_truth_file: str = "train_ground_truth.tsv"
    file_format: str = "tsv"
    separator: str = "\t"

    def train_path(self, source: str) -> Path:
        return self.data_root / self.train_dir / self.train_source_files[source]

    def test_path(self, source: str) -> Path:
        return self.data_root / self.test_dir / self.test_source_files[source]

    def ground_truth_path(self) -> Path:
        return self.data_root / self.train_dir / self.ground_truth_file


@dataclass(frozen=True)
class SchemaConfig:
    source_column: str | None = None
    entity_id_column: str = "entity_id"
    name_column: str = "business_name"
    address_column: str = "business_address"
    country_column: str = "country"
    ground_truth_source_id_column: str = "source1_entity_id"
    ground_truth_matches_column: str = "matched_entity_ids"
    output_source_id_column: str = "source1_entity_id"
    output_matches_column: str = "matched_entity_ids"
    candidate_matches_column: str = "candidate_entity_ids"
    source_id_prefixes: dict[str, str] = field(default_factory=lambda: {"S1": "S1-", "S2": "S2-", "S3": "S3-"})


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = True
    unicode_normalization: str = "NFKD"
    transliterate_ascii: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True
    normalize_ampersand: bool = True
    expand_common_abbreviations: bool = True
    remove_legal_suffixes_from_names: bool = True
    preserve_original_values: bool = True
    name_abbreviations: dict[str, str] = field(default_factory=dict)
    address_abbreviations: dict[str, str] = field(default_factory=dict)
    legal_name_suffixes: tuple[str, ...] = ("limited", "ltd", "llc", "inc", "incorporated", "corp", "corporation", "company", "co", "plc", "pvt", "private", "llp")


@dataclass(frozen=True)
class BlockingConfig:
    enabled_methods: tuple[str, ...] = ("country", "normalized_name_exact", "name_token", "address_token", "postal_token", "combined")
    use_country_as_blocking_key: bool = True
    use_exact_normalized_name: bool = True
    use_name_tokens: bool = True
    use_address_tokens: bool = True
    use_character_ngrams: bool = False
    use_phonetic_keys: bool = False
    use_combined_keys: bool = True
    ngram_size: int = 3
    minimum_token_length: int = 2
    max_candidates_per_source1: int = 500
    max_candidates_per_block: int = 500
    max_candidate_batch_size: int = 10_000
    minimum_similarity_for_fallback: float = 0.0
    retain_blocking_method: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    enabled_features: tuple[str, ...] = ("name_exact", "address_exact", "country_exact", "name_token_jaccard", "address_token_jaccard", "name_character_similarity", "address_character_similarity", "postal_overlap", "name_shared_token", "address_shared_token", "name_length_difference", "address_length_difference")
    include_missingness_features: bool = True
    include_blocking_method_features: bool = True
    include_source_features: bool = True
    tfidf_ngram_range: tuple[int, int] = (2, 5)
    max_tfidf_features: int = 100_000
    minimum_similarity_threshold: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "logistic_regression"
    hyperparameters: dict[str, Any] = field(default_factory=lambda: {"class_weight": "balanced", "max_iter": 300})
    probability_output: bool = True
    positive_label: int = 1
    negative_sampling_ratio: float = 5.0
    hard_negative_sampling: bool = True
    model_artifact_path: Path = Path("artifacts/models/entity_matcher.joblib")


@dataclass(frozen=True)
class ValidationConfig:
    metric_name: str = "f0.5"
    beta: float = 0.5
    validation_fraction: float = 0.2
    random_state: int = 42
    stratify_by_match_presence: bool = True
    include_singletons: bool = True
    evaluate_blocking_recall: bool = True
    evaluate_false_merge_rate: bool = True
    experiment_artifact_dir: Path = Path("artifacts/experiments")


@dataclass(frozen=True)
class InferenceConfig:
    probability_threshold: float = 0.86
    allow_zero_matches: bool = True
    allow_multiple_matches: bool = True
    require_independent_support: bool = True
    minimum_independent_signals: int = 1
    score_batch_size: int = 10_000
    preserve_candidate_order: bool = False


@dataclass(frozen=True)
class ThresholdSearchConfig:
    enabled: bool = True
    minimum: float = 0.50
    maximum: float = 0.98
    step: float = 0.02
    metric_name: str = "f0.5"
    prefer_precision_on_tie: bool = True
    artifact_path: Path = Path("artifacts/experiments/threshold_search.json")


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = Path("output")
    matching_results_file: str = "matching_results.tsv"
    candidate_pairs_file: str = "candidate_pairs.tsv"
    report_dir: Path = Path("artifacts/reports")
    methodology_file: Path = Path("Documentation_template.md")
    separator: str = "\t"

    @property
    def matching_results_path(self) -> Path:
        return self.output_dir / self.matching_results_file

    @property
    def candidate_pairs_path(self) -> Path:
        return self.output_dir / self.candidate_pairs_file


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int = 42
    read_chunk_size: int = 100_000
    write_batch_size: int = 10_000
    sqlite_index_path: Path = Path("artifacts/er_index.sqlite")
    logging_level: str = "INFO"
    logging_format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    n_jobs: int = 1
    fail_on_missing_input: bool = True


@dataclass(frozen=True)
class EntityResolutionConfig:
    """Single master configuration object for every pipeline stage."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    threshold_search: ThresholdSearchConfig = field(default_factory=ThresholdSearchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> "EntityResolutionConfig":
        errors: list[str] = []
        if self.dataset.separator != "\t":
            errors.append("dataset.separator must be '\\t' for TSV inputs")
        if len(self.dataset.train_source_files) < 2:
            errors.append("dataset.train_source_files must define a reference and candidate source")
        if set(self.dataset.train_source_files) != set(self.dataset.test_source_files):
            errors.append("train_source_files and test_source_files must define the same sources")
        if not self.schema.entity_id_column.strip():
            errors.append("schema.entity_id_column must be non-empty")
        for name in (self.schema.name_column, self.schema.address_column, self.schema.country_column):
            if not name.strip():
                errors.append("record schema column names must be non-empty")
        if self.blocking.ngram_size < 2:
            errors.append("blocking.ngram_size must be at least 2")
        if self.blocking.minimum_token_length < 1:
            errors.append("blocking.minimum_token_length must be positive")
        if min(self.blocking.max_candidates_per_source1, self.blocking.max_candidates_per_block, self.blocking.max_candidate_batch_size) < 1:
            errors.append("blocking candidate limits must be positive")
        if not 0 <= self.inference.probability_threshold <= 1:
            errors.append("inference.probability_threshold must be between 0 and 1")
        if self.inference.minimum_independent_signals < 0:
            errors.append("inference.minimum_independent_signals cannot be negative")
        if self.validation.beta <= 0:
            errors.append("validation.beta must be positive")
        if not 0 < self.validation.validation_fraction < 1:
            errors.append("validation.validation_fraction must be between 0 and 1")
        if self.threshold_search.enabled:
            if not 0 <= self.threshold_search.minimum <= self.threshold_search.maximum <= 1:
                errors.append("threshold search bounds must satisfy 0 <= minimum <= maximum <= 1")
            if self.threshold_search.step <= 0:
                errors.append("threshold_search.step must be positive")
        if min(self.runtime.read_chunk_size, self.runtime.write_batch_size) < 1:
            errors.append("runtime chunk and batch sizes must be positive")
        if self.runtime.n_jobs == 0:
            errors.append("runtime.n_jobs cannot be zero")
        if self.runtime.logging_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            errors.append("runtime.logging_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        lo, hi = self.features.tfidf_ngram_range
        if lo < 1 or lo > hi:
            errors.append("features.tfidf_ngram_range must be an ordered pair of positive integers")
        if errors:
            raise ConfigurationError("Invalid entity-resolution configuration:\n- " + "\n- ".join(errors))
        return self

    def resolved(self, project_root: Path | str) -> "EntityResolutionConfig":
        """Return a validated copy with relative paths rooted at ``project_root``."""
        root = Path(project_root).expanduser().resolve()
        def rooted(path: Path) -> Path:
            return path if path.is_absolute() else root / path

        # train_dir/test_dir are intentionally relative to DatasetConfig.data_root.
        dataset = replace(self.dataset, data_root=rooted(self.dataset.data_root))
        model = replace(self.model, model_artifact_path=rooted(self.model.model_artifact_path))
        validation = replace(self.validation, experiment_artifact_dir=rooted(self.validation.experiment_artifact_dir))
        threshold = replace(self.threshold_search, artifact_path=rooted(self.threshold_search.artifact_path))
        output = replace(self.output, output_dir=rooted(self.output.output_dir), report_dir=rooted(self.output.report_dir), methodology_file=rooted(self.output.methodology_file))
        runtime = replace(self.runtime, sqlite_index_path=rooted(self.runtime.sqlite_index_path))
        return replace(self, dataset=dataset, model=model, validation=validation, threshold_search=threshold, output=output, runtime=runtime).validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/log-friendly representation with string paths."""
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value):
                return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value
        return convert(self)


def default_config() -> EntityResolutionConfig:
    """Construct validated defaults for the current Amazon ML challenge."""
    return EntityResolutionConfig().validate()


def config_from_mapping(values: Mapping[str, Any]) -> EntityResolutionConfig:
    """Build a validated config from nested section mappings."""
    classes = {"dataset": DatasetConfig, "schema": SchemaConfig, "normalization": NormalizationConfig, "blocking": BlockingConfig, "features": FeatureConfig, "model": ModelConfig, "validation": ValidationConfig, "inference": InferenceConfig, "threshold_search": ThresholdSearchConfig, "output": OutputConfig, "runtime": RuntimeConfig}
    unknown = set(values) - set(classes)
    if unknown:
        raise ConfigurationError(f"Unknown configuration sections: {sorted(unknown)}")
    kwargs = {}
    for name, cls in classes.items():
        section = values.get(name, {})
        if not isinstance(section, Mapping):
            raise ConfigurationError(f"Configuration section '{name}' must be a mapping")
        kwargs[name] = cls(**dict(section))
    return EntityResolutionConfig(**kwargs).validate()


__all__ = ["BlockingConfig", "ConfigurationError", "DatasetConfig", "EntityResolutionConfig", "FeatureConfig", "InferenceConfig", "ModelConfig", "NormalizationConfig", "OutputConfig", "RuntimeConfig", "SchemaConfig", "ThresholdSearchConfig", "ValidationConfig", "config_from_mapping", "default_config"]
