"""Configuration-driven, read-only data loading for entity resolution.

The loader preserves raw values and identifiers.  Normalization, blocking,
label construction, and modeling belong to later pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from config import EntityResolutionConfig, SchemaConfig, default_config


class DataLoadingError(ValueError):
    """Raised when an input file cannot be safely loaded or validated."""


@dataclass(frozen=True)
class TrainingData:
    """Training source tables and the untouched ground-truth table."""

    sources: dict[str, pd.DataFrame]
    ground_truth: pd.DataFrame


@dataclass(frozen=True)
class TestData:
    """Test source tables keyed by configured source name."""

    sources: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class EntityResolutionData:
    """Complete loaded dataset bundle."""

    training: TrainingData
    test: TestData


def _dtype_map(schema: SchemaConfig, dtype: Mapping[str, str] | None) -> dict[str, str]:
    """Build safe default dtypes while allowing caller overrides."""
    result = {
        schema.entity_id_column: "string",
        schema.name_column: "string",
        schema.address_column: "string",
        schema.country_column: "string",
    }
    if dtype:
        result.update(dtype)
    return result


def validate_schema(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str = "table",
) -> pd.DataFrame:
    """Validate required columns without renaming or reordering the frame."""
    required = list(required_columns)
    if len(df.columns) != len(set(df.columns)):
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise DataLoadingError(f"{table_name} contains duplicate column names: {duplicates}")
    blank_columns = [str(c) for c in df.columns if not str(c).strip()]
    if blank_columns:
        raise DataLoadingError(f"{table_name} contains blank column names")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataLoadingError(
            f"{table_name} is missing required columns {missing}; "
            f"available columns are {list(df.columns)}"
        )
    if df.shape[1] == 0:
        raise DataLoadingError(f"{table_name} has no columns")
    return df


def load_table(
    path: Path | str,
    *,
    required_columns: Iterable[str] = (),
    table_name: str | None = None,
    dtype: Mapping[str, str] | None = None,
    separator: str = "\t",
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load one UTF-8 TSV without changing the source file.

    Empty strings are retained as empty strings (`keep_default_na=False`).
    Identifier columns are strings by default.  With ``chunksize`` supplied,
    a validated iterator of DataFrame chunks is returned for future large-file
    workflows; the normal API returns one DataFrame.
    """
    file_path = Path(path)
    label = table_name or str(file_path)
    if not file_path.exists():
        raise DataLoadingError(f"Input file for {label} does not exist: {file_path}")
    if not file_path.is_file():
        raise DataLoadingError(f"Input path for {label} is not a file: {file_path}")
    if file_path.stat().st_size == 0:
        raise DataLoadingError(f"Input file for {label} is empty: {file_path}")
    if not separator:
        raise DataLoadingError(f"Separator for {label} must not be empty")
    if chunksize is not None and chunksize < 1:
        raise DataLoadingError("chunksize must be a positive integer")

    read_kwargs = {
        "sep": separator,
        "dtype": dict(dtype or {}),
        "keep_default_na": False,
        "na_filter": False,
        "encoding": "utf-8",
        "on_bad_lines": "error",
        "chunksize": chunksize,
    }
    try:
        loaded = pd.read_csv(file_path, **read_kwargs)
    except EmptyDataError as exc:
        raise DataLoadingError(f"Input file for {label} has no parseable header/data: {file_path}") from exc
    except (ParserError, UnicodeError) as exc:
        raise DataLoadingError(f"Could not parse {label} at {file_path}: {exc}") from exc
    except OSError as exc:
        raise DataLoadingError(f"Could not read {label} at {file_path}: {exc}") from exc

    required = list(required_columns)
    if chunksize is None:
        frame = validate_schema(loaded, required, label)
        if frame.empty:
            raise DataLoadingError(f"Input table {label} contains a header but zero data rows: {file_path}")
        return frame

    def checked_chunks() -> Iterator[pd.DataFrame]:
        saw_rows = False
        for chunk in loaded:
            saw_rows = True
            yield validate_schema(chunk, required, label)
        if not saw_rows:
            raise DataLoadingError(f"Input table {label} contains a header but zero data rows: {file_path}")

    return checked_chunks()


def load_source(
    path: Path | str,
    schema: SchemaConfig,
    *,
    table_name: str | None = None,
    dtype: Mapping[str, str] | None = None,
    separator: str = "\t",
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load one entity source using the configured record schema."""
    required = (
        schema.entity_id_column,
        schema.name_column,
        schema.address_column,
        schema.country_column,
    )
    return load_table(
        path,
        required_columns=required,
        table_name=table_name,
        dtype=_dtype_map(schema, dtype),
        separator=separator,
        chunksize=chunksize,
    )


def load_ground_truth(
    config: EntityResolutionConfig,
    *,
    project_root: Path | str | None = None,
    dtype: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Load and validate training ground truth without deriving labels."""
    active = config.resolved(project_root) if project_root is not None else config
    schema, dataset = active.schema, active.dataset
    path = dataset.ground_truth_path()
    ground_truth_dtype = {
        schema.ground_truth_source_id_column: "string",
        schema.ground_truth_matches_column: "string",
    }
    if dtype:
        ground_truth_dtype.update(dtype)
    return load_table(
        path,
        required_columns=(schema.ground_truth_source_id_column, schema.ground_truth_matches_column),
        table_name="training ground truth",
        dtype=ground_truth_dtype,
        separator=dataset.separator,
    )


def _load_sources(
    paths: Mapping[str, Path],
    config: EntityResolutionConfig,
    split_name: str,
    *,
    dtype: Mapping[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    if not paths:
        raise DataLoadingError(f"No {split_name} source files are configured")
    loaded: dict[str, pd.DataFrame] = {}
    for source_name, path in paths.items():
        loaded[source_name] = load_source(
            path,
            config.schema,
            table_name=f"{split_name} source '{source_name}'",
            dtype=dtype,
            separator=config.dataset.separator,
        )
    return loaded


def load_training_data(
    config: EntityResolutionConfig | None = None,
    *,
    project_root: Path | str | None = None,
    dtype: Mapping[str, str] | None = None,
) -> TrainingData:
    """Load all configured training sources and ground truth."""
    active = (config or default_config()).resolved(project_root) if project_root is not None else (config or default_config()).validate()
    paths = {source: active.dataset.train_path(source) for source in active.dataset.train_source_files}
    return TrainingData(_load_sources(paths, active, "training", dtype=dtype), load_ground_truth(active, dtype=dtype))


def load_test_data(
    config: EntityResolutionConfig | None = None,
    *,
    project_root: Path | str | None = None,
    dtype: Mapping[str, str] | None = None,
) -> TestData:
    """Load all configured test sources."""
    active = (config or default_config()).resolved(project_root) if project_root is not None else (config or default_config()).validate()
    paths = {source: active.dataset.test_path(source) for source in active.dataset.test_source_files}
    return TestData(_load_sources(paths, active, "test", dtype=dtype))


def load_all(
    config: EntityResolutionConfig | None = None,
    *,
    project_root: Path | str | None = None,
    dtype: Mapping[str, str] | None = None,
) -> EntityResolutionData:
    """Load training data and test data as one typed dataset bundle."""
    active = config or default_config()
    return EntityResolutionData(
        training=load_training_data(active, project_root=project_root, dtype=dtype),
        test=load_test_data(active, project_root=project_root, dtype=dtype),
    )


__all__ = [
    "DataLoadingError", "EntityResolutionData", "TestData", "TrainingData",
    "load_all", "load_ground_truth", "load_source", "load_table",
    "load_test_data", "load_training_data", "validate_schema",
]
