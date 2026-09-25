"""Deterministic, reusable preprocessing for entity-resolution records.

Raw values are never overwritten.  This module only adds normalized and
deterministic helper columns; it does not generate labels, compare records, or
perform blocking.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Iterable

import pandas as pd

from config import NormalizationConfig, SchemaConfig, default_config


_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_POSTAL_RE = re.compile(r"(?<!\d)\d{4,6}(?!\d)")

# Conservative, language-agnostic address forms.  Dataset-specific additions
# should be supplied through NormalizationConfig.address_abbreviations.
_DEFAULT_ADDRESS_ABBREVIATIONS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "highway": "hwy", "drive": "dr", "lane": "ln", "parkway": "pkwy",
    "apartment": "apt", "building": "bldg", "mount": "mt", "saint": "st",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _value(value: object) -> str:
    return "" if _is_missing(value) else str(value)


def _clean_whitespace(value: str) -> str:
    return _SPACE_RE.sub(" ", value.replace("\u00a0", " ").strip())


def _abbreviation_map(config: NormalizationConfig, field: str) -> dict[str, str]:
    if field == "address":
        result = dict(_DEFAULT_ADDRESS_ABBREVIATIONS)
        result.update({str(k).casefold(): str(v).casefold() for k, v in config.address_abbreviations.items()})
        return result
    return {str(k).casefold(): str(v).casefold() for k, v in config.name_abbreviations.items()}


def normalize_text(value: object, config: NormalizationConfig | None = None, *, field: str = "generic") -> str:
    """Normalize one text value without removing numeric information."""
    cfg = config or default_config().normalization
    text = _value(value)
    if not text:
        return ""
    text = unicodedata.normalize(cfg.unicode_normalization, text)
    if cfg.transliterate_ascii:
        text = text.encode("ascii", "ignore").decode("ascii")
    if cfg.normalize_ampersand:
        text = text.replace("&", " and ")
    if cfg.lowercase:
        text = text.casefold()
    # Convert punctuation to separators rather than deleting it so adjacent
    # tokens do not get accidentally concatenated.
    if cfg.strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    text = _clean_whitespace(text)
    if cfg.expand_common_abbreviations and field in {"name", "address"}:
        replacements = _abbreviation_map(cfg, field)
        text = " ".join(replacements.get(token, token) for token in text.split())
    return _clean_whitespace(text) if cfg.collapse_whitespace else text


def normalize_business_name(value: object, config: NormalizationConfig | None = None) -> str:
    """Normalize a business name while conservatively removing legal suffixes."""
    cfg = config or default_config().normalization
    text = normalize_text(value, cfg, field="name")
    if not text or not cfg.remove_legal_suffixes_from_names:
        return text
    suffixes = set(cfg.legal_name_suffixes)
    tokens = [token for token in text.split() if token not in suffixes]
    return " ".join(tokens)


def normalize_business_address(value: object, config: NormalizationConfig | None = None) -> str:
    """Normalize address formatting while preserving numbers and token order."""
    return normalize_text(value, config or default_config().normalization, field="address")


def normalize_country(value: object, config: NormalizationConfig | None = None) -> str:
    """Normalize country labels without mapping them to a fixed country set."""
    return normalize_text(value, config or default_config().normalization, field="country")


def extract_tokens(value: object, *, sort_tokens: bool = False) -> str:
    """Return unique alphanumeric tokens as a space-separated representation."""
    text = _value(value)
    values = list(dict.fromkeys(_TOKEN_RE.findall(text.casefold())))
    if sort_tokens:
        values.sort()
    return " ".join(values)


def extract_numeric_tokens(value: object) -> str:
    """Return numeric-bearing tokens, preserving their original token text."""
    return " ".join(token for token in _TOKEN_RE.findall(_value(value)) if any(char.isdigit() for char in token))


def extract_postal_tokens(value: object) -> str:
    """Extract generic 4--6 digit postal/PIN-like tokens without country rules."""
    return " ".join(dict.fromkeys(_POSTAL_RE.findall(_value(value))))


def extract_locality_tokens(value: object, *, config: NormalizationConfig | None = None) -> str:
    """Extract conservative trailing comma-separated address components.

    This is intentionally structural rather than geographic: the final one or
    two comma-delimited components are retained as locality-like text, with no
    external lookup or country-specific assumptions.
    """
    raw = _value(value)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return ""
    # With only two comma-delimited components, the first is often the full
    # street address; use only the final component in that case.
    tail = parts[-2:] if len(parts) >= 3 else parts[-1:]
    normalized = " ".join(normalize_text(part, config, field="address") for part in tail)
    return extract_tokens(normalized, sort_tokens=True)


def _series_normalize(series: pd.Series, function, config: NormalizationConfig) -> pd.Series:
    return series.map(lambda value: function(value, config)).astype("string")


def build_normalized_columns(
    df: pd.DataFrame,
    schema: SchemaConfig | None = None,
    config: NormalizationConfig | None = None,
) -> pd.DataFrame:
    """Return a copy with normalized and deterministic helper columns added."""
    schema = schema or default_config().schema
    config = config or default_config().normalization
    required = [schema.name_column, schema.address_column, schema.country_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot preprocess data; missing columns: {missing}")
    result = df.copy()
    names = _series_normalize(result[schema.name_column], normalize_business_name, config)
    addresses = _series_normalize(result[schema.address_column], normalize_business_address, config)
    countries = _series_normalize(result[schema.country_column], normalize_country, config)
    result["name_normalized"] = names
    result["address_normalized"] = addresses
    result["country_normalized"] = countries
    result["name_tokens"] = names.map(extract_tokens).astype("string")
    result["name_tokens_sorted"] = names.map(lambda value: extract_tokens(value, sort_tokens=True)).astype("string")
    result["address_tokens"] = addresses.map(extract_tokens).astype("string")
    result["address_tokens_sorted"] = addresses.map(lambda value: extract_tokens(value, sort_tokens=True)).astype("string")
    result["address_numeric_tokens"] = addresses.map(extract_numeric_tokens).astype("string")
    result["address_postal_tokens"] = addresses.map(extract_postal_tokens).astype("string")
    result["address_locality_tokens"] = result[schema.address_column].map(lambda value: extract_locality_tokens(value, config=config)).astype("string")
    result["name_missing"] = names.eq("")
    result["address_missing"] = addresses.eq("")
    result["country_missing"] = countries.eq("")
    return result


def preprocess_sources(
    sources: Mapping[str, pd.DataFrame],
    schema: SchemaConfig | None = None,
    config: NormalizationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply identical preprocessing to each configured source."""
    return {source: build_normalized_columns(frame, schema, config) for source, frame in sources.items()}


def validate_preprocessed_schema(df: pd.DataFrame, *, require_helpers: bool = True) -> pd.DataFrame:
    """Validate normalized columns and return the frame unchanged."""
    required = {"name_normalized", "address_normalized", "country_normalized"}
    if require_helpers:
        required |= {"name_tokens", "address_tokens", "address_postal_tokens"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Preprocessed frame is missing required helper columns: {missing}")
    return df


__all__ = [
    "build_normalized_columns", "extract_locality_tokens", "extract_numeric_tokens",
    "extract_postal_tokens", "extract_tokens", "normalize_business_address",
    "normalize_business_name", "normalize_country", "normalize_text",
    "preprocess_sources", "validate_preprocessed_schema",
]
