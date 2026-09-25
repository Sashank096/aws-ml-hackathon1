"""Deterministic pairwise features for entity-resolution classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import EntityResolutionConfig, FeatureConfig, SchemaConfig, default_config
from preprocessing import build_normalized_columns


_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)


@dataclass
class FeatureTransformers:
    """Learned text transforms fitted only on caller-provided training data."""

    name_vectorizer: TfidfVectorizer
    address_vectorizer: TfidfVectorizer


@dataclass(frozen=True)
class FeatureMatrix:
    """Numerical features and their stable column names."""

    matrix: sparse.csr_matrix
    feature_names: tuple[str, ...]


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _tokens(value: object) -> set[str]:
    return set(_WORD_RE.findall(_text(value).casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _char_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return 1.0 if left == right else SequenceMatcher(None, left, right, autojunk=False).ratio()


def _edit_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0 if left else 0.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _prefix_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count / max(len(left), len(right))


def _suffix_similarity(left: str, right: str) -> float:
    return _prefix_similarity(left[::-1], right[::-1])


def _safe_length_difference(left: str, right: str) -> float:
    return abs(len(left) - len(right)) / max(1, max(len(left), len(right)))


def _rowwise_cosine(left: sparse.spmatrix, right: sparse.spmatrix) -> np.ndarray:
    values = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    left_norm = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).ravel())
    right_norm = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).ravel())
    denominator = left_norm * right_norm
    return np.divide(values, denominator, out=np.zeros_like(values, dtype=float), where=denominator != 0)


def _prepare(frame: pd.DataFrame, schema: SchemaConfig, config: EntityResolutionConfig) -> pd.DataFrame:
    return frame if {"name_normalized", "address_normalized", "country_normalized"}.issubset(frame.columns) else build_normalized_columns(frame, schema, config.normalization)


def _record_index(reference: pd.DataFrame, candidates: Mapping[str, pd.DataFrame], schema: SchemaConfig, config: EntityResolutionConfig) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    ref = _prepare(reference, schema, config)
    for row in ref.to_dict(orient="records"):
        index[("reference", _text(row.get(schema.entity_id_column)))] = row
    for source, frame in candidates.items():
        for row in _prepare(frame, schema, config).to_dict(orient="records"):
            index[(str(source), _text(row.get(schema.entity_id_column)))] = row
    return index


def fit_feature_transformers(
    reference: pd.DataFrame,
    candidate_sources: Mapping[str, pd.DataFrame],
    config: EntityResolutionConfig | None = None,
) -> FeatureTransformers:
    """Fit TF-IDF vocabularies on supplied training records only."""
    cfg = config or default_config()
    schema = cfg.schema
    frames = [_prepare(reference, schema, cfg), *[_prepare(frame, schema, cfg) for frame in candidate_sources.values()]]
    names = pd.concat([frame["name_normalized"] for frame in frames], ignore_index=True).fillna("").astype(str).drop_duplicates().tolist()
    addresses = pd.concat([frame["address_normalized"] for frame in frames], ignore_index=True).fillna("").astype(str).drop_duplicates().tolist()
    vectorizer_kwargs = {"analyzer": "char", "ngram_range": cfg.features.tfidf_ngram_range, "max_features": cfg.features.max_tfidf_features, "lowercase": False, "dtype": np.float32}
    name_vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    address_vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    name_vectorizer.fit(names or ["empty"])
    address_vectorizer.fit(addresses or ["empty"])
    return FeatureTransformers(name_vectorizer, address_vectorizer)


def get_feature_names(config: EntityResolutionConfig | None = None) -> tuple[str, ...]:
    """Return the stable feature order used by ``transform_features``."""
    cfg = config or default_config()
    names = [
        "name_exact", "name_token_jaccard", "name_character_similarity", "name_edit_similarity", "name_tfidf_cosine", "name_token_overlap", "name_length_difference", "name_token_count_difference", "name_prefix_similarity", "name_suffix_similarity", "name_order_insensitive_similarity",
        "address_exact", "address_token_jaccard", "address_character_similarity", "address_edit_similarity", "address_tfidf_cosine", "address_token_overlap", "address_length_difference", "address_postal_agreement", "address_locality_overlap",
        "country_exact", "country_compatible", "name_address_exact", "name_address_similarity", "reference_source", "candidate_source", "missing_name", "missing_address", "missing_country",
    ]
    enabled = set(cfg.features.enabled_features)
    if not enabled:
        return tuple(names)
    groups = {
        "name": [n for n in names if n.startswith("name_")],
        "address": [n for n in names if n.startswith("address_")],
        "country": [n for n in names if n.startswith("country_")],
        "combined": [n for n in names if n.startswith("name_address")],
        "missingness": [n for n in names if n.startswith("missing_")],
        "source": [n for n in names if n.endswith("_source")],
    }
    selected = []
    for name in names:
        if name in enabled or any(group in enabled and name in members for group, members in groups.items()):
            selected.append(name)
    # The default config uses granular names that do not enumerate every
    # requested similarity; include the complete family when its family is on.
    if any(value.startswith("name_") for value in enabled):
        selected.extend(groups["name"])
    if any(value.startswith("address_") for value in enabled):
        selected.extend(groups["address"])
    selected = list(dict.fromkeys(selected))
    return tuple(selected or names)


def transform_features(
    pairs: pd.DataFrame,
    reference: pd.DataFrame,
    candidate_sources: Mapping[str, pd.DataFrame],
    transformers: FeatureTransformers,
    config: EntityResolutionConfig | None = None,
) -> FeatureMatrix:
    """Transform blocked pairs using already-fitted text transformers."""
    cfg = config or default_config()
    schema = cfg.schema
    index = _record_index(reference, candidate_sources, schema, cfg)
    feature_names = get_feature_names(cfg)
    source_names = sorted({str(value) for value in pairs.get("reference_source", pd.Series(dtype=str)).tolist()} | {str(value) for value in pairs.get("candidate_source", pd.Series(dtype=str)).tolist()} | {str(value) for value in candidate_sources})
    source_codes = {source: float(i) for i, source in enumerate(source_names)}
    rows: list[dict[str, float]] = []
    name_cache: dict[str, set[str]] = {}
    address_cache: dict[str, set[str]] = {}
    for pair in pairs.to_dict(orient="records"):
        ref_key = (str(pair.get("reference_source", "reference")), _text(pair.get("reference_entity_id")))
        cand_key = (str(pair.get("candidate_source", "")), _text(pair.get("candidate_entity_id")))
        left = index.get(ref_key) or index.get(("reference", ref_key[1]), {})
        right = index.get(cand_key, {})
        name_left, name_right = _text(left.get("name_normalized")), _text(right.get("name_normalized"))
        addr_left, addr_right = _text(left.get("address_normalized")), _text(right.get("address_normalized"))
        country_left, country_right = _text(left.get("country_normalized")), _text(right.get("country_normalized"))
        name_cache.setdefault(name_left, _tokens(left.get("name_tokens", name_left)))
        name_cache.setdefault(name_right, _tokens(right.get("name_tokens", name_right)))
        address_cache.setdefault(addr_left, _tokens(left.get("address_tokens", addr_left)))
        address_cache.setdefault(addr_right, _tokens(right.get("address_tokens", addr_right)))
        ntl, ntr = name_cache[name_left], name_cache[name_right]
        atl, atr = address_cache[addr_left], address_cache[addr_right]
        name_sorted_similarity = _jaccard(ntl, ntr)
        address_locality = _jaccard(_tokens(left.get("address_locality_tokens", "")), _tokens(right.get("address_locality_tokens", "")))
        name_tfidf = _rowwise_cosine(transformers.name_vectorizer.transform([name_left]), transformers.name_vectorizer.transform([name_right]))[0]
        address_tfidf = _rowwise_cosine(transformers.address_vectorizer.transform([addr_left]), transformers.address_vectorizer.transform([addr_right]))[0]
        values = {
            "name_exact": float(bool(name_left) and name_left == name_right), "name_token_jaccard": _jaccard(ntl, ntr), "name_character_similarity": _char_similarity(name_left, name_right), "name_edit_similarity": _edit_similarity(name_left, name_right), "name_tfidf_cosine": float(name_tfidf), "name_token_overlap": float(bool(ntl & ntr)), "name_length_difference": _safe_length_difference(name_left, name_right), "name_token_count_difference": float(abs(len(ntl) - len(ntr))), "name_prefix_similarity": _prefix_similarity(name_left, name_right), "name_suffix_similarity": _suffix_similarity(name_left, name_right), "name_order_insensitive_similarity": name_sorted_similarity,
            "address_exact": float(bool(addr_left) and addr_left == addr_right), "address_token_jaccard": _jaccard(atl, atr), "address_character_similarity": _char_similarity(addr_left, addr_right), "address_edit_similarity": _edit_similarity(addr_left, addr_right), "address_tfidf_cosine": float(address_tfidf), "address_token_overlap": float(bool(atl & atr)), "address_length_difference": _safe_length_difference(addr_left, addr_right), "address_postal_agreement": float(bool(_tokens(left.get("address_postal_tokens", "")) & _tokens(right.get("address_postal_tokens", "")))), "address_locality_overlap": address_locality,
            "country_exact": float(bool(country_left) and country_left == country_right), "country_compatible": float(not country_left or not country_right or country_left == country_right), "name_address_exact": float(bool(name_left and addr_left and name_left == name_right and addr_left == addr_right)), "name_address_similarity": (name_sorted_similarity + _jaccard(atl, atr)) / 2.0, "reference_source": source_codes.get(str(pair.get("reference_source", "reference")), -1.0), "candidate_source": source_codes.get(str(pair.get("candidate_source", "")), -1.0), "missing_name": float(not name_left or not name_right), "missing_address": float(not addr_left or not addr_right), "missing_country": float(not country_left or not country_right),
        }
        rows.append({name: float(values.get(name, 0.0)) for name in feature_names})
    dense = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=np.float32) if rows else np.empty((0, len(feature_names)), dtype=np.float32)
    dense[~np.isfinite(dense)] = 0.0
    return FeatureMatrix(sparse.csr_matrix(dense), feature_names)


def build_pair_features(*args, **kwargs) -> FeatureMatrix:
    """Alias for ``transform_features`` used by downstream pipeline code."""
    return transform_features(*args, **kwargs)


def build_name_features(*args, **kwargs) -> FeatureMatrix:
    return transform_features(*args, **kwargs)


def build_address_features(*args, **kwargs) -> FeatureMatrix:
    return transform_features(*args, **kwargs)


def build_country_features(*args, **kwargs) -> FeatureMatrix:
    return transform_features(*args, **kwargs)


def build_combined_features(*args, **kwargs) -> FeatureMatrix:
    return transform_features(*args, **kwargs)


__all__ = ["FeatureMatrix", "FeatureTransformers", "build_address_features", "build_combined_features", "build_name_features", "build_pair_features", "build_country_features", "fit_feature_transformers", "get_feature_names", "transform_features"]
