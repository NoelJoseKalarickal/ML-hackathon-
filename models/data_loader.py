"""Data loading, schema validation, leakage checks, and road-network loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from models.config import (
    REQUIRED_CANDIDATE_COLS,
    RESERVED_NON_FEATURE_COLS,
)

DataInput = Union[pd.DataFrame, str, Path]


def read_table(source: DataInput) -> pd.DataFrame:
    """Accept an in-memory DataFrame or load a CSV/Parquet table from disk."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except ImportError as exc:
            raise ImportError(
                f"Cannot read Parquet file {p}: install pyarrow (`pip install pyarrow`) "
                "or pass a CSV / in-memory DataFrame."
            ) from exc
    raise ValueError(f"Unsupported file format '{suffix}' for {p}. Use .csv, .parquet, or DataFrame.")


def normalize_trip_id(series: pd.Series) -> pd.Series:
    """Normalize trip_id values to clean strings (8-digit zero-padded if numeric) so CSV ints match .arcs/.route filenames."""
    def _fmt(val: object) -> str:
        s = str(val).strip()
        if s.isdigit():
            return f"{int(s):08d}"
        return s

    return series.map(_fmt)


def infer_feature_columns(
    df: pd.DataFrame,
    explicit_feature_cols: Optional[list[str]] = None,
) -> list[str]:
    """Infer numeric feature columns while excluding all reserved ID/target columns."""
    if explicit_feature_cols is not None:
        missing = [c for c in explicit_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from DataFrame: {missing}")
        forbidden = [c for c in explicit_feature_cols if c in RESERVED_NON_FEATURE_COLS]
        if forbidden:
            raise ValueError(
                f"Forbidden non-feature columns included in feature_cols: {forbidden}"
            )
        return list(explicit_feature_cols)

    feature_cols: list[str] = []
    for col in df.columns:
        if col in RESERVED_NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found after excluding reserved metadata columns."
        )
    return feature_cols


def sanitize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a clean float32 feature matrix with +/-inf replaced by NaN."""
    X = df[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(np.float32)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X


def prepare_candidate_dataset(
    source: DataInput,
    feature_cols: Optional[list[str]] = None,
    split_name: str = "dataset",
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and standardize a candidate-level DataFrame (from memory or disk)."""
    df = read_table(source)
    missing_req = REQUIRED_CANDIDATE_COLS - set(df.columns)
    if missing_req:
        raise ValueError(
            f"[{split_name}] Missing required columns: {sorted(missing_req)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["trip_id"] = normalize_trip_id(df["trip_id"])
    df["obs_idx"] = pd.to_numeric(df["obs_idx"], errors="raise").astype(np.int64)
    df["arc_id"] = pd.to_numeric(df["arc_id"], errors="raise").astype(np.int64)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(np.int64)

    unique_labels = set(df["label"].unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"[{split_name}] Column 'label' must be binary {{0, 1}}, found: {unique_labels}"
        )

    inferred_cols = infer_feature_columns(df, explicit_feature_cols=feature_cols)
    df[inferred_cols] = sanitize_features(df, inferred_cols)

    # Deterministic ordering for reproducible tie-breaks and evaluation
    df.sort_values(["trip_id", "obs_idx", "arc_id"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, inferred_cols


def prepare_train_and_student_test(
    train_data: DataInput,
    student_test_data: Optional[DataInput] = None,
    explicit_feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], list[str]]:
    """Load/validate train and optional student_test tables with strict zero trip overlap."""
    train_df, feature_cols = prepare_candidate_dataset(
        train_data, feature_cols=explicit_feature_cols, split_name="train"
    )

    if student_test_data is None:
        return train_df, None, feature_cols

    test_df, test_feature_cols = prepare_candidate_dataset(
        student_test_data, feature_cols=feature_cols, split_name="student_test"
    )
    if feature_cols != test_feature_cols:
        raise ValueError(
            f"Feature mismatch between train and student_test: {feature_cols} vs {test_feature_cols}"
        )

    overlap = set(train_df["trip_id"].unique()) & set(test_df["trip_id"].unique())
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} trip_id(s) appear in BOTH train and "
            f"student_test: {sorted(list(overlap))[:10]}"
        )

    return train_df, test_df, feature_cols


def load_obs_meta(source: Optional[DataInput]) -> Optional[pd.DataFrame]:
    """Load optional per-observation metadata for exact Candidate Miss Rate computation."""
    if source is None:
        return None
    df = read_table(source)
    missing = {"trip_id", "obs_idx"} - set(df.columns)
    if missing:
        raise ValueError(f"obs_meta missing required columns: {sorted(missing)}")
    df["trip_id"] = normalize_trip_id(df["trip_id"])
    df["obs_idx"] = pd.to_numeric(df["obs_idx"], errors="raise").astype(np.int64)
    if "candidate_found" in df.columns:
        df["candidate_found"] = pd.to_numeric(df["candidate_found"], errors="raise").astype(np.int64)
    if "true_arc_id" in df.columns:
        df["true_arc_id"] = pd.to_numeric(df["true_arc_id"], errors="coerce").fillna(-1).astype(np.int64)
    return df


def load_road_arcs(
    raw_data_dir: Optional[str | Path] = None,
    arcs_csv: Optional[DataInput] = None,
    arcs_by_trip_dict: Optional[dict[str, dict[int, tuple[int, int]]]] = None,
    trip_ids: Optional[set[str]] = None,
) -> dict[str, dict[int, tuple[int, int]]]:
    """
    Load per-trip road-network arc topology: trip_id -> {arc_id: (from_node_id, to_node_id)}.
    Supports an in-memory dict, a DataFrame/CSV (`trip_id, arc_id, from_node_id, to_node_id`),
    or a raw directory containing `<trip_id>.arcs` files.
    """
    if arcs_by_trip_dict is not None:
        return {str(k): v for k, v in arcs_by_trip_dict.items()}

    arcs_by_trip: dict[str, dict[int, tuple[int, int]]] = {}

    if arcs_csv is not None:
        df = read_table(arcs_csv)
        req = {"trip_id", "arc_id", "from_node_id", "to_node_id"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"arcs_csv missing columns: {sorted(missing)}")
        df["trip_id"] = normalize_trip_id(df["trip_id"])
        for tid, grp in df.groupby("trip_id", sort=False):
            tid_s = str(tid)
            if trip_ids is not None and tid_s not in trip_ids:
                continue
            arcs_by_trip[tid_s] = {
                int(a): (int(u), int(v))
                for a, u, v in zip(grp["arc_id"], grp["from_node_id"], grp["to_node_id"])
            }
        return arcs_by_trip

    if raw_data_dir is not None:
        root = Path(raw_data_dir)
        if not root.exists():
            raise FileNotFoundError(f"raw_data_dir not found: {root}")
        for arcs_file in sorted(root.rglob("*.arcs")):
            raw_keys = [arcs_file.stem, arcs_file.parent.name]
            candidate_keys = [f"{int(k):08d}" if k.isdigit() else k for k in raw_keys]
            matched_key = None
            if trip_ids is not None:
                for k in candidate_keys:
                    if k in trip_ids:
                        matched_key = k
                        break
                if matched_key is None:
                    continue
            else:
                matched_key = candidate_keys[0]
            raw = np.loadtxt(arcs_file, dtype=np.int64, ndmin=2)
            arcs_by_trip[matched_key] = {
                int(idx): (int(row[0]), int(row[1])) for idx, row in enumerate(raw)
            }

    return arcs_by_trip


def load_reference_routes(
    raw_data_dir: Optional[str | Path] = None,
    routes_csv: Optional[DataInput] = None,
    routes_by_trip_dict: Optional[dict[str, list[int]]] = None,
    trip_ids: Optional[set[str]] = None,
) -> dict[str, list[int]]:
    """Load per-trip chronological reference routes: trip_id -> [arc_id, ...]."""
    if routes_by_trip_dict is not None:
        return {str(k): [int(x) for x in v] for k, v in routes_by_trip_dict.items()}

    routes_by_trip: dict[str, list[int]] = {}

    if routes_csv is not None:
        df = read_table(routes_csv)
        if "trip_id" not in df.columns or "arc_id" not in df.columns:
            raise ValueError("routes_csv must contain 'trip_id' and 'arc_id' columns.")
        df["trip_id"] = normalize_trip_id(df["trip_id"])
        if "step_idx" in df.columns:
            df = df.sort_values(["trip_id", "step_idx"], kind="mergesort")
        for tid, grp in df.groupby("trip_id", sort=False):
            tid_s = str(tid)
            if trip_ids is not None and tid_s not in trip_ids:
                continue
            routes_by_trip[tid_s] = grp["arc_id"].astype(int).tolist()
        return routes_by_trip

    if raw_data_dir is not None:
        root = Path(raw_data_dir)
        if not root.exists():
            return routes_by_trip
        for route_file in sorted(root.rglob("*.route")):
            raw_keys = [route_file.stem, route_file.parent.name]
            candidate_keys = [f"{int(k):08d}" if k.isdigit() else k for k in raw_keys]
            matched_key = None
            if trip_ids is not None:
                for k in candidate_keys:
                    if k in trip_ids:
                        matched_key = k
                        break
                if matched_key is None:
                    continue
            else:
                matched_key = candidate_keys[0]
            raw = np.loadtxt(route_file, dtype=np.int64, ndmin=1)
            routes_by_trip[matched_key] = [int(x) for x in raw.tolist()]

    return routes_by_trip


def summarize_dataset(df: pd.DataFrame, feature_cols: list[str], name: str) -> dict:
    """Return a clean diagnostic summary dictionary of a candidate dataset."""
    n_rows = int(len(df))
    n_trips = int(df["trip_id"].nunique())
    n_obs = int(df.groupby(["trip_id", "obs_idx"], sort=False).ngroups)
    pos_count = int((df["label"] == 1).sum())
    neg_count = int((df["label"] == 0).sum())
    missing_per_feat = {c: int(df[c].isna().sum()) for c in feature_cols if df[c].isna().any()}
    return {
        "name": name,
        "num_rows": n_rows,
        "num_trips": n_trips,
        "num_observations_in_candidates": n_obs,
        "avg_candidates_per_obs": round(n_rows / max(n_obs, 1), 3),
        "positive_candidates": pos_count,
        "negative_candidates": neg_count,
        "positive_rate": round(pos_count / max(n_rows, 1), 5),
        "num_features": len(feature_cols),
        "missing_values_by_feature": missing_per_feat,
    }
