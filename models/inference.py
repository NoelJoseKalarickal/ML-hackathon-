"""Production inference wrapper for Person 4 (FastAPI backend + HMM/Viterbi decoder).

Guarantees:
  - Preserves exact (trip_id, obs_idx, arc_id) candidate correspondence.
  - Validates and aligns feature columns identically to training.
  - Enforces strict zero-leakage (refuses/strips ground-truth label columns).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from models.config import (
    DEFAULT_ARTIFACTS_DIR,
    FORBIDDEN_INFERENCE_COLS,
    REQUIRED_INFERENCE_COLS,
)
from models.data_loader import normalize_trip_id, sanitize_features
from models.metrics import build_scored_candidates_df


class CandidateScorer:
    """Loads saved model artifacts and scores candidate road segments for live trips."""

    def __init__(
        self,
        artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
        model_filename: str = "final_model.joblib",
        features_filename: str = "feature_columns.json",
        metadata_filename: str = "model_metadata.json",
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        model_path = self.artifacts_dir / model_filename
        feats_path = self.artifacts_dir / features_filename
        meta_path = self.artifacts_dir / metadata_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found at {model_path}")
        if not feats_path.exists():
            raise FileNotFoundError(f"Saved feature_columns.json not found at {feats_path}")

        self.model = joblib.load(model_path)
        self.feature_columns: list[str] = json.loads(feats_path.read_text(encoding="utf-8"))
        self.metadata: dict[str, Any] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )

    def score_candidates(
        self,
        candidates_df: pd.DataFrame,
        strict_no_labels: bool = True,
    ) -> tuple[pd.DataFrame, float]:
        """
        Score candidate road segments for one or more trips.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Must contain `trip_id`, `obs_idx`, `arc_id`, and all `self.feature_columns`.
        strict_no_labels : bool
            If True (default for evaluator inference), raises ValueError if any ground-truth
            column (`label`, `true_arc_id`, etc.) is present in `candidates_df`.

        Returns
        -------
        scored_df : pd.DataFrame
            Columns: `trip_id`, `obs_idx`, `arc_id`, `pred_prob`, `is_top1`.
        inference_time_ms : float
            Wall-clock model inference latency in milliseconds.
        """
        present_forbidden = FORBIDDEN_INFERENCE_COLS & set(candidates_df.columns)
        if present_forbidden:
            if strict_no_labels:
                raise ValueError(
                    f"EVALUATION INTEGRITY ERROR: Ground-truth columns {sorted(present_forbidden)} "
                    "must NOT enter the prediction pipeline!"
                )
            candidates_df = candidates_df.drop(columns=list(present_forbidden))

        missing_req = REQUIRED_INFERENCE_COLS - set(candidates_df.columns)
        if missing_req:
            raise ValueError(f"Missing required inference ID columns: {sorted(missing_req)}")

        missing_feats = [c for c in self.feature_columns if c not in candidates_df.columns]
        if missing_feats:
            raise ValueError(f"Missing required model feature columns: {missing_feats}")

        work = candidates_df[list(REQUIRED_INFERENCE_COLS) + self.feature_columns].copy()
        work["trip_id"] = normalize_trip_id(work["trip_id"])
        work["obs_idx"] = pd.to_numeric(work["obs_idx"], errors="raise").astype(np.int64)
        work["arc_id"] = pd.to_numeric(work["arc_id"], errors="raise").astype(np.int64)
        X = sanitize_features(work, self.feature_columns)

        t0 = time.perf_counter()
        probs = self.model.predict_proba(X)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        scored_df = build_scored_candidates_df(work[["trip_id", "obs_idx", "arc_id"]], probs)
        return scored_df, float(latency_ms)

    def get_top1_route(
        self,
        candidates_df: pd.DataFrame,
        strict_no_labels: bool = True,
    ) -> tuple[pd.DataFrame, float]:
        """Convenience helper returning only the Top-1 predicted arc per (trip_id, obs_idx)."""
        scored_df, latency_ms = self.score_candidates(
            candidates_df, strict_no_labels=strict_no_labels
        )
        top1_df = (
            scored_df[scored_df["is_top1"] == 1]
            .sort_values(["trip_id", "obs_idx"], kind="mergesort")
            .reset_index(drop=True)
        )
        return top1_df, latency_ms
