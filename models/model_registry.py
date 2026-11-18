"""Model registry across the 5 team-selected algorithm families for PathSnap (Person 3).

Core 5 Algorithm Families:
  1. Logistic Regression   (Linear)
  2. K-Nearest Neighbors   (Instance-based)
  3. Decision Tree         (Tree-based)
  4. Random Forest         (Ensemble)
  5. HistGradientBoosting  (Boosting — genuine iteration train/val log-loss curves)

Includes `DirectedBearingTransformer`, a stateless zero-leakage preprocessing step
inside every Pipeline that derives rotation-invariant directed circular bearing
alignment from `gps_bearing`, `road_bearing`, `speed_mps`, and `perp_dist_m`
(resolving opposite-direction arc ties on two-way roads without altering Person 2's schema).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from models.config import RANDOM_SEED


class DirectedBearingTransformer(BaseEstimator, TransformerMixin):
    """
    Stateless, zero-leakage feature transformer inside each sklearn Pipeline.

    When `gps_bearing` and `road_bearing` are present, derives:
      - `directed_heading_diff_deg`: circular angle in [0, 180] distinguishing forward vs reverse arcs
      - `cos_directed_heading_diff`: cos(directed_heading_diff_deg) in [-1, +1]
      - `speed_x_cos_dir`:           log1p(max(speed_mps, 0)) * cos_directed_heading_diff (if speed_mps present)
      - `inv_perp_dist`:             1 / (1 + max(perp_dist_m, 0)) (if perp_dist_m present)
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[np.ndarray] = None) -> "DirectedBearingTransformer":
        if isinstance(X, pd.DataFrame):
            self.input_cols_ = list(X.columns)
            extra: list[str] = []
            if "gps_bearing" in X.columns and "road_bearing" in X.columns:
                extra.extend(["directed_heading_diff_deg", "cos_directed_heading_diff"])
                if "speed_mps" in X.columns:
                    extra.append("speed_x_cos_dir")
            if "perp_dist_m" in X.columns:
                extra.append("inv_perp_dist")
            self.output_cols_ = self.input_cols_ + extra
        else:
            self.input_cols_ = None
            self.output_cols_ = None
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            return np.asarray(X, dtype=np.float32)

        cols = [X[c].to_numpy(dtype=np.float32, copy=False) for c in X.columns]
        if "gps_bearing" in X.columns and "road_bearing" in X.columns:
            gps_b = X["gps_bearing"].to_numpy(dtype=np.float32, copy=False)
            road_b = X["road_bearing"].to_numpy(dtype=np.float32, copy=False)
            raw_diff = np.abs(gps_b - road_b) % np.float32(360.0)
            dir_diff = np.minimum(raw_diff, np.float32(360.0) - raw_diff).astype(np.float32)
            cos_dir = np.cos(np.deg2rad(dir_diff)).astype(np.float32)
            cols.append(dir_diff)
            cols.append(cos_dir)
            if "speed_mps" in X.columns:
                spd = np.maximum(X["speed_mps"].to_numpy(dtype=np.float32, copy=False), np.float32(0.0))
                cols.append((np.log1p(spd) * cos_dir).astype(np.float32))
        if "perp_dist_m" in X.columns:
            d = np.maximum(X["perp_dist_m"].to_numpy(dtype=np.float32, copy=False), np.float32(0.0))
            cols.append((np.float32(1.0) / (np.float32(1.0) + d)).astype(np.float32))

        return np.column_stack(cols).astype(np.float32, copy=False)

    def get_feature_names_out(self, input_features: Optional[list[str]] = None) -> list[str]:
        if getattr(self, "output_cols_", None) is not None:
            return list(self.output_cols_)
        return list(input_features or [])


class BoundedKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible K-Nearest Neighbors classifier safe for ~2M-row tabular datasets.

    Uses 100% of training rows when n_samples <= max_train_samples, and deterministically
    draws a stratified subsample of size `max_train_samples` otherwise (while ALWAYS
    evaluating on 100% of validation/test rows). Transparently records subsampling metadata.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        weights: str = "distance",
        metric: str = "minkowski",
        p: int = 2,
        leaf_size: int = 40,
        algorithm: str = "kd_tree",
        max_train_samples: Optional[int] = 40000,
        n_jobs: int = -1,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.leaf_size = leaf_size
        self.algorithm = algorithm
        self.max_train_samples = max_train_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoundedKNNClassifier":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = int(X_arr.shape[1])
        self.n_train_total_ = int(len(y_arr))

        if self.max_train_samples is not None and len(y_arr) > int(self.max_train_samples):
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=int(self.max_train_samples),
                random_state=int(self.random_state),
            )
            sub_idx, _ = next(splitter.split(X_arr, y_arr))
            X_fit = X_arr[sub_idx]
            y_fit = y_arr[sub_idx]
            self.subsampled_ = True
        else:
            X_fit = X_arr
            y_fit = y_arr
            self.subsampled_ = False

        self.n_train_fitted_ = int(len(y_fit))
        k_eff = min(int(self.n_neighbors), max(1, self.n_train_fitted_))
        self.knn_ = KNeighborsClassifier(
            n_neighbors=k_eff,
            weights=self.weights,
            metric=self.metric,
            p=int(self.p),
            leaf_size=int(self.leaf_size),
            algorithm=self.algorithm,
            n_jobs=int(self.n_jobs),
        )
        self.knn_.fit(X_fit, y_fit)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "knn_"):
            raise RuntimeError("BoundedKNNClassifier must be fitted before predict_proba.")
        X_arr = np.asarray(X, dtype=np.float32)
        probs = self.knn_.predict_proba(X_arr).astype(np.float64)
        return np.clip(probs, 1e-7, 1.0 - 1e-7)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    algorithm_family: str
    key_configuration: str
    initial_hyperparameters: dict[str, Any]
    loss_function: str
    accepts_groups_in_fit: bool
    builder: Callable[[dict[str, Any]], Pipeline]


def build_logistic_regression(params: dict[str, Any]) -> Pipeline:
    cfg = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": RANDOM_SEED}
    cfg.update(params)
    return Pipeline(
        [
            ("bearing_feats", DirectedBearingTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**cfg)),
        ]
    )


def build_knn(params: dict[str, Any]) -> Pipeline:
    cfg = {
        "n_neighbors": 15,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
        "leaf_size": 40,
        "algorithm": "kd_tree",
        "max_train_samples": 40000,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }
    cfg.update(params)
    return Pipeline(
        [
            ("bearing_feats", DirectedBearingTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", BoundedKNNClassifier(**cfg)),
        ]
    )


def build_decision_tree(params: dict[str, Any]) -> Pipeline:
    cfg = {
        "max_depth": 12,
        "min_samples_leaf": 50,
        "criterion": "gini",
        "random_state": RANDOM_SEED,
    }
    cfg.update(params)
    return Pipeline(
        [
            ("bearing_feats", DirectedBearingTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(**cfg)),
        ]
    )


def build_random_forest(params: dict[str, Any]) -> Pipeline:
    cfg = {
        "n_estimators": 120,
        "max_depth": 16,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "max_samples": 0.4,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }
    cfg.update(params)
    return Pipeline(
        [
            ("bearing_feats", DirectedBearingTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**cfg)),
        ]
    )


def build_hist_gradient_boosting(params: dict[str, Any]) -> Pipeline:
    cfg = {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 15,
        "scoring": "loss",
        "random_state": RANDOM_SEED,
    }
    cfg.update(params)
    return Pipeline(
        [
            ("bearing_feats", DirectedBearingTransformer()),
            ("clf", HistGradientBoostingClassifier(**cfg)),
        ]
    )


def get_model_registry(include_optional: bool = False) -> dict[str, ModelSpec]:
    """Return the exact 5 team-selected model families (plus optional LightGBM if installed)."""
    registry: dict[str, ModelSpec] = {
        "logistic_regression": ModelSpec(
            key="logistic_regression",
            display_name="Logistic Regression",
            algorithm_family="Linear",
            key_configuration="C=1.0, L2, lbfgs, StandardScaler + directed-bearing transform",
            initial_hyperparameters={"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
            loss_function="Binary Cross-Entropy (Log Loss) with L2 regularisation",
            accepts_groups_in_fit=False,
            builder=build_logistic_regression,
        ),
        "k_nearest_neighbors": ModelSpec(
            key="k_nearest_neighbors",
            display_name="K-Nearest Neighbors",
            algorithm_family="Instance-based",
            key_configuration="k=15, distance-weighted, kd_tree, StandardScaler (cap=40k)",
            initial_hyperparameters={
                "n_neighbors": 15,
                "weights": "distance",
                "metric": "minkowski",
                "p": 2,
                "max_train_samples": 40000,
            },
            loss_function="Non-parametric (distance-weighted k-NN probability voting)",
            accepts_groups_in_fit=False,
            builder=build_knn,
        ),
        "decision_tree": ModelSpec(
            key="decision_tree",
            display_name="Decision Tree",
            algorithm_family="Tree-based",
            key_configuration="max_depth=12, min_samples_leaf=50, Gini",
            initial_hyperparameters={"max_depth": 12, "min_samples_leaf": 50, "criterion": "gini"},
            loss_function="Gini Impurity",
            accepts_groups_in_fit=False,
            builder=build_decision_tree,
        ),
        "random_forest": ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            algorithm_family="Ensemble",
            key_configuration="n_estimators=120, max_depth=16, min_samples_leaf=20, max_samples=0.4",
            initial_hyperparameters={
                "n_estimators": 120,
                "max_depth": 16,
                "min_samples_leaf": 20,
                "max_features": "sqrt",
                "max_samples": 0.4,
            },
            loss_function="Gini Impurity (bootstrap-aggregated ensemble)",
            accepts_groups_in_fit=False,
            builder=build_random_forest,
        ),
        "hist_gradient_boosting": ModelSpec(
            key="hist_gradient_boosting",
            display_name="HistGradientBoosting",
            algorithm_family="Boosting",
            key_configuration="max_iter=200, lr=0.05, max_leaf_nodes=31, min_leaf=50, L2=1.0, early_stop",
            initial_hyperparameters={
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 1.0,
            },
            loss_function="Binary Log Loss (Stage-wise Gradient Boosting)",
            accepts_groups_in_fit=False,
            builder=build_hist_gradient_boosting,
        ),
    }

    if include_optional:
        try:
            from lightgbm import LGBMClassifier

            def _build_lgbm(params: dict[str, Any]) -> Pipeline:
                cfg = {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "colsample_bytree": 0.8,
                    "subsample": 0.8,
                    "random_state": RANDOM_SEED,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                cfg.update(params)
                return Pipeline(
                    [
                        ("bearing_feats", DirectedBearingTransformer()),
                        ("clf", LGBMClassifier(**cfg)),
                    ]
                )

            registry["lightgbm"] = ModelSpec(
                key="lightgbm",
                display_name="LightGBM",
                algorithm_family="Boosting",
                key_configuration="n_estimators=300, lr=0.05, num_leaves=31",
                initial_hyperparameters={"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
                loss_function="Binary Log Loss",
                accepts_groups_in_fit=False,
                builder=_build_lgbm,
            )
        except ImportError:
            pass

    return registry


def fit_pipeline(
    pipeline: Pipeline,
    spec: ModelSpec,
    X: Any,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Pipeline:
    """Fit a pipeline, passing trip groups when supported."""
    if spec.accepts_groups_in_fit and groups is not None:
        pipeline.fit(X, y, clf__groups=groups)
    else:
        pipeline.fit(X, y)
    return pipeline


def extract_training_history(pipeline: Pipeline) -> Optional[dict[str, Any]]:
    """Extract ONLY genuine iterative training & validation loss histories."""
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return None

    if isinstance(clf, HistGradientBoostingClassifier):
        tr_scores = getattr(clf, "train_score_", None)
        val_scores = getattr(clf, "validation_score_", None)
        if tr_scores is not None and len(tr_scores) > 0:
            tr_loss = [float(-s) for s in tr_scores]
            val_loss = [float(-s) for s in val_scores] if val_scores is not None else []
            return {
                "unit": "boosting_iteration",
                "metric": "Binary Log Loss",
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "best_iteration": int(np.argmin(val_loss)) if val_loss else len(tr_loss) - 1,
                "iterations_run": int(len(tr_loss)),
            }

    return None


def extract_feature_importance(
    pipeline: Pipeline,
    feature_cols: list[str],
) -> Optional[dict[str, float]]:
    """Extract feature importances (tree-based or linear |coef|) with derived feature names."""
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return None

    bearing_step = pipeline.named_steps.get("bearing_feats")
    names = (
        bearing_step.get_feature_names_out(feature_cols)
        if bearing_step is not None
        else list(feature_cols)
    )

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=np.float64)
        if len(imp) == len(names):
            return {c: float(v) for c, v in zip(names, imp)}
    if hasattr(clf, "coef_"):
        coef = np.abs(np.asarray(clf.coef_, dtype=np.float64).ravel())
        if len(coef) == len(names):
            return {c: float(v) for c, v in zip(names, coef)}
    return None
