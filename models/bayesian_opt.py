"""Bayesian Hyperparameter Optimisation for PathSnap (Person 3).

Implements true Bayesian Optimisation using:
  1. Gaussian Process Surrogate (Matérn 5/2 + WhiteKernel) + Expected Improvement (EI)
     acquisition function (`backend="gp_ei"`, zero extra dependencies, works offline).
  2. Optuna TPE Sampler (`backend="optuna"`, if optuna is installed).

Strictly uses trip-grouped StratifiedGroupKFold on the training partition ONLY.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import StratifiedGroupKFold

from models.config import N_CV_SPLITS, RANDOM_SEED
from models.data_loader import sanitize_features
from models.metrics import build_scored_candidates_df, evaluate_predictions
from models.model_registry import (
    ModelSpec,
    extract_training_history,
    fit_pipeline,
    get_model_registry,
)


@dataclass(frozen=True)
class ParamDim:
    name: str
    kind: str  # "int", "float", "log_float", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[tuple[Any, ...]] = None

    def decode(self, u: float) -> Any:
        """Map u in [0, 1] to concrete hyperparameter value."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            return int(round(self.low + u * (self.high - self.low)))
        if self.kind == "float":
            assert self.low is not None and self.high is not None
            return float(self.low + u * (self.high - self.low))
        if self.kind == "log_float":
            assert self.low is not None and self.high is not None
            log_val = np.log10(self.low) + u * (np.log10(self.high) - np.log10(self.low))
            return float(10.0**log_val)
        if self.kind == "categorical":
            assert self.choices is not None
            idx = min(int(u * len(self.choices)), len(self.choices) - 1)
            return self.choices[idx]
        raise ValueError(f"Unknown ParamDim kind: {self.kind}")

    def describe(self) -> dict[str, Any]:
        if self.kind == "categorical":
            return {"name": self.name, "distribution": "categorical", "choices": list(self.choices or [])}
        return {
            "name": self.name,
            "distribution": self.kind,
            "range": [self.low, self.high],
        }


SEARCH_SPACES: dict[str, list[ParamDim]] = {
    "hist_gradient_boosting": [
        ParamDim("learning_rate", "log_float", 0.01, 0.25),
        ParamDim("max_leaf_nodes", "int", 15, 127),
        ParamDim("min_samples_leaf", "int", 10, 100),
        ParamDim("l2_regularization", "log_float", 1e-4, 20.0),
        ParamDim("max_iter", "int", 120, 400),
        ParamDim("max_bins", "int", 64, 255),
    ],
    "random_forest": [
        ParamDim("n_estimators", "int", 100, 400),
        ParamDim("max_depth", "int", 8, 32),
        ParamDim("min_samples_leaf", "int", 1, 25),
        ParamDim("min_samples_split", "int", 2, 20),
        ParamDim("max_features", "categorical", choices=("sqrt", "log2", 0.5, 0.75)),
    ],
    "k_nearest_neighbors": [
        ParamDim("n_neighbors", "int", 3, 45),
        ParamDim("weights", "categorical", choices=("uniform", "distance")),
        ParamDim("p", "categorical", choices=(1, 2)),
    ],
    "decision_tree": [
        ParamDim("max_depth", "int", 4, 28),
        ParamDim("min_samples_leaf", "int", 2, 80),
        ParamDim("min_samples_split", "int", 2, 40),
        ParamDim("criterion", "categorical", choices=("gini", "entropy")),
    ],
    "logistic_regression": [
        ParamDim("C", "log_float", 1e-3, 100.0),
    ],
}


def _expected_improvement(
    X_cand: np.ndarray,
    gp: GaussianProcessRegressor,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute Expected Improvement (EI) for maximization."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - best_y - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 1e-11] = 0.0
    return ei


def _evaluate_cv_params(
    spec: ModelSpec,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Evaluate a parameter configuration via fixed trip-grouped CV splits."""
    X_all = train_df[feature_cols]
    y_all = train_df["label"].to_numpy(dtype=np.int64)
    groups_all = train_df["trip_id"].to_numpy()

    fold_f1s: list[float] = []
    fold_top1s: list[float] = []
    fold_losses: list[float] = []

    for tr_idx, val_idx in splits:
        pipe = spec.builder(params)
        fit_pipeline(
            pipe,
            spec,
            X_all.iloc[tr_idx],
            y_all[tr_idx],
            groups=groups_all[tr_idx],
        )
        val_probs = pipe.predict_proba(X_all.iloc[val_idx])[:, 1]
        val_df = train_df.iloc[val_idx]
        ev = evaluate_predictions(val_df, val_probs)
        fold_f1s.append(ev["candidate"]["macro_f1"])
        fold_top1s.append(ev["observation"]["top1_accuracy"])
        fold_losses.append(ev["candidate"]["log_loss"])

    return {
        "cv_macro_f1": float(np.mean(fold_f1s)),
        "cv_macro_f1_std": float(np.std(fold_f1s)),
        "cv_top1_accuracy": float(np.mean(fold_top1s)),
        "cv_log_loss": float(np.mean(fold_losses)),
    }


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert tuples/numpy scalars into JSON-safe Python types."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def run_bayesian_optimization(
    model_key: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    student_test_df: Optional[pd.DataFrame] = None,
    student_obs_meta: Optional[pd.DataFrame] = None,
    arcs_by_trip: Optional[dict[str, dict[int, tuple[int, int]]]] = None,
    routes_by_trip: Optional[dict[str, list[int]]] = None,
    n_trials: int = 20,
    n_init: int = 5,
    n_cv_splits: int = N_CV_SPLITS,
    objective: str = "cv_macro_f1",
    backend: str = "auto",
    random_state: int = RANDOM_SEED,
    save_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Run Bayesian Hyperparameter Optimisation on `model_key` using trip-grouped CV on train_df.

    Returns a dictionary of Python objects:
      - `tuned_model`: fitted final sklearn Pipeline on all train_df
      - `untuned_model`: fitted untuned baseline Pipeline on all train_df
      - `best_params`: dict of best hyperparameters
      - `trials_df`: DataFrame of every trial (params, CV metrics, duration)
      - `untuned_vs_tuned_df`: DataFrame comparing untuned vs tuned (Report Sec 4)
      - `summary`: complete JSON-serializable summary dict
      - `student_test_predictions_df`: DataFrame of tuned candidate probabilities (if student_test_df given)
    """
    registry = get_model_registry(include_optional=True)
    if model_key not in registry:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from {list(registry.keys())}")
    if model_key not in SEARCH_SPACES:
        raise ValueError(f"No Bayesian search space defined for '{model_key}'.")

    spec = registry[model_key]
    space = SEARCH_SPACES[model_key]
    dim = len(space)

    # Defensively sanitize features in case run_bayesian_optimization is called directly
    train_df = train_df.copy()
    train_df[feature_cols] = sanitize_features(train_df, feature_cols)
    if student_test_df is not None:
        student_test_df = student_test_df.copy()
        student_test_df[feature_cols] = sanitize_features(student_test_df, feature_cols)

    n_unique_trips = int(train_df["trip_id"].nunique())
    actual_splits = min(int(n_cv_splits), max(2, n_unique_trips))
    sgkf = StratifiedGroupKFold(n_splits=actual_splits)
    splits = list(
        sgkf.split(
            train_df[feature_cols],
            train_df["label"].to_numpy(dtype=np.int64),
            groups=train_df["trip_id"].to_numpy(),
        )
    )

    if objective not in {"cv_macro_f1", "cv_top1_accuracy", "neg_cv_log_loss"}:
        raise ValueError(
            "objective must be one of 'cv_macro_f1', 'cv_top1_accuracy', 'neg_cv_log_loss'"
        )

    def _extract_obj(cv_res: dict[str, float]) -> float:
        if objective == "neg_cv_log_loss":
            return -cv_res["cv_log_loss"]
        return cv_res[objective]

    # Resolve backend ("optuna" if requested/available, else offline GP-EI)
    chosen_backend = backend
    if backend == "auto":
        try:
            import optuna  # noqa: F401
            chosen_backend = "optuna"
        except ImportError:
            chosen_backend = "gp_ei"

    # Step 1: Evaluate untuned baseline on the exact same CV splits
    t0_untuned = time.perf_counter()
    untuned_cv = _evaluate_cv_params(spec, {}, train_df, feature_cols, splits)
    untuned_model = spec.builder({})
    t0_fit_u = time.perf_counter()
    fit_pipeline(
        untuned_model,
        spec,
        train_df[feature_cols],
        train_df["label"].to_numpy(dtype=np.int64),
        groups=train_df["trip_id"].to_numpy(),
    )
    untuned_fit_sec = time.perf_counter() - t0_fit_u

    untuned_test_eval = None
    untuned_inf_sec = None
    if student_test_df is not None:
        t0_inf = time.perf_counter()
        u_probs = untuned_model.predict_proba(student_test_df[feature_cols])[:, 1]
        untuned_inf_sec = time.perf_counter() - t0_inf
        untuned_test_eval = evaluate_predictions(
            student_test_df,
            u_probs,
            obs_meta=student_obs_meta,
            arcs_by_trip=arcs_by_trip,
            routes_by_trip=routes_by_trip,
        )

    # Step 2: Run Bayesian Optimisation trials
    t0_bo = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []
    best_obj = -float("inf")
    best_params: dict[str, Any] = {}
    best_cv_metrics: dict[str, float] = {}

    if chosen_backend == "optuna":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=min(n_init, n_trials))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        for trial_num in range(1, n_trials + 1):
            t_tr = time.perf_counter()
            trial = study.ask()
            params: dict[str, Any] = {}
            for p in space:
                if p.kind == "int":
                    params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
                elif p.kind == "float":
                    params[p.name] = trial.suggest_float(p.name, float(p.low), float(p.high))
                elif p.kind == "log_float":
                    params[p.name] = trial.suggest_float(p.name, float(p.low), float(p.high), log=True)
                elif p.kind == "categorical":
                    idx = trial.suggest_categorical(f"{p.name}_idx", list(range(len(p.choices))))
                    params[p.name] = p.choices[idx]

            cv_res = _evaluate_cv_params(spec, params, train_df, feature_cols, splits)
            obj_val = _extract_obj(cv_res)
            study.tell(trial, obj_val)
            dur = time.perf_counter() - t_tr

            if obj_val > best_obj:
                best_obj = obj_val
                best_params = dict(params)
                best_cv_metrics = dict(cv_res)

            print(
                f"  [BO Trial {trial_num:02d}/{n_trials} | {phase}] "
                f"{objective}={obj_val:.5f} (best={best_obj:.5f}) in {dur:.1f}s | params={_serialize_params(params)}",
                flush=True,
            )

            row = {
                "trial": trial_num,
                "phase": "startup_random" if trial_num <= n_init else "bayesian_tpe",
                "objective_value": obj_val,
                "best_so_far": best_obj,
                **cv_res,
                "duration_sec": dur,
                "params_json": json.dumps(_serialize_params(params)),
            }
            for k, v in _serialize_params(params).items():
                row[f"param_{k}"] = v
            trial_rows.append(row)

    else:
        # Genuine Gaussian Process (Matérn 5/2) + Expected Improvement Bayesian Optimisation
        rng = np.random.default_rng(random_state)
        X_obs: list[np.ndarray] = []
        y_obs: list[float] = []
        n_startup = max(2, min(n_init, n_trials))

        for trial_num in range(1, n_trials + 1):
            t_tr = time.perf_counter()
            if trial_num <= n_startup or len(X_obs) < 2:
                # Stratified Latin-style random point in [0, 1]^d
                u_vec = rng.uniform(0.0, 1.0, size=dim)
                phase = "initial_exploration"
            else:
                phase = "bayesian_gp_ei"
                X_mat = np.vstack(X_obs)
                y_vec = np.asarray(y_obs, dtype=np.float64)
                y_std = float(np.std(y_vec))
                y_norm = (y_vec - np.mean(y_vec)) / (y_std if y_std > 1e-9 else 1.0)

                kernel = (
                    ConstantKernel(1.0, (1e-3, 1e3))
                    * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 10.0), nu=2.5)
                    + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-7, 1e-1))
                )
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=3,
                    random_state=random_state + trial_num,
                    normalize_y=False,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(X_mat, y_norm)

                # Maximize Expected Improvement over a dense quasi-random candidate pool
                cand_pool = rng.uniform(0.0, 1.0, size=(2048, dim))
                ei_vals = _expected_improvement(cand_pool, gp, best_y=float(np.max(y_norm)), xi=0.01)

                # Avoid picking an already-evaluated identical point
                order = np.argsort(-ei_vals)
                u_vec = cand_pool[order[0]]
                for idx in order[:50]:
                    cand = cand_pool[idx]
                    dists = np.linalg.norm(X_mat - cand, axis=1)
                    if float(np.min(dists)) > 1e-3:
                        u_vec = cand
                        break

            params = {p.name: p.decode(u_vec[i]) for i, p in enumerate(space)}
            cv_res = _evaluate_cv_params(spec, params, train_df, feature_cols, splits)
            obj_val = _extract_obj(cv_res)
            dur = time.perf_counter() - t_tr

            X_obs.append(u_vec)
            y_obs.append(obj_val)

            if obj_val > best_obj:
                best_obj = obj_val
                best_params = dict(params)
                best_cv_metrics = dict(cv_res)

            print(
                f"  [BO Trial {trial_num:02d}/{n_trials} | {phase}] "
                f"{objective}={obj_val:.5f} (best={best_obj:.5f}) in {dur:.1f}s | params={_serialize_params(params)}",
                flush=True,
            )

            row = {
                "trial": trial_num,
                "phase": phase,
                "objective_value": obj_val,
                "best_so_far": best_obj,
                **cv_res,
                "duration_sec": dur,
                "params_json": json.dumps(_serialize_params(params)),
            }
            for k, v in _serialize_params(params).items():
                row[f"param_{k}"] = v
            trial_rows.append(row)

    total_bo_sec = time.perf_counter() - t0_bo
    trials_df = pd.DataFrame(trial_rows)

    # Step 3: Refit final tuned model on all training trips
    tuned_model = spec.builder(best_params)
    t0_fit_t = time.perf_counter()
    fit_pipeline(
        tuned_model,
        spec,
        train_df[feature_cols],
        train_df["label"].to_numpy(dtype=np.int64),
        groups=train_df["trip_id"].to_numpy(),
    )
    tuned_fit_sec = time.perf_counter() - t0_fit_t
    tuned_history = extract_training_history(tuned_model)

    tuned_test_eval = None
    tuned_inf_sec = None
    student_test_preds_df = None
    if student_test_df is not None:
        t0_inf = time.perf_counter()
        t_probs = tuned_model.predict_proba(student_test_df[feature_cols])[:, 1]
        tuned_inf_sec = time.perf_counter() - t0_inf
        tuned_test_eval = evaluate_predictions(
            student_test_df,
            t_probs,
            obs_meta=student_obs_meta,
            arcs_by_trip=arcs_by_trip,
            routes_by_trip=routes_by_trip,
        )
        student_test_preds_df = build_scored_candidates_df(student_test_df, t_probs)

    # Step 4: Build Untuned vs Tuned comparison table (Report Section 4 exact fields)
    comp_rows = [
        {
            "Version": f"{spec.display_name} (Untuned)",
            "CV Macro F1-score": untuned_cv["cv_macro_f1"],
            "CV Macro F1 Std": untuned_cv["cv_macro_f1_std"],
            "Student-Test Top-1 Accuracy": (
                untuned_test_eval["observation"]["top1_accuracy"] if untuned_test_eval else None
            ),
            "Student-Test Macro F1-score": (
                untuned_test_eval["candidate"]["macro_f1"] if untuned_test_eval else None
            ),
            "Student-Test Log Loss": (
                untuned_test_eval["candidate"]["log_loss"] if untuned_test_eval else None
            ),
            "Student-Test Route Continuity": (
                untuned_test_eval["continuity"]["route_continuity"] if untuned_test_eval else None
            ),
            "Train Time (s)": untuned_fit_sec,
            "Inference Time (s)": untuned_inf_sec,
        },
        {
            "Version": f"{spec.display_name} (Bayesian Tuned)",
            "CV Macro F1-score": best_cv_metrics["cv_macro_f1"],
            "CV Macro F1 Std": best_cv_metrics["cv_macro_f1_std"],
            "Student-Test Top-1 Accuracy": (
                tuned_test_eval["observation"]["top1_accuracy"] if tuned_test_eval else None
            ),
            "Student-Test Macro F1-score": (
                tuned_test_eval["candidate"]["macro_f1"] if tuned_test_eval else None
            ),
            "Student-Test Log Loss": (
                tuned_test_eval["candidate"]["log_loss"] if tuned_test_eval else None
            ),
            "Student-Test Route Continuity": (
                tuned_test_eval["continuity"]["route_continuity"] if tuned_test_eval else None
            ),
            "Train Time (s)": tuned_fit_sec,
            "Inference Time (s)": tuned_inf_sec,
        },
    ]
    untuned_vs_tuned_df = pd.DataFrame(comp_rows)

    summary = {
        "model_key": model_key,
        "model_display_name": spec.display_name,
        "algorithm_family": spec.algorithm_family,
        "backend": chosen_backend,
        "surrogate_and_acquisition": (
            "Optuna TPE (Tree-structured Parzen Estimator)"
            if chosen_backend == "optuna"
            else "Gaussian Process (Matérn 5/2 + WhiteKernel) with Expected Improvement (EI)"
        ),
        "objective_optimised": objective,
        "cv_strategy": f"{actual_splits}-fold trip-grouped StratifiedGroupKFold (training partition only)",
        "n_trials": int(n_trials),
        "n_initial_random_trials": int(min(n_init, n_trials)),
        "total_optimisation_time_sec": float(total_bo_sec),
        "baseline_eval_time_sec": float(time.perf_counter() - t0_untuned - total_bo_sec),
        "search_space": [p.describe() for p in space],
        "best_hyperparameters": _serialize_params(best_params),
        "untuned_cv_metrics": untuned_cv,
        "tuned_cv_metrics": best_cv_metrics,
        "untuned_student_test_metrics": untuned_test_eval,
        "tuned_student_test_metrics": tuned_test_eval,
        "tuned_training_history": tuned_history,
        "improved_cv_objective": bool(_extract_obj(best_cv_metrics) > _extract_obj(untuned_cv)),
        "random_seed": int(random_state),
    }

    artifact_paths: dict[str, str] = {}
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        trials_path = save_dir / "bayesian_opt_trials.csv"
        comp_path = save_dir / "untuned_vs_tuned_comparison.csv"
        summary_path = save_dir / "bayesian_opt_summary.json"
        trials_df.to_csv(trials_path, index=False)
        untuned_vs_tuned_df.to_csv(comp_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        artifact_paths["trials_csv"] = str(trials_path)
        artifact_paths["untuned_vs_tuned_csv"] = str(comp_path)
        artifact_paths["bayesian_summary_json"] = str(summary_path)
        if student_test_preds_df is not None:
            preds_path = save_dir / "final_model_student_test_predictions.csv"
            student_test_preds_df.to_csv(preds_path, index=False)
            artifact_paths["final_predictions_csv"] = str(preds_path)

    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / "final_model.joblib"
        feats_path = artifacts_dir / "feature_columns.json"
        meta_path = artifacts_dir / "model_metadata.json"
        joblib.dump(tuned_model, model_path, compress=3)
        feats_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
        meta = {
            "model_key": model_key,
            "model_display_name": spec.display_name,
            "algorithm_family": spec.algorithm_family,
            "best_hyperparameters": _serialize_params(best_params),
            "feature_columns": feature_cols,
            "random_seed": int(random_state),
            "tuned_cv_metrics": best_cv_metrics,
            "tuned_student_test_summary": (
                {
                    "top1_accuracy": tuned_test_eval["observation"]["top1_accuracy"],
                    "macro_f1": tuned_test_eval["candidate"]["macro_f1"],
                    "log_loss": tuned_test_eval["candidate"]["log_loss"],
                    "route_continuity": tuned_test_eval["continuity"]["route_continuity"],
                    "candidate_miss_rate": tuned_test_eval["observation"]["candidate_miss_rate"],
                }
                if tuned_test_eval
                else None
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        artifact_paths["final_model_joblib"] = str(model_path)
        artifact_paths["feature_columns_json"] = str(feats_path)
        artifact_paths["model_metadata_json"] = str(meta_path)

    return {
        "tuned_model": tuned_model,
        "untuned_model": untuned_model,
        "best_params": _serialize_params(best_params),
        "trials_df": trials_df,
        "untuned_vs_tuned_df": untuned_vs_tuned_df,
        "summary": summary,
        "student_test_predictions_df": student_test_preds_df,
        "artifact_paths": artifact_paths,
    }
