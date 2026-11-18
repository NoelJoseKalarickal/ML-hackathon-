"""Person 3 Entry Point: Model Training, 5-Model Comparison & Bayesian Optimisation.

Follows the PathSnap team contract:
  - Exposes `train_model(...)` returning structured Python objects (trained model Pipeline,
    comparison DataFrames, metrics dictionaries, candidate probability DataFrames, etc.).
  - Supports three clean execution modes:
      * `mode="rf_baseline"`: Phase 1 fast Random Forest baseline + trip-grouped CV.
      * `mode="compare"`:     Phase 2 comparison across all 5 distinct algorithm families.
      * `mode="full"`:        Phase 2 full pipeline (5-model comparison + Bayesian HPO on the
                              best model + plots + final production artifact bundle).
  - Also executable directly from the command line (`python -m models.train_model ...`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from models.bayesian_opt import run_bayesian_optimization
from models.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_COMPARISON_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_RESULTS_DIR,
    N_CV_SPLITS,
    RANDOM_SEED,
)
from models.data_loader import (
    DataInput,
    load_obs_meta,
    load_reference_routes,
    load_road_arcs,
    prepare_train_and_student_test,
    summarize_dataset,
)
from models.metrics import build_scored_candidates_df, evaluate_predictions
from models.model_registry import (
    ModelSpec,
    extract_feature_importance,
    extract_training_history,
    fit_pipeline,
    get_model_registry,
)
from models.plotting import generate_all_plots


def _diagnose_fit(train_f1: float, cv_f1: float, test_f1: Optional[float]) -> dict[str, Any]:
    """Provide quantitative overfitting / underfitting diagnostics for Report Section 3.2 Q9."""
    train_cv_gap = float(train_f1 - cv_f1)
    cv_test_gap = float(cv_f1 - test_f1) if test_f1 is not None else None
    if cv_f1 < 0.70 and train_f1 < 0.75:
        status = "possible_underfitting"
    elif train_cv_gap > 0.08:
        status = "noticeable_overfitting"
    elif train_cv_gap > 0.03:
        status = "mild_overfitting"
    else:
        status = "well_generalised"
    return {
        "status": status,
        "train_macro_f1": float(train_f1),
        "cv_macro_f1": float(cv_f1),
        "student_test_macro_f1": float(test_f1) if test_f1 is not None else None,
        "train_minus_cv_gap": train_cv_gap,
        "cv_minus_test_gap": cv_test_gap,
    }


def _evaluate_single_model(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    student_test_df: Optional[pd.DataFrame],
    train_obs_meta: Optional[pd.DataFrame],
    student_obs_meta: Optional[pd.DataFrame],
    arcs_by_trip: dict[str, dict[int, tuple[int, int]]],
    routes_by_trip: dict[str, list[int]],
    params_override: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run trip-grouped CV + full training fit + student-test evaluation for one model."""
    params = dict(params_override or {})
    X_train = train_df[feature_cols]
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    groups_train = train_df["trip_id"].to_numpy()

    # 1. Trip-grouped K-fold Cross-Validation on the 80 training trips
    fold_records: list[dict[str, Any]] = []
    for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
        pipe_cv = spec.builder(params)
        t0_f = time.perf_counter()
        fit_pipeline(
            pipe_cv,
            spec,
            X_train.iloc[tr_idx],
            y_train[tr_idx],
            groups=groups_train[tr_idx],
        )
        fold_fit_s = time.perf_counter() - t0_f

        t0_inf = time.perf_counter()
        val_probs = pipe_cv.predict_proba(X_train.iloc[val_idx])[:, 1]
        fold_inf_s = time.perf_counter() - t0_inf

        val_df = train_df.iloc[val_idx]
        val_eval = evaluate_predictions(
            val_df,
            val_probs,
            obs_meta=train_obs_meta,
            arcs_by_trip=arcs_by_trip,
            routes_by_trip=routes_by_trip,
        )
        fold_records.append(
            {
                "model_key": spec.key,
                "model_name": spec.display_name,
                "fold": fold_idx,
                "n_val_trips": int(val_df["trip_id"].nunique()),
                "n_val_rows": int(len(val_df)),
                "val_macro_f1": val_eval["candidate"]["macro_f1"],
                "val_binary_f1": val_eval["candidate"]["binary_f1"],
                "val_accuracy": val_eval["candidate"]["accuracy"],
                "val_log_loss": val_eval["candidate"]["log_loss"],
                "val_top1_accuracy": val_eval["observation"]["top1_accuracy"],
                "val_route_continuity": val_eval["continuity"]["route_continuity"],
                "fit_time_sec": fold_fit_s,
                "inference_time_sec": fold_inf_s,
            }
        )

    cv_f1s = [r["val_macro_f1"] for r in fold_records]
    cv_top1s = [r["val_top1_accuracy"] for r in fold_records]
    cv_losses = [r["val_log_loss"] for r in fold_records]
    cv_conts = [r["val_route_continuity"] for r in fold_records if r["val_route_continuity"] is not None]

    cv_summary = {
        "n_splits": len(splits),
        "cv_macro_f1_mean": float(np.mean(cv_f1s)),
        "cv_macro_f1_std": float(np.std(cv_f1s)),
        "cv_top1_accuracy_mean": float(np.mean(cv_top1s)),
        "cv_top1_accuracy_std": float(np.std(cv_top1s)),
        "cv_log_loss_mean": float(np.mean(cv_losses)),
        "cv_log_loss_std": float(np.std(cv_losses)),
        "cv_route_continuity_mean": float(np.mean(cv_conts)) if cv_conts else None,
        "mean_fold_fit_time_sec": float(np.mean([r["fit_time_sec"] for r in fold_records])),
    }

    # 2. Fit full untuned model on all training trips
    fitted_model = spec.builder(params)
    t0_full = time.perf_counter()
    fit_pipeline(fitted_model, spec, X_train, y_train, groups=groups_train)
    train_time_sec = time.perf_counter() - t0_full

    # Genuine training history (None for non-iterative models — never fabricated)
    history = extract_training_history(fitted_model)
    feat_imp = extract_feature_importance(fitted_model, feature_cols)

    # Evaluate on training partition (for overfitting diagnostics only)
    tr_probs = fitted_model.predict_proba(X_train)[:, 1]
    train_eval = evaluate_predictions(
        train_df,
        tr_probs,
        obs_meta=train_obs_meta,
        arcs_by_trip=arcs_by_trip,
        routes_by_trip=routes_by_trip,
    )

    # 3. Evaluate on the 10 student-testing trips
    student_test_eval = None
    inference_time_sec = None
    student_preds_df = None
    if student_test_df is not None:
        X_test = student_test_df[feature_cols]
        t0_te = time.perf_counter()
        test_probs = fitted_model.predict_proba(X_test)[:, 1]
        inference_time_sec = time.perf_counter() - t0_te

        student_test_eval = evaluate_predictions(
            student_test_df,
            test_probs,
            obs_meta=student_obs_meta,
            arcs_by_trip=arcs_by_trip,
            routes_by_trip=routes_by_trip,
        )
        student_preds_df = build_scored_candidates_df(student_test_df, test_probs)

    fit_diag = _diagnose_fit(
        train_f1=train_eval["candidate"]["macro_f1"],
        cv_f1=cv_summary["cv_macro_f1_mean"],
        test_f1=(
            student_test_eval["candidate"]["macro_f1"] if student_test_eval is not None else None
        ),
    )

    # Row matching Report Table 3.1 + Person 3 required Accuracy / Precision / Recall / F1
    comparison_row = {
        "model_key": spec.key,
        "Model": spec.display_name,
        "Algorithm Family": spec.algorithm_family,
        "Key Configuration": spec.key_configuration,
        "CV Macro F1-score": cv_summary["cv_macro_f1_mean"],
        "CV Macro F1 Std": cv_summary["cv_macro_f1_std"],
        "Top-1 Accuracy": (
            student_test_eval["observation"]["top1_accuracy"] if student_test_eval else None
        ),
        "Route Continuity": (
            student_test_eval["continuity"]["route_continuity"] if student_test_eval else None
        ),
        "Student-Test Macro F1-score": (
            student_test_eval["candidate"]["macro_f1"] if student_test_eval else None
        ),
        "Student-Test Accuracy": (
            student_test_eval["candidate"]["accuracy"] if student_test_eval else None
        ),
        "Student-Test Macro Precision": (
            student_test_eval["candidate"]["macro_precision"] if student_test_eval else None
        ),
        "Student-Test Macro Recall": (
            student_test_eval["candidate"]["macro_recall"] if student_test_eval else None
        ),
        "Log Loss": (
            student_test_eval["candidate"]["log_loss"] if student_test_eval else None
        ),
        "Inference Time (s)": inference_time_sec,
        "Train Time (s)": train_time_sec,
        "Candidate Miss Rate": (
            student_test_eval["observation"]["candidate_miss_rate"] if student_test_eval else None
        ),
        "Sequence LCS F1": (
            student_test_eval["sequence"]["sequence_lcs_f1"] if student_test_eval else None
        ),
    }

    detailed = {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "algorithm_family": spec.algorithm_family,
        "key_configuration": spec.key_configuration,
        "initial_hyperparameters": {**spec.initial_hyperparameters, **params},
        "loss_function": spec.loss_function,
        "random_seed": RANDOM_SEED,
        "train_time_sec": float(train_time_sec),
        "inference_time_sec": float(inference_time_sec) if inference_time_sec is not None else None,
        "cv_summary": cv_summary,
        "train_metrics": {
            "candidate": train_eval["candidate"],
            "observation": train_eval["observation"],
        },
        "student_test_metrics": student_test_eval,
        "fit_diagnostic": fit_diag,
        "has_genuine_training_history": history is not None,
        "feature_importance": feat_imp,
    }

    return {
        "fitted_model": fitted_model,
        "fold_records": fold_records,
        "comparison_row": comparison_row,
        "detailed": detailed,
        "history": history,
        "student_preds_df": student_preds_df,
    }


def _select_best_model_key(comparison_df: pd.DataFrame) -> str:
    """Select best model by CV Macro F1-score (tie-breaking on lowest Log Loss, highest Top-1)."""
    ranked = comparison_df.copy()
    ranked["_log_loss_sort"] = ranked["Log Loss"].fillna(999.0)
    ranked["_top1_sort"] = ranked["Top-1 Accuracy"].fillna(-1.0)
    ranked.sort_values(
        ["CV Macro F1-score", "_top1_sort", "_log_loss_sort"],
        ascending=[False, False, True],
        kind="mergesort",
        inplace=True,
    )
    return str(ranked.iloc[0]["model_key"])


def train_model(
    train_data: DataInput,
    student_test_data: Optional[DataInput] = None,
    mode: str = "full",
    feature_cols: Optional[list[str]] = None,
    model_keys: Optional[list[str]] = None,
    train_obs_meta: Optional[DataInput] = None,
    student_obs_meta: Optional[DataInput] = None,
    raw_data_dir: Optional[str | Path] = None,
    arcs_csv: Optional[DataInput] = None,
    arcs_by_trip_dict: Optional[dict[str, dict[int, tuple[int, int]]]] = None,
    routes_csv: Optional[DataInput] = None,
    routes_by_trip_dict: Optional[dict[str, list[int]]] = None,
    n_cv_splits: int = N_CV_SPLITS,
    bo_model_key: Optional[str] = None,
    bo_trials: int = 20,
    bo_init: int = 5,
    bo_objective: str = "cv_macro_f1",
    bo_backend: str = "auto",
    include_optional_models: bool = False,
    save_outputs: bool = True,
    generate_plots: bool = True,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """
    Main Person 3 entry point callable by Person 4 (`integration/pipeline.py`) or via CLI.

    Returns a dictionary of structured Python objects:
      - `model`:                  Final selected fitted sklearn Pipeline (tuned if mode=="full")
      - `best_model_key`:         Key of the selected winning model
      - `models`:                 dict[model_key -> fitted untuned sklearn Pipeline]
      - `comparison_df`:          pd.DataFrame matching Report Table 3.1
      - `cv_folds_df`:            pd.DataFrame of per-fold trip-grouped CV metrics
      - `metrics`:                dict of full detailed metrics per model + dataset summaries
      - `predictions_df`:         pd.DataFrame of student-test candidate probabilities for best model
      - `all_predictions`:        dict[model_key -> student-test candidate probability DataFrame]
      - `feature_importance_df`:  pd.DataFrame of feature importances for best model (if available)
      - `training_histories`:     dict of genuine epoch/iteration loss histories ONLY
      - `bayesian_opt`:           dict with `trials_df`, `untuned_vs_tuned_df`, `summary` (if mode=="full")
      - `feature_columns`:        list[str] of exact input feature names
      - `plot_paths`:             dict[plot_name -> file path]
      - `artifact_paths`:         dict[artifact_name -> file path]
    """
    if mode not in {"rf_baseline", "compare", "full"}:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'rf_baseline', 'compare', 'full'.")

    train_df, test_df, inferred_feature_cols = prepare_train_and_student_test(
        train_data=train_data,
        student_test_data=student_test_data,
        explicit_feature_cols=feature_cols,
    )

    tr_meta_df = load_obs_meta(train_obs_meta)
    te_meta_df = load_obs_meta(student_obs_meta)

    all_trips = set(train_df["trip_id"].unique())
    if test_df is not None:
        all_trips |= set(test_df["trip_id"].unique())

    arcs_by_trip = load_road_arcs(
        raw_data_dir=raw_data_dir,
        arcs_csv=arcs_csv,
        arcs_by_trip_dict=arcs_by_trip_dict,
        trip_ids=all_trips,
    )
    routes_by_trip = load_reference_routes(
        raw_data_dir=raw_data_dir,
        routes_csv=routes_csv,
        routes_by_trip_dict=routes_by_trip_dict,
        trip_ids=all_trips,
    )

    dataset_summaries = {
        "train": summarize_dataset(train_df, inferred_feature_cols, "train"),
        "student_test": (
            summarize_dataset(test_df, inferred_feature_cols, "student_test")
            if test_df is not None
            else None
        ),
        "road_arcs_trips_loaded": len(arcs_by_trip),
        "reference_routes_trips_loaded": len(routes_by_trip),
    }

    registry = get_model_registry(include_optional=include_optional_models)
    if mode == "rf_baseline":
        selected_keys = ["random_forest"]
    elif model_keys is not None:
        unknown = [k for k in model_keys if k not in registry]
        if unknown:
            raise ValueError(f"Unknown model_keys: {unknown}. Available: {list(registry.keys())}")
        selected_keys = list(model_keys)
    else:
        selected_keys = list(registry.keys())

    # Build deterministic trip-grouped StratifiedGroupKFold splits ONCE so all models share exact folds
    n_unique_trips = int(train_df["trip_id"].nunique())
    actual_splits = min(int(n_cv_splits), max(2, n_unique_trips))
    sgkf = StratifiedGroupKFold(n_splits=actual_splits)
    splits = list(
        sgkf.split(
            train_df[inferred_feature_cols],
            train_df["label"].to_numpy(dtype=np.int64),
            groups=train_df["trip_id"].to_numpy(),
        )
    )

    fitted_models: dict[str, Any] = {}
    all_fold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    detailed_metrics: dict[str, Any] = {}
    training_histories: dict[str, dict[str, Any]] = {}
    all_predictions: dict[str, pd.DataFrame] = {}

    for key in selected_keys:
        spec = registry[key]
        print(f"[Person 3] Training & 5-fold trip-grouped CV: {spec.display_name} ({spec.algorithm_family})...", flush=True)
        res = _evaluate_single_model(
            spec=spec,
            train_df=train_df,
            feature_cols=inferred_feature_cols,
            splits=splits,
            student_test_df=test_df,
            train_obs_meta=tr_meta_df,
            student_obs_meta=te_meta_df,
            arcs_by_trip=arcs_by_trip,
            routes_by_trip=routes_by_trip,
        )
        cr = res["comparison_row"]
        print(
            f"  -> Done {spec.display_name}: CV Macro F1={cr['CV Macro F1-score']:.4f} (±{cr['CV Macro F1 Std']:.4f}), "
            f"Test Top-1={cr['Top-1 Accuracy'] if cr['Top-1 Accuracy'] is not None else 'N/A'}, "
            f"Test Macro F1={cr['Student-Test Macro F1-score'] if cr['Student-Test Macro F1-score'] is not None else 'N/A'}, "
            f"LogLoss={cr['Log Loss'] if cr['Log Loss'] is not None else 'N/A'}, "
            f"Cont={cr['Route Continuity'] if cr['Route Continuity'] is not None else 'N/A'}",
            flush=True,
        )
        fitted_models[key] = res["fitted_model"]
        all_fold_rows.extend(res["fold_records"])
        comparison_rows.append(res["comparison_row"])
        detailed_metrics[key] = res["detailed"]
        if res["history"] is not None:
            training_histories[key] = res["history"]
        if res["student_preds_df"] is not None:
            all_predictions[key] = res["student_preds_df"]

    comparison_df = pd.DataFrame(comparison_rows)
    cv_folds_df = pd.DataFrame(all_fold_rows)
    best_model_key = _select_best_model_key(comparison_df)
    final_model = fitted_models[best_model_key]
    best_predictions_df = all_predictions.get(best_model_key)

    artifact_paths: dict[str, str] = {}
    res_dir = Path(results_dir)
    art_dir = Path(artifacts_dir)

    if save_outputs:
        res_dir.mkdir(parents=True, exist_ok=True)
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "models").mkdir(parents=True, exist_ok=True)
        # Mirror comparison tables and experiment logs into Person 4's experiments/ layout
        comp_dir = (
            DEFAULT_COMPARISON_DIR
            if res_dir == DEFAULT_RESULTS_DIR
            else res_dir
        )
        logs_dir = (
            DEFAULT_LOGS_DIR
            if res_dir == DEFAULT_RESULTS_DIR
            else res_dir
        )
        comp_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        comp_path = comp_dir / "model_comparison.csv"
        folds_path = logs_dir / "cv_fold_details.csv"
        detail_path = logs_dir / "detailed_model_metrics.json"
        hist_path = logs_dir / "training_histories.json"
        ds_path = logs_dir / "dataset_summary.json"

        comparison_df.to_csv(comp_path, index=False)
        cv_folds_df.to_csv(folds_path, index=False)
        detail_path.write_text(json.dumps(detailed_metrics, indent=2), encoding="utf-8")
        hist_path.write_text(json.dumps(training_histories, indent=2), encoding="utf-8")
        ds_path.write_text(json.dumps(dataset_summaries, indent=2), encoding="utf-8")

        artifact_paths["model_comparison_csv"] = str(comp_path)
        artifact_paths["cv_fold_details_csv"] = str(folds_path)
        artifact_paths["detailed_model_metrics_json"] = str(detail_path)
        artifact_paths["training_histories_json"] = str(hist_path)
        artifact_paths["dataset_summary_json"] = str(ds_path)

        for key, mdl in fitted_models.items():
            mp = art_dir / "models" / f"{key}_untuned.joblib"
            joblib.dump(mdl, mp, compress=3)
            artifact_paths[f"untuned_model_{key}"] = str(mp)
            if key in all_predictions:
                pp = res_dir / f"student_test_predictions_{key}.csv"
                all_predictions[key].to_csv(pp, index=False)

    # Optional Bayesian Hyperparameter Optimisation (Phase 2 full mode)
    bayesian_res: Optional[dict[str, Any]] = None
    if mode == "full":
        target_bo_key = bo_model_key or best_model_key
        bayesian_res = run_bayesian_optimization(
            model_key=target_bo_key,
            train_df=train_df,
            feature_cols=inferred_feature_cols,
            student_test_df=test_df,
            student_obs_meta=te_meta_df,
            arcs_by_trip=arcs_by_trip,
            routes_by_trip=routes_by_trip,
            n_trials=bo_trials,
            n_init=bo_init,
            n_cv_splits=actual_splits,
            objective=bo_objective,
            backend=bo_backend,
            random_state=random_state,
            save_dir=res_dir if save_outputs else None,
            artifacts_dir=art_dir if save_outputs else None,
        )
        final_model = bayesian_res["tuned_model"]
        best_model_key = target_bo_key
        if bayesian_res["student_test_predictions_df"] is not None:
            best_predictions_df = bayesian_res["student_test_predictions_df"]
        artifact_paths.update(bayesian_res.get("artifact_paths", {}))
    elif save_outputs:
        # Save best untuned (or RF baseline) directly as final_model.joblib for Phase 1 demo
        model_path = art_dir / "final_model.joblib"
        feats_path = art_dir / "feature_columns.json"
        meta_path = art_dir / "model_metadata.json"
        joblib.dump(final_model, model_path)
        feats_path.write_text(json.dumps(inferred_feature_cols, indent=2), encoding="utf-8")
        meta = {
            "mode": mode,
            "model_key": best_model_key,
            "model_display_name": registry[best_model_key].display_name,
            "algorithm_family": registry[best_model_key].algorithm_family,
            "feature_columns": inferred_feature_cols,
            "random_seed": random_state,
            "cv_summary": detailed_metrics[best_model_key]["cv_summary"],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        artifact_paths["final_model_joblib"] = str(model_path)
        artifact_paths["feature_columns_json"] = str(feats_path)
        artifact_paths["model_metadata_json"] = str(meta_path)

    # Feature importance DataFrame for the selected model (if natively available)
    best_imp = extract_feature_importance(final_model, inferred_feature_cols)
    feature_importance_df = (
        pd.DataFrame(
            sorted(best_imp.items(), key=lambda kv: kv[1], reverse=True),
            columns=["feature", "importance"],
        )
        if best_imp is not None
        else None
    )

    plot_paths: dict[str, str] = {}
    if generate_plots and save_outputs:
        plot_paths = generate_all_plots(
            plots_dir=plots_dir,
            comparison_df=comparison_df,
            detailed_metrics=detailed_metrics,
            training_histories=training_histories,
            bayesian_trials_df=bayesian_res["trials_df"] if bayesian_res else None,
            untuned_vs_tuned_df=bayesian_res["untuned_vs_tuned_df"] if bayesian_res else None,
            bayesian_objective=bo_objective,
        )

    return {
        "model": final_model,
        "best_model_key": best_model_key,
        "models": fitted_models,
        "comparison_df": comparison_df,
        "cv_folds_df": cv_folds_df,
        "metrics": {
            "datasets": dataset_summaries,
            "models": detailed_metrics,
        },
        "predictions_df": best_predictions_df,
        "all_predictions": all_predictions,
        "feature_importance_df": feature_importance_df,
        "training_histories": training_histories,
        "bayesian_opt": bayesian_res,
        "feature_columns": inferred_feature_cols,
        "plot_paths": plot_paths,
        "artifact_paths": artifact_paths,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PathSnap Person 3: Train, Compare, and Bayesian-Optimise Map-Matching Models"
    )
    parser.add_argument("--train", required=True, help="Path to training candidates (.csv or .parquet)")
    parser.add_argument("--student-test", default=None, help="Path to student-test candidates (.csv or .parquet)")
    parser.add_argument(
        "--mode",
        choices=["rf_baseline", "compare", "full"],
        default="full",
        help="rf_baseline (Phase 1), compare (5 models), or full (5 models + Bayesian HPO)",
    )
    parser.add_argument("--train-obs-meta", default=None, help="Optional train observation metadata CSV")
    parser.add_argument("--student-obs-meta", default=None, help="Optional student-test observation metadata CSV")
    parser.add_argument("--raw-data-dir", default=None, help="Optional raw trip directory with .arcs and .route files")
    parser.add_argument("--arcs-csv", default=None, help="Optional combined arcs CSV")
    parser.add_argument("--routes-csv", default=None, help="Optional combined reference routes CSV")
    parser.add_argument("--cv-splits", type=int, default=N_CV_SPLITS, help="Number of trip-grouped CV folds")
    parser.add_argument("--bo-model", default=None, help="Specific model key for Bayesian HPO (default: auto best)")
    parser.add_argument("--bo-trials", type=int, default=20, help="Number of Bayesian optimisation trials")
    parser.add_argument("--bo-init", type=int, default=5, help="Initial random exploration trials")
    parser.add_argument(
        "--bo-objective",
        choices=["cv_macro_f1", "cv_top1_accuracy", "neg_cv_log_loss"],
        default="cv_macro_f1",
    )
    parser.add_argument("--bo-backend", choices=["auto", "gp_ei", "optuna"], default="auto")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = train_model(
        train_data=args.train,
        student_test_data=args.student_test,
        mode=args.mode,
        train_obs_meta=args.train_obs_meta,
        student_obs_meta=args.student_obs_meta,
        raw_data_dir=args.raw_data_dir,
        arcs_csv=args.arcs_csv,
        routes_csv=args.routes_csv,
        n_cv_splits=args.cv_splits,
        bo_model_key=args.bo_model,
        bo_trials=args.bo_trials,
        bo_init=args.bo_init,
        bo_objective=args.bo_objective,
        bo_backend=args.bo_backend,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print("\n=== PathSnap Model Comparison (Report Table 3.1) ===")
    print(out["comparison_df"].to_string(index=False))
    if out["bayesian_opt"] is not None:
        print("\n=== Untuned vs. Bayesian-Tuned Comparison (Report Section 4) ===")
        print(out["bayesian_opt"]["untuned_vs_tuned_df"].to_string(index=False))
    print(f"\nSelected Final Model: {out['best_model_key']}")
    print(f"Saved artifacts: {out['artifact_paths'].get('final_model_joblib')}")
