"""Publication-quality visualisations for Report Sections 3.3 & 4.

Strictly uses real experimental results only — training/validation loss curves are
generated ONLY for models that produce genuine epoch/iteration histories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.4f}", y_offset: float = 0.008) -> None:
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=8.5,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_cv_macro_f1(comparison_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    models = comparison_df["Model"].tolist()
    means = comparison_df["CV Macro F1-score"].astype(float).values
    stds = (
        comparison_df["CV Macro F1 Std"].astype(float).values
        if "CV Macro F1 Std" in comparison_df.columns
        else np.zeros_like(means)
    )
    palette = sns.color_palette("viridis", len(models))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=palette, edgecolor="black", alpha=0.9)
    ax.set_ylabel("5-Fold Trip-Grouped CV Macro F1-score")
    ax.set_title("Cross-Validation Macro F1-score Comparison Across Models (Mean ± 1 SD)")
    ax.set_ylim(0, min(1.08, max(means + stds) + 0.12))
    plt.xticks(rotation=20, ha="right")
    for bar, m, s in zip(bars, means, stds):
        ax.annotate(
            f"{m:.4f}\n(±{s:.4f})",
            (bar.get_x() + bar.get_width() / 2.0, m + s),
            ha="center",
            va="bottom",
            fontsize=8.5,
            xytext=(0, 4),
            textcoords="offset points",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_student_test_top1_and_f1(comparison_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if (
        comparison_df["Top-1 Accuracy"].isna().all()
        and comparison_df["Student-Test Macro F1-score"].isna().all()
    ):
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5))
    models = comparison_df["Model"].tolist()
    x = np.arange(len(models))
    w = 0.36
    top1 = comparison_df["Top-1 Accuracy"].fillna(0.0).astype(float).values
    f1 = comparison_df["Student-Test Macro F1-score"].fillna(0.0).astype(float).values

    ax.bar(x - w / 2, top1, width=w, label="Observation-Level Top-1 Accuracy", color="#2b6cb0", edgecolor="black")
    ax.bar(x + w / 2, f1, width=w, label="Candidate-Level Macro F1-score", color="#38a169", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Student-Testing Performance: Top-1 Accuracy vs. Macro F1-score")
    ax.legend(loc="lower right")
    _annotate_bars(ax, fmt="{:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_log_loss_comparison(comparison_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    valid = comparison_df.dropna(subset=["Log Loss"])
    if valid.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8))
    models = valid["Model"].tolist()
    losses = valid["Log Loss"].astype(float).values
    palette = sns.color_palette("flare", len(models))
    ax.bar(models, losses, color=palette, edgecolor="black")
    ax.set_ylabel("Binary Log Loss (Lower is Better)")
    ax.set_title("Student-Testing Probability Calibration: Log Loss Comparison")
    ax.set_ylim(0, max(losses) * 1.18 + 0.02)
    plt.xticks(rotation=20, ha="right")
    _annotate_bars(ax, fmt="{:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_continuity_and_inference_time(comparison_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    has_cont = (
        "Route Continuity" in comparison_df.columns
        and pd.to_numeric(comparison_df["Route Continuity"], errors="coerce").notna().any()
    )
    has_inf = (
        "Inference Time (s)" in comparison_df.columns
        and pd.to_numeric(comparison_df["Inference Time (s)"], errors="coerce").notna().any()
    )
    if not has_cont and not has_inf:
        return None

    n_panels = int(has_cont) + int(has_inf)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]
    idx = 0
    models = comparison_df["Model"].tolist()

    if has_cont:
        ax = axes[idx]
        idx += 1
        vals = pd.to_numeric(comparison_df["Route Continuity"], errors="coerce").fillna(0.0).values
        ax.bar(models, vals, color="#805ad5", edgecolor="black")
        ax.set_ylabel("Strict Directed Route Continuity")
        ax.set_title("Route Continuity (Classifier-Only)")
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=22)
        _annotate_bars(ax, fmt="{:.4f}")

    if has_inf:
        ax = axes[idx]
        vals = pd.to_numeric(comparison_df["Inference Time (s)"], errors="coerce").fillna(0.0).values
        ax.bar(models, vals, color="#dd6b20", edgecolor="black")
        ax.set_ylabel("Student-Test Inference Time (seconds)")
        ax.set_title("Inference Speed Comparison (Lower is Faster)")
        ax.set_ylim(0, max(vals) * 1.2 + 0.005)
        ax.tick_params(axis="x", rotation=22)
        _annotate_bars(ax, fmt="{:.4f}s")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_overfitting_diagnostic(detailed_metrics: dict[str, Any], out_path: Path) -> Optional[Path]:
    rows = []
    for m_key, info in detailed_metrics.items():
        tr_f1 = info.get("train_metrics", {}).get("candidate", {}).get("macro_f1")
        cv_f1 = info.get("cv_summary", {}).get("cv_macro_f1_mean")
        te_f1 = (info.get("student_test_metrics") or {}).get("candidate", {}).get("macro_f1")
        if tr_f1 is not None and cv_f1 is not None:
            rows.append(
                {
                    "Model": info.get("display_name", m_key),
                    "Train Macro F1": tr_f1,
                    "5-Fold CV Macro F1": cv_f1,
                    "Student-Test Macro F1": te_f1 if te_f1 is not None else np.nan,
                }
            )
    if not rows:
        return None
    df_plot = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_plot))
    has_test = df_plot["Student-Test Macro F1"].notna().any()
    w = 0.26 if has_test else 0.35

    if has_test:
        ax.bar(x - w, df_plot["Train Macro F1"], width=w, label="Train Macro F1", color="#4a5568", edgecolor="black")
        ax.bar(x, df_plot["5-Fold CV Macro F1"], width=w, label="5-Fold CV Macro F1", color="#3182ce", edgecolor="black")
        ax.bar(x + w, df_plot["Student-Test Macro F1"], width=w, label="Student-Test Macro F1", color="#38a169", edgecolor="black")
    else:
        ax.bar(x - w / 2, df_plot["Train Macro F1"], width=w, label="Train Macro F1", color="#4a5568", edgecolor="black")
        ax.bar(x + w / 2, df_plot["5-Fold CV Macro F1"], width=w, label="5-Fold CV Macro F1", color="#3182ce", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Generalisation Diagnostic: Train vs. 5-Fold CV vs. Student-Test Macro F1")
    ax.legend(loc="lower right")
    _annotate_bars(ax, fmt="{:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    cm_dict: dict[str, int],
    model_name: str,
    split_label: str,
    out_path: Path,
) -> Path:
    mat = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]], dtype=np.int64)
    row_sums = np.maximum(mat.sum(axis=1, keepdims=True), 1)
    pct = mat / row_sums * 100.0
    annot = np.array([[f"{mat[i, j]:,}\n({pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(
        mat,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Pred 0 (Non-Match)", "Pred 1 (Match)"],
        yticklabels=["True 0 (Non-Match)", "True 1 (Match)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name} ({split_label})")
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_training_history(
    history: dict[str, Any],
    model_name: str,
    out_path: Path,
) -> Path:
    """Plot genuine epoch-wise / iteration-wise training and validation loss."""
    tr = history.get("train_loss", [])
    val = history.get("val_loss", [])
    unit = history.get("unit", "epoch").replace("_", " ").title()
    metric = history.get("metric", "Loss")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = np.arange(1, len(tr) + 1)
    ax.plot(xs, tr, label="Training Loss", linewidth=2.0, marker="o", markersize=3.5, color="#2b6cb0")
    if val:
        ax.plot(xs[: len(val)], val, label="Validation Loss", linewidth=2.0, marker="s", markersize=3.5, color="#e53e3e")
        best_idx = int(np.argmin(val)) + 1
        ax.axvline(best_idx, linestyle="--", color="gray", alpha=0.7, label=f"Best Val ({unit} {best_idx})")
    ax.set_xlabel(unit)
    ax.set_ylabel(metric)
    ax.set_title(f"{model_name} — Genuine {unit}-wise Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_feature_importance(
    imp_dict: dict[str, float],
    model_name: str,
    out_path: Path,
    top_k: int = 15,
) -> Path:
    items = sorted(imp_dict.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names = [k for k, _ in reversed(items)]
    vals = [v for _, v in reversed(items)]
    fig, ax = plt.subplots(figsize=(8, max(4.2, 0.35 * len(names) + 1.5)))
    ax.barh(names, vals, color="#2c7a7b", edgecolor="black")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top Feature Importances — {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_bayesian_convergence(trials_df: pd.DataFrame, objective_name: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    trials = trials_df["trial"].values
    vals = trials_df["objective_value"].values
    best = trials_df["best_so_far"].values

    is_init = trials_df["phase"].str.contains("init|startup", case=False, na=False).values
    if is_init.any():
        ax.scatter(trials[is_init], vals[is_init], color="#718096", s=55, label="Initial Exploration Trials", zorder=3)
    if (~is_init).any():
        ax.scatter(trials[~is_init], vals[~is_init], color="#3182ce", s=55, label="Bayesian Optimised Trials", zorder=3)
    ax.plot(trials, best, color="#e53e3e", linewidth=2.2, label="Best Objective So Far", zorder=4)

    ax.set_xlabel("Optimisation Trial")
    ax.set_ylabel(objective_name)
    ax.set_title(f"Bayesian Hyperparameter Optimisation Convergence ({objective_name})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_untuned_vs_tuned(untuned_vs_tuned_df: pd.DataFrame, out_path: Path) -> Path:
    metrics_to_plot = [
        ("CV Macro F1-score", "CV Macro F1"),
        ("Student-Test Top-1 Accuracy", "Test Top-1 Acc"),
        ("Student-Test Macro F1-score", "Test Macro F1"),
        ("Student-Test Route Continuity", "Route Continuity"),
    ]
    available = [
        (col, label)
        for col, label in metrics_to_plot
        if col in untuned_vs_tuned_df.columns
        and pd.to_numeric(untuned_vs_tuned_df[col], errors="coerce").notna().all()
    ]
    if not available:
        available = [("CV Macro F1-score", "CV Macro F1")]

    labels = [lbl for _, lbl in available]
    untuned_vals = [float(untuned_vs_tuned_df.iloc[0][c]) for c, _ in available]
    tuned_vals = [float(untuned_vs_tuned_df.iloc[1][c]) for c, _ in available]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, untuned_vals, width=w, label=untuned_vs_tuned_df.iloc[0]["Version"], color="#718096", edgecolor="black")
    ax.bar(x + w / 2, tuned_vals, width=w, label=untuned_vs_tuned_df.iloc[1]["Version"], color="#2b6cb0", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Untuned Baseline vs. Bayesian-Tuned Model Performance")
    ax.legend(loc="lower right")
    _annotate_bars(ax, fmt="{:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_all_plots(
    plots_dir: str | Path,
    comparison_df: pd.DataFrame,
    detailed_metrics: dict[str, Any],
    training_histories: dict[str, dict[str, Any]],
    bayesian_trials_df: Optional[pd.DataFrame] = None,
    untuned_vs_tuned_df: Optional[pd.DataFrame] = None,
    bayesian_objective: str = "cv_macro_f1",
) -> dict[str, str]:
    """Generate every Report Section 3.3 & Section 4 plot and return saved path map."""
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    if len(comparison_df) > 0:
        p = plot_cv_macro_f1(comparison_df, out_dir / "cv_macro_f1_comparison.png")
        saved["cv_macro_f1_comparison"] = str(p)

        p2 = plot_student_test_top1_and_f1(
            comparison_df, out_dir / "student_test_top1_and_f1_comparison.png"
        )
        if p2:
            saved["student_test_top1_and_f1_comparison"] = str(p2)

        p3 = plot_log_loss_comparison(comparison_df, out_dir / "log_loss_comparison.png")
        if p3:
            saved["log_loss_comparison"] = str(p3)

        p4 = plot_continuity_and_inference_time(
            comparison_df, out_dir / "route_continuity_and_inference_time.png"
        )
        if p4:
            saved["route_continuity_and_inference_time"] = str(p4)

    p5 = plot_overfitting_diagnostic(detailed_metrics, out_dir / "overfitting_train_cv_test.png")
    if p5:
        saved["overfitting_train_cv_test"] = str(p5)

    for m_key, info in detailed_metrics.items():
        display = info.get("display_name", m_key)
        test_m = info.get("student_test_metrics")
        if test_m and "candidate" in test_m:
            cm_path = plot_confusion_matrix(
                test_m["candidate"]["confusion_matrix"],
                display,
                "Student-Test",
                out_dir / f"confusion_matrix_{m_key}.png",
            )
            saved[f"confusion_matrix_{m_key}"] = str(cm_path)
        else:
            tr_m = info.get("train_metrics")
            if tr_m and "candidate" in tr_m:
                cm_path = plot_confusion_matrix(
                    tr_m["candidate"]["confusion_matrix"],
                    display,
                    "Train",
                    out_dir / f"confusion_matrix_{m_key}.png",
                )
                saved[f"confusion_matrix_{m_key}"] = str(cm_path)

        imp = info.get("feature_importance")
        if imp:
            fi_path = plot_feature_importance(
                imp, display, out_dir / f"feature_importance_{m_key}.png"
            )
            saved[f"feature_importance_{m_key}"] = str(fi_path)

    # ONLY genuine iterative/epoch histories — strictly zero synthetic curves
    for m_key, hist in training_histories.items():
        if not hist or not hist.get("train_loss"):
            continue
        display = detailed_metrics.get(m_key, {}).get("display_name", m_key)
        h_path = plot_training_history(hist, display, out_dir / f"training_loss_{m_key}.png")
        saved[f"training_loss_{m_key}"] = str(h_path)

    if bayesian_trials_df is not None and len(bayesian_trials_df) > 0:
        bp = plot_bayesian_convergence(
            bayesian_trials_df, bayesian_objective, out_dir / "bayesian_opt_convergence.png"
        )
        saved["bayesian_opt_convergence"] = str(bp)

    if untuned_vs_tuned_df is not None and len(untuned_vs_tuned_df) >= 2:
        up = plot_untuned_vs_tuned(untuned_vs_tuned_df, out_dir / "untuned_vs_tuned_comparison.png")
        saved["untuned_vs_tuned_comparison"] = str(up)

    return saved
