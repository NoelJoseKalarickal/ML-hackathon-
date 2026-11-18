"""End-to-end verification of Person 3's PathSnap ML pipeline on synthetic data.

Writes all outputs to /tmp/pathsnap_verify so the repository stays clean.
Tests:
  1. Strict zero-leakage guard (train/student_test trip overlap raises ValueError).
  2. Strict evaluator integrity guard (passing ground-truth labels to CandidateScorer raises ValueError).
  3. Phase 1 mode (`mode="rf_baseline"`) with in-memory DataFrames.
  4. Phase 2 full mode (`mode="full"`): 5 diverse model families, 5-fold trip-grouped CV,
     GP-EI Bayesian HPO, genuine-only training histories, all plots, and saved artifact inference.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from models.data_loader import prepare_train_and_student_test
from models.inference import CandidateScorer
from models.train_model import train_model


def make_synthetic_pathsnap_data(
    n_train_trips: int = 20,
    n_test_trips: int = 5,
    obs_per_trip: int = 25,
    k_candidates: int = 5,
    miss_rate: float = 0.04,
    seed: int = 42,
):
    """Generate realistic synthetic candidate tables, obs_meta, directed .arcs, and .route."""
    rng = np.random.default_rng(seed)

    def _build_split(trip_names: list[str]):
        cand_rows = []
        meta_rows = []
        arcs_by_trip: dict[str, dict[int, tuple[int, int]]] = {}
        routes_by_trip: dict[str, list[int]] = {}

        for tid in trip_names:
            # Build a directed road chain + parallel distractor arcs
            n_route_arcs = 8
            arc_map: dict[int, tuple[int, int]] = {}
            for a in range(n_route_arcs):
                arc_map[a] = (a, a + 1)
            # Distractor parallel/disconnected arcs
            for a in range(n_route_arcs, n_route_arcs + 12):
                arc_map[a] = (100 + a, 101 + a)
            arcs_by_trip[tid] = arc_map
            routes_by_trip[tid] = list(range(n_route_arcs))

            for obs_idx in range(obs_per_trip):
                true_arc = min(obs_idx * n_route_arcs // obs_per_trip, n_route_arcs - 1)
                missed = rng.random() < miss_rate
                distractors = rng.choice(
                    [a for a in arc_map if a != true_arc],
                    size=k_candidates - (0 if missed else 1),
                    replace=False,
                ).tolist()
                cand_arcs = distractors if missed else [true_arc] + distractors
                rng.shuffle(cand_arcs)

                meta_rows.append(
                    {
                        "trip_id": tid,
                        "obs_idx": obs_idx,
                        "true_arc_id": true_arc,
                        "candidate_found": 0 if missed else 1,
                    }
                )

                # Realistic non-linear map-matching features
                dists = []
                for arc_id in cand_arcs:
                    is_true = int(arc_id == true_arc)
                    perp_dist = abs(rng.normal(4.0 if is_true else 18.0, 4.5 if is_true else 9.0))
                    heading_diff = abs(rng.normal(8.0 if is_true else 55.0, 12.0 if is_true else 35.0))
                    heading_diff = min(heading_diff, 180.0)
                    speed = max(0.0, rng.normal(11.0, 3.0))
                    proj_ratio = float(np.clip(rng.normal(0.5 if is_true else 0.9, 0.25), 0.0, 1.0))
                    arc_len = float(rng.uniform(25.0, 180.0))
                    dists.append(perp_dist)
                    cand_rows.append(
                        {
                            "trip_id": tid,
                            "obs_idx": obs_idx,
                            "arc_id": int(arc_id),
                            "label": is_true,
                            "perp_dist_m": perp_dist,
                            "heading_diff_deg": heading_diff,
                            "cos_heading_diff": float(np.cos(np.deg2rad(heading_diff))),
                            "speed_mps": speed,
                            "proj_ratio": proj_ratio,
                            "arc_length_m": arc_len,
                            "dist_x_heading": perp_dist * (heading_diff / 180.0),
                        }
                    )

                # Add relative candidate distance feature
                min_d = min(dists)
                for row in cand_rows[-len(cand_arcs) :]:
                    row["dist_diff_from_min"] = row["perp_dist_m"] - min_d

        df = pd.DataFrame(cand_rows)
        # Inject a few NaNs and one inf to verify robust pipeline sanitization
        df.loc[df.index[:5], "heading_diff_deg"] = np.nan
        df.loc[df.index[5:7], "speed_mps"] = np.inf
        return df, pd.DataFrame(meta_rows), arcs_by_trip, routes_by_trip

    tr_trips = [f"trip_train_{i:02d}" for i in range(n_train_trips)]
    te_trips = [f"trip_test_{i:02d}" for i in range(n_test_trips)]

    tr_df, tr_meta, tr_arcs, tr_routes = _build_split(tr_trips)
    te_df, te_meta, te_arcs, te_routes = _build_split(te_trips)
    all_arcs = {**tr_arcs, **te_arcs}
    all_routes = {**tr_routes, **te_routes}
    return tr_df, te_df, tr_meta, te_meta, all_arcs, all_routes


def main() -> None:
    tmp_dir = Path("/tmp/pathsnap_verify")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    tr_df, te_df, tr_meta, te_meta, all_arcs, all_routes = make_synthetic_pathsnap_data()

    # 1. Verify trip-overlap leakage check
    leaked_test = pd.concat([te_df, tr_df[tr_df["trip_id"] == "trip_train_00"]], ignore_index=True)
    try:
        prepare_train_and_student_test(tr_df, leaked_test)
        raise AssertionError("Leakage check failed to catch overlapping trip_id!")
    except ValueError as e:
        assert "DATA LEAKAGE DETECTED" in str(e)
        print("[PASS] 1. Trip-overlap data leakage guard verified.")

    # 2. Verify Phase 1 RF baseline mode with in-memory DataFrames
    res_p1 = train_model(
        train_data=tr_df,
        student_test_data=te_df,
        mode="rf_baseline",
        train_obs_meta=tr_meta,
        student_obs_meta=te_meta,
        arcs_by_trip_dict=all_arcs,
        routes_by_trip_dict=all_routes,
        results_dir=tmp_dir / "p1_results",
        plots_dir=tmp_dir / "p1_plots",
        artifacts_dir=tmp_dir / "p1_artifacts",
    )
    assert res_p1["best_model_key"] == "random_forest"
    assert isinstance(res_p1["comparison_df"], pd.DataFrame) and len(res_p1["comparison_df"]) == 1
    assert isinstance(res_p1["predictions_df"], pd.DataFrame)
    assert res_p1["feature_importance_df"] is not None
    print("[PASS] 2. Phase 1 rf_baseline mode verified (returns Python objects + artifacts).")

    # 3. Verify Phase 2 full mode (5 models + GP-EI Bayesian HPO + genuine histories + plots)
    res_p2 = train_model(
        train_data=tr_df,
        student_test_data=te_df,
        mode="full",
        train_obs_meta=tr_meta,
        student_obs_meta=te_meta,
        arcs_by_trip_dict=all_arcs,
        routes_by_trip_dict=all_routes,
        bo_trials=8,
        bo_init=4,
        bo_backend="gp_ei",
        results_dir=tmp_dir / "p2_results",
        plots_dir=tmp_dir / "p2_plots",
        artifacts_dir=tmp_dir / "p2_artifacts",
    )
    assert len(res_p2["comparison_df"]) == 5, f"Expected 5 models, got {len(res_p2['comparison_df'])}"
    assert set(res_p2["models"].keys()) == {
        "logistic_regression",
        "k_nearest_neighbors",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert set(res_p2["training_histories"].keys()) == {
        "hist_gradient_boosting",
    }, f"Unexpected training_histories keys: {set(res_p2['training_histories'].keys())}"
    assert res_p2["bayesian_opt"] is not None
    assert len(res_p2["bayesian_opt"]["trials_df"]) == 8
    assert len(res_p2["bayesian_opt"]["untuned_vs_tuned_df"]) == 2
    assert len(res_p2["plot_paths"]) >= 10
    for pname, ppath in res_p2["plot_paths"].items():
        assert Path(ppath).exists(), f"Missing plot {pname}: {ppath}"
    print("[PASS] 3. Phase 2 full mode verified (5 models, GP-EI HPO, genuine histories, all plots).")

    # 4. Verify CandidateScorer inference & strict no-label guard
    scorer = CandidateScorer(artifacts_dir=tmp_dir / "p2_artifacts")
    try:
        scorer.score_candidates(te_df, strict_no_labels=True)
        raise AssertionError("CandidateScorer failed to block ground-truth 'label' column!")
    except ValueError as e:
        assert "EVALUATION INTEGRITY ERROR" in str(e)
        print("[PASS] 4. CandidateScorer strict no-label integrity guard verified.")

    unlabeled_eval = te_df.drop(columns=["label"])
    scored_df, latency_ms = scorer.score_candidates(unlabeled_eval, strict_no_labels=True)
    assert list(scored_df.columns) == ["trip_id", "obs_idx", "arc_id", "pred_prob", "is_top1"]
    assert len(scored_df) == len(unlabeled_eval)
    assert scored_df["pred_prob"].between(0.0, 1.0).all()
    assert int(scored_df["is_top1"].sum()) == unlabeled_eval.groupby(["trip_id", "obs_idx"]).ngroups
    print(f"[PASS] 5. CandidateScorer inference verified ({len(scored_df)} candidates in {latency_ms:.2f} ms).")

    print("\n--- Synthetic Verification Comparison Table ---")
    cols = [
        "Model",
        "Algorithm Family",
        "CV Macro F1-score",
        "Top-1 Accuracy",
        "Route Continuity",
        "Student-Test Macro F1-score",
        "Log Loss",
        "Inference Time (s)",
    ]
    print(res_p2["comparison_df"][cols].to_string(index=False))
    print("\n--- Synthetic Untuned vs Tuned ---")
    print(res_p2["bayesian_opt"]["untuned_vs_tuned_df"].to_string(index=False))

    shutil.rmtree(tmp_dir)
    print("\nALL CHECKS PASSED CLEANLY!")


if __name__ == "__main__":
    main()
