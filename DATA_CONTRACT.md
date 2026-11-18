# PathSnap — Team Data & Interface Contract (Persons 1 ↔ 2 ↔ 3 ↔ 4)

Aligned with the final `ML-hackathon-` repository layout so all 4 teammates integrate with zero merge conflicts or schema mismatches.

---

## 1. Strict Partition & Zero-Leakage Rules
- **Training (`data/train/`)**: 80 trips — used for trip-grouped CV, preprocessing fit, model training, and Bayesian HPO.
- **Student Testing (`data/student_test/`)**: 10 trips — used ONLY for development-stage model comparison, HMM evaluation, and untuned-vs-tuned comparison.
- **Evaluator Testing**: 10 trips — **NEVER** used for training, feature selection, tuning, or threshold selection. Ground-truth `.route` files remain server-side on Laptop 1 only.

---

## 2. Person 2 (`features/feature_engineering.py`) → Person 3 (`models/train_model.py`)

Person 2's `extract_features()` returns DataFrames (or exports `.csv` / `.parquet` to `data/processed/`):
- `train_df` (80 training trips)
- `student_test_df` (10 student-testing trips)

### Required Columns in Candidate DataFrames
| Column | Type | Meaning | Used as ML Feature? |
|---|---|---|---|
| `trip_id` | `str` or `int` | Trip folder ID (e.g., `"00000034"`) for **trip-grouped CV** | **NO** |
| `obs_idx` | `int` | 0-based GPS observation index within `trip_id` | **NO** |
| `arc_id` | `int` | Candidate road segment ID (0-based row index from `.arcs`) | **NO** |
| `label` | `int` (`0` or `1`) | `1` if `arc_id` matches the reference segment for `(trip_id, obs_idx)`, else `0` | **Target only** |

### Recommended Observation Metadata (for exact Candidate Miss Rate)
Optional `train_obs_meta` and `student_obs_meta` DataFrames/CSVs with one row per GPS observation:
| Column | Type | Meaning |
|---|---|---|
| `trip_id` | `str`/`int` | Trip ID |
| `obs_idx` | `int` | 0-based GPS observation index |
| `true_arc_id` | `int` | Reference `arc_id` (`-1` if unknown) |
| `candidate_found` | `int` (`0`/`1`) | `1` if `true_arc_id` was included in candidates, else `0` |

### Feature Columns & Projection Note for Person 2
- Any numeric column other than the metadata columns above is automatically used as a model feature (e.g., `gps_to_road_dist_m`, `gps_bearing`, `road_bearing`, `heading_diff_deg`, `cos_heading_diff`, `speed_mps`, `displacement_m`, `proj_ratio`, `dist_diff_from_min`, etc.).
- **Important**: The 90 trips span globally (`lon [-100.4°, 137.2°]`, `lat [-34.6°, 61.7°]`). Use a **per-trip** local metric projection (or per-trip UTM zone), not a single global UTM zone.
- **Do NOT scale features across train + student_test** — Person 3 fits median imputation and `StandardScaler` strictly inside each CV fold.

---

## 3. Person 3 (`models/`) → Person 4 (`evaluation/`, `reconstruction/`, `integration/`, `deployment/`)

### A. Training & 5-Model Comparison Entry Point (`models/train_model.py`)
```python
from models.train_model import train_model

# Modes: "rf_baseline" (Phase 1), "compare" (5 models), "full" (5 models + Bayesian HPO)
result = train_model(
    train_data=train_df,                 # pd.DataFrame or .csv/.parquet path
    student_test_data=student_test_df,   # pd.DataFrame or .csv/.parquet path
    mode="full",
    raw_data_dir="data",                 # reads .arcs & .route for Route Continuity / Sequence F1
    bo_trials=25,
)
```
Returns clean Python objects:
- `result["model"]`: final fitted sklearn `Pipeline` (tuned when `mode="full"`)
- `result["models"]`: dict of all 5 fitted untuned `Pipeline`s (`logistic_regression`, `k_nearest_neighbors`, `decision_tree`, `random_forest`, `hist_gradient_boosting`)
- `result["comparison_df"]`: `pd.DataFrame` (Report Table 3.1 + Accuracy, Precision, Recall, F1, Log Loss, Top-1, Route Continuity, Inference Time)
- `result["cv_folds_df"]`: `pd.DataFrame` of per-fold trip-grouped CV results
- `result["predictions_df"]`: `pd.DataFrame` (`trip_id`, `obs_idx`, `arc_id`, `pred_prob`, `is_top1`) — ready for `reconstruction/hmm.py` & `viterbi.py`
- `result["bayesian_opt"]`: dict containing `trials_df`, `untuned_vs_tuned_df`, `best_params`, `summary`
- `result["metrics"]`: nested dict with full candidate, observation, continuity, sequence, confusion matrix, and overfitting diagnostics

### B. Saved Files Mapped to Person 4's `experiments/` & `artifacts/` Layout
- `experiments/comparison_tables/model_comparison.csv`
- `experiments/logs/cv_fold_details.csv`, `detailed_model_metrics.json`, `training_histories.json`, `dataset_summary.json`
- `experiments/results/student_test_predictions_<model>.csv`, `bayesian_opt_trials.csv`, `untuned_vs_tuned_comparison.csv`, `final_model_student_test_predictions.csv`, `plots/*.png`
- `artifacts/final_model.joblib`, `artifacts/feature_columns.json`, `artifacts/model_metadata.json`

### C. Live Inference for FastAPI & HMM/Viterbi (`deployment/main.py`, `reconstruction/`)
```python
from models.inference import CandidateScorer

scorer = CandidateScorer(artifacts_dir="artifacts")
scored_df, latency_ms = scorer.score_candidates(unlabeled_candidates_df)
# scored_df preserves (trip_id, obs_idx, arc_id) and returns:
#   - pred_prob: calibrated P(label = 1 | features) for HMM emission probabilities
#   - is_top1:   1 for highest-probability candidate per (trip_id, obs_idx), else 0
```
`CandidateScorer` raises a `ValueError` if any ground-truth column (`label`, `true_arc_id`, `gt_arc_id`) enters live inference.
