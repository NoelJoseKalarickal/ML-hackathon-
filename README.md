# PathSnap — Distributed Map-Matching & Route Reconstruction

PES University — UE24CS352A: Machine Learning Hackathon

## Repository Structure (`main`)

```text
ML-hackathon-/
├── preprocessing/               # Person 1: Raw data loading, GPS cleaning, validation
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── gps_cleaner.py
│   └── data_validation.py
├── features/                    # Person 2: Candidate generation, feature engineering, alignment
│   ├── feature_engineering.py
│   ├── label_alignment.py
│   ├── process_all_trips.py
│   ├── check_candidate_coverage.py
│   └── inspect_route.py
├── models/                      # Person 3: 5-model trip-grouped CV, Bayesian HPO, inference
│   ├── train_model.py           # Main entry point: train_model(...)
│   ├── model_registry.py        # 5 families + DirectedBearingTransformer + BoundedKNN
│   ├── bayesian_opt.py          # Gaussian Process (Matérn 5/2) + Expected Improvement
│   ├── metrics.py               # Candidate, observation Top-1, continuity & sequence metrics
│   ├── plotting.py              # Report Sections 3.3 & 4 figures (genuine curves only)
│   ├── inference.py             # CandidateScorer for FastAPI + HMM/Viterbi (zero-leakage)
│   ├── data_loader.py
│   └── config.py
├── evaluation/                  # Person 4: Group-aware CV, metrics, comparison & logging
├── reconstruction/              # Person 4: HMM & Viterbi route reconstruction
├── integration/                 # Person 4: End-to-end pipeline integration
├── deployment/                  # Person 4: FastAPI backend & evaluator UI
├── artifacts/                   # Exported final model bundle (271 KB HistGradientBoosting)
│   ├── final_model.joblib
│   ├── feature_columns.json
│   └── model_metadata.json
├── experiments/
│   ├── comparison_tables/       # model_comparison.csv (Report Table 3.1)
│   ├── logs/                    # cv_fold_details.csv, detailed_model_metrics.json, histories
│   └── results/                 # Bayesian trials, untuned vs tuned, HMM probs, 16 PNG plots
├── tests/
│   ├── test_group_split.py
│   └── verify_pipeline_synthetic.py
├── DATA_CONTRACT.md             # Shared team schema & interface contract
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
PYTHONPATH=. python3 tests/test_group_split.py
PYTHONPATH=. python3 tests/verify_pipeline_synthetic.py
```

## Model Training & Evaluation Summary (80 Train / 10 Student-Test Trips)

| Model | Algorithm Family | 5-Fold CV Macro F1 | Student-Test Top-1 Acc | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Linear | 0.8229 ± 0.0177 | 0.7972 | 0.8937 | 0.8605 | 0.1242 | 0.024s |
| K-Nearest Neighbors | Instance-based | 0.8254 ± 0.0164 | 0.7879 | 0.8819 | 0.8603 | 0.1709 | 5.488s |
| Decision Tree | Tree-based | 0.8314 ± 0.0366 | 0.8015 | 0.8937 | 0.8843 | 0.1130 | 0.025s |
| Random Forest | Ensemble | 0.8539 ± 0.0235 | 0.8050 | 0.8979 | 0.8886 | 0.0952 | 0.204s |
| **HistGradientBoosting** | **Boosting** | **0.8560 ± 0.0260** | **0.8042** | **0.8975** | **0.8890** | **0.0958** | **0.204s** |

After 20-trial Gaussian Process + Expected Improvement Bayesian HPO (`artifacts/final_model.joblib`, 271 KB):
- **Student-Test Candidate Accuracy**: `96.03%` | **Observation Top-1 Accuracy**: `80.45%` | **Route Continuity**: `89.95%` | **ROC-AUC**: `0.9827` | **Latency**: `24.5 ms/trip`.

## Quick Inference Usage (`models/inference.py`)

```python
from models.inference import CandidateScorer

scorer = CandidateScorer(artifacts_dir="artifacts")
scored_df, latency_ms = scorer.score_candidates(unlabeled_candidates_df)
# scored_df columns: trip_id, obs_idx, arc_id, pred_prob, is_top1
```
