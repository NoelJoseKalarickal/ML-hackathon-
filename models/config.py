"""Central configuration, constants, and reserved columns for PathSnap models (Person 3)."""

from pathlib import Path

RANDOM_SEED: int = 42
N_CV_SPLITS: int = 5

# Columns that must NEVER be passed as ML input features
RESERVED_NON_FEATURE_COLS: set[str] = {
    "trip_id",
    "obs_idx",
    "arc_id",
    "label",
    "target",
    "y",
    "true_arc_id",
    "gt_arc_id",
    "ground_truth_arc_id",
    "split",
    "partition",
    "pred_prob",
    "pred_label",
    "is_top1",
}

FORBIDDEN_INFERENCE_COLS: set[str] = {
    "label",
    "target",
    "y",
    "true_arc_id",
    "gt_arc_id",
    "ground_truth_arc_id",
}

REQUIRED_CANDIDATE_COLS: set[str] = {"trip_id", "obs_idx", "arc_id", "label"}
REQUIRED_INFERENCE_COLS: set[str] = {"trip_id", "obs_idx", "arc_id"}

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "experiments" / "results"
DEFAULT_LOGS_DIR: Path = REPO_ROOT / "experiments" / "logs"
DEFAULT_COMPARISON_DIR: Path = REPO_ROOT / "experiments" / "comparison_tables"
DEFAULT_PLOTS_DIR: Path = REPO_ROOT / "experiments" / "results" / "plots"
DEFAULT_ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
