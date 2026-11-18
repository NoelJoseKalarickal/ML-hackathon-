"""PathSnap Person 3 Modeling Package.

Primary team interface:
    from models.train_model import train_model
    from models.inference import CandidateScorer
"""

from models.bayesian_opt import run_bayesian_optimization
from models.inference import CandidateScorer
from models.metrics import evaluate_predictions
from models.plotting import generate_all_plots
from models.train_model import train_model

__all__ = [
    "train_model",
    "run_bayesian_optimization",
    "CandidateScorer",
    "evaluate_predictions",
    "generate_all_plots",
]
