
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

TRIP_DIR = (
    BASE_DIR / "Student folder"
    / "Student folder" / "train" / "00000001"
)

# Load generated candidates
candidates = pd.read_csv(
    BASE_DIR / "candidate_features.csv"
)

# Load ground-truth route edge IDs
route = pd.read_csv(
    TRIP_DIR / "00000001.route",
    header=None
)[0].astype(int)

route_edges = set(route.tolist())
candidate_edges = set(candidates["edge_id"].tolist())

# Check how many reference edges appear
# anywhere in the generated candidate set.
covered = route_edges.intersection(candidate_edges)
missing = route_edges - candidate_edges

coverage = len(covered) / len(route_edges) * 100

print("\n========== CANDIDATE COVERAGE ==========")

print("Reference route edges:", len(route_edges))
print("Unique candidate edges:", len(candidate_edges))
print("Reference edges found:", len(covered))
print("Reference edges missing:", len(missing))

print(f"\nCandidate coverage: {coverage:.2f}%")

if missing:
    print("\nFirst missing reference edges:")
    print(list(missing)[:20])

print("\n========================================")
print("Coverage check completed.")
