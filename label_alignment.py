
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

TRIP_ID = "00000001"

TRIP_DIR = (
    BASE_DIR
    / "Student folder"
    / "Student folder"
    / "train"
    / TRIP_ID
)

CANDIDATE_FILE = BASE_DIR / "candidate_features.csv"

OUTPUT_FILE = BASE_DIR / "candidate_features_labeled.csv"

EARTH_RADIUS = 6371000

# Alignment parameters
EMISSION_SCALE = 12.0
PROGRESS_SCALE = 25.0
MAX_ADVANCE = 100


# ==========================================
# DISTANCE FUNCTION
# ==========================================

def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * EARTH_RADIUS
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


# ==========================================
# LOAD DATA
# ==========================================

print("Loading data...")

track = np.loadtxt(
    TRIP_DIR / f"{TRIP_ID}.track"
)

nodes = np.loadtxt(
    TRIP_DIR / f"{TRIP_ID}.nodes"
)

arcs = np.loadtxt(
    TRIP_DIR / f"{TRIP_ID}.arcs",
    dtype=int
)

route = np.loadtxt(
    TRIP_DIR / f"{TRIP_ID}.route",
    dtype=int
)

candidates = pd.read_csv(CANDIDATE_FILE)

gps_count = len(track)
route_count = len(route)

print("GPS observations:", gps_count)
print("Reference route edges:", route_count)


# ==========================================
# VALIDATE ROUTE
# ==========================================

if np.any(route < 0) or np.any(route >= len(arcs)):
    raise ValueError("Invalid arc ID in route file")

route_arcs = arcs[route]

if np.any(route_arcs < 0) or np.any(route_arcs >= len(nodes)):
    raise ValueError("Invalid node ID in route")

for i in range(len(route_arcs) - 1):

    if route_arcs[i, 1] != route_arcs[i + 1, 0]:
        raise ValueError(
            f"Route discontinuity at position {i}"
        )

print("Route connectivity validated.")


# ==========================================
# CONVERT COORDINATES TO METRES
# ==========================================

print("Preparing route geometry...")

reference_lat = np.mean(nodes[:, 1])

scale_x = (
    111320 * np.cos(np.radians(reference_lat))
)

scale_y = 111320

node_x = nodes[:, 0] * scale_x
node_y = nodes[:, 1] * scale_y

gps_x = track[:, 0] * scale_x
gps_y = track[:, 1] * scale_y

gps_xy = np.column_stack((gps_x, gps_y))

# Get the actual route edges in chronological order

route_nodes = arcs[route]

route_start = nodes[route_nodes[:, 0], :2]
route_end = nodes[route_nodes[:, 1], :2]

route_start_xy = np.column_stack((
    route_start[:, 0] * scale_x,
    route_start[:, 1] * scale_y
))

route_end_xy = np.column_stack((
    route_end[:, 0] * scale_x,
    route_end[:, 1] * scale_y
))

vectors = route_end_xy - route_start_xy

segment_lengths_xy = np.linalg.norm(
    vectors, axis=1
)

# Geographic segment lengths in metres

route_lengths = haversine(
    route_start[:, 0],
    route_start[:, 1],
    route_end[:, 0],
    route_end[:, 1]
)

cumulative_lengths = np.concatenate((
    [0],
    np.cumsum(route_lengths[:-1])
))


# ==========================================
# GPS DISTANCE BETWEEN OBSERVATIONS
# ==========================================

gps_steps = np.zeros(gps_count)

gps_steps[1:] = haversine(
    track[:-1, 0],
    track[:-1, 1],
    track[1:, 0],
    track[1:, 1]
)


# ==========================================
# CALCULATE GPS-TO-ROUTE DISTANCES
# ==========================================

print("Calculating GPS-to-route distances...")

distance_matrix = np.zeros(
    (gps_count, route_count)
)

projection_matrix = np.zeros(
    (gps_count, route_count)
)

for i, point in enumerate(gps_xy):

    relative = point - route_start_xy

    denominator = np.sum(
        vectors * vectors,
        axis=1
    )

    projection = np.divide(
        np.sum(relative * vectors, axis=1),
        denominator,
        out=np.zeros(route_count),
        where=denominator > 0
    )

    projection = np.clip(
        projection, 0, 1
    )

    closest_points = (
        route_start_xy
        + projection[:, None] * vectors
    )

    distances = np.linalg.norm(
        closest_points - point,
        axis=1
    )

    distance_matrix[i] = distances
    projection_matrix[i] = projection


# ==========================================
# ROUTE POSITION OF EACH GPS OBSERVATION
# ==========================================

route_positions = (
    cumulative_lengths[None, :]
    + projection_matrix * route_lengths[None, :]
)


# ==========================================
# SEQUENCE ALIGNMENT USING DYNAMIC PROGRAMMING
# ==========================================

print("Running sequence alignment...")

dp = np.full(
    (gps_count, route_count),
    np.inf
)

backtrack = np.full(
    (gps_count, route_count),
    -1,
    dtype=int
)

# First observation: initialise using
# geographic distance to each route edge.

dp[0] = (
    distance_matrix[0] / EMISSION_SCALE
)

# Process remaining GPS observations

for i in range(1, gps_count):

    for j in range(route_count):

        # Route index cannot move backwards.
        # Limit the number of route edges skipped
        # between consecutive GPS observations.

        lower = max(
            0,
            j - MAX_ADVANCE
        )

        previous_indices = np.arange(
            lower,
            j + 1
        )

        previous_positions = route_positions[
            i - 1, previous_indices
        ]

        current_position = route_positions[i, j]

        route_progress = (
            current_position - previous_positions
        )

        progress_error = np.abs(
            route_progress - gps_steps[i]
        )

        transition_cost = (
            progress_error / PROGRESS_SCALE
        )

        total_cost = (
            dp[i - 1, previous_indices]
            + transition_cost
        )

        best_position = np.argmin(total_cost)

        best_previous_index = (
            previous_indices[best_position]
        )

        dp[i, j] = (
            distance_matrix[i, j] / EMISSION_SCALE
            + total_cost[best_position]
        )

        backtrack[i, j] = best_previous_index


# ==========================================
# BACKTRACK OPTIMAL ROUTE ALIGNMENT
# ==========================================

print("Backtracking optimal alignment...")

aligned_route_indices = np.zeros(
    gps_count,
    dtype=int
)

aligned_route_indices[-1] = np.argmin(
    dp[-1]
)

for i in range(gps_count - 1, 0, -1):

    aligned_route_indices[i - 1] = (
        backtrack[
            i,
            aligned_route_indices[i]
        ]
    )

aligned_edge_ids = route[
    aligned_route_indices
]

aligned_distances = distance_matrix[
    np.arange(gps_count),
    aligned_route_indices
]


# ==========================================
# ADD LABELS TO CANDIDATES
# ==========================================

print("Generating training labels...")

candidates["aligned_route_index"] = (
    candidates["gps_point"]
    .map(dict(enumerate(aligned_route_indices)))
)

candidates["reference_edge_id"] = (
    candidates["gps_point"]
    .map(dict(enumerate(aligned_edge_ids)))
)

candidates["alignment_distance_m"] = (
    candidates["gps_point"]
    .map(dict(enumerate(aligned_distances)))
)

candidates["label"] = (
    candidates["edge_id"]
    == candidates["reference_edge_id"]
).astype(int)


# ==========================================
# EVALUATE CANDIDATE COVERAGE PER OBSERVATION
# ==========================================

point_hits = (
    candidates.groupby("gps_point")["label"]
    .max()
)

matched_points = int(point_hits.sum())

missed_points = gps_count - matched_points

coverage = (
    matched_points / gps_count * 100
)

candidate_miss_rate = (
    missed_points / gps_count * 100
)

# ==========================================
# SAVE LABELLED DATASET
# ==========================================

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("LABEL ALIGNMENT COMPLETED")
print("===================================")

print("Trip ID:", TRIP_ID)

print("GPS observations:", gps_count)

print("Reference route edges:", route_count)

print(
    "Aligned route observations:",
    len(aligned_route_indices)
)

print(
    "Positive candidate samples:",
    int(candidates["label"].sum())
)

print(
    "Negative candidate samples:",
    int((candidates["label"] == 0).sum())
)

print(
    "Observations with a positive candidate:",
    matched_points
)

print(
    "Observations without a positive candidate:",
    missed_points
)

print(
    f"Observation-level candidate coverage: {coverage:.2f}%"
)

print(
    f"Candidate miss rate: {candidate_miss_rate:.2f}%"
)

print(
    "Median alignment distance:",
    round(float(np.median(aligned_distances)), 2),
    "metres"
)

print(
    "95th percentile alignment distance:",
    round(float(np.percentile(aligned_distances, 95)), 2),
    "metres"
)

print("\nLabel distribution:")

print(candidates["label"].value_counts())

print("\nSaved:", OUTPUT_FILE)

print("\n===================================")
