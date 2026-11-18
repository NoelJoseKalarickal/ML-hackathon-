
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import math

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "Student folder"
    / "Student folder"
    / "train"
    / "00000001"
)

TRIP_ID = "00000001"

K = 10
SEARCH_K = 200

EARTH_RADIUS = 6371000


# ==========================================
# LOAD DATA
# ==========================================

print("Loading trip data...")

track = np.loadtxt(DATA_DIR / f"{TRIP_ID}.track")
nodes = np.loadtxt(DATA_DIR / f"{TRIP_ID}.nodes")
arcs = np.loadtxt(
    DATA_DIR / f"{TRIP_ID}.arcs",
    dtype=int
)

# Validate input shapes
if track.ndim != 2 or track.shape[1] < 3:
    raise ValueError("Invalid GPS track format")

if nodes.ndim != 2 or nodes.shape[1] < 2:
    raise ValueError("Invalid nodes format")

if arcs.ndim != 2 or arcs.shape[1] < 2:
    raise ValueError("Invalid arcs format")

gps_lon = track[:, 0]
gps_lat = track[:, 1]
timestamps = track[:, 2]

start_nodes = arcs[:, 0]
end_nodes = arcs[:, 1]

if (
    np.any(start_nodes < 0)
    or np.any(end_nodes < 0)
    or np.any(start_nodes >= len(nodes))
    or np.any(end_nodes >= len(nodes))
):
    raise ValueError("Invalid node index in arcs")

if not (
    np.isfinite(track[:, :3]).all()
    and np.isfinite(nodes[:, :2]).all()
):
    raise ValueError("Non-finite coordinates or timestamps")

if np.any(np.abs(gps_lat) > 90):
    raise ValueError("Invalid GPS latitude")

if np.any(np.abs(nodes[:, 1]) > 90):
    raise ValueError("Invalid road node latitude")

if len(arcs) == 0:
    raise ValueError("No road segments found")

print("GPS observations:", len(track))
print("Road segments:", len(arcs))


# ==========================================
# DISTANCE FUNCTIONS
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

    return 2 * EARTH_RADIUS * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def bearing(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    return (
        np.degrees(np.arctan2(y, x)) + 360
    ) % 360


def angle_difference(a, b):

    return abs(
        (a - b + 180) % 360 - 180
    )


# ==========================================
# ROAD NETWORK PREPARATION
# ==========================================

print("Preparing road network...")

reference_lat = np.mean(nodes[:, 1])

scale_x = (
    111320
    * np.cos(np.radians(reference_lat))
)

scale_y = 111320

node_x = nodes[:, 0] * scale_x
node_y = nodes[:, 1] * scale_y

mid_x = (
    node_x[start_nodes]
    + node_x[end_nodes]
) / 2

mid_y = (
    node_y[start_nodes]
    + node_y[end_nodes]
) / 2

midpoints = np.column_stack(
    (mid_x, mid_y)
)

tree = cKDTree(midpoints)


# ==========================================
# GPS MOVEMENT FEATURES
# ==========================================

print("Calculating GPS movement features...")

gps_distance = np.zeros(len(track))
gps_speed = np.zeros(len(track))

gps_bearing = np.full(
    len(track),
    np.nan
)

for i in range(1, len(track)):

    gps_distance[i] = haversine(
        gps_lon[i - 1],
        gps_lat[i - 1],
        gps_lon[i],
        gps_lat[i]
    )

    time_diff = (
        timestamps[i]
        - timestamps[i - 1]
    )

    if time_diff > 0:

        gps_speed[i] = (
            gps_distance[i] / time_diff
        )

        gps_bearing[i] = bearing(
            gps_lon[i - 1],
            gps_lat[i - 1],
            gps_lon[i],
            gps_lat[i]
        )

# First point has no previous observation.
# Use the next observation to estimate its direction.

if len(track) > 1:

    gps_bearing[0] = bearing(
        gps_lon[0],
        gps_lat[0],
        gps_lon[1],
        gps_lat[1]
    )

    time_diff = timestamps[1] - timestamps[0]

    if time_diff > 0:

        gps_speed[0] = (
            gps_distance[1] / time_diff
        )


# ==========================================
# IMPROVED CANDIDATE GENERATION
# ==========================================

print("Generating road candidates...")

rows = []

for i in range(len(track)):

    point_x = gps_lon[i] * scale_x
    point_y = gps_lat[i] * scale_y

    _, nearby_ids = tree.query(
        [point_x, point_y],
        k=min(SEARCH_K, len(arcs))
    )

    nearby_ids = np.atleast_1d(
        nearby_ids
    )

    point_candidates = []

    for edge_id in nearby_ids:

        start = start_nodes[edge_id]
        end = end_nodes[edge_id]

        lon1, lat1 = nodes[start, :2]
        lon2, lat2 = nodes[end, :2]

        ax = lon1 * scale_x
        ay = lat1 * scale_y

        bx = lon2 * scale_x
        by = lat2 * scale_y

        dx = bx - ax
        dy = by - ay

        px = point_x - ax
        py = point_y - ay

        denominator = (
            dx * dx + dy * dy
        )

        if denominator > 0:

            t = np.clip(
                (px * dx + py * dy)
                / denominator,
                0,
                1
            )

        else:

            t = 0

        closest_x = ax + t * dx
        closest_y = ay + t * dy

        distance_to_road = math.hypot(
            point_x - closest_x,
            point_y - closest_y
        )

        road_length = haversine(
            lon1,
            lat1,
            lon2,
            lat2
        )

        road_direction = bearing(
            lon1,
            lat1,
            lon2,
            lat2
        )

        # Compare both directions because
        # roads can be traversed either way.

        if np.isfinite(gps_bearing[i]):

            forward_diff = angle_difference(
                gps_bearing[i],
                road_direction
            )

            reverse_diff = angle_difference(
                gps_bearing[i],
                (road_direction + 180) % 360
            )

            direction_difference = min(
                forward_diff,
                reverse_diff
            )

        else:

            direction_difference = np.nan

        point_candidates.append({

            "edge_id": int(edge_id),

            "distance_to_road": distance_to_road,

            "gps_distance": gps_distance[i],

            "gps_speed": gps_speed[i],

            "gps_bearing": gps_bearing[i],

            "road_bearing": road_direction,

            "bearing_difference": direction_difference,

            "road_length": road_length

        })

    # Sort by actual point-to-segment distance

    point_candidates.sort(
        key=lambda x: x["distance_to_road"]
    )

    # Keep the 10 closest roads

    for rank, candidate in enumerate(
        point_candidates[:K],
        start=1
    ):

        rows.append({

            "trip_id": TRIP_ID,

            "gps_point": i,

            "candidate_rank": rank,

            **candidate

        })


# ==========================================
# SAVE FEATURE DATASET
# ==========================================

print("Creating feature dataset...")

df = pd.DataFrame(rows)

output_path = (
    BASE_DIR / "candidate_features.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# RESULTS
# ==========================================

print("\n================================")
print("FEATURE ENGINEERING COMPLETED")
print("================================")

print("Trip ID:", TRIP_ID)

print("GPS observations:", len(track))

print("Road segments:", len(arcs))

print("Candidate samples:", len(df))

print("Features:", len(df.columns))

print("\nMissing values:")

print(df.isnull().sum())

print("\nSample output:")

print(df.head())

print("\nSaved file:", output_path)
