"""
PathSnap batch candidate generation + route alignment
-----------------------------------------------------
Run from the project root:
    python process_all_trips.py

Reads:
    Student folder/Student folder/train/<trip_id>/
    Student folder/Student folder/student_test/<trip_id>/

Each trip folder must contain:
    <trip_id>.track  (longitude latitude timestamp)
    <trip_id>.nodes  (longitude latitude)
    <trip_id>.arcs   (from_node_index to_node_index)
    <trip_id>.route  (arc_id, one per line)

Writes:
    data/processed/train_candidates.csv
    data/processed/student_test_candidates.csv
    data/processed/train_obs_meta.csv
    data/processed/student_test_obs_meta.csv

The labels are produced by monotonic sequence alignment against each
trip's supplied .route. They are reference-derived alignment labels,
not independent proof that every GPS observation has been mapped perfectly.
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("ERROR: scipy is required. Install it with:")
    print("    python -m pip install scipy")
    sys.exit(1)


# ----------------------------- CONFIGURATION -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "Student folder" / "Student folder"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

CANDIDATES_PER_OBSERVATION = 10

# Midpoint shortlist used before exact point-to-segment distance calculation.
# Increase this (e.g. 800) if candidate coverage is too low, at the cost of speed.
MIDPOINT_SHORTLIST = 500

# Sequence-alignment settings.
# The maximum number of route positions that alignment may advance per GPS row.
MAX_ROUTE_ADVANCE_PER_OBS = 20
PROGRESS_PENALTY_METRES = 25.0

# Keep train and student_test strictly separate in output files.
DATASETS = {
    "train": DATA_ROOT / "train",
    "student_test": DATA_ROOT / "student_test",
}


# ----------------------------- FILE READING ------------------------------

def load_matrix(path, min_columns):
    """Load whitespace-separated numeric rows, tolerating a single-row file."""
    arr = np.loadtxt(path, dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < min_columns:
        raise ValueError(
            f"{path.name} has {arr.shape[1]} columns; expected at least {min_columns}"
        )
    return arr


def load_trip(trip_dir):
    trip_id = trip_dir.name

    track = load_matrix(trip_dir / f"{trip_id}.track", 3)[:, :3]
    nodes = load_matrix(trip_dir / f"{trip_id}.nodes", 2)[:, :2]
    arcs_raw = load_matrix(trip_dir / f"{trip_id}.arcs", 2)[:, :2]
    route_raw = np.loadtxt(trip_dir / f"{trip_id}.route", dtype=np.int64)
    route = np.atleast_1d(route_raw).astype(np.int64)

    arcs = arcs_raw.astype(np.int64)

    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0 or len(route) == 0:
        raise ValueError(f"Empty track/nodes/arcs/route file in trip {trip_id}")

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Trip {trip_id}: .arcs contains an invalid node index")

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError(f"Trip {trip_id}: .route contains an invalid arc_id")

    return trip_id, track, nodes, arcs, route


# ----------------------------- GEOMETRY ----------------------------------

def project_lonlat(lonlat, lon0, lat0):
    """Local equirectangular projection; suitable for trip-scale distances."""
    lon = np.asarray(lonlat, dtype=float)[:, 0]
    lat = np.asarray(lonlat, dtype=float)[:, 1]
    x = (lon - lon0) * 111320.0 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 110540.0
    return np.column_stack((x, y))


def bearing_degrees(points):
    """Bearing from each point to the next; final point uses previous-to-current."""
    n = len(points)
    result = np.zeros(n, dtype=float)
    if n <= 1:
        return result

    delta = np.diff(points, axis=0)
    seg_bearing = (np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) + 360.0) % 360.0
    result[:-1] = seg_bearing
    result[-1] = seg_bearing[-1]
    return result


def haversine_distances_m(track):
    """Distance between consecutive GPS observations."""
    lon = np.deg2rad(track[:, 0])
    lat = np.deg2rad(track[:, 1])
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_segment_measurements(points, starts, ends):
    """
    Vectorized point-to-segment projection for one point against many segments.
    Returns perpendicular distances and projection ratios.
    """
    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    rel = points[:, None, :] - starts[None, :, :]
    numerator = np.einsum("ijk,jk->ij", rel, vectors)
    ratios = np.zeros_like(numerator, dtype=float)

    valid = lengths_sq > 1e-12
    ratios[:, valid] = numerator[:, valid] / lengths_sq[valid]
    ratios = np.clip(ratios, 0.0, 1.0)

    closest = starts[None, :, :] + ratios[:, :, None] * vectors[None, :, :]
    diff = points[:, None, :] - closest
    distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return distances, ratios


def build_network_geometry(nodes_xy, arcs):
    starts = nodes_xy[arcs[:, 0]]
    ends = nodes_xy[arcs[:, 1]]
    vectors = ends - starts
    lengths = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    midpoints = (starts + ends) / 2.0
    road_bearings = (
        np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360.0
    ) % 360.0

    degree = np.bincount(arcs.reshape(-1), minlength=len(nodes_xy))
    node_degree = degree[arcs[:, 0]] + degree[arcs[:, 1]]

    return starts, ends, lengths, midpoints, road_bearings, node_degree


def angle_difference_degrees(a, b):
    """Smallest bidirectional angular difference, in [0, 180]."""
    diff = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return np.minimum(diff, 180.0 - diff)


# ----------------------------- ALIGNMENT ---------------------------------

def align_gps_to_route(track_xy, route_starts, route_ends):
    """
    Monotonic dynamic-programming alignment of GPS observations to ordered
    route edges. The output is a route-position index for every GPS row.
    """
    n_obs = len(track_xy)
    n_route = len(route_starts)

    # Emission cost: distance from GPS point to each edge in the reference route.
    emission, _ = point_segment_measurements(
        track_xy, route_starts, route_ends
    )

    if n_obs == 1 or n_route == 1:
        return np.argmin(emission, axis=1).astype(np.int64), emission

    expected_advance = (n_route - 1) / max(1, n_obs - 1)
    max_advance = min(
        MAX_ROUTE_ADVANCE_PER_OBS,
        max(1, n_route - 1)
    )

    inf = np.inf
    dp_prev = emission[0].copy()
    # Encourage the first GPS point to align near the start of the route.
    dp_prev += np.arange(n_route, dtype=float) * PROGRESS_PENALTY_METRES

    back = np.full((n_obs, n_route), -1, dtype=np.int32)

    offsets = np.arange(max_advance + 1, dtype=np.int64)

    for i in range(1, n_obs):
        dp_curr = np.full(n_route, inf, dtype=float)
        parent = np.full(n_route, -1, dtype=np.int32)

        for advance in offsets:
            if advance == 0:
                prev_values = dp_prev
                curr_slice = slice(0, n_route)
                prev_slice = slice(0, n_route)
            else:
                if advance >= n_route:
                    continue
                curr_slice = slice(advance, n_route)
                prev_slice = slice(0, n_route - advance)
                prev_values = dp_prev[prev_slice]

            penalty = abs(float(advance) - expected_advance) * PROGRESS_PENALTY_METRES
            proposed = prev_values + penalty
            current_values = dp_curr[curr_slice]
            better = proposed < current_values

            if np.any(better):
                updated = current_values.copy()
                updated[better] = proposed[better]
                dp_curr[curr_slice] = updated

                parent_values = parent[curr_slice].copy()
                previous_indices = np.arange(n_route, dtype=np.int32)[prev_slice]
                parent_values[better] = previous_indices[better]
                parent[curr_slice] = parent_values

        dp_curr += emission[i]
        back[i] = parent
        dp_prev = dp_curr

    # Encourage the final GPS point to align near the end of the reference route.
    final_cost = dp_prev + (
        (n_route - 1 - np.arange(n_route)) * PROGRESS_PENALTY_METRES
    )
    route_pos = int(np.argmin(final_cost))

    aligned = np.empty(n_obs, dtype=np.int64)
    aligned[-1] = route_pos

    for i in range(n_obs - 1, 0, -1):
        previous = int(back[i, aligned[i]])
        if previous < 0:
            previous = aligned[i]
        aligned[i - 1] = previous

    return aligned, emission


# ----------------------------- CANDIDATES --------------------------------

def generate_candidates(
    trip_id, track, nodes, arcs, route, aligned_route_positions
):
    lon0 = float(np.mean(nodes[:, 0]))
    lat0 = float(np.mean(nodes[:, 1]))

    nodes_xy = project_lonlat(nodes, lon0, lat0)
    track_xy = project_lonlat(track[:, :2], lon0, lat0)

    starts, ends, lengths, midpoints, road_bearings, node_degree = (
        build_network_geometry(nodes_xy, arcs)
    )

    midpoint_tree = cKDTree(midpoints)
    shortlist_size = min(MIDPOINT_SHORTLIST, len(arcs))
    gps_bearings = bearing_degrees(track_xy)

    # GPS speed in m/s. Timestamp is the third .track column.
    segment_distances = haversine_distances_m(track)
    time_diffs = np.diff(track[:, 2])
    speeds_between = np.divide(
        segment_distances,
        time_diffs,
        out=np.zeros_like(segment_distances, dtype=float),
        where=time_diffs > 0,
    )
    gps_speeds = np.zeros(len(track), dtype=float)
    if len(track) > 1:
        gps_speeds[:-1] = speeds_between
        gps_speeds[-1] = speeds_between[-1]

    rows = []
    obs_meta = []

    for obs_idx, point in enumerate(track_xy):
        _, shortlist = midpoint_tree.query(point, k=shortlist_size)
        shortlist = np.atleast_1d(shortlist).astype(np.int64)

        local_dist, local_ratio = point_segment_measurements(
            point.reshape(1, 2),
            starts[shortlist],
            ends[shortlist],
        )
        local_dist = local_dist[0]
        local_ratio = local_ratio[0]

        order = np.argsort(local_dist)[: min(CANDIDATES_PER_OBSERVATION, len(shortlist))]
        selected_arc_ids = shortlist[order]
        selected_distances = local_dist[order]
        selected_ratios = local_ratio[order]

        min_dist = float(selected_distances[0])
        gps_heading = float(gps_bearings[obs_idx])

        true_route_pos = int(aligned_route_positions[obs_idx])
        true_arc_id = int(route[true_route_pos])

        candidate_found = int(np.any(selected_arc_ids == true_arc_id))
        obs_meta.append({
            "trip_id": str(trip_id),
            "obs_idx": int(obs_idx),
            "true_arc_id": true_arc_id,
            "candidate_found": candidate_found,
        })

        for rank, (arc_id, distance, ratio) in enumerate(
            zip(selected_arc_ids, selected_distances, selected_ratios), start=1
        ):
            arc_id = int(arc_id)
            heading_diff = float(
                angle_difference_degrees(gps_heading, road_bearings[arc_id])
            )

            rows.append({
                "trip_id": str(trip_id),
                "obs_idx": int(obs_idx),
                "arc_id": arc_id,
                "label": int(arc_id == true_arc_id),
                "perp_dist_m": float(distance),
                "heading_diff_deg": heading_diff,
                "cos_heading_diff": float(np.cos(np.deg2rad(heading_diff))),
                "speed_mps": float(gps_speeds[obs_idx]),
                "proj_ratio": float(ratio),
                "arc_length_m": float(lengths[arc_id]),
                "dist_rank": int(rank),
                "dist_diff_from_min": float(distance - min_dist),
                "node_degree": int(node_degree[arc_id]),
                "gps_bearing": gps_heading,
                "road_bearing": float(road_bearings[arc_id]),
            })

    candidates_df = pd.DataFrame(rows)
    obs_meta_df = pd.DataFrame(obs_meta)

    # Candidate-level diagnostic distance of the aligned reference edge.
    aligned_route_arc_ids = route[aligned_route_positions]
    aligned_starts = starts[aligned_route_arc_ids]
    aligned_ends = ends[aligned_route_arc_ids]
    aligned_distances, _ = point_segment_measurements(
        track_xy, aligned_starts, aligned_ends
    )
    per_obs_alignment_distance = np.diag(aligned_distances)

    diagnostics = {
        "gps_observations": len(track),
        "route_edges": len(route),
        "candidate_rows": len(candidates_df),
        "positive_samples": int(candidates_df["label"].sum()),
        "negative_samples": int((candidates_df["label"] == 0).sum()),
        "observations_with_positive": int(obs_meta_df["candidate_found"].sum()),
        "observations_without_positive": int((obs_meta_df["candidate_found"] == 0).sum()),
        "coverage_pct": float(obs_meta_df["candidate_found"].mean() * 100.0),
        "median_alignment_distance_m": float(np.median(per_obs_alignment_distance)),
        "p95_alignment_distance_m": float(np.percentile(per_obs_alignment_distance, 95)),
    }

    return candidates_df, obs_meta_df, diagnostics


# ----------------------------- DATASET RUNNER ----------------------------

def process_dataset(dataset_name, dataset_dir):
    trip_dirs = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )

    if not trip_dirs:
        raise RuntimeError(f"No trip folders found in {dataset_dir}")

    all_candidates = []
    all_obs_meta = []

    print(f"\n{'=' * 72}")
    print(f"PROCESSING {dataset_name.upper()} — {len(trip_dirs)} TRIPS")
    print(f"{'=' * 72}")

    for trip_number, trip_dir in enumerate(trip_dirs, start=1):
        print(f"\n[{trip_number}/{len(trip_dirs)}] Trip {trip_dir.name}")

        trip_id, track, nodes, arcs, route = load_trip(trip_dir)

        # Validate route connectivity before alignment.
        route_arcs = arcs[route]
        if len(route_arcs) > 1:
            discontinuities = int(
                np.sum(route_arcs[:-1, 1] != route_arcs[1:, 0])
            )
            if discontinuities:
                print(
                    f"WARNING: {discontinuities} route transitions are not "
                    "node-contiguous."
                )
        else:
            discontinuities = 0

        lon0 = float(np.mean(nodes[:, 0]))
        lat0 = float(np.mean(nodes[:, 1]))
        nodes_xy = project_lonlat(nodes, lon0, lat0)
        track_xy = project_lonlat(track[:, :2], lon0, lat0)

        starts, ends, _, _, _, _ = build_network_geometry(nodes_xy, arcs)
        route_starts = starts[route]
        route_ends = ends[route]

        aligned_positions, _ = align_gps_to_route(
            track_xy, route_starts, route_ends
        )

        candidates_df, obs_meta_df, diag = generate_candidates(
            trip_id, track, nodes, arcs, route, aligned_positions
        )

        all_candidates.append(candidates_df)
        all_obs_meta.append(obs_meta_df)

        print(
            f"  GPS observations: {diag['gps_observations']:,} | "
            f"route edges: {diag['route_edges']:,} | "
            f"candidate rows: {diag['candidate_rows']:,}"
        )
        print(
            f"  Positive: {diag['positive_samples']:,} | "
            f"negative: {diag['negative_samples']:,}"
        )
        print(
            f"  Observation coverage: {diag['coverage_pct']:.2f}% "
            f"({diag['observations_with_positive']}/"
            f"{diag['gps_observations']})"
        )
        print(
            f"  Alignment distance median / p95: "
            f"{diag['median_alignment_distance_m']:.2f} m / "
            f"{diag['p95_alignment_distance_m']:.2f} m"
        )
        if discontinuities:
            print(f"  Route connectivity warnings: {discontinuities}")

    candidates = pd.concat(all_candidates, ignore_index=True)
    obs_meta = pd.concat(all_obs_meta, ignore_index=True)

    # Ensure contract metadata types.
    candidates["trip_id"] = candidates["trip_id"].astype(str)
    candidates["obs_idx"] = candidates["obs_idx"].astype(np.int64)
    candidates["arc_id"] = candidates["arc_id"].astype(np.int64)
    candidates["label"] = candidates["label"].astype(np.int8)

    obs_meta["trip_id"] = obs_meta["trip_id"].astype(str)
    obs_meta["obs_idx"] = obs_meta["obs_idx"].astype(np.int64)
    obs_meta["true_arc_id"] = obs_meta["true_arc_id"].astype(np.int64)
    obs_meta["candidate_found"] = obs_meta["candidate_found"].astype(np.int8)

    return candidates, obs_meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_dir in DATASETS.items():
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_candidates, train_obs_meta = process_dataset(
        "train", DATASETS["train"]
    )
    student_candidates, student_obs_meta = process_dataset(
        "student_test", DATASETS["student_test"]
    )

    outputs = {
        "train_candidates.csv": train_candidates,
        "student_test_candidates.csv": student_candidates,
        "train_obs_meta.csv": train_obs_meta,
        "student_test_obs_meta.csv": student_obs_meta,
    }

    print(f"\n{'=' * 72}")
    print("SAVING CONTRACT FILES")
    print(f"{'=' * 72}")

    for filename, dataframe in outputs.items():
        output_path = OUTPUT_DIR / filename
        dataframe.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(f"  Rows: {len(dataframe):,} | Columns: {len(dataframe.columns)}")

    print("\nTRAIN SUMMARY")
    print(f"  Trips: {train_candidates['trip_id'].nunique()}")
    print(f"  Candidate samples: {len(train_candidates):,}")
    print(f"  Positive labels: {int(train_candidates['label'].sum()):,}")
    print(
        "  Observation candidate coverage: "
        f"{train_obs_meta['candidate_found'].mean() * 100:.2f}%"
    )

    print("\nSTUDENT TEST SUMMARY")
    print(f"  Trips: {student_candidates['trip_id'].nunique()}")
    print(f"  Candidate samples: {len(student_candidates):,}")
    print(f"  Positive labels: {int(student_candidates['label'].sum()):,}")
    print(
        "  Observation candidate coverage: "
        f"{student_obs_meta['candidate_found'].mean() * 100:.2f}%"
    )

    print("\nIMPORTANT:")
    print("- Use train_candidates.csv for model fitting and grouped CV.")
    print("- Use student_test_candidates.csv only for development evaluation.")
    print("- Do not use student_test labels to fit the model.")
    print("- Do not access or process evaluator_test.")
    print("- Exclude trip_id, obs_idx, arc_id, and label from model features.")
    print("\nBATCH PROCESSING COMPLETED.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
