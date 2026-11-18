
from pathlib import Path
import pandas as pd
import numpy as np

from data_loader import load_and_preprocess_data


def validate_trip(trip):

    gps = pd.DataFrame(trip["gps"])
    nodes = pd.DataFrame(trip["nodes"])
    arcs = pd.DataFrame(trip["arcs"])
    route = trip["route"]

    report = {
        "trip_id": trip["trip_id"],
        "gps_count": len(gps),
        "missing_gps": 0,
        "invalid_coordinates": 0,
        "duplicate_gps_rows": 0,
        "duplicate_timestamps": 0,
        "non_increasing_timestamps": 0,
        "irregular_gaps": 0,
        "invalid_node_coordinates": 0,
        "invalid_arc_connections": 0,
        "invalid_route_ids": 0
    }

    # GPS validation
    if not gps.empty:

        report["missing_gps"] = int(
            gps.isna().any(axis=1).sum()
        )

        valid = (
            gps["longitude"].between(-180, 180)
            & gps["latitude"].between(-90, 90)
            & np.isfinite(gps["longitude"])
            & np.isfinite(gps["latitude"])
            & np.isfinite(gps["timestamp"])
        )

        report["invalid_coordinates"] = int((~valid).sum())

        report["duplicate_gps_rows"] = int(
            gps.duplicated().sum()
        )

        report["duplicate_timestamps"] = int(
            gps["timestamp"].duplicated().sum()
        )

        # Preserve original order when checking timestamps
        differences = gps["timestamp"].diff().dropna()

        report["non_increasing_timestamps"] = int(
            (differences <= 0).sum()
        )

        report["irregular_gaps"] = int(
            (differences != 1.0).sum()
        )

    # Node coordinate validation
    if not nodes.empty:

        valid_nodes = (
            nodes["longitude"].between(-180, 180)
            & nodes["latitude"].between(-90, 90)
            & np.isfinite(nodes["longitude"])
            & np.isfinite(nodes["latitude"])
        )

        report["invalid_node_coordinates"] = int(
            (~valid_nodes).sum()
        )

    # Arc validation: node IDs are zero-based row indices
    for arc in trip["arcs"]:

        source = arc["from_node_id"]
        destination = arc["to_node_id"]

        if not (
            0 <= source < len(nodes)
            and 0 <= destination < len(nodes)
        ):
            report["invalid_arc_connections"] += 1

    # Route validation: arc IDs are zero-based row indices
    for arc_id in route:

        if not (0 <= arc_id < len(arcs)):
            report["invalid_route_ids"] += 1

    return report


def main():

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "Student folder"

    output_dir = base_dir / "results" / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    trips = load_and_preprocess_data(data_dir)

    reports = [validate_trip(trip) for trip in trips]

    report_df = pd.DataFrame(reports)

    output_file = output_dir / "validation_report.csv"

    report_df.to_csv(output_file, index=False)

    print("\nVALIDATION SUMMARY")
    print(report_df.to_string(index=False))

    print("\nTotal trips inspected:", len(trips))
    print("Report saved to:", output_file)


if __name__ == "__main__":
    main()
