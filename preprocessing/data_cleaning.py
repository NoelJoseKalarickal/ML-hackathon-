
from pathlib import Path
import pandas as pd
import numpy as np

from data_loader import load_and_preprocess_data


def clean_gps(gps_data):

    df = pd.DataFrame(gps_data)

    original_count = len(df)

    # Remove missing observations
    df = df.dropna(
        subset=["latitude", "longitude", "timestamp"]
    ).copy()

    # Convert columns to numeric
    for column in ["latitude", "longitude", "timestamp"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=["latitude", "longitude", "timestamp"]
    )

    # Remove invalid and non-finite coordinates/timestamps
    valid = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
        & np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
        & np.isfinite(df["timestamp"])
    )

    df = df.loc[valid].copy()

    # Remove exact duplicate observations
    df = df.drop_duplicates()

    # Do not reorder timestamps
    cleaned_gps = [
        {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "timestamp": float(row.timestamp)
        }
        for row in df.itertuples(index=False)
    ]

    removed = original_count - len(cleaned_gps)

    return cleaned_gps, removed


def main():

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "Student folder"

    output_dir = base_dir / "results" / "preprocessing" / "cleaned_gps"
    output_dir.mkdir(parents=True, exist_ok=True)

    trips = load_and_preprocess_data(data_dir)

    cleaning_summary = []

    for trip in trips:

        cleaned_gps, removed = clean_gps(trip["gps"])

        output_file = (
            output_dir / f"{trip['trip_id']}_gps.csv"
        )

        pd.DataFrame(cleaned_gps).to_csv(
            output_file, index=False
        )

        cleaning_summary.append({
            "trip_id": trip["trip_id"],
            "original_gps_count": len(trip["gps"]),
            "cleaned_gps_count": len(cleaned_gps),
            "removed_records": removed
        })

        print(
            f"{trip['trip_id']}: "
            f"{len(trip['gps'])} -> {len(cleaned_gps)}"
        )

    summary = pd.DataFrame(cleaning_summary)

    summary.to_csv(
        output_dir / "cleaning_summary.csv",
        index=False
    )

    print("\nGPS cleaning completed.")
    print("Trips processed:", len(trips))


if __name__ == "__main__":
    main()
