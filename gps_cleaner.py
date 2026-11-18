
import pandas as pd
import numpy as np


def clean_gps_data(gps_data):

    df = pd.DataFrame(gps_data)

    original_count = len(df)

    df = df.dropna(
        subset=["latitude", "longitude", "timestamp"]
    ).copy()

    for col in ["latitude", "longitude", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["latitude", "longitude", "timestamp"]
    )

    valid = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
        & np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
        & np.isfinite(df["timestamp"])
    )

    df = df.loc[valid].drop_duplicates()

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
