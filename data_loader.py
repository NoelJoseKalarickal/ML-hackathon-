
from pathlib import Path
import pandas as pd
try:
    from .gps_cleaner import clean_gps_data
except ImportError:
    from gps_cleaner import clean_gps_data


def read_file(file_path, columns):
    """Read whitespace-separated files without headers."""

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns
    )

    return df


def load_gps(file_path):
    """Load GPS observations into a list of dictionaries."""

    df = read_file(
        file_path,
        ["longitude", "latitude", "timestamp"]
    )

    gps_data = []

    for row in df.itertuples(index=False):

        gps_data.append({
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "timestamp": float(row.timestamp)
        })

    return gps_data


def load_nodes(file_path):
    """Load road network nodes."""

    df = read_file(
        file_path,
        ["longitude", "latitude"]
    )

    nodes = []

    for row in df.itertuples(index=False):

        nodes.append({
            "latitude": float(row.latitude),
            "longitude": float(row.longitude)
        })

    return nodes


def load_arcs(file_path):
    """Load road network connections."""

    df = read_file(
        file_path,
        ["from_node_id", "to_node_id"]
    )

    return [
        {
            "from_node_id": int(row.from_node_id),
            "to_node_id": int(row.to_node_id)
        }
        for row in df.itertuples(index=False)
    ]


def load_route(file_path):
    """Load reference route arc IDs."""

    df = read_file(file_path, ["arc_id"])

    return df["arc_id"].astype(int).tolist()


def load_trip(trip_folder):
    """Load all files associated with one trip."""

    trip_folder = Path(trip_folder)
    trip_id = trip_folder.name

    gps_file = trip_folder / f"{trip_id}.track"
    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"
    route_file = trip_folder / f"{trip_id}.route"

    if not all([gps_file.exists(),
                nodes_file.exists(),
                arcs_file.exists(),
                route_file.exists()]):

        raise FileNotFoundError(
            f"Incomplete files for trip {trip_id}"
        )

    gps_data = load_gps(gps_file)

    cleaned_gps, removed_count = clean_gps_data(gps_data)

    return {
        "trip_id": trip_id,
        "gps": cleaned_gps,
        "nodes": load_nodes(nodes_file),
        "arcs": load_arcs(arcs_file),
        "route": load_route(route_file),
        "removed_gps_records": removed_count
    }


def load_and_preprocess_data(data_dir):
    """
    Load all 80 training trips.
    Returns a list containing one dictionary per trip.
    """

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    training_data = []

    for trip_folder in sorted(train_dir.iterdir()):

        if not trip_folder.is_dir():
            continue

        trip = load_trip(trip_folder)
        training_data.append(trip)

        print("Loaded:", trip["trip_id"])

    print("\nTotal training trips:", len(training_data))

    return training_data


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[2]

    DATA_DIR = BASE_DIR / "data" / "Student folder"

    training_data = load_and_preprocess_data(DATA_DIR)

    if training_data:

        first_trip = training_data[0]

        print("\nFirst GPS observation:")
        print(first_trip["gps"][0])

        print("\nTrip ID:", first_trip["trip_id"])
        print("GPS observations:", len(first_trip["gps"]))
        print("Nodes:", len(first_trip["nodes"]))
        print("Arcs:", len(first_trip["arcs"]))
        print("Route length:", len(first_trip["route"]))
