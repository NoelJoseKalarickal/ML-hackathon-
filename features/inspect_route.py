
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent

TRIP = (
    BASE / "Student folder"
    / "Student folder" / "train" / "00000001"
)

track = np.loadtxt(TRIP / "00000001.track")
nodes = np.loadtxt(TRIP / "00000001.nodes")
arcs = np.loadtxt(TRIP / "00000001.arcs", dtype=int)
route = np.loadtxt(TRIP / "00000001.route", dtype=int)

print("GPS observations:", len(track))
print("Route edges:", len(route))

# Convert route arc IDs to their endpoint node IDs
route_arcs = arcs[route]

print("\nFirst 10 route edges:")
for i, edge in enumerate(route_arcs[:10]):
    print(i, "Start node:", edge[0], "End node:", edge[1])

# Check route continuity
continuous = 0
for i in range(len(route_arcs) - 1):
    if route_arcs[i][1] == route_arcs[i + 1][0]:
        continuous += 1

print("\nContinuous transitions:", continuous)
print("Total transitions:", len(route_arcs) - 1)

# Check GPS timestamps
time_diffs = np.diff(track[:, 2])

print("\nTimestamp differences:")
print("Minimum:", time_diffs.min())
print("Maximum:", time_diffs.max())
print("Mean:", time_diffs.mean())

# Check route edge lengths
start = route_arcs[:, 0]
end = route_arcs[:, 1]

lon1, lat1 = nodes[start, 0], nodes[start, 1]
lon2, lat2 = nodes[end, 0], nodes[end, 1]

print("\nRoute start coordinates:", nodes[start[0], :2])
print("Route end coordinates:", nodes[end[-1], :2])

print("\nDiagnostic completed.")
