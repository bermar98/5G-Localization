'''import pandas as pd
import numpy as np

df = pd.read_csv("austria.csv", header=None, names=[
    "radio", "mcc", "mnc", "lac", "cell_id",
    "unit", "lon", "lat", "range", "samples",
    "changeable", "created", "updated", "average_signal"
])

# Deine Serving Cell Position als Referenz
ref_lat, ref_lon = 47.2646, 11.2628

# Alle T-Mobile LTE Zellen in der Nähe
tmobile = df[(df["mcc"] == 232) & (df["mnc"] == 3) & (df["radio"] == "LTE")]

nearby = tmobile[
    (tmobile["lat"].between(ref_lat - 0.05, ref_lat + 0.05)) &
    (tmobile["lon"].between(ref_lon - 0.05, ref_lon + 0.05))
].copy()

# Distanz zur Serving Cell berechnen (in Metern)
nearby["dist_m"] = np.sqrt(
    ((nearby["lat"] - ref_lat) * 111320) ** 2 +
    ((nearby["lon"] - ref_lon) * 111320 * np.cos(np.radians(ref_lat))) ** 2
)

# Nach Distanz sortieren
nearby = nearby.sort_values("dist_m")

print(nearby[["cell_id", "lat", "lon", "dist_m", "range"]].to_string())'''

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "austria.csv"

df = pd.read_csv("austria.csv", header=None, names=[
    "radio", "mcc", "mnc", "lac", "cell_id",
    "unit", "lon", "lat", "range", "samples",
    "changeable", "created", "updated", "average_signal"
])

tmobile = df[
    (df["mcc"] == 232) &
    (df["mnc"] == 3) &
    (df["radio"] == "LTE")
]

ref_lat, ref_lon = 47.2646, 11.2628

nearby = tmobile[
    (tmobile["lat"].between(ref_lat - 0.05, ref_lat + 0.05)) &
    (tmobile["lon"].between(ref_lon - 0.05, ref_lon + 0.05))
].copy()

nearby["dist_m"] = np.sqrt(
    ((nearby["lat"] - ref_lat) * 111320) ** 2 +
    ((nearby["lon"] - ref_lon) * 111320 * np.cos(np.radians(ref_lat))) ** 2
)

# unit Spalte = PCI in manchen Einträgen?
print("Eindeutige 'unit' Werte in der Nähe:")
print(sorted(nearby["unit"].unique()))

print("\nZellen sortiert nach Distanz:")
print(nearby[["cell_id", "unit", "lat", "lon", "dist_m"]].sort_values("dist_m").head(20).to_string())