from pathlib import Path
import json
import numpy as np

BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "fingerprints.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Anzahl Fingerprints: {len(data)}")


positions = [(s["position"]["x"], s["position"]["y"]) for s in data]
xs = [p[0] for p in positions]
ys = [p[1] for p in positions]

print(f"Anzahl Fingerprints:    {len(data)}")
print(f"X-Bereich:              {min(xs):.2f}m bis {max(xs):.2f}m")
print(f"Y-Bereich:              {min(ys):.2f}m bis {max(ys):.2f}m")

# Wie viele einzigartige Positionen (gerundet auf 0.5m Raster)?
unique = set(
    (round(x * 2) / 2, round(y * 2) / 2)
    for x, y in positions
)
print(f"Einzigartige Positionen (0.5m Raster): {len(unique)}")
print(f"Durchschnittliche Samples pro Position: {len(data) / len(unique):.1f}")