# wifi_radio_map.py
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, ConvexHull

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
FINGERPRINTS_PATH = DATA_PATH / "fingerprints.json"
AUGMENTED_PATH = DATA_PATH / "fingerprints_augmented.json"
MAPS_DIR = DATA_PATH / "radio_maps"

# Mindestanzahl an Messpunkten, die für eine BSSID vorliegen müssen, damit
# überhaupt interpoliert wird (weniger als 3-4 Punkte lassen keine
# sinnvolle 2D-Interpolation zu).
MIN_OBSERVATIONS = 4

# Sentinel-Wert für "BSSID an dieser Position nicht sichtbar" -- konsistent
# mit DatasetBuilder.SENTINEL_RSSI aus der ML-Pipeline.
SENTINEL_RSSI = -100


class RadioMapBuilder:

    def __init__(self, data, min_observations=MIN_OBSERVATIONS):
        """
        :param data: Liste von Fingerprint-Samples, wie aus fingerprints.json
                      geladen (siehe DatasetBuilder für das erwartete Format).
        :param min_observations: Mindestanzahl Messpunkte pro BSSID, um sie
                      zu interpolieren.
        """
        self.min_observations = min_observations
        self.samples = self._extract_samples(data)
        self.bssid_points = self._group_by_bssid(self.samples)
        self.interpolators = {}   # bssid -> LinearNDInterpolator
        self.all_positions = np.array([[s["x"], s["y"]] for s in self.samples])
        self.hull_delaunay = Delaunay(self.all_positions) if len(self.samples) >= 3 else None

    @staticmethod
    def _extract_samples(data):
        samples = []
        for entry in data:
            if "position" not in entry or "fingerprint" not in entry:
                continue
            if entry["position"] is None or entry["fingerprint"] is None:
                continue

            x, y = entry["position"]["x"], entry["position"]["y"]
            for bssid, values in entry["fingerprint"].items():
                normalized = bssid.upper().rstrip(":")
                samples.append({
                    "bssid": normalized,
                    "x": x,
                    "y": y,
                    "rssi": values["rssi"],
                })
        return samples

    @staticmethod
    def _group_by_bssid(samples):
        grouped = {}
        for s in samples:
            grouped.setdefault(s["bssid"], []).append((s["x"], s["y"], s["rssi"]))
        return grouped

    def build_interpolators(self):
        """
        Baut für jede BSSID mit ausreichend Messpunkten einen
        LinearNDInterpolator. BSSIDs mit zu wenigen Beobachtungen oder
        kollinearen Punkten (Triangulation nicht möglich) werden
        übersprungen und im Log vermerkt.
        """
        skipped = []

        for bssid, points in self.bssid_points.items():
            unique_points = {(x, y) for x, y, _ in points}

            if len(points) < self.min_observations or len(unique_points) < 3:
                skipped.append((bssid, len(points), "zu wenige Messpunkte"))
                continue

            xy = np.array([(x, y) for x, y, _ in points])
            rssi = np.array([r for _, _, r in points])

            try:
                interp = LinearNDInterpolator(xy, rssi)
                self.interpolators[bssid] = interp
            except Exception as e:
                skipped.append((bssid, len(points), f"Triangulation fehlgeschlagen: {e}"))

        print(f"Radio Maps erstellt für {len(self.interpolators)} von "
              f"{len(self.bssid_points)} BSSIDs.")
        if skipped:
            print(f"Übersprungen ({len(skipped)}):")
            for bssid, n, reason in skipped[:10]:
                print(f"  {bssid}: {n} Messungen -- {reason}")
            if len(skipped) > 10:
                print(f"  ... und {len(skipped) - 10} weitere")

        return self.interpolators

    def _grid_for_bssid(self, bssid, resolution=0.15, margin=0.3):
        """Erzeugt ein Raster über die konvexe Hülle der Messpunkte einer BSSID."""
        points = np.array([(x, y) for x, y, _ in self.bssid_points[bssid]])
        x_min, y_min = points.min(axis=0) - margin
        x_max, y_max = points.max(axis=0) + margin

        xs = np.arange(x_min, x_max, resolution)
        ys = np.arange(y_min, y_max, resolution)
        return np.meshgrid(xs, ys)

    def plot_bssid_heatmap(self, bssid, save_dir=MAPS_DIR, show=False):
        """Visualisiert die interpolierte Radio Map einer einzelnen BSSID."""
        if bssid not in self.interpolators:
            raise ValueError(f"Keine Radio Map für BSSID {bssid} vorhanden "
                              f"(build_interpolators() ausgeführt?)")

        interp = self.interpolators[bssid]
        X, Y = self._grid_for_bssid(bssid)
        Z = interp(X, Y)
        Z_masked = np.ma.masked_invalid(Z)

        points = np.array([(x, y) for x, y, _ in self.bssid_points[bssid]])
        rssi_values = np.array([r for _, _, r in self.bssid_points[bssid]])

        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")

        mesh = ax.pcolormesh(X, Y, Z_masked, cmap=cmap, shading="auto")
        fig.colorbar(mesh, ax=ax, label="Interpolierter RSSI (dBm)")

        ax.scatter(
            points[:, 0], points[:, 1],
            c=rssi_values, cmap=cmap,
            edgecolors="black", linewidths=0.8, s=60, zorder=3,
            vmin=np.nanmin(Z), vmax=np.nanmax(Z)
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Radio Map: {bssid}\n({len(points)} Messpunkte)")
        ax.axis("equal")

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            safe_name = bssid.replace(":", "-")
            out_path = save_dir / f"radio_map_{safe_name}.png"
            plt.savefig(out_path, dpi=150)
            print(f"Radio Map gespeichert: {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_top_n(self, n=6, save_dir=MAPS_DIR, show=False):
        """Erzeugt Heatmaps für die n BSSIDs mit den meisten Messpunkten."""
        ranked = sorted(
            self.interpolators.keys(),
            key=lambda b: len(self.bssid_points[b]),
            reverse=True
        )
        for bssid in ranked[:n]:
            self.plot_bssid_heatmap(bssid, save_dir=save_dir, show=show)

    def _in_hull(self, points):
        """Prüft, welche Punkte innerhalb der Gesamt-Messhülle liegen."""
        if self.hull_delaunay is None:
            return np.zeros(len(points), dtype=bool)
        return self.hull_delaunay.find_simplex(points) >= 0

    def generate_synthetic_fingerprints(self, n_samples=200, min_bssids=2,
                                         seed=42, max_attempts_factor=20):
        """
        Erzeugt synthetische Fingerprints an zufälligen Positionen innerhalb
        der konvexen Hülle der Originalmessungen.

        Für jede Zufallsposition wird für jede BSSID, deren eigene Hülle
        die Position einschließt, ein interpolierter RSSI-Wert berechnet.
        BSSIDs, deren Interpolation an dieser Stelle NaN liefert (außerhalb
        ihrer eigenen Hülle), werden für diesen Punkt einfach weggelassen --
        das entspricht exakt der Behandlung "nicht sichtbarer" BSSIDs in der
        bestehenden ML-Pipeline (DatasetBuilder).

        :param n_samples: gewünschte Anzahl synthetischer Fingerprints
        :param min_bssids: Mindestanzahl an BSSIDs, die an einer Position
                            interpolierbar sein müssen, sonst wird der
                            Punkt verworfen
        :param seed: Zufalls-Seed für Reproduzierbarkeit
        """
        if self.hull_delaunay is None:
            raise RuntimeError("Zu wenige Messpunkte für eine konvexe Hülle.")

        rng = random.Random(seed)
        x_min, y_min = self.all_positions.min(axis=0)
        x_max, y_max = self.all_positions.max(axis=0)

        synthetic_samples = []
        attempts = 0
        max_attempts = n_samples * max_attempts_factor

        while len(synthetic_samples) < n_samples and attempts < max_attempts:
            attempts += 1

            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)

            if not self._in_hull(np.array([[x, y]]))[0]:
                continue

            fingerprint = {}
            for bssid, interp in self.interpolators.items():
                value = interp(x, y)
                if np.isnan(value):
                    continue
                fingerprint[bssid] = {"rssi": round(float(value), 1)}

            if len(fingerprint) < min_bssids:
                continue

            synthetic_samples.append({
                "position": {"x": round(x, 3), "y": round(y, 3)},
                "fingerprint": fingerprint,
                "source": "interpolated"
            })

        if len(synthetic_samples) < n_samples:
            print(f"Warnung: nur {len(synthetic_samples)}/{n_samples} "
                  f"synthetische Samples erzeugt (max_attempts erreicht).")
        else:
            print(f"{len(synthetic_samples)} synthetische Fingerprints erzeugt.")

        return synthetic_samples

    def save_augmented_dataset(self, synthetic_samples, original_data,
                                path=AUGMENTED_PATH):
        """
        Kombiniert Original- und synthetische Fingerprints und speichert sie
        als separate Datei (überschreibt NICHT fingerprints.json).
        """
        original_marked = []
        for entry in original_data:
            entry_copy = dict(entry)
            entry_copy.setdefault("source", "measured")
            original_marked.append(entry_copy)

        combined = original_marked + synthetic_samples

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        n_measured = len(original_marked)
        n_synthetic = len(synthetic_samples)
        print(f"Augmentierter Datensatz gespeichert: {path}")
        print(f"  Gemessen:    {n_measured}")
        print(f"  Synthetisch: {n_synthetic}")
        print(f"  Gesamt:      {n_measured + n_synthetic}")


def main():
    with open(FINGERPRINTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    builder = RadioMapBuilder(data)
    builder.build_interpolators()

    # 1. Visualisierung: Heatmaps für die BSSIDs mit den meisten Messpunkten
    builder.plot_top_n(n=6)

    # 2. Datenaugmentation: synthetische Fingerprints erzeugen und speichern
    synthetic = builder.generate_synthetic_fingerprints(n_samples=200)
    builder.save_augmented_dataset(synthetic, data)


if __name__ == "__main__":
    main()