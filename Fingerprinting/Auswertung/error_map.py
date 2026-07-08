# spatial_error_map.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
RESULTS_FILE = DATA_PATH / "position_comparisons.json"
MAP_PATH = DATA_PATH / "spatial_error_map.png"

# Schwellenwerte für die Fehler-Kategorien (in Metern)
LOW_THRESHOLD = 1     # < 0.5m  -> gut (grün)
HIGH_THRESHOLD = 2.5    # 0.5-1.5m -> mittel (gelb), > 1.5m -> schlecht (rot)

COLOR_GOOD = "#2ca02c"     # grün
COLOR_MEDIUM = "#f2c94c"   # gelb
COLOR_BAD = "#d62728"      # rot


def load_comparisons(path=RESULTS_FILE):
    """Lädt die gespeicherten Positionsvergleiche aus der JSON-Datei."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} nicht gefunden. Wurde get_position.py bereits ausgeführt?"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    if not data:
        raise ValueError(f"{path} enthält keine Einträge.")

    return data


def error_to_color(error, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    """Ordnet einen Fehlerwert einer der drei Kategorien (Farben) zu."""
    if error < low:
        return COLOR_GOOD
    elif error < high:
        return COLOR_MEDIUM
    else:
        return COLOR_BAD


def plot_spatial_error_map(data, save_path=MAP_PATH, show=True, annotate_values=True):
    """
    Visualisiert die AGV-Referenzpositionen, eingefärbt nach dem an dieser
    Stelle gemessenen Positionierungsfehler (grün/gelb/rot je nach
    Schwellenwert). Zusätzlich werden mittlerer und maximaler Fehler
    berechnet und im Plot angezeigt.
    """
    agv_x = np.array([entry["agv"]["x"] for entry in data])
    agv_y = np.array([entry["agv"]["y"] for entry in data])
    errors = np.array([entry["error_m"] for entry in data])

    colors = [error_to_color(e) for e in errors]

    mean_error = errors.mean()
    max_error = errors.max()

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(
        agv_x, agv_y,
        c=colors, s=140, edgecolors="black", linewidths=0.6,
        alpha=0.9, zorder=3
    )

    if annotate_values:
        for x, y, e in zip(agv_x, agv_y, errors):
            ax.annotate(
                f"{e:.2f}", (x, y),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, zorder=4
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Räumliche Fehlerkarte der Positionsschätzung (n={len(data)} Messungen)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=COLOR_GOOD, label=f"< {LOW_THRESHOLD:.1f} m"),
        mpatches.Patch(color=COLOR_MEDIUM, label=f"{LOW_THRESHOLD:.1f}\u2013{HIGH_THRESHOLD:.1f} m"),
        mpatches.Patch(color=COLOR_BAD, label=f"> {HIGH_THRESHOLD:.1f} m"),
    ]
    ax.legend(handles=legend_handles, title="Fehler", loc="upper right")

    stats_text = (
        f"Mittlerer Fehler: {mean_error:.2f} m\n"
        f"Maximaler Fehler: {max_error:.2f} m"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Fehlerkarte gespeichert: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "mean_error_m": float(mean_error),
        "max_error_m": float(max_error),
        "n": len(data),
        "n_good": int(sum(1 for e in errors if e < LOW_THRESHOLD)),
        "n_medium": int(sum(1 for e in errors if LOW_THRESHOLD <= e < HIGH_THRESHOLD)),
        "n_bad": int(sum(1 for e in errors if e >= HIGH_THRESHOLD)),
    }


def main():
    data = load_comparisons()
    stats = plot_spatial_error_map(data)

    print(f"\nAnzahl Messungen:  {stats['n']}")
    print(f"Mittlerer Fehler:  {stats['mean_error_m']:.2f} m")
    print(f"Maximaler Fehler:  {stats['max_error_m']:.2f} m")
    print(f"Kategorie gut (< {LOW_THRESHOLD}m):    {stats['n_good']}")
    print(f"Kategorie mittel:                 {stats['n_medium']}")
    print(f"Kategorie schlecht (> {HIGH_THRESHOLD}m): {stats['n_bad']}")


if __name__ == "__main__":
    main()