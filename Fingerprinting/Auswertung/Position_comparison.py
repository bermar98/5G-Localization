# plot_positions.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
RESULTS_FILE = DATA_PATH / "position_comparisons.json"
PLOT_PATH = DATA_PATH / "position_comparison_plot.png"


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


def plot_positions(data, save_path=PLOT_PATH, show=True):
    """
    Visualisiert AGV-Position vs. geschätzte Position für alle Einträge
    und berechnet mittleren sowie maximalen Fehler über alle Messungen.
    """
    agv_x = [entry["agv"]["x"] for entry in data]
    agv_y = [entry["agv"]["y"] for entry in data]
    est_x = [entry["estimated"]["x"] for entry in data]
    est_y = [entry["estimated"]["y"] for entry in data]
    errors = np.array([entry["error_m"] for entry in data])

    mean_error = errors.mean()
    max_error = errors.max()
    min_error = errors.min()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(agv_x, agv_y, c="blue", label="AGV-Position (real)", alpha=0.7, zorder=3)
    ax.scatter(est_x, est_y, c="red", label="Geschätzte Position", alpha=0.7, zorder=3)

    for i in range(len(data)):
        ax.plot(
            [agv_x[i], est_x[i]],
            [agv_y[i], est_y[i]],
            color="gray", alpha=0.4, linewidth=0.8, zorder=1
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Positionsvergleich: AGV vs. Schätzung (n={len(data)} Messungen)")
    ax.legend(loc="upper right")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    stats_text = (
        f"Mittlerer Fehler: {mean_error:.2f} m\n"
        f"Maximaler Fehler: {max_error:.2f} m\n"
        f"Minimaler Fehler: {min_error:.2f} m"
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
        print(f"Plot gespeichert: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"mean_error_m": float(mean_error), "max_error_m": float(max_error), "min_error_m": float(min_error), "n": len(data)}


def main():
    data = load_comparisons()
    stats = plot_positions(data)

    print(f"\nAnzahl Messungen:  {stats['n']}")
    print(f"Mittlerer Fehler:  {stats['mean_error_m']:.2f} m")
    print(f"Maximaler Fehler:  {stats['max_error_m']:.2f} m")
    print(f"Minimaler Fehler:  {stats['min_error_m']:.2f} m")


if __name__ == "__main__":
    main()