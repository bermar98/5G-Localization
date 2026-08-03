# =============================================================================
#  position_comparison_5g.py
#  Visualisiert wahre vs. geschätzte UE-Position aus der 5G DL-OTDOA Simulation.
#  Analoges Format zu position_comparison.py des WiFi-Fingerprinting.
#
#  Eingabe: output/grid_results.json (erzeugt von evaluate_grid.py)
#  Ausgabe: output/position_comparison_5g.png
# =============================================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = BASE_DIR / "output" / "grid_results.json"
PLOT_PATH   = BASE_DIR / "output" / "position_comparison_5g.png"


def load_results(path=RESULTS_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nicht gefunden. Bitte zuerst evaluate_grid.py ausführen."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = [r for r in data["results"] if r["success"]]
    if not results:
        raise ValueError("Keine erfolgreichen Schätzungen in der Datei.")
    return results


def plot_positions(results, save_path=PLOT_PATH, show=True):
    """
    Visualisiert wahre UE-Position vs. geschätzte Position.
    Verbindungslinien zeigen den Fehler je Rasterpunkt.
    Analog zu plot_positions() in position_comparison.py.
    """
    agv_x  = [r["ue_true"][0]      for r in results]
    agv_y  = [r["ue_true"][1]      for r in results]
    est_x  = [r["ue_estimated"][0] for r in results]
    est_y  = [r["ue_estimated"][1] for r in results]
    errors = np.array([r["error_2d_m"] for r in results])

    mean_error = errors.mean()
    max_error  = errors.max()
    min_error  = errors.min()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(agv_x, agv_y,
               c="blue", label="Wahre UE-Position", alpha=0.7, zorder=3, s=60)
    ax.scatter(est_x, est_y,
               c="red",  label="Geschätzte Position", alpha=0.7, zorder=3, s=60)

    for i in range(len(results)):
        ax.plot(
            [agv_x[i], est_x[i]],
            [agv_y[i], est_y[i]],
            color="gray", alpha=0.4, linewidth=0.8, zorder=1
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Positionsvergleich: Wahre vs. geschätzte Position "
        f"(n={len(results)} Messpunkte)"
    )
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

    return {
        "mean_error_m": float(mean_error),
        "max_error_m":  float(max_error),
        "min_error_m":  float(min_error),
        "n":            len(results),
    }


def main():
    results = load_results()
    stats   = plot_positions(results)

    print(f"\nAnzahl Messpunkte: {stats['n']}")
    print(f"Mittlerer Fehler:  {stats['mean_error_m']:.2f} m")
    print(f"Maximaler Fehler:  {stats['max_error_m']:.2f} m")
    print(f"Minimaler Fehler:  {stats['min_error_m']:.2f} m")


if __name__ == "__main__":
    main()