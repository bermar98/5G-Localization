# =============================================================================
#  spatial_error_map_5g.py
#  Räumliche Fehlerkarte der 5G DL-OTDOA Positionsschätzung.
#  Analoges Format zu spatial_error_map.py des WiFi-Fingerprinting.
#
#  Eingabe: output/grid_results.json (erzeugt von evaluate_grid.py)
#  Ausgabe: output/spatial_error_map_5g.png
# =============================================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = BASE_DIR / "output" / "grid_results.json"
MAP_PATH     = BASE_DIR / "output" / "spatial_error_map_5g.png"

# Schwellenwerte für Fehler-Kategorien (in Metern)
LOW_THRESHOLD  = 1.0    # < 1m   → gut (grün)
HIGH_THRESHOLD = 2.5   # 1–2.5m  → mittel (gelb), > 2.5m → schlecht (rot)

COLOR_GOOD   = "#2ca02c"  # grün
COLOR_MEDIUM = "#f2c94c"  # gelb
COLOR_BAD    = "#d62728"  # rot


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


def error_to_color(error, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    if error < low:
        return COLOR_GOOD
    elif error < high:
        return COLOR_MEDIUM
    else:
        return COLOR_BAD


def plot_spatial_error_map(results, save_path=MAP_PATH,
                           show=True, annotate_values=True):
    """
    Visualisiert die Rasterpunkte eingefärbt nach Positionierungsfehler.
    Analog zu plot_spatial_error_map() in spatial_error_map.py.
    """
    agv_x  = np.array([r["ue_true"][0] for r in results])
    agv_y  = np.array([r["ue_true"][1] for r in results])
    errors = np.array([r["error_2d_m"] for r in results])
    colors = [error_to_color(e) for e in errors]

    mean_error = errors.mean()
    max_error  = errors.max()

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(
        agv_x, agv_y,
        c=colors, s=140, edgecolors="black", linewidths=0.6,
        alpha=0.9, zorder=3
    )

    if annotate_values:
        for x, y, e in zip(agv_x, agv_y, errors):
            ax.annotate(
                f"{e:.1f}", (x, y),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, zorder=4
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Räumliche Fehlerkarte der 5G DL-OTDOA Positionsschätzung "
        f"(n={len(results)} Messpunkte)"
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=COLOR_GOOD,
                       label=f"< {LOW_THRESHOLD:.0f} m (gut)"),
        mpatches.Patch(color=COLOR_MEDIUM,
                       label=f"{LOW_THRESHOLD:.0f}–{HIGH_THRESHOLD:.0f} m (mittel)"),
        mpatches.Patch(color=COLOR_BAD,
                       label=f"> {HIGH_THRESHOLD:.0f} m (schlecht)"),
    ]
    ax.legend(handles=legend_handles, title="Fehler-Kategorie",
              loc="upper right")

    stats_text = (
        f"Mittlerer Fehler: {mean_error:.2f} m\n"
        f"Maximaler Fehler: {max_error:.2f} m"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.85, edgecolor="gray"),
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
        "max_error_m":  float(max_error),
        "n":            len(results),
        "n_good":   int(sum(1 for e in errors if e < LOW_THRESHOLD)),
        "n_medium": int(sum(1 for e in errors if LOW_THRESHOLD <= e < HIGH_THRESHOLD)),
        "n_bad":    int(sum(1 for e in errors if e >= HIGH_THRESHOLD)),
    }


def main():
    results = load_results()
    stats   = plot_spatial_error_map(results)

    print(f"\nAnzahl Messpunkte: {stats['n']}")
    print(f"Mittlerer Fehler:  {stats['mean_error_m']:.2f} m")
    print(f"Maximaler Fehler:  {stats['max_error_m']:.2f} m")
    print(f"Gut   (< {LOW_THRESHOLD:.0f}m):  {stats['n_good']}")
    print(f"Mittel:            {stats['n_medium']}")
    print(f"Schlecht (>{HIGH_THRESHOLD:.0f}m): {stats['n_bad']}")


if __name__ == "__main__":
    main()