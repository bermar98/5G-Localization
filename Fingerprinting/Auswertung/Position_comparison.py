# position_comparison.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

file_name = "position_comparisons_12.08.optimiertesModell.json"
suffix    = Path(file_name).stem.removeprefix("position_comparison_")
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "data"
RESULTS_FILE = DATA_PATH / "Positionsvergleich/Messungen" / file_name
PLOT_PATH    = DATA_PATH / f"Positionsvergleich/Auswertungen/position_comparisons_{suffix}.png"

# Pfad zur Hallenkarte
MAP_FILE = DATA_PATH / "FTS_Map.png"

# --- Karten-Kalibrierung ---
# extent = [x_min, x_max, y_min, y_max] in Metern
# Anpassen bis Messpunkte korrekt auf der Karte liegen.
# y ist gespiegelt: origin='upper' dreht die Karte automatisch.
MAP_EXTENT = [-8, 11.0, -9.7, 6.3]
MAP_ALPHA  = 0.25   # Transparenz der Karte (0=unsichtbar, 1=voll)

# --- Schriftgrößen ---
FONTSIZE_STATS  = 10
FONTSIZE_LABELS = 11
FONTSIZE_TITLE  = 12
FONTSIZE_LEGEND = 10
FONTSIZE_TICKS  = 10


def load_comparisons(path=RESULTS_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} nicht gefunden.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    if not data:
        raise ValueError(f"{path} enthält keine Einträge.")
    return data


def plot_positions(data, save_path=PLOT_PATH, show=True):
    agv_x  = [e["agv"]["x"]       for e in data]
    agv_y  = [-e["agv"]["y"]       for e in data]
    est_x  = [e["estimated"]["x"] for e in data]
    est_y  = [-e["estimated"]["y"] for e in data]
    errors = np.array([e["error_m"] for e in data])

    mean_error = errors.mean()
    max_error  = errors.max()
    min_error  = errors.min()

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Hallenkarte als Hintergrund ---
    # origin='upper': Pixel-y=0 oben → y-Achse gespiegelt wie FTS-Koordinaten
    if MAP_FILE.exists():
        img = mpimg.imread(str(MAP_FILE))
        ax.imshow(
            img,
            extent=MAP_EXTENT,
            origin="upper",       # Kartenursprung oben links
            alpha=MAP_ALPHA,
            zorder=0,
            aspect="auto",
        )
    else:
        print(f"[Info] Karte nicht gefunden: {MAP_FILE}")

    # --- Verbindungslinien ---
    for i in range(len(data)):
        ax.plot(
            [agv_x[i], est_x[i]],
            [agv_y[i], est_y[i]],
            color="gray", alpha=0.5, linewidth=0.8, zorder=1,
        )

    # --- Messpunkte ---
    ax.scatter(agv_x, agv_y,
               c="blue", s=50, label="FTS-Position (real)",
               alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4)
    ax.scatter(est_x, est_y,
               c="red",  s=50, label="Geschätzte Position",
               alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4)

    ax.set_xlabel("x (m)", fontsize=FONTSIZE_LABELS)
    ax.set_ylabel("y (m)", fontsize=FONTSIZE_LABELS)
    ax.set_title(
        f"Positionsvergleich: FTS vs. Schätzung (n={len(data)} Messungen)",
        fontsize=FONTSIZE_TITLE,
    )
    ax.legend(loc="upper right", fontsize=FONTSIZE_LEGEND)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)

    stats_text = (
        f"Mittlerer Fehler: {mean_error:.2f} m\n"
        f"Maximaler Fehler: {max_error:.2f} m\n"
        f"Minimaler Fehler: {min_error:.2f} m"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=FONTSIZE_STATS,
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
        "n":            len(data),
    }


def main():
    data  = load_comparisons()
    stats = plot_positions(data)
    print(f"\nAnzahl Messungen:  {stats['n']}")
    print(f"Mittlerer Fehler:  {stats['mean_error_m']:.2f} m")
    print(f"Maximaler Fehler:  {stats['max_error_m']:.2f} m")
    print(f"Minimaler Fehler:  {stats['min_error_m']:.2f} m")


if __name__ == "__main__":
    main()