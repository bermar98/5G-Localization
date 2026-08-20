# spatial_error_map.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np

file_name = "position_comparison_Grundmessung.json"
suffix    = Path(file_name).stem.removeprefix("position_comparison_")
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "data"
RESULTS_FILE = DATA_PATH / "Positionsvergleich/Messungen" / file_name
MAP_PATH     = DATA_PATH / f"Positionsvergleich/Auswertungen/spatial_error_{suffix}.png"

# Pfad zur Hallenkarte
MAP_FILE = DATA_PATH / "FTS_Map.png"

# --- Karten-Kalibrierung ---
MAP_EXTENT = [-8, 11.0, -9.7, 6.2]  # [x_min, x_max, y_min, y_max] in Metern
MAP_ALPHA  = 0.25   # Transparenz

# --- Fehler-Schwellenwerte ---
LOW_THRESHOLD  = 1.0
HIGH_THRESHOLD = 2.5

COLOR_GOOD   = "#2ca02c"
COLOR_MEDIUM = "#f2c94c"
COLOR_BAD    = "#d62728"

# --- Schriftgrößen ---
FONTSIZE_ANNOTATION = 7
FONTSIZE_STATS      = 10
FONTSIZE_LABELS     = 11
FONTSIZE_TITLE      = 12
FONTSIZE_LEGEND     = 10
FONTSIZE_TICKS      = 10


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


def error_to_color(error):
    if error < LOW_THRESHOLD:
        return COLOR_GOOD
    elif error < HIGH_THRESHOLD:
        return COLOR_MEDIUM
    return COLOR_BAD


def plot_spatial_error_map(data, save_path=MAP_PATH, show=True, annotate_values=True):
    agv_x  = np.array([e["agv"]["x"] for e in data])
    agv_y  = np.array([-e["agv"]["y"] for e in data])  # y gespiegelt
    errors = np.array([e["error_m"]  for e in data])
    colors = [error_to_color(e) for e in errors]

    mean_error = errors.mean()
    max_error  = errors.max()

    fig, ax = plt.subplots(figsize=(9, 9))

    # --- Hallenkarte als Hintergrund ---
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

    # --- Scatter ---
    ax.scatter(
        agv_x, agv_y,
        c=colors, s=140,
        edgecolors="black", linewidths=0.6,
        alpha=0.9, zorder=3,
    )

    # --- Kollisionsfreie Beschriftung ---
    if annotate_values:
        CLUSTER_THRESH = 0.35
        ARROW_LEN      = 26
        TEXT_W         = 52
        TEXT_H         = 14
        POINT_RADIUS   = 10

        angles_deg = [90, 45, 135, 0, 180, 315, 225, 270,
                      60, 120, 30, 150, 330, 210, 300, 240]
        directions = [(np.cos(np.radians(a)), np.sin(np.radians(a)))
                      for a in angles_deg]

        occupied = []

        def rect_collides(cx, cy, w, h):
            for ox, oy, ow, oh in occupied:
                if (abs(cx-ox) < (w+ow)/2+4 and abs(cy-oy) < (h+oh)/2+4):
                    return True
            return False

        # Cluster bilden
        used = [False] * len(agv_x)
        clusters = []
        for i in range(len(agv_x)):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(len(agv_x)):
                if not used[j]:
                    d = np.sqrt((agv_x[i]-agv_x[j])**2 + (agv_y[i]-agv_y[j])**2)
                    if d < CLUSTER_THRESH:
                        group.append(j)
                        used[j] = True
            clusters.append(group)

        # Alle Punkte als belegt vorregistrieren
        fig.canvas.draw()
        trans = ax.transData
        for xi, yi in zip(agv_x, agv_y):
            px, py = trans.transform((xi, yi))
            occupied.append((px, py, POINT_RADIUS*2, POINT_RADIUS*2))

        for group in clusters:
            rep = max(group, key=lambda i: errors[i])
            x, y, e = agv_x[rep], agv_y[rep], errors[rep]
            px, py = trans.transform((x, y))

            best_off = None
            for dx_n, dy_n in directions:
                tx = px + dx_n * (ARROW_LEN + TEXT_W/2)
                ty = py + dy_n * (ARROW_LEN + TEXT_H/2)
                if not rect_collides(tx, ty, TEXT_W, TEXT_H):
                    best_off = (dx_n*ARROW_LEN, dy_n*ARROW_LEN, tx, ty)
                    break

            if best_off is None:
                best_off = (0, ARROW_LEN, px, py + ARROW_LEN + TEXT_H/2)

            off_x, off_y, tx, ty = best_off
            occupied.append((tx, ty, TEXT_W, TEXT_H))

            has_arrow = len(group) > 1 or abs(off_x) > 10 or abs(off_y) > 10
            ax.annotate(
                f"{e:.2f}", (x, y),
                textcoords="offset points",
                xytext=(off_x, off_y),
                ha="center",
                va="bottom" if off_y >= 0 else "top",
                fontsize=FONTSIZE_ANNOTATION,
                zorder=5,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6)
                          if has_arrow else None,
            )

    ax.set_xlabel("x (m)", fontsize=FONTSIZE_LABELS)
    ax.set_ylabel("y (m)", fontsize=FONTSIZE_LABELS)
    ax.set_title(
        f"Räumliche Fehlerkarte der Positionsschätzung (n={len(data)} Messungen)",
        fontsize=FONTSIZE_TITLE,
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)

    legend_handles = [
        mpatches.Patch(color=COLOR_GOOD,   label=f"< {LOW_THRESHOLD:.1f} m"),
        mpatches.Patch(color=COLOR_MEDIUM, label=f"{LOW_THRESHOLD:.1f}–{HIGH_THRESHOLD:.1f} m"),
        mpatches.Patch(color=COLOR_BAD,    label=f"> {HIGH_THRESHOLD:.1f} m"),
    ]
    ax.legend(handles=legend_handles, title="Fehler",
              loc="upper right", fontsize=FONTSIZE_LEGEND)

    ax.text(
        0.02, 0.02,
        f"Mittlerer Fehler: {mean_error:.2f} m\nMaximaler Fehler: {max_error:.2f} m",
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
        print(f"Fehlerkarte gespeichert: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "mean_error_m": float(mean_error),
        "max_error_m":  float(max_error),
        "n":            len(data),
        "n_good":   int(sum(1 for e in errors if e <  LOW_THRESHOLD)),
        "n_medium": int(sum(1 for e in errors if LOW_THRESHOLD <= e < HIGH_THRESHOLD)),
        "n_bad":    int(sum(1 for e in errors if e >= HIGH_THRESHOLD)),
    }


def main():
    data  = load_comparisons()
    stats = plot_spatial_error_map(data)
    print(f"\nAnzahl Messungen:  {stats['n']}")
    print(f"Mittlerer Fehler:  {stats['mean_error_m']:.2f} m")
    print(f"Maximaler Fehler:  {stats['max_error_m']:.2f} m")
    print(f"Gut  (<{LOW_THRESHOLD}m):  {stats['n_good']}")
    print(f"Mittel:            {stats['n_medium']}")
    print(f"Schlecht(>{HIGH_THRESHOLD}m): {stats['n_bad']}")


if __name__ == "__main__":
    main()