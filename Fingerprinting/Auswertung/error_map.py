# spatial_error_map.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

file_name = "position_comparison_Grundmessung.json"
suffix = Path(file_name).stem.removeprefix("position_comparison_")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
RESULTS_FILE = DATA_PATH / "Positionsvergleich/Messungen" /file_name
MAP_PATH = DATA_PATH / f"Positionsvergleich/Auswertungen/spatial_error_{suffix}.png"

# --- Schriftgrößen (zentral anpassbar) ---
FONTSIZE_ANNOTATION = 11    # Fehlerwerte an den Messpunkten
FONTSIZE_STATS      = 11   # Statistikbox unten links
FONTSIZE_LABELS     = 11   # Achsenbeschriftungen
FONTSIZE_TITLE      = 12   # Titel
FONTSIZE_LEGEND     = 11   # Legende

# Schwellenwerte für die Fehler-Kategorien (in Metern)
LOW_THRESHOLD = 1     # < 1m  -> gut (grün)
HIGH_THRESHOLD = 2.5    # 1-2.5m -> mittel (gelb), > 2.5m -> schlecht (rot)

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
        CLUSTER_THRESH  = 0.35   # m  – Punkte enger als dies → Cluster
        ARROW_LEN       = 26     # pt – Abstand Text zum Punkt
        TEXT_W          = 52     # pt – geschätzte Textbreite (4 Zeichen)
        TEXT_H          = 14     # pt – geschätzte Texthöhe
        POINT_RADIUS    = 10     # pt – Radius der Scatter-Kreise (s=140 → r≈7pt, +Puffer)

        # 16 Richtungen – dicht gesät für bessere Auswahl
        angles_deg = [90, 45, 135, 0, 180, 315, 225, 270,
                      60, 120, 30, 150, 330, 210, 300, 240]
        directions = [(np.cos(np.radians(a)), np.sin(np.radians(a)))
                      for a in angles_deg]

        # Belegungsrechtecke: (cx, cy, w, h) in Pixel
        occupied = []

        def rect_collides(cx, cy, w, h):
            """Prüft ob ein Rechteck (cx,cy,w,h) mit bestehenden Rechtecken kollidiert."""
            for ox, oy, ow, oh in occupied:
                if (abs(cx - ox) < (w + ow) / 2 + 4 and
                        abs(cy - oy) < (h + oh) / 2 + 4):
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

        # Alle Scatter-Punkte als belegte Kreise vorregistrieren
        fig.canvas.draw()
        trans = ax.transData
        for xi, yi in zip(agv_x, agv_y):
            px, py = trans.transform((xi, yi))
            occupied.append((px, py, POINT_RADIUS * 2, POINT_RADIUS * 2))

        for group in clusters:
            # Pro Cluster nur den Punkt mit dem größten Fehler beschriften
            rep = max(group, key=lambda i: errors[i])
            x, y, e = agv_x[rep], agv_y[rep], errors[rep]
            px, py = trans.transform((x, y))

            best_off = None
            for dx_n, dy_n in directions:
                # Text-Mittelpunkt in Pixel
                tx = px + dx_n * (ARROW_LEN + TEXT_W / 2)
                ty = py + dy_n * (ARROW_LEN + TEXT_H / 2)
                if not rect_collides(tx, ty, TEXT_W, TEXT_H):
                    best_off = (dx_n * ARROW_LEN, dy_n * ARROW_LEN, tx, ty)
                    break

            if best_off is None:
                # Alle Richtungen belegt → trotzdem platzieren (oben)
                best_off = (0, ARROW_LEN,
                            px, py + ARROW_LEN + TEXT_H / 2)

            off_x, off_y, tx, ty = best_off
            occupied.append((tx, ty, TEXT_W, TEXT_H))

            has_arrow = len(group) > 1 or abs(off_x) > 10 or abs(off_y) > 10
            ax.annotate(
                f"{e:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(off_x, off_y),
                ha="center",
                va="bottom" if off_y >= 0 else "top",
                fontsize=FONTSIZE_ANNOTATION,
                zorder=5,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.6,
                ) if has_arrow else None,
            )

    ax.set_xlabel("x (m)", fontsize=FONTSIZE_LABELS)
    ax.set_ylabel("y (m)", fontsize=FONTSIZE_LABELS)
    ax.set_title(f"Räumliche Fehlerkarte der Positionsschätzung (n={len(data)} Messungen)", fontsize=FONTSIZE_TITLE)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=COLOR_GOOD, label=f"< {LOW_THRESHOLD:.1f} m"),
        mpatches.Patch(color=COLOR_MEDIUM, label=f"{LOW_THRESHOLD:.1f}\u2013{HIGH_THRESHOLD:.1f} m"),
        mpatches.Patch(color=COLOR_BAD, label=f"> {HIGH_THRESHOLD:.1f} m"),
    ]
    ax.legend(handles=legend_handles, title="Fehler", loc="upper right", fontsize=FONTSIZE_LEGEND)

    stats_text = (
        f"Mittlerer Fehler: {mean_error:.2f} m\n"
        f"Maximaler Fehler: {max_error:.2f} m"
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