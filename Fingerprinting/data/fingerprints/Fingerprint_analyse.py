"""
Analyseskript fuer WiFi-Fingerprint-Rohdaten

Zwei Analyseblöcke:
  1. Raeumliche Verteilung der Messpunkte
     - Scatterplot aller Messpositionen
     - Dichte-Heatmap (2D-Histogramm)
     - Naechste-Nachbar-Distanz pro Punkt

  2. RSSI-Schwankung pro BSSID
     - Globale und lokale RSSI-Streuung je BSSID

Input:  fingerprints_<suffix>.json
Output: 5 PNG-Plots + raw_data_analysis_report_<suffix>.json
"""

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import cKDTree

# ── Konfiguration ─────────────────────────────────────────────────────
file_name = "fingerprints_Grundmessung.json"

BASE_DIR   = Path(__file__).resolve().parent
INPUT_PATH  = BASE_DIR / "Messungen" / file_name
OUTPUT_DIR  = BASE_DIR / "Auswertung"
suffix      = Path(file_name).stem.removeprefix("fingerprints_")

# Hallenkarte
MAP_FILE   = BASE_DIR.parent / "FTS_Map.png"
MAP_EXTENT = [-8, 11.0, -9.7, 6.3]   # [x_min, x_max, y_min, y_max] in Metern
MAP_ALPHA  = 0.25

GRID_CELL_SIZE           = 0.5
NEIGHBOR_RADIUS          = 0.5
MIN_NEIGHBORS_FOR_STATS  = 3

# ── Schriftgrößen ─────────────────────────────────────────────────────
FONTSIZE_TITLE  = 11
FONTSIZE_LABELS = 11
FONTSIZE_TICKS  = 11
FONTSIZE_LEGEND = 11
FONTSIZE_CBAR   = 11


# ── Hilfsfunktionen ───────────────────────────────────────────────────

def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_bssid(bssid: str) -> str:
    return bssid.upper().rstrip(":")


def _add_map(ax):
    """Hinterlegt die Hallenkarte auf einer Achse."""
    if MAP_FILE.exists():
        img = mpimg.imread(str(MAP_FILE))
        ax.imshow(
            img,
            extent=MAP_EXTENT,
            origin="upper",   # Pixel-y=0 oben → passt zum FTS-System
            alpha=MAP_ALPHA,
            zorder=0,
            aspect="auto",
        )


def _apply_style(ax, title, xlabel="x (m)", ylabel="y (m)"):
    """Einheitliche Achsen-Formatierung."""
    ax.set_title(f"{title}\n({suffix})", fontsize=FONTSIZE_TITLE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)


def _save(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {path}")


# ── Block 1: Räumliche Verteilung ─────────────────────────────────────

def analyze_spatial_distribution(positions: np.ndarray):
    x, y = positions[:, 0], positions[:, 1]

    # y-Koordinaten für Kartendarstellung spiegeln
    y_map = -y

    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)
    nn_dist = dists[:, 1]

    report = {
        "n_points": len(positions),
        "x_range": [float(x.min()), float(x.max())],
        "y_range": [float(y.min()), float(y.max())],
        "nearest_neighbor_dist_m": {
            "mean":   float(nn_dist.mean()),
            "median": float(np.median(nn_dist)),
            "max":    float(nn_dist.max()),
            "p90":    float(np.percentile(nn_dist, 90)),
        },
    }

    # ── Plot (a): Scatter ─────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(7, 6))
    _add_map(ax_a)
    ax_a.scatter(x, y_map, s=10, alpha=0.6, c="steelblue", zorder=3)
    _apply_style(ax_a, "Messpunkte im Raum")
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.grid(alpha=0.25)
    fig_a.tight_layout()
    _save(fig_a, f"analysis_spatial_scatter_{suffix}.png")

    # ── Plot (b): Dichte-Heatmap ──────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(7, 6))
    _add_map(ax_b)
    x_bins = np.arange(x.min() - GRID_CELL_SIZE, x.max() + GRID_CELL_SIZE, GRID_CELL_SIZE)
    y_bins = np.arange(y_map.min() - GRID_CELL_SIZE, y_map.max() + GRID_CELL_SIZE, GRID_CELL_SIZE)
    h = ax_b.hist2d(x, y_map, bins=[x_bins, y_bins], cmap="viridis", zorder=2, alpha=0.75)
    cb = fig_b.colorbar(h[3], ax=ax_b, label="Anzahl Messungen")
    cb.ax.tick_params(labelsize=FONTSIZE_CBAR)
    cb.set_label("Anzahl Messungen", fontsize=FONTSIZE_CBAR)
    _apply_style(ax_b, f"Punktdichte (Zellgröße {GRID_CELL_SIZE} m)")
    ax_b.set_aspect("equal", adjustable="box")
    fig_b.tight_layout()
    _save(fig_b, f"analysis_spatial_density_{suffix}.png")

    # ── Plot (c): Nächste-Nachbar-Distanz ─────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(7, 6))
    _add_map(ax_c)
    sc = ax_c.scatter(x, y_map, s=15, c=nn_dist, cmap="magma", zorder=3)
    cb2 = fig_c.colorbar(sc, ax=ax_c, label="Distanz zum nächsten Nachbarn (m)")
    cb2.ax.tick_params(labelsize=FONTSIZE_CBAR)
    cb2.set_label("Distanz zum nächsten Nachbarn (m)", fontsize=FONTSIZE_CBAR)
    _apply_style(ax_c, "Nächste-Nachbar-Distanz je Punkt")
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.grid(alpha=0.25)
    fig_c.tight_layout()
    _save(fig_c, f"analysis_spatial_nn_dist_{suffix}.png")

    return report


# ── Block 2: RSSI-Schwankung ──────────────────────────────────────────

def analyze_rssi_variation(data, positions: np.ndarray):
    n_records = len(data)
    tree = cKDTree(positions)

    rows = []
    for i, rec in enumerate(data):
        for bssid, v in rec["fingerprint"].items():
            rows.append({
                "record_idx": i,
                "bssid": normalize_bssid(bssid),
                "rssi": v["rssi"],
            })
    df = pd.DataFrame(rows)

    # Globale Statistik
    global_stats = df.groupby("bssid")["rssi"].agg(
        count="count", mean="mean", std="std", min="min", max="max"
    ).reset_index()
    global_stats["visibility_ratio"] = global_stats["count"] / n_records
    global_stats = global_stats.sort_values("visibility_ratio", ascending=False)

    # Lokale Statistik
    local_rows = []
    neighbor_lists  = tree.query_ball_point(positions, r=NEIGHBOR_RADIUS)
    bssid_to_records = df.groupby("bssid")["record_idx"].apply(set).to_dict()

    for bssid, rec_idx_set in bssid_to_records.items():
        local_stds = []
        for rec_idx in rec_idx_set:
            neighbors = [n for n in neighbor_lists[rec_idx] if n in rec_idx_set]
            if len(neighbors) >= MIN_NEIGHBORS_FOR_STATS:
                rssi_vals = df[
                    (df["bssid"] == bssid) &
                    (df["record_idx"].isin(neighbors))
                ]["rssi"]
                if len(rssi_vals) >= MIN_NEIGHBORS_FOR_STATS:
                    local_stds.append(rssi_vals.std())
        if local_stds:
            local_rows.append({
                "bssid": bssid,
                "local_rssi_std_mean": float(np.mean(local_stds)),
                "n_local_groups": len(local_stds),
            })

    local_stats = pd.DataFrame(local_rows)
    merged = global_stats.merge(local_stats, on="bssid", how="left")

    # ── Plot: RSSI-Streuung ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_n = merged.sort_values("visibility_ratio", ascending=False).head(25)
    axes[0].barh(top_n["bssid"], top_n["std"], color="steelblue")
    axes[0].set_title(
        f"Globale RSSI-Streuung pro BSSID (Top 25)\n({suffix})",
        fontsize=FONTSIZE_TITLE,
    )
    axes[0].set_xlabel("Std.-Abw. RSSI (dB)", fontsize=FONTSIZE_LABELS)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].tick_params(axis="x", labelsize=FONTSIZE_TICKS)

    comparable = merged.dropna(subset=["local_rssi_std_mean"]).sort_values(
        "visibility_ratio", ascending=False
    ).head(25)
    y_pos = np.arange(len(comparable))
    axes[1].barh(y_pos - 0.2, comparable["std"],
                 height=0.4, label="Global (alle Orte)", color="steelblue")
    axes[1].barh(y_pos + 0.2, comparable["local_rssi_std_mean"],
                 height=0.4,
                 label=f"Lokal (Nachbarn < {NEIGHBOR_RADIUS} m)",
                 color="darkorange")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(comparable["bssid"], fontsize=7)
    axes[1].set_title(
        f"Globale vs. lokale RSSI-Streuung\n({suffix})",
        fontsize=FONTSIZE_TITLE,
    )
    axes[1].set_xlabel("Std.-Abw. RSSI (dB)", fontsize=FONTSIZE_LABELS)
    axes[1].tick_params(axis="x", labelsize=FONTSIZE_TICKS)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=FONTSIZE_LEGEND)

    fig.tight_layout()
    _save(fig, f"analysis_rssi_variation_{suffix}.png")

    return merged


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data = load_data(INPUT_PATH)
    positions = np.array([
        [rec["position"]["x"], rec["position"]["y"]] for rec in data
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spatial_report = analyze_spatial_distribution(positions)
    rssi_stats_df  = analyze_rssi_variation(data, positions)

    report = {
        "spatial":        spatial_report,
        "rssi_per_bssid": json.loads(rssi_stats_df.to_json(orient="records")),
    }

    with open(
        OUTPUT_DIR / f"raw_data_analysis_report_{suffix}.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nMesspunkte: {spatial_report['n_points']}")
    print(f"x-Bereich : {spatial_report['x_range']}")
    print(f"y-Bereich : {spatial_report['y_range']}")
    nn = spatial_report["nearest_neighbor_dist_m"]
    print(f"NN-Distanz: Mittel={nn['mean']:.3f}m  P90={nn['p90']:.3f}m  Max={nn['max']:.3f}m")
    print()
    print("BSSIDs mit größter globaler RSSI-Streuung (Top 5):")
    print(rssi_stats_df.sort_values("std", ascending=False)[
        ["bssid", "visibility_ratio", "std", "local_rssi_std_mean"]
    ].head(5).to_string(index=False))
    print()
    print(f"Plots gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()