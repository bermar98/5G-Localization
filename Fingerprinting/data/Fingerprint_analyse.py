"""
Analyseskript fuer WiFi-Fingerprint-Rohdaten

Zwei Analyseblöcke:
  1. Raeumliche Verteilung der Messpunkte
     - Scatterplot aller Messpositionen
     - Dichte-Heatmap (2D-Histogramm), um duenn abgedeckte Bereiche
       (v.a. Raender) sichtbar zu machen
     - Naechste-Nachbar-Distanz pro Punkt als Mass fuer lokale Punktdichte

  2. RSSI-Schwankung pro BSSID
     - Fuer jede BSSID: wie stark schwankt der RSSI ueber alle Messungen,
       in denen sie sichtbar ist (Std.-Abw., Min/Max, Sichtbarkeitsanteil)
     - Zusaetzlich: Schwankung pro BSSID INNERHALB rraeumlich eng benachbarter
       Messpunkte (z.B. < 0.5 m auseinander) - das trennt "Schwankung durch
       echten Ortswechsel" von "Schwankung/Rauschen am selben Ort", was fuer
       die Einschaetzung von Ausreissern (siehe unsere vorherige Diskussion)
       deutlich aussagekraeftiger ist als eine globale z-Score-Betrachtung.

Input:  fingerprints.json
Output: PNG-Plots + raw_data_analysis_report.json im outputs-Verzeichnis
"""

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ---- Konfiguration ----
BASE_DIR   = Path(__file__).resolve().parent
INPUT_PATH  = BASE_DIR / "fingerprints_Grundmessung.json"
OUTPUT_DIR = BASE_DIR 

GRID_CELL_SIZE = 0.5          # Meter, fuer Dichte-Heatmap
NEIGHBOR_RADIUS = 0.5         # Meter, fuer "raeumlich benachbarte Messungen"
MIN_NEIGHBORS_FOR_STATS = 3   # Mindestanzahl Nachbarpunkte fuer lokale RSSI-Stats


def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_bssid(bssid: str) -> str:
    return bssid.upper().rstrip(":")


# ---------------------------------------------------------------------
# Block 1: raeumliche Verteilung der Messpunkte
# ---------------------------------------------------------------------

def analyze_spatial_distribution(positions: np.ndarray):
    x, y = positions[:, 0], positions[:, 1]

    # Naechste-Nachbar-Distanz pro Punkt (Mass fuer lokale Punktdichte)
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)  # k=2, da naechster "Nachbar" bei k=1 der Punkt selbst ist
    nn_dist = dists[:, 1]

    report = {
        "n_points": len(positions),
        "x_range": [float(x.min()), float(x.max())],
        "y_range": [float(y.min()), float(y.max())],
        "nearest_neighbor_dist_m": {
            "mean": float(nn_dist.mean()),
            "median": float(np.median(nn_dist)),
            "max": float(nn_dist.max()),
            "p90": float(np.percentile(nn_dist, 90)),
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Scatter aller Messpunkte
    axes[0].scatter(x, y, s=15, alpha=0.6, c="steelblue")
    axes[0].set_title("Messpunkte im Raum")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.3)

    # (b) Dichte-Heatmap (2D-Histogramm)
    x_bins = np.arange(x.min() - GRID_CELL_SIZE, x.max() + GRID_CELL_SIZE, GRID_CELL_SIZE)
    y_bins = np.arange(y.min() - GRID_CELL_SIZE, y.max() + GRID_CELL_SIZE, GRID_CELL_SIZE)
    h = axes[1].hist2d(x, y, bins=[x_bins, y_bins], cmap="viridis")
    axes[1].set_title(f"Punktdichte (Zellgröße {GRID_CELL_SIZE} m)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal", adjustable="box")
    fig.colorbar(h[3], ax=axes[1], label="Anzahl Messungen")

    # (c) Naechste-Nachbar-Distanz als Farbe je Punkt (zeigt duenne Randbereiche)
    sc = axes[2].scatter(x, y, s=20, c=nn_dist, cmap="magma")
    axes[2].set_title("Nächste-Nachbar-Distanz je Punkt")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    axes[2].set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes[2], label="Distanz zum nächsten Nachbarn (m)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "analysis_spatial_distribution.png", dpi=150)
    plt.close(fig)

    return report


# ---------------------------------------------------------------------
# Block 2: RSSI-Schwankung pro BSSID
# ---------------------------------------------------------------------

def analyze_rssi_variation(data, positions: np.ndarray):
    n_records = len(data)
    tree = cKDTree(positions)

    # Alle (bssid, rssi, record_index) Tripel einsammeln
    rows = []
    for i, rec in enumerate(data):
        for bssid, v in rec["fingerprint"].items():
            rows.append({"record_idx": i, "bssid": normalize_bssid(bssid), "rssi": v["rssi"]})
    df = pd.DataFrame(rows)

    # --- Globale Statistik pro BSSID ---
    global_stats = df.groupby("bssid")["rssi"].agg(
        count="count", mean="mean", std="std", min="min", max="max"
    ).reset_index()
    global_stats["visibility_ratio"] = global_stats["count"] / n_records
    global_stats = global_stats.sort_values("visibility_ratio", ascending=False)

    # --- Lokale Statistik: RSSI-Schwankung an raeumlich fast identischen Orten ---
    # Fuer jeden Punkt: alle anderen Punkte im Radius NEIGHBOR_RADIUS finden,
    # und pro BSSID die Streuung des RSSI innerhalb dieser lokalen Gruppe messen.
    local_rows = []
    neighbor_lists = tree.query_ball_point(positions, r=NEIGHBOR_RADIUS)

    bssid_to_records = df.groupby("bssid")["record_idx"].apply(set).to_dict()

    for bssid, rec_idx_set in bssid_to_records.items():
        # Fuer jeden Record, in dem diese BSSID sichtbar ist: Nachbarn sammeln,
        # die diese BSSID ebenfalls sehen
        local_stds = []
        for rec_idx in rec_idx_set:
            neighbors = [n for n in neighbor_lists[rec_idx] if n in rec_idx_set]
            if len(neighbors) >= MIN_NEIGHBORS_FOR_STATS:
                rssi_vals = df[(df["bssid"] == bssid) & (df["record_idx"].isin(neighbors))]["rssi"]
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

    # --- Plot: globale Std vs. lokale Std pro BSSID ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_n = merged.sort_values("visibility_ratio", ascending=False).head(25)
    axes[0].barh(top_n["bssid"], top_n["std"], color="steelblue")
    axes[0].set_title("Globale RSSI-Streuung pro BSSID (Top 25 nach Sichtbarkeit)")
    axes[0].set_xlabel("Std.-Abw. RSSI (dB), über alle Messungen")
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=7)

    comparable = merged.dropna(subset=["local_rssi_std_mean"]).sort_values(
        "visibility_ratio", ascending=False
    ).head(25)
    y_pos = np.arange(len(comparable))
    axes[1].barh(y_pos - 0.2, comparable["std"], height=0.4, label="Global (alle Orte)", color="steelblue")
    axes[1].barh(y_pos + 0.2, comparable["local_rssi_std_mean"], height=0.4,
                 label=f"Lokal (Nachbarn < {NEIGHBOR_RADIUS} m)", color="darkorange")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(comparable["bssid"], fontsize=7)
    axes[1].set_title("Globale vs. lokale RSSI-Streuung")
    axes[1].set_xlabel("Std.-Abw. RSSI (dB)")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "analysis_rssi_variation.png", dpi=150)
    plt.close(fig)

    return merged


def main():
    data = load_data(INPUT_PATH)
    positions = np.array([[rec["position"]["x"], rec["position"]["y"]] for rec in data])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spatial_report = analyze_spatial_distribution(positions)
    rssi_stats_df = analyze_rssi_variation(data, positions)

    report = {
        "spatial": spatial_report,
        "rssi_per_bssid": json.loads(rssi_stats_df.to_json(orient="records")),
    }

    with open(OUTPUT_DIR / "raw_data_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Kurze Konsolenausgabe der wichtigsten Kennzahlen
    print(f"Messpunkte: {spatial_report['n_points']}")
    print(f"x-Bereich: {spatial_report['x_range']}")
    print(f"y-Bereich: {spatial_report['y_range']}")
    print(f"Nächste-Nachbar-Distanz - Mittel: {spatial_report['nearest_neighbor_dist_m']['mean']:.2f} m, "
          f"P90: {spatial_report['nearest_neighbor_dist_m']['p90']:.2f} m, "
          f"Max: {spatial_report['nearest_neighbor_dist_m']['max']:.2f} m "
          f"(hoher Max-Wert = isolierter/duenn abgedeckter Punkt)")
    print()
    print("BSSIDs mit größter globaler RSSI-Streuung (Top 5):")
    print(rssi_stats_df.sort_values("std", ascending=False)[
        ["bssid", "visibility_ratio", "std", "local_rssi_std_mean"]
    ].head(5).to_string(index=False))
    print()
    print(f"Plots gespeichert in: {OUTPUT_DIR}")
    print(f"Report: {OUTPUT_DIR / 'raw_data_analysis_report.json'}")


if __name__ == "__main__":
    main()