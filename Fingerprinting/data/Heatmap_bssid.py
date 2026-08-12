import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR/"fingerprints.json"    # Pfad zu deinen Messdaten
OUTPUT_DIR = BASE_DIR/"radio_maps"            # Zielordner für die erzeugten Plots
RSSI_MISSING = -100                # Sentinel-Wert für "BSSID nicht empfangen"
GRID_RESOLUTION = 200              # Auflösung des Interpolationsgitters
INTERP_METHOD = "cubic"            # 'linear', 'cubic' oder 'nearest'
TOP_N_BSSIDS = None                # z.B. 10 -> nur die N häufigsten BSSIDs plotten, None = alle
MIN_POINTS = 4                     # Mindestanzahl an Messpunkten pro BSSID für Interpolation


"""
Erstellt für jede BSSID:
  1) eine Heatmap der GEMITTELTEN RSSI-Werte über den gemessenen Positionen
  2) eine Heatmap der SCHWANKUNGSBREITE (max - min, sowie Std) pro Position

Warum Mittelung nötig ist:
Wenn an (fast) derselben Position mehrere Messungen existieren, "zittert"
die interpolierte Fläche, da griddata() jeden Einzelwert exakt trifft.
Daher werden Messpunkte zunächst nach Position geclustert (POSITION_ROUND)
und pro Cluster gemittelt, bevor interpoliert wird.

Erwartetes Eingabeformat (JSON), Liste von Messpunkten:
[
    {
        "timestamp": "2026-07-01 13:55:01",
        "position": {"x": 6.065, "y": 1.729, "theta": 0.303},
        "fingerprint": {
            "38:C0:EA:46:3D:71": {"ssid": "MEDIC", "rssi": -62, "distance": 6.53},
            ...
        }
    },
    ...
]
"""
POSITION_ROUND = 1

ANNOTATE_RANGE = True              # Schwankungsbreite (min-max) als Text neben Messpunkten anzeigen


def load_data(json_path: str) -> pd.DataFrame:
    """Liest die verschachtelte JSON-Struktur und flacht sie zu einem
    DataFrame mit Spalten x, y, bssid, ssid, rssi, distance, timestamp ab."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
 
    rows = []
    for rec in records:
        pos = rec["position"]
        ts = rec.get("timestamp")
        for bssid, info in rec["fingerprint"].items():
            rows.append({
                "timestamp": ts,
                "x": pos["x"],
                "y": pos["y"],
                "bssid": bssid,
                "ssid": info.get("ssid", ""),
                "rssi": info["rssi"],
                "distance": info.get("distance", np.nan),
            })
 
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Keine Messpunkte in der JSON-Datei gefunden.")
    return df
 
 
def aggregate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Clustert Messpunkte nach gerundeter Position und mittelt pro
    (Cluster, BSSID). Liefert zusätzlich Std/Min/Max/Count als Maß für
    die Schwankungsbreite."""
    df = df.copy()
    df["x_cluster"] = df["x"].round(POSITION_ROUND)
    df["y_cluster"] = df["y"].round(POSITION_ROUND)
 
    agg = (
        df.groupby(["x_cluster", "y_cluster", "bssid"], as_index=False)
        .agg(
            x=("x", "mean"),           # tatsächliche mittlere Koordinate im Cluster
            y=("y", "mean"),
            ssid=("ssid", "first"),
            rssi_mean=("rssi", "mean"),
            rssi_std=("rssi", "std"),
            rssi_min=("rssi", "min"),
            rssi_max=("rssi", "max"),
            n_samples=("rssi", "count"),
        )
    )
    agg["rssi_std"] = agg["rssi_std"].fillna(0.0)
    agg["rssi_range"] = agg["rssi_max"] - agg["rssi_min"]
    return agg
 
 
def select_bssids(df: pd.DataFrame, top_n) -> list:
    if top_n is None:
        return sorted(df["bssid"].unique())
    counts = (
        df[df["rssi_mean"] > RSSI_MISSING]
        .groupby("bssid")
        .size()
        .sort_values(ascending=False)
    )
    return counts.head(top_n).index.tolist()
 
 
def _interp_grid(x, y, values, method=INTERP_METHOD):
    grid_x, grid_y = np.mgrid[
        x.min():x.max():complex(GRID_RESOLUTION),
        y.min():y.max():complex(GRID_RESOLUTION),
    ]
    grid_z = griddata(points=(x, y), values=values, xi=(grid_x, grid_y), method=method)
    return grid_x, grid_y, grid_z
 
 
def plot_mean_heatmap(sub: pd.DataFrame, bssid: str, ssid: str, output_dir: str) -> None:
    x = sub["x"].to_numpy()
    y = sub["y"].to_numpy()
    rssi = sub["rssi_mean"].to_numpy()
 
    grid_x, grid_y, grid_z = _interp_grid(x, y, rssi)
 
    fig, ax = plt.subplots(figsize=(7, 6))
 
    im = ax.imshow(
        grid_z.T,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="RdYlGn",
        vmin=RSSI_MISSING,
        vmax=max(rssi.max(), -30),
        aspect="equal",
    )
 
    ax.scatter(
        x, y, c=rssi, cmap="RdYlGn",
        vmin=RSSI_MISSING, vmax=max(rssi.max(), -30),
        edgecolors="black", linewidths=0.6, s=45, zorder=3,
    )
 
    if ANNOTATE_RANGE:
        for _, row in sub.iterrows():
            label = f"±{row['rssi_range']/2:.1f}\n(n={int(row['n_samples'])})"
            ax.annotate(
                label, (row["x"], row["y"]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=6, color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
            )
 
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mittlerer RSSI [dBm]")
 
    title = f"Ø RSSI-Heatmap – {ssid} ({bssid})" if ssid else f"Ø RSSI-Heatmap – {bssid}"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
 
    fname = bssid.replace(":", "-") + "_mean.png"
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"[ok] Mittelwert-Heatmap gespeichert: {fname}")
 
 
def plot_variability_heatmap(sub: pd.DataFrame, bssid: str, ssid: str, output_dir: str) -> None:
    """Zeigt räumlich, wie stark die RSSI-Werte pro Position schwanken
    (Std-Interpolation über die Fläche)."""
    x = sub["x"].to_numpy()
    y = sub["y"].to_numpy()
    std = sub["rssi_std"].to_numpy()
 
    if np.allclose(std, 0):
        print(f"[skip-var] {bssid}: keine Mehrfachmessungen, Std überall 0")
        return
 
    grid_x, grid_y, grid_z = _interp_grid(x, y, std, method="linear")
 
    fig, ax = plt.subplots(figsize=(7, 6))
 
    im = ax.imshow(
        grid_z.T,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=max(std.max(), 1.0),
        aspect="equal",
    )
 
    sc = ax.scatter(
        x, y, c=std, cmap="viridis",
        vmin=0, vmax=max(std.max(), 1.0),
        edgecolors="black", linewidths=0.6, s=45, zorder=3,
    )
 
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RSSI-Standardabweichung [dB]")
 
    title = f"Schwankungsbreite – {ssid} ({bssid})" if ssid else f"Schwankungsbreite – {bssid}"
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
 
    fname = bssid.replace(":", "-") + "_std.png"
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"[ok] Schwankungs-Heatmap gespeichert: {fname}")
 
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_raw = load_data(JSON_PATH)
    df = aggregate_positions(df_raw)
 
    n_positions = df[["x", "y"]].drop_duplicates().shape[0]
    print(f"Geladen: {len(df_raw)} Rohmessungen -> {len(df)} (Position, BSSID)-Cluster "
          f"über {df['bssid'].nunique()} BSSIDs und {n_positions} Positionen.")
 
    bssids = select_bssids(df, TOP_N_BSSIDS)
    print(f"Erzeuge Heatmaps für {len(bssids)} BSSID(s)...")
 
    for bssid in bssids:
        sub = df[df["bssid"] == bssid]
        ssid = sub["ssid"].iloc[0] if not sub.empty else ""
 
        if len(sub) < MIN_POINTS:
            print(f"[skip] {bssid} ({ssid}): zu wenige distinkte Positionen ({len(sub)})")
            continue
 
        plot_mean_heatmap(sub, bssid, ssid, OUTPUT_DIR)
        plot_variability_heatmap(sub, bssid, ssid, OUTPUT_DIR)
 
 
if __name__ == "__main__":
    main()
 222222222222