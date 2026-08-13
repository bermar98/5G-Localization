"""
Filterskript für WiFi-Fingerprint-Rohdaten (Schritt 1: Rohdatenbereinigung)

Wendet folgende Filterschritte an, wie besprochen:
  1. Seltene/instabile BSSIDs entfernen (Sichtbarkeit unter einem Schwellwert)
  2. Ausreißer pro BSSID entfernen (RSSI-Werte weit außerhalb der 3-Sigma-Umgebung
     dieses BSSIDs, um Multipath-/Störeinflüsse abzufangen)
  3. Fehlende BSSIDs bleiben in dieser Stufe implizit fehlend (kein RSSI-Eintrag) -
     die feste Vektorlänge und der Fill-Value (-100 dBm) werden erst im nächsten
     Schritt (Feature-Building / DatasetBuilder) erzeugt, damit dieses Skript
     ausschließlich für die Rohdatenbereinigung zuständig bleibt.

Input:  fingerprints.json  (Liste von {timestamp, position, fingerprint})
Output: fingerprints_filtered.json (gleiche Struktur, bereinigt)
        filter_report.json (Statistik darüber, was entfernt wurde)
"""

import json
import statistics
from pathlib import Path

# ---- Konfiguration ----
BASE_DIR   = Path(__file__).resolve().parent
INPUT_PATH  = BASE_DIR / "fingerprints.json"
OUTPUT_PATH = BASE_DIR / "fingerprints_filtered.json"
REPORT_PATH = BASE_DIR / "fingerprints_filter_report.json"

MIN_VISIBILITY_RATIO = 0.010   # BSSID muss in mind. 10% aller Messungen sichtbar sein
OUTLIER_Z_THRESHOLD = 30.0     # RSSI-Ausreißer: |z-score| > 3 pro BSSID wird entfernt


def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_bssid_visibility(data):
    """Zählt, in wie vielen Messungen jede BSSID vorkommt."""
    counts = {}
    for rec in data:
        for bssid in rec["fingerprint"]:
            counts[bssid] = counts.get(bssid, 0) + 1
    return counts


def compute_bssid_rssi_stats(data, keep_bssids):
    """Mittelwert und Standardabweichung des RSSI pro BSSID (nur für behaltene BSSIDs)."""
    values = {}
    for rec in data:
        for bssid, v in rec["fingerprint"].items():
            if bssid in keep_bssids:
                values.setdefault(bssid, []).append(v["rssi"])

    stats = {}
    for bssid, rssis in values.items():
        if len(rssis) >= 2:
            mean = statistics.mean(rssis)
            stdev = statistics.pstdev(rssis)
        else:
            mean, stdev = rssis[0], 0.0
        stats[bssid] = (mean, stdev)
    return stats


def filter_data(data):
    report = {}

    # --- Schritt 1: seltene BSSIDs bestimmen und verwerfen ---
    total_records = len(data)
    visibility = compute_bssid_visibility(data)
    min_count = MIN_VISIBILITY_RATIO * total_records

    kept_bssids = {b for b, c in visibility.items() if c >= min_count}
    dropped_bssids = {b for b in visibility if b not in kept_bssids}

    report["total_records"] = total_records
    report["total_bssids_before"] = len(visibility)
    report["bssids_dropped_rare"] = sorted(dropped_bssids)
    report["bssids_kept"] = len(kept_bssids)

    # --- Schritt 2: RSSI-Ausreißer pro BSSID bestimmen (z-Score) ---
    rssi_stats = compute_bssid_rssi_stats(data, kept_bssids)

    outlier_count = 0
    cleaned_data = []

    for rec in data:
        new_fp = {}
        for bssid, v in rec["fingerprint"].items():
            if bssid not in kept_bssids:
                continue  # bereits als seltene BSSID verworfen

            mean, stdev = rssi_stats[bssid]
            if stdev > 0:
                z = (v["rssi"] - mean) / stdev
                if abs(z) > OUTLIER_Z_THRESHOLD:
                    outlier_count += 1
                    continue  # Ausreißer verwerfen

            new_fp[bssid] = v

        cleaned_rec = {
            "timestamp": rec["timestamp"],
            "position": rec["position"],
            "fingerprint": new_fp,
        }
        cleaned_data.append(cleaned_rec)

    report["rssi_outliers_removed"] = outlier_count

    # --- Statistik zur Fingerprint-Länge nach Filterung ---
    fp_lens = [len(rec["fingerprint"]) for rec in cleaned_data]
    report["fingerprint_length_after"] = {
        "min": min(fp_lens),
        "max": max(fp_lens),
        "mean": round(statistics.mean(fp_lens), 2),
    }

    return cleaned_data, report


def main():
    data = load_data(INPUT_PATH)
    cleaned_data, report = filter_data(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Fertig. {report['total_records']} Messungen verarbeitet.")
    print(f"BSSIDs: {report['total_bssids_before']} -> {report['bssids_kept']} "
          f"({len(report['bssids_dropped_rare'])} entfernt wegen Seltenheit)")
    print(f"RSSI-Ausreißer entfernt: {report['rssi_outliers_removed']}")
    print(f"Gefiltertes Ergebnis: {OUTPUT_PATH}")
    print(f"Bericht: {REPORT_PATH}")


if __name__ == "__main__":
    main()