import subprocess
import re
import csv
import time
from datetime import datetime


CSV_FILE = "wifi_rssi_dataset.csv"


def scan_wifi():

    result = subprocess.check_output(
        ["netsh", "wlan", "show", "networks", "mode=bssid"],
        text=True,
        encoding="cp1252",
        errors="ignore"
    )

    raw_text = result.splitlines()

    networks = []

    current_ssid = None
    current_bssid = None

    for line in raw_text:

        line = line.strip()

        # -----------------------------
        # SSID
        # -----------------------------
        ssid_match = re.search(
            r"SSID\s+\d+\s+:\s(.+)",
            line
        )

        if ssid_match:
            current_ssid = ssid_match.group(1).strip()

        # -----------------------------
        # BSSID
        # -----------------------------
        bssid_match = re.search(
            r"BSSID\s+\d+\s+:\s([0-9A-Fa-f:]{17})",
            line
        )

        if bssid_match:
            current_bssid = bssid_match.group(1)

        # -----------------------------
        # Signal
        # -----------------------------
        signal_match = re.search(
            r"Signal\s+:\s+(\d+)%",
            line
        )

        if signal_match:

            signal_percent = int(signal_match.group(1))

            # Approximation
            rssi_dbm = (signal_percent / 2) - 100

            networks.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ssid": current_ssid,
                "bssid": current_bssid,
                "signal_percent": signal_percent,
                "rssi_dbm": round(rssi_dbm, 2)
            })

    return networks


def save_to_csv(networks):

    file_exists = False

    try:
        with open(CSV_FILE, "r", encoding="utf-8"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:

        fieldnames = [
            "timestamp",
            "ssid",
            "bssid",
            "signal_percent",
            "rssi_dbm"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for network in networks:
            writer.writerow(network)


def main():

    print("📡 WLAN Scanner gestartet...\n")

    while True:

        networks = scan_wifi()

        print(f"\nGefundene Netzwerke: {len(networks)}\n")

        for n in networks:

            print(
                f"SSID: {n['ssid']:25} "
                f"BSSID: {n['bssid']} "
                f"RSSI: {n['rssi_dbm']} dBm"
            )

        save_to_csv(networks)

        print("\nCSV gespeichert.")
        print("-" * 70)

        time.sleep(5)


if __name__ == "__main__":
    main()