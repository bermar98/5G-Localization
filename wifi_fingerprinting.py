import csv
import time
from datetime import datetime
from pywifi import PyWiFi


CSV_FILE = "wifi_rssi_dataset.csv"


def scan_wifi():

    wifi = PyWiFi()

    iface = wifi.interfaces()[0]

    iface.scan()

    time.sleep(3)

    results = iface.scan_results()

    networks = {}

    for r in results:

        ssid = r.ssid.strip()

        # Leere SSIDs ignorieren
        if ssid == "":
            continue

        networks[ssid] = r.signal

    return networks


def load_existing_ssids():

    try:
        with open(CSV_FILE, "r", encoding="utf-8") as file:

            reader = csv.reader(file)

            header = next(reader)

            return header[1:]

    except FileNotFoundError:
        return []


def update_csv(networks):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing_ssids = load_existing_ssids()

    # Neue SSIDs ergänzen
    all_ssids = sorted(
        list(set(existing_ssids + list(networks.keys())))
    )

    rows = []

    # Alte Daten laden
    try:
        with open(CSV_FILE, "r", encoding="utf-8") as file:

            reader = csv.reader(file)

            rows = list(reader)

    except FileNotFoundError:
        pass

    # Neue Header erzeugen
    header = ["timestamp"] + all_ssids

    # Neue Messzeile
    row = [timestamp]

    for ssid in all_ssids:

        if ssid in networks:
            row.append(networks[ssid])
        else:
            row.append("")

    # Datei neu schreiben
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(header)

        # Alte Daten übernehmen
        if len(rows) > 1:

            old_header = rows[0]

            for old_row in rows[1:]:

                old_dict = dict(zip(old_header, old_row))

                new_row = []

                for col in header:
                    new_row.append(old_dict.get(col, ""))

                writer.writerow(new_row)

        # Neue Messung hinzufügen
        writer.writerow(row)


def main():

    print("📡 WLAN Fingerprint Logger gestartet...\n")

    while True:

        networks = scan_wifi()

        print("Gefundene Netzwerke:\n")

        for ssid, rssi in networks.items():
            print(f"{ssid:30} {rssi} dBm")

        update_csv(networks)

        print("\nCSV aktualisiert.")
        print("-" * 70)

        #time.sleep(5)


if __name__ == "__main__":
    main()