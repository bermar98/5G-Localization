import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "fingerprints.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "fingerprints_restructured.json")


def restructure_fingerprint(fingerprint_list):

    fingerprint_dict = {}

    for network in fingerprint_list:

        bssid = network.get("bssid")

        if not bssid:
            continue

        fingerprint_dict[bssid] = {
            "ssid": network.get("ssid"),
            "rssi": network.get("rssi")
        }

    return fingerprint_dict

def main():

    # Datei laden
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Daten umstrukturieren
    for sample in data:

        sample["fingerprint"] = restructure_fingerprint(
            sample["fingerprint"]
        )

    # Neue Datei speichern
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"✔ Datei erfolgreich umgeschrieben: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()