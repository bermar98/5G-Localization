# main Datei zur Fingerprinting von WiFi Signalstärken
# Aus den anderen Dateien die einzelnen Klassen importieren
from agv_position import AGVPosition
from wifi_scan import WiFiScanner
from fingerprint import FingerprintBuilder
from json_storage import Storage
import time


def main():
    # Instanz der Klassen definieren:
    storage = Storage("C:\Dokumente\Studium\Master\Masterarbeit\Code\Fingerprinting/fingerprints.json")
    agv = AGVPosition()
    wifi = WiFiScanner()
    fingerprint_builder = FingerprintBuilder()
    print("Programm gestartet...\n")

    while True:
        # Funktionen der Instanzen hinzugeügt
        position = agv.get_position()
        networks = wifi.scan_networks()
        fingerprint = fingerprint_builder.create_fingerprint(networks)
        storage.save(
            fingerprint,
            position
        )
        print(position)
        print(fingerprint)
        
        #time.sleep(5)


if __name__ == "__main__":
    main()