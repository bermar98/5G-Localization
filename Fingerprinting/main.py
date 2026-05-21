# main Datei zur Fingerprinting von WiFi Signalstärken
# Aus den anderen Dateien die einzelnen Klassen importieren
from positioning.agv_position import AGVPosition
from wifi.wifi_scan import WiFiScanner
from wifi.fingerprint import FingerprintBuilder
from storage.json_storage import Storage
import time


def main():
    # Instanz der Klassen definieren:
    storage = Storage("C:\Dokumente\Studium\Master\Masterarbeit\Code\Fingerprinting\data/fingerprints.json")
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