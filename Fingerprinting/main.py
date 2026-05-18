# main Datei zur Fingerprinting von WiFi Signalstärken
from agv_position import AGVPosition
from wifi_scan import WiFiScanner
from json_storage import JSONStorage
import time


def main():
    storage = JSONStorage("C:\Dokumente\Studium\Master\Masterarbeit\Code\Fingerprinting/fingerprints.json")
    agv = AGVPosition()
    wifi = WiFiScanner()
    print("Programm gestartet...\n")

    while True:
        position = agv.get_position()
        networks = wifi.scan_networks()
        storage.save(
            networks,
            position
        )
        print(position)
        print(networks)
        
        time.sleep(5)


if __name__ == "__main__":
    main()