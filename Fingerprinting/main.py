# main Datei zur Fingerprinting von WiFi Signalstärken
# Aus den anderen Dateien die einzelnen Klassen importieren
from positioning.agv_position import AGVPosition
from wifi.wifi_scan import WiFiScanner
from wifi.fingerprint import FingerprintBuilder
from storage.json_storage import Storage
from api.job_client import JobClient
import time
from pathlib import Path

print(Path(__file__).resolve())
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "fingerprints.json"

# --- Konfiguration für den Job-API-Aufruf ---
API_BASE_URL = "https://192.168.10.1:9089"
API_KEY = "198a79262aa221793baa8c87dd26601d2dac0706512d8ba83f34e6035c45dc05"
SECTOR_REFERENCE = "6a43b1fdc6575ba89db7599b"
JOB_NAME = "WiFi-Messung"

def main():
    # Instanz der Klassen definieren:
    storage = Storage(DATA_PATH)
    agv = AGVPosition()
    wifi = WiFiScanner()
    fingerprint_builder = FingerprintBuilder()
    job_client = JobClient(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        sector_reference=SECTOR_REFERENCE,
        job_name=JOB_NAME,
    )

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
        
        
        # Nach erfolgreicher Positions- und Netzwerkmessung
        # die wartende Trigger-Action im Job auflösen
        job_client.resolve_wait_action()
        
        
        time.sleep(30)


if __name__ == "__main__":
    main()