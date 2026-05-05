import requests
import urllib3
import json
import time
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://192.168.10.1:9089/v2/droid"

params = {
    "apikey": "198a79262aa221793baa8c87dd26601d2dac0706512d8ba83f34e6035c45dc05" # 
}

headers = {
    "accept": "application/json"
}

datei = r"C:\Dokumente\Studium\Master\Masterarbeit\Code\agv_tracking.json"

print("Tracking gestartet... STRG+C zum Stoppen")

try:
    while True:
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                verify=False,
                timeout=5
            )

            data = response.json()
            droid = data[0]

            pos = droid["fromDroid"]["agvPosition"]

            output = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "x": pos["x"],
                "y": pos["y"],
                "theta": pos["theta"]
            }

            with open(datei, "a", encoding="utf-8") as f:
                f.write(json.dumps(output) + "\n")

            print(output)

        except Exception as e:
            print("Fehler bei Abfrage:", e)

        time.sleep(1)   # jede 1 Sekunde messen

except KeyboardInterrupt:
    print("Tracking beendet.")