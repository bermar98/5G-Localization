'''from pywifi import PyWiFi
import time


class WiFiScanner:

    def __init__(self):

        wifi = PyWiFi()

        self.iface = wifi.interfaces()[0]

    def scan_networks(self):

        self.iface.scan()

        time.sleep(3)

        results = self.iface.scan_results()

        networks = []

        for r in results:

            if r.ssid.strip() == "":
                continue

            networks.append({
                "ssid": r.ssid, "bssid": r.bssid, "rssi": r.signal
            })

        return networks'''
        
import subprocess
import re

class WiFiScanner:

    def __init__(self):
        self.interface = "wlan0"

    def scan_networks(self):
        try:
            result = subprocess.run(
                ["sudo", "iwlist", self.interface, "scan"],
                capture_output=True,
                text=True,
                timeout=10
            )

            networks = []
            current = {}

            for line in result.stdout.split("\n"):
                line = line.strip()

                if "Cell" in line and "Address" in line:
                    if current:
                        networks.append(current)
                    current = {"bssid": line.split("Address: ")[1]}

                elif "ESSID" in line:
                    ssid = line.split('"')[1]
                    if ssid.strip() == "":
                        continue
                    current["ssid"] = ssid

                elif "Signal level" in line:
                    match = re.search(r"Signal level=(-?\d+)", line)
                    if match:
                        current["rssi"] = int(match.group(1))

            if current and "ssid" in current:
                networks.append(current)

            return networks

        except Exception as e:
            print("Fehler beim WLAN-Scan:", e)
            return []