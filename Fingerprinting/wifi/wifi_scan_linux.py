import subprocess
import re

class WiFiScanner:

    def __init__(self):
        self.interface = "wlan0"
        
    def _rssi_to_distance(self, rssi, tx_power=-40, path_loss_exp=2.7):
        '''
        tx_power:       RSSI bei 1m Abstand (typisch -40 bis -50 dBm)
        path_loss_exp:  2.7 typisch für Innenräume (2.0 = Freifläche)
        '''
        distance = 10 ** ((tx_power - rssi) / (10 * path_loss_exp))
        return round(distance, 2)

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
                    if current and "ssid" in current and "bssid" in current:
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
                        current["distance_m"] = self._rssi_to_distance(rssi)

            if current and "ssid" in current and "bssid" in current:
                networks.append(current)

            return networks

        except Exception as e:
            print("Fehler beim WLAN-Scan:", e)
            return []