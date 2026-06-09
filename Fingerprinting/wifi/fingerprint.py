class FingerprintBuilder:
    
    @staticmethod
    def create_fingerprint(networks):

        fingerprint = {}

        for network in networks:

           fingerprint[network["bssid"]] = {
                "ssid": network["ssid"],
                "rssi": network["rssi"],
                "distance": network["distance_m"]
            }
        return fingerprint