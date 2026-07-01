import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class AGVPosition:
    
    def __init__(self):

        self.url = "https://192.168.10.1:9089/v2/droid"

        self.params = {
            "apikey": "198a79262aa221793baa8c87dd26601d2dac0706512d8ba83f34e6035c45dc05"
        }

        self.headers = {
            "accept": "application/json"
        }

    def get_position(self):
        
    

        try:

            response = requests.get(
                self.url,
                params=self.params,
                headers=self.headers,
                verify=False,
                timeout=5
            )

            data = response.json()

            droid = data[0]

            pos = droid["fromDroid"]["agvPosition"]

            return {
                "x": pos["x"],
                "y": pos["y"],
                "theta": pos["theta"]
            }
        

        except Exception as e:

            print("Fehler bei Positionsabfrage:", e)

            return None