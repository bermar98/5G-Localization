# Nach erfolgreicher Positions- und Netzwerkmessung
 # Modul zum Ansprechen der Job-API (z.B. resolveWaitAction)
import requests
from urllib3.exceptions import InsecureRequestWarning

# Da 192.168.10.1 vermutlich ein selbstsigniertes Zertifikat verwendet,
# unterdrücken wir die entsprechende Warnung von urllib3.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


class JobClient:
    """
    Kleiner API-Client, der Job-bezogene Endpunkte des AGV-Systems anspricht.
    """

    def __init__(self, base_url: str, job_id: str, api_key: str, verify_ssl: bool = False, timeout: float = 5.0):
        """
        :param base_url: z.B. "https://192.168.10.1:9089"
        :param job_id: die Job-ID, z.B. "6a44c9b5c7395a0493930b6c"
        :param api_key: der API-Key als Query-Parameter
        :param verify_ssl: ob das SSL-Zertifikat geprüft werden soll (bei self-signed: False)
        :param timeout: Timeout in Sekunden für den Request
        """
        self.base_url = base_url.rstrip("/")
        self.job_id = job_id
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def resolve_wait_action(self) -> bool:
        """
        Ruft /v2/job/{job_id}/resolveWaitAction auf, um eine wartende
        TriggerWait-Action im Job aufzulösen.

        :return: True bei Erfolg (HTTP 2xx), False bei Fehler
        """
        url = f"{self.base_url}/v2/job/{self.job_id}/resolveWaitAction"
        params = {"apikey": self.api_key}

        try:
            response = requests.post(
                url,
                params=params,
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()
            print(f"[JobClient] resolveWaitAction erfolgreich (HTTP {response.status_code})")
            return True

        except requests.exceptions.RequestException as e:
            print(f"[JobClient] Fehler beim Aufruf von resolveWaitAction: {e}")
            return False
