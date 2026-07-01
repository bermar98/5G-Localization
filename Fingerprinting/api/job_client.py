# Modul zum Ansprechen der Job-API (Job-ID ermitteln + resolveWaitAction)
import requests
from urllib3.exceptions import InsecureRequestWarning

# Da 192.168.10.1 vermutlich ein selbstsigniertes Zertifikat verwendet,
# unterdrücken wir die entsprechende Warnung von urllib3.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


class JobClient:
    """
    Kleiner API-Client, der Job-bezogene Endpunkte des AGV-Systems anspricht.

    Ablauf:
      1. get_job_id() -> holt alle Jobs zu einer sectorReference und sucht
         darin per Namen (z.B. "WiFi-Messung") die passende Job-ID.
      2. resolve_wait_action() -> ruft zuerst get_job_id() auf und danach
         /v2/job/{job_id}/resolveWaitAction für genau diesen Job.
    """

    def __init__(
        self,
        base_url,
        api_key,
        sector_reference,
        job_name="WiFi-Messung",
        start_range="2024-07-04T12:34:56Z",
        include_archived=False,
        verify_ssl=False,
        timeout=5.0,
    ):
        """
        :param base_url: z.B. "https://192.168.10.1:9089"
        :param api_key: der API-Key als Query-Parameter
        :param sector_reference: sectorReference, in dem nach dem Job gesucht wird
        :param job_name: Name des gesuchten Jobs (exakter Vergleich)
        :param start_range: unterer Zeit-Grenzwert (ISO 8601, z.B. "2024-07-04T12:34:56Z")
        :param include_archived: ob archivierte Jobs mit einbezogen werden sollen
        :param verify_ssl: ob das SSL-Zertifikat geprüft werden soll (bei self-signed: False)
        :param timeout: Timeout in Sekunden für die Requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.sector_reference = sector_reference
        self.job_name = job_name
        self.start_range = start_range
        self.include_archived = include_archived
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def get_job_id(self):
        """
        Fragt alle Jobs zur konfigurierten sectorReference ab und sucht darin
        per Namen (self.job_name) die passende Job-ID.

        :return: die Job-ID als String, oder None falls kein passender Job gefunden wurde
        """
        url = f"{self.base_url}/v2/job/bySectorReference/{self.sector_reference}"
        params = {
            "includeArchived": str(self.include_archived).lower(),
            "startRange": self.start_range,
            "apikey": self.api_key,
        }
        headers = {"accept": "application/json"}

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Je nach API-Antwortformat kann die Job-Liste direkt oder
            # unter einem Schlüssel (z.B. "jobs") verschachtelt sein.
            jobs = data if isinstance(data, list) else data.get("jobs", [])

            for job in jobs:
                if job.get("name", "").strip() == self.job_name:
                    job_id = job.get("_id")
                    print(f"[JobClient] Job '{self.job_name}' gefunden -> _id={job_id}")
                    return job_id

            print(f"[JobClient] Kein Job mit Namen '{self.job_name}' gefunden.")
            return None

        except requests.exceptions.RequestException as e:
            print(f"[JobClient] Fehler beim Abfragen der Job-Liste: {e}")
            return None

    def resolve_wait_action(self):
        """
        Ermittelt zuerst die aktuelle Job-ID (per Namen) und ruft danach
        /v2/job/{job_id}/resolveWaitAction auf, um eine wartende
        TriggerWait-Action im Job aufzulösen.

        :return: True bei Erfolg (HTTP 2xx), False bei Fehler
        """
        job_id = self.get_job_id()
        if not job_id:
            print("[JobClient] Abbruch: keine gültige Job-ID gefunden, resolveWaitAction wird nicht aufgerufen.")
            return False

        url = f"{self.base_url}/v2/job/{job_id}/resolveWaitAction"
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
