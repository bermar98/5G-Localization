# =============================================================================
#  lmf_client.py
#  LMF (Location Management Function) API Client
#
#  Implementiert die 3GPP TS 29.572 Nlmf_Location Service API.
#
#  Zwei Modi:
#  1. LIVE  → sendet echte HTTP-Anfrage an Open5GS LMF
#  2. MOCK  → simuliert LMF-Antwort lokal (zum Testen ohne Campusnetz)
#
#  3GPP Ablauf (vereinfacht):
#    Client → POST /nlmf-loc/v1/provide-loc-info  → LMF
#    LMF    → koordiniert gNBs via NRPPa
#    LMF    → antwortet mit geschätzter Position
#
#  Installation:
#   sudo apt install python3-httpx
#   sudo apt install python3-h2
#   
# =============================================================================

import json
import time
import uuid
import datetime
import argparse
from typing import Optional

try:
    import httpx
    # HTTP/2 braucht das h2-Paket: pip install "httpx[http2]"
    import h2  # noqa: F401
    HTTP2_AVAILABLE = True
except ImportError:
    HTTP2_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

if not HTTP2_AVAILABLE and not REQUESTS_AVAILABLE:
    raise ImportError("Bitte installieren: pip install requests")

if not HTTP2_AVAILABLE:
    print("[Info] HTTP/2 nicht verfügbar – nutze HTTP/1.1 (requests)")
    print("       Für echtes Open5GS: pip install 'httpx[http2]'")


# =============================================================================
#  Konfiguration
# =============================================================================

LMF_BASE_URL    = "http://127.0.0.16:7777"   # Open5GS Standard-LMF-Adresse
LMF_API_VERSION = "v1"
TIMEOUT_S       = 10.0

# Endpunkte nach 3GPP TS 29.572
ENDPOINTS = {
    "provide_location":    f"/nlmf-loc/{LMF_API_VERSION}/provide-loc-info",
    "cancel_location":     f"/nlmf-loc/{LMF_API_VERSION}/cancel-loc-info",
    "location_deferred":   f"/nlmf-loc/{LMF_API_VERSION}/loc-context-transfer",
}


# =============================================================================
#  LMF Request Body nach 3GPP TS 29.572
# =============================================================================

def build_location_request(
    supi:           str,
    plmn_id:        dict,
    cell_id:        str,
    lcs_client_type: str = "EMERGENCY_SERVICES",
    loc_type:       str = "CURRENT_LOCATION",
    accuracy:       str = "MA",              # MA = Most Accurate
    max_age_s:      int = 0,
    response_time:  str = "LOW_DELAY",
) -> dict:
    """
    Baut den JSON-Request-Body für POST /nlmf-loc/v1/provide-loc-info.

    Nach 3GPP TS 29.572 §6.1.3.2.2 InputData:

    supi            : UE-Identifier z.B. "imsi-999700000021635"
    plmn_id         : {"mcc": "999", "mnc": "70"}
    cell_id         : Global Cell ID des Serving-gNB
    lcs_client_type : Wer fragt die Position an
    loc_type        : CURRENT_LOCATION / CURRENT_OR_LAST_KNOWN_LOCATION
    accuracy        : MA (Most Accurate) / PREFERRED_LOW_DELAY
    response_time   : LOW_DELAY / DELAY_TOLERANT / NO_DELAY
    """
    return {
        # UE-Identifikation
        "supi": supi,

        # Serving Cell Information (woher kommt die Anfrage)
        "servingCellId": {
            "plmnId":   plmn_id,
            "cellId":   cell_id,
        },

        # LCS Client (wer fragt die Position an)
        "lcsClientType": lcs_client_type,

        # Positionsanforderung
        "locationType": loc_type,

        # Qualitäts-Anforderungen (QoS)
        "locationQos": {
            "hAccuracy":     10.0,   # Horizontale Genauigkeit [m]
            "vAccuracy":     5.0,    # Vertikale Genauigkeit [m]
            "responseTime":  response_time,
            "maxAge":        max_age_s,
        },

        # Unterstützte Positioning-Methoden (LMF wählt aus)
        "supportedFeatures": "PRS_MEAS",

        # Referenz-ID für diese Anfrage
        "correlationID": str(uuid.uuid4()),
    }


# =============================================================================
#  API-Antwort parsen
# =============================================================================

def parse_location_response(response_json: dict) -> dict:
    """
    Parst die LMF-Antwort nach 3GPP TS 29.572 §6.1.3.2.3.

    Mögliche Positionsformate in der Antwort:
    - locationEstimate.point           → einfacher 2D-Punkt
    - locationEstimate.pointWithUncert → Punkt mit Unsicherheitsradius
    - locationEstimate.ellipsoidPoint  → Ellipsoid
    - civicAddress                     → Adresse (Gebäude, Raum)
    """
    result = {
        "success":    False,
        "latitude":   None,
        "longitude":  None,
        "altitude":   None,
        "accuracy_m": None,
        "method":     None,
        "timestamp":  None,
        "raw":        response_json,
    }

    # Positionsschätzung extrahieren
    loc = response_json.get("locationEstimate", {})
    if not loc:
        result["error"] = "Keine Positionsschätzung in der Antwort"
        return result

    shape = loc.get("shape", "")

    # Punkt (einfachstes Format)
    if shape in ("POINT", "POINT_UNCERTAINTY_CIRCLE"):
        point = loc.get("point", {})
        result["latitude"]  = point.get("lat")
        result["longitude"] = point.get("lon")
        result["altitude"]  = loc.get("altitude")
        result["accuracy_m"] = loc.get("uncertainty")
        result["success"]   = True

    # Ellipse
    elif shape == "POINT_UNCERTAINTY_ELLIPSE":
        point = loc.get("point", {})
        result["latitude"]  = point.get("lat")
        result["longitude"] = point.get("lon")
        result["accuracy_m"] = loc.get("uncertaintySemiMajor")
        result["success"]   = True

    # Polygon (nimm Schwerpunkt)
    elif shape == "POLYGON":
        points = loc.get("polygon", {}).get("pointList", [])
        if points:
            result["latitude"]  = sum(p["lat"] for p in points) / len(points)
            result["longitude"] = sum(p["lon"] for p in points) / len(points)
            result["success"]   = True

    # Positioning-Methode
    result["method"]    = response_json.get("positioningDataList",
                                            [{}])[0].get("positioningMethod")
    result["timestamp"] = response_json.get("ageOfLocationEstimate",
                                            datetime.datetime.now().isoformat())
    return result


# =============================================================================
#  LMF HTTP-Client
# =============================================================================

class LMFClient:
    """
    HTTP-Client für die Open5GS LMF API.
    Unterstützt HTTP/2 (httpx) und HTTP/1.1 (requests) automatisch.
    """

    def __init__(self, base_url: str = LMF_BASE_URL,
                 timeout: float = TIMEOUT_S):
        self.base_url     = base_url.rstrip("/")
        self.timeout      = timeout
        self.is_reachable = False
        self._mock        = MockLMFClient()
        self._check_connection()

    def _check_connection(self):
        """Prüft ob die LMF erreichbar ist und setzt is_reachable."""
        try:
            url = f"{self.base_url}/nlmf-loc/{LMF_API_VERSION}"
            if HTTP2_AVAILABLE:
                with httpx.Client(http2=True, timeout=2.0) as client:
                    client.get(url)
            elif REQUESTS_AVAILABLE:
                requests.get(url, timeout=2.0)
            self.is_reachable = True
            print(f"[LMF] Verbunden: {self.base_url}")
        except Exception as e:
            self.is_reachable = False
            print(f"[LMF] Nicht erreichbar: {self.base_url}")
            print(f"[LMF] Grund: {type(e).__name__}: {e}")
            print(f"[LMF] Automatischer Fallback auf Mock-Modus")

    def request_location(self, request_body: dict) -> dict:
        """
        POST /nlmf-loc/v1/provide-loc-info
        Fällt automatisch auf Mock zurück wenn LMF nicht erreichbar.
        """
        if not self.is_reachable:
            print("[LMF] Nutze Mock (LMF nicht erreichbar)")
            return self._mock.request_location(request_body)

        url     = self.base_url + ENDPOINTS["provide_location"]
        headers = {
            "Content-Type": "application/json",
            "Accept":        "application/json",
        }

        print(f"\n[LMF] POST {url}")
        print(f"[LMF] SUPI: {request_body.get('supi')}")
        print(f"[LMF] Korrelations-ID: {request_body.get('correlationID')}")

        try:
            if HTTP2_AVAILABLE:
                with httpx.Client(http2=True, timeout=self.timeout) as client:
                    response = client.post(url, json=request_body,
                                           headers=headers)
            else:
                response = requests.post(url, json=request_body,
                                         headers=headers,
                                         timeout=self.timeout)

            print(f"[LMF] HTTP Status: {response.status_code}")

            if response.status_code in (200, 201):
                return parse_location_response(response.json())
            else:
                return {
                    "success": False,
                    "error":   f"HTTP {response.status_code}: {response.text}"
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cancel_location(self, correlation_id: str) -> bool:
        """DELETE – bricht laufende Positionsanfrage ab."""
        url = (self.base_url
               + ENDPOINTS["cancel_location"]
               + f"/{correlation_id}")
        try:
            if HTTP2_AVAILABLE:
                with httpx.Client(http2=True, timeout=self.timeout) as c:
                    r = c.delete(url)
            else:
                r = requests.delete(url, timeout=self.timeout)
            return r.status_code == 204
        except Exception:
            return False


# =============================================================================
#  Mock-LMF für Tests ohne echtes Campusnetz
# =============================================================================

class MockLMFClient:
    """
    Simuliert LMF-Antworten lokal.
    Nützlich zum Testen der API-Integration ohne echtes Open5GS.
    Antwortet mit realistischen Werten basierend auf dem Request.
    """

    def request_location(self, request_body: dict) -> dict:
        import random
        time.sleep(0.1)   # simulierte Latenz

        # Simulierte Position (leicht verrauscht)
        base_lat = 48.1374 + random.gauss(0, 0.0001)
        base_lon = 11.5755 + random.gauss(0, 0.0001)

        mock_response = {
            "locationEstimate": {
                "shape":      "POINT_UNCERTAINTY_CIRCLE",
                "point":      {"lat": base_lat, "lon": base_lon},
                "altitude":   1.5,
                "uncertainty": round(random.uniform(5.0, 25.0), 2),
            },
            "positioningDataList": [
                {"positioningMethod": "DL_TDOA_PRS"}
            ],
            "ageOfLocationEstimate": datetime.datetime.now().isoformat(),
            "correlationID":  request_body.get("correlationID"),
        }

        print(f"\n[MockLMF] Simulierte Antwort:")
        print(f"[MockLMF] Position: ({base_lat:.6f}, {base_lon:.6f})")

        return parse_location_response(mock_response)

    def cancel_location(self, correlation_id: str) -> bool:
        return True


# =============================================================================
#  PRS-Signalauslesen aus LMF-Assistenzdaten
# =============================================================================

def request_prs_assistance_data(
    client,
    supi:    str,
    plmn_id: dict,
    cell_id: str,
) -> dict:
    """
    Fordert PRS-Assistenzdaten von der LMF an.

    Die LMF gibt zurück welche gNBs PRS senden und mit welchen Parametern:
    - NPRSID pro gNB
    - Slot-Offsets
    - CombSize, NumSymbols
    - gNB-Koordinaten

    Das ist die Grundlage für Modell B – der Server fragt die LMF
    nach den PRS-Parametern statt sie statisch in config.py zu haben.

    Nach 3GPP TS 37.355 §6.5.1 (LPP: Provide Capabilities)
    und TS 38.455 (NRPPa: Positioning Information Exchange)
    """
    request_body = build_location_request(
        supi            = supi,
        plmn_id         = plmn_id,
        cell_id         = cell_id,
        lcs_client_type = "VALUE_ADDED_SERVICES",
        loc_type        = "CURRENT_LOCATION",
        accuracy        = "MA",
        response_time   = "LOW_DELAY",
    )
    # Assistenzdaten-Flag hinzufügen
    request_body["supportedFeatures"] = "PRS_ASSIST_DATA"

    response = client.request_location(request_body)

    # PRS-Assistenzdaten aus Antwort extrahieren
    prs_params = []
    raw = response.get("raw", {})
    assist = raw.get("prsAssistanceData", {})
    for gnb_data in assist.get("nrPRSResourceList", []):
        prs_params.append({
            "gnb_id":       gnb_data.get("gNB-ID"),
            "nprs_id":      gnb_data.get("nPRS-ID"),
            "slot_offset":  gnb_data.get("slotOffset"),
            "comb_size":    gnb_data.get("combSize"),
            "num_symbols":  gnb_data.get("numPRSSymbols"),
            "gnb_position": gnb_data.get("gNBPosition"),
        })

    response["prs_params"] = prs_params
    return response


# =============================================================================
#  Hilfsfunktionen: Signalqualität auslesen
# =============================================================================

def print_location_result(result: dict):
    """Gibt das Ergebnis einer Positionsanfrage übersichtlich aus."""
    print("\n" + "="*55)
    print("  LMF Positionierungsergebnis")
    print("="*55)

    if not result["success"]:
        print(f"  Status  : FEHLER")
        print(f"  Ursache : {result.get('error', 'Unbekannt')}")
        print("="*55)
        return

    print(f"  Status      : OK")
    print(f"  Breitengrad : {result['latitude']:.6f}°")
    print(f"  Längengrad  : {result['longitude']:.6f}°")
    if result.get("altitude"):
        print(f"  Höhe        : {result['altitude']:.1f} m")
    if result.get("accuracy_m"):
        print(f"  Genauigkeit : ± {result['accuracy_m']:.1f} m")
    if result.get("method"):
        print(f"  Methode     : {result['method']}")
    print(f"  Zeitstempel : {result['timestamp']}")
    print("="*55 + "\n")


# =============================================================================
#  Hauptprogramm
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="5G LMF API Client – Positionsanfrage an Open5GS LMF"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock-Modus: simuliert LMF lokal (kein Campusnetz nötig)"
    )
    parser.add_argument(
        "--url", default=LMF_BASE_URL,
        help=f"LMF-URL (Standard: {LMF_BASE_URL})"
    )
    parser.add_argument(
        "--supi", default="imsi-999700000021635",
        help="UE-Identifikation (SUPI / IMSI)"
    )
    parser.add_argument(
        "--mcc", default="999",
        help="Mobile Country Code"
    )
    parser.add_argument(
        "--mnc", default="70",
        help="Mobile Network Code"
    )
    parser.add_argument(
        "--cell", default="99970190001",
        help="Global Cell ID des Serving-gNB"
    )
    parser.add_argument(
        "--prs-assist", action="store_true",
        help="PRS-Assistenzdaten von LMF abrufen"
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Anfrage N-mal wiederholen (für kontinuierliches Tracking)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Wartezeit zwischen Anfragen [s]"
    )
    args = parser.parse_args()

    # Client wählen
    if args.mock:
        print("[Modus] MOCK – lokale Simulation")
        client = MockLMFClient()
    else:
        print(f"[Modus] LIVE – verbinde zu {args.url}")
        client = LMFClient(base_url=args.url)

    plmn_id = {"mcc": args.mcc, "mnc": args.mnc}

    # PRS-Assistenzdaten abrufen (optional)
    if args.prs_assist:
        print("\n--- PRS-Assistenzdaten abrufen ---")
        result = request_prs_assistance_data(
            client, args.supi, plmn_id, args.cell)
        if result.get("prs_params"):
            print("\nEmpfangene PRS-Parameter von LMF:")
            for i, p in enumerate(result["prs_params"]):
                print(f"  gNB {i+1}: NPRSID={p['nprs_id']} | "
                      f"Slot={p['slot_offset']} | "
                      f"Comb={p['comb_size']} | "
                      f"Pos={p['gnb_position']}")
            print("\n→ Diese Parameter können direkt in config.py übernommen werden")
        else:
            print("  (Keine PRS-Assistenzdaten in Antwort –")
            print("   LMF unterstützt dies möglicherweise noch nicht)")

    # Positionsanfragen
    print(f"\n--- Positionsanfrage(n): {args.repeat}x ---")
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n[{i+1}/{args.repeat}]")

        request_body = build_location_request(
            supi            = args.supi,
            plmn_id         = plmn_id,
            cell_id         = args.cell,
            lcs_client_type = "VALUE_ADDED_SERVICES",
            loc_type        = "CURRENT_LOCATION",
        )

        result = client.request_location(request_body)
        print_location_result(result)

        if i < args.repeat - 1:
            time.sleep(args.interval)

    print("Fertig.")


if __name__ == "__main__":
    main()