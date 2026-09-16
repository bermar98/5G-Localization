# get_position.py
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ml.predict_light_Testdaten import PositionEstimator

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
RESULTS_FILE = DATA_PATH / "Positionsvergleich/Messungen/position_comparisons_Trainingsdaten.json"
FINGERPRINTS_FILE = DATA_PATH / "fingerprints/Messungen/fingerprints_Grundmessung_Zeitstempel.json"


class PositionComparator:

    def __init__(self, fingerprints_path=FINGERPRINTS_FILE):
        self.estimator = PositionEstimator()

        with open(fingerprints_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

    def compare(self, record):

        agv_pos = record["position"]
        estimated_pos = self.estimator.estimate(record["fingerprint"])

        error = np.sqrt(
            (agv_pos["x"] - estimated_pos["x"])**2 +
            (agv_pos["y"] - estimated_pos["y"])**2
        )

        print(f"AGV-Position:        x={agv_pos['x']:.2f}m, y={agv_pos['y']:.2f}m")
        print(f"Geschätzte Position: x={estimated_pos['x']:.2f}m, y={estimated_pos['y']:.2f}m")
        print(f"Abweichung:          {error:.2f}m")

        return {
            "record_timestamp": record.get("timestamp"),
            "agv":       agv_pos,
            "estimated": estimated_pos,
            "error_m":   float(error)   # np.float64 ist nicht JSON-serialisierbar -> float
        }

    def save_result(self, result, path=RESULTS_FILE):
        """
        Hängt das Ergebnis von compare() mit Zeitstempel an eine JSON-Datei
        im data-Ordner an. Die Datei enthält eine Liste aller bisherigen
        Vergleiche (nicht nur den letzten).
        """
        if result is None:
            print("Kein Ergebnis zum Speichern vorhanden (result ist None).")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result
        }

        # Bestehende Einträge laden, falls die Datei schon existiert
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except (json.JSONDecodeError, OSError):
                print(f"Warnung: {path} war leer oder beschädigt, wird neu angelegt.")
                data = []
        else:
            data = []

        data.append(entry)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Ergebnis gespeichert in: {path}")


def main():

    print("Programm gestartet...\n")

    comparator = PositionComparator()

    for record in comparator.records:
        result = comparator.compare(record)
        comparator.save_result(result)



if __name__ == "__main__":
    main()
