# get_position.py
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ml.predict_light import PositionEstimator
from positioning.agv_position import AGVPosition

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
RESULTS_FILE = DATA_PATH / "Positionsvergleich/Messungen/position_comparisons_Systemtest.json"


class PositionComparator:

    def __init__(self):
        self.agv       = AGVPosition()
        self.estimator = PositionEstimator()

    def compare(self):

        agv_pos       = self.agv.get_position()
        estimated_pos = self.estimator.estimate()

        if agv_pos is None:
            print("AGV-Position nicht verfügbar.")
            return None

        error = np.sqrt(
            (agv_pos["x"] - estimated_pos["x"])**2 +
            (agv_pos["y"] - estimated_pos["y"])**2
        )

        print(f"AGV-Position:        x={agv_pos['x']:.2f}m, y={agv_pos['y']:.2f}m")
        print(f"Geschätzte Position: x={estimated_pos['x']:.2f}m, y={estimated_pos['y']:.2f}m")
        print(f"Abweichung:          {error:.2f}m")

        return {
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

    while True:
        comparator = PositionComparator()
        result = comparator.compare()
        comparator.save_result(result)
        
        
        time.sleep(20)
if __name__ == "__main__":
    main()