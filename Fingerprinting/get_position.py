# get_position.py
import numpy as np
from ml.predict import PositionEstimator
from positioning.agv_position import AGVPosition


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
            "error_m":   error
        }

if __name__ == "__main__":
    comparator = PositionComparator()
    comparator.compare()