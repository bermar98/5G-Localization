#ml.predict.py'
import numpy as np


class Predictor:

    @staticmethod
    def predict(
        model,
        fingerprint,
        all_bssids
    ):

        vector = []

        for bssid in all_bssids:

            if bssid in fingerprint:

                vector.append(
                    fingerprint[bssid]["rssi"]
                )

            else:

                vector.append(-100)

        x = np.array([vector])

        prediction = model.predict(x)

        return {
            "x": prediction[0][0],
            "y": prediction[0][1]
        }