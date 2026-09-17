#ml.dataset_builder.py
from datetime import datetime
import numpy as np


class DatasetBuilder:

    SENTINEL_RSSI = -100
    TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def _time_features(timestamp):
        """
        Wandelt einen Zeitstempel-String in zwei zyklische Merkmale um
        (Sinus/Cosinus der Tageszeit, Wertebereich jeweils [-1, 1]).
        Dadurch werden z.B. 23:59 Uhr und 00:01 Uhr als benachbart erkannt,
        statt als maximal unterschiedlich (wie bei einem rohen Sekundenwert).
        """
        dt = datetime.strptime(timestamp, DatasetBuilder.TIMESTAMP_FORMAT)
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        fraction_of_day = seconds_since_midnight / 86400.0
        angle = 2 * np.pi * fraction_of_day
        return [float(np.sin(angle)), float(np.cos(angle))]

    @staticmethod
    def build_feature_vector(sample, all_bssids, use_presence_feature=True, use_time_feature=True):

        vector = []

        fingerprint = sample["fingerprint"]

        for bssid in all_bssids:

            if bssid in fingerprint:
                rssi = fingerprint[bssid]["rssi"]
                presence = 1
            else:
                rssi = DatasetBuilder.SENTINEL_RSSI
                presence = 0

            vector.append(rssi)

            if use_presence_feature:
                vector.append(presence)

        if use_time_feature:
            timestamp = sample.get("timestamp")
            if timestamp:
                vector.extend(DatasetBuilder._time_features(timestamp))
            else:
                # Fallback, falls kein Zeitstempel vorhanden ist
                print("[DatasetBuilder] Warnung: Kein Zeitstempel im Sample, verwende Fallback (0.0, 0.0)")
                vector.extend([0.0, 0.0])

        return vector

    @staticmethod
    def build_dataset(data, use_presence_feature=True, use_time_feature=True):

        all_bssids = set()

        for sample in data:
            for bssid in sample["fingerprint"].keys():
                # Einheitlich: Großbuchstaben, kein Doppelpunkt am Ende
                normalized = bssid.upper().rstrip(":")
                all_bssids.add(normalized)

        all_bssids = sorted(list(all_bssids))

        X = []
        y = []

        for sample in data:

            # Sicherheitscheck
            if "position" not in sample or "fingerprint" not in sample:
                continue
            if sample["position"] is None or sample["fingerprint"] is None:
                continue
            if use_time_feature and not sample.get("timestamp"):
                continue

            # Fingerprint auch normalisieren
            normalized_fp = {
                k.upper().rstrip(":"): v
                for k, v in sample["fingerprint"].items()
            }

            X.append(
                DatasetBuilder.build_feature_vector(
                    {"fingerprint": normalized_fp, "timestamp": sample.get("timestamp")},
                    all_bssids,
                    use_presence_feature=use_presence_feature,
                    use_time_feature=use_time_feature
                )
            )

            y.append([
                sample["position"]["x"],
                sample["position"]["y"]
            ])

        return (
            np.array(X),
            np.array(y),
            all_bssids
        )
