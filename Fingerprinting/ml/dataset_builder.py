#ml.dataset_builder.py
import numpy as np


class DatasetBuilder:

    SENTINEL_RSSI = -100

    @staticmethod
    def build_feature_vector(sample, all_bssids, use_presence_feature=True):

        vector = []

        fingerprint = sample["fingerprint"]

        for bssid in all_bssids:

            if bssid in fingerprint:

                vector.append(
                    fingerprint[bssid]["rssi"]
                )

            else:

                vector.append(-100)

        return vector

    @staticmethod
    def build_dataset(data, use_presence_feature=True):

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
            # Fingerprint auch normalisieren
            normalized_fp = {
                k.upper().rstrip(":"): v
                for k, v in sample["fingerprint"].items()
            }

            X.append(
                DatasetBuilder.build_feature_vector(
                    {"fingerprint": normalized_fp},
                    all_bssids,
                    use_presence_feature=use_presence_feature
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