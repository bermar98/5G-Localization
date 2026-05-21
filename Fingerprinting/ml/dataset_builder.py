import numpy as np


class DatasetBuilder:

    @staticmethod
    def build_feature_vector(sample, all_bssids):

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
    def build_dataset(data):

        all_bssids = set()

        for sample in data:

            all_bssids.update(
                sample["fingerprint"].keys()
            )

        all_bssids = sorted(list(all_bssids))

        X = []
        y = []

        for sample in data:

            X.append(
                DatasetBuilder.build_feature_vector(
                    sample,
                    all_bssids
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