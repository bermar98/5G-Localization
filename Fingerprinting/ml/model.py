#ml.model.py
from tensorflow.keras import models
from tensorflow.keras import layers


class IndoorLocalizationModel:

    @staticmethod
    def create_model(input_dim):

        model = models.Sequential()

        model.add(
            layers.Dense(
                128,
                activation='relu',
                input_shape=(input_dim,)
            )
        )

        model.add(
            layers.Dense(
                64,
                activation='relu'
            )
        )

        model.add(
            layers.Dense(
                32,
                activation='relu'
            )
        )

        model.add(
            layers.Dense(
                2,
                activation='linear'
            )
        )

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model