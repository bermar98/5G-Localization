#ml.model.py
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers


class IndoorLocalizationModel:

    @staticmethod
    def create_model(input_dim, l2_factor=1e-4, dropout_rate=0.5):

        model = models.Sequential()

        model.add(
            layers.Dense(
                128,
                activation='relu',
                input_shape=(input_dim,),
                kernel_regularizer=regularizers.l2(l2_factor)
            )
        )
        model.add(layers.Dropout(0.2))
        model.add(
            layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_factor)
            )
        )
        model.add(layers.Dropout(0.2))
        model.add(
            layers.Dense(
                32,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_factor)
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