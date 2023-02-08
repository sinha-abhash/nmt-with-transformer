import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        dff,
        dropout_rate=0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )

        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)

        return x
