import tensorflow as tf

from nmt_with_transformer.model import Encoder, Decoder


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        dropout_rate=0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            d_model=d_model,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_heads=num_heads,
            num_layers=num_layers,
            d_model=d_model,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)

        x = self.decoder(x, context)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
