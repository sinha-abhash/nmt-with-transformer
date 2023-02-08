import tensorflow as tf

from nmt_with_transformer.model import (
    GlobalSelfAttention,
    FeedForward,
    PositionalEmbedding,
)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        dff,
        dropout_rate=0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        dropout_rate=0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x
