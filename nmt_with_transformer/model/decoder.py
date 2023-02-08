import tensorflow as tf
from nmt_with_transformer.model import (
    CausalSelfAttention,
    CrossAttention,
    FeedForward,
    PositionalEmbedding,
)


class DecoderLayer(tf.keras.layers.Layer):
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

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model=d_model, dff=dff)

    def call(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x=x, context=context)

        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
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
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(self.num_layers)
        ]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x
