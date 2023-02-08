import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x
