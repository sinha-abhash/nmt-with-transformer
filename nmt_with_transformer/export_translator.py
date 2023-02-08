import tensorflow as tf


from nmt_with_transformer.translator import Translator
from nmt_with_transformer.config import MAX_TOKENS


class ExportTranslator(tf.Module):
    def __init__(self, translator: Translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        result, tokens, attention_weights = self.translator(
            sentence, max_length=MAX_TOKENS
        )

        return result
