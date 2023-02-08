import logging
from pathlib import Path
from typing import Optional
import tensorflow as tf
import tensorflow_text

from nmt_with_transformer.config import MAX_TOKENS, BUFFER_SIZE, BATCH_SIZE


class Preprocessor:
    def __init__(
        self,
        tokenizer_model_path: Optional[str],
        tokenizer_model_name: str = "ted_hrlr_translate_pt_en_converter",
    ) -> None:
        self.logger = logging.getLogger("preprocessor")
        if tokenizer_model_path is None and tokenizer_model_name is None:
            raise ValueError(
                "Please provide either the path to Tokenizer model or name of the Tokenizer model"
            )

        self.tokenizer_model_name = tokenizer_model_name
        if tokenizer_model_path is not None:
            self.tokenizer_model_path = Path(tokenizer_model_path)

        self.logger.info(
            f"Loading tokenizer model: {self.tokenizer_model_name}, path: {self.tokenizer_model_path}"
        )
        self.tokenizers = self._load_tokenizer_model()

    def _load_tokenizer_model(self):
        if not self.tokenizer_model_path.exists():
            tf.keras.utils.get_file(
                f"{self.tokenizer_model_name}.zip",
                f"https://storage.googleapis.com/download.tensorflow.org/models/{self.tokenizer_model_name}.zip",
                cache_dir=".",
                cache_subdir="",
                extract=True,
            )
        else:
            self.tokenizer_model_name = self.tokenizer_model_path.name
        tokenizers = tf.saved_model.load(self.tokenizer_model_name)
        return tokenizers

    def prepare_batch(self, pt, en):
        pt = self.tokenizers.pt.tokenize(pt)
        pt = pt[:, :MAX_TOKENS]
        pt = pt.to_tensor()

        en = self.tokenizers.en.tokenize(en)
        en = en[:, : (MAX_TOKENS + 1)]
        en_inputs = en[:, :-1].to_tensor()
        en_labels = en[:, 1:].to_tensor()

        return (pt, en_inputs), en_labels

    def make_batch(self, ds: tf.data.Dataset):
        return (
            ds.shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(self.prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    def get_vocab_size_for_input_and_target(self):
        return (
            self.tokenizers.pt.get_vocab_size().numpy(),
            self.tokenizers.en.get_vocab_size().numpy(),
        )
