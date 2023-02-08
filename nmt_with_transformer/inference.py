from argparse import ArgumentParser
from pathlib import Path
import logging
import tensorflow as tf
import tensorflow_text as tf_text

from nmt_with_transformer.translator import Translator


logging.basicConfig(level=logging.INFO)


class NotCorrectModelPathException(Exception):
    pass


def check_if_model_exists(path: Path) -> bool:
    if not path.exists:
        raise FileNotFoundError(f"Provided path does not exists: {str(path)}")

    all_files = list(path.iterdir())
    model_files = [f for f in all_files if f.suffix == ".pb"]

    if len(model_files) == 0:
        raise NotCorrectModelPathException(f"Provided path does not have model files.")

    return True


def load_model(path) -> Translator:
    model = tf.saved_model.load(path)
    return model


def infer(args):
    logger = logging.getLogger("infer")
    check_if_model_exists(Path(args.model_path))
    logger.info("Loading Model!")
    translator: Translator = load_model(args.model_path)
    logger.info("Model loaded")

    logger.info(
        f"Input Sentence: {args.input_text}\nTranslation: {translator(args.input_text).numpy()}"
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        default="./nmt_with_transformer/trained_models/translator",
        help="Provide saved model path",
    )
    parser.add_argument(
        "--input_text",
        "-i",
        type=str,
        required=True,
        help="Input string to try the model",
    )

    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
