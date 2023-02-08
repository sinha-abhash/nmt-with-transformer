from argparse import ArgumentParser
import logging

from nmt_with_transformer.data import DataReader, Preprocessor
from nmt_with_transformer.model import Transformer
from nmt_with_transformer import config
from nmt_with_transformer.helper import (
    get_optimizer,
    masked_accuracy,
    masked_loss,
    save_model,
)
from nmt_with_transformer.translator import Translator
from nmt_with_transformer.export_translator import ExportTranslator


logging.basicConfig(level=logging.INFO)


def run(args):
    logger = logging.getLogger("run")

    logger.info(f"Loading dataset: {args.dataset_name}")
    dr = DataReader(dataset=args.dataset_name)
    train, validation = dr.load_data()

    logger.info("Preparing training and validation batches")
    preprocessor = Preprocessor(
        tokenizer_model_path=args.tokenizer_model_path,
        tokenizer_model_name=args.tokenizer_model_name,
    )
    train_batches = preprocessor.make_batch(train)
    val_batches = preprocessor.make_batch(validation)

    (
        input_vocab_size,
        target_vocab_size,
    ) = preprocessor.get_vocab_size_for_input_and_target()

    logger.info("Create Model")
    transformer = Transformer(
        num_layers=config.NUM_LAYERS,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        dff=config.DFF,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        dropout_rate=config.DROPOUT_RATE,
    )

    optimizer = get_optimizer(d_model=config.D_MODEL)

    logger.info("Compiling the model")
    transformer.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )

    logger.info("Train the model")
    transformer.fit(train_batches, epochs=20, validation_data=val_batches)

    logger.info("Export the model")
    translator = Translator(tokenizers=preprocessor.tokenizers, transformer=transformer)
    translator = ExportTranslator(translator=translator)

    save_model(translator=translator)

    logger.info("Model saved")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        required=True,
        help="dataset name to be downloaded from Tensorflow Dataset",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        "-t",
        type=str,
        required=False,
        help="path to downloaded tokenizer model if available",
    )
    parser.add_argument(
        "--tokenizer_model_name",
        "-t_name",
        type=str,
        required=False,
        help="tokenizer model name if model is not already downloaded",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
