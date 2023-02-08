import tensorflow_datasets as tfds


class DataReader:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset

    def load_data(self):
        examples, _ = tfds.load(self.dataset, with_info=True, as_supervised=True)
        return examples["train"], examples["validation"]
