import tensorflow as tf

from nmt_with_transformer.export_translator import ExportTranslator


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model: int, warmup_steps=4000) -> None:
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_optimizer(d_model: int):
    learning_rate = CustomSchedule(d_model=d_model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    return optimizer


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)

    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def save_model(translator: ExportTranslator):
    tf.saved_model.save(
        translator, export_dir="./nmt_with_transformer/trained_models/translator"
    )
