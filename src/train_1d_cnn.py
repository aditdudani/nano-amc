"""Training script for the lightweight 1D CNN AMC model."""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

from data_loader_1d import build_tf_datasets, load_iq_dataset, filter_samples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_1d_cnn")


def build_model(input_shape: tuple[int, int] = (1024, 2), num_classes: int = 8) -> tf.keras.Model:
    """Define the small convolutional model described in the README."""

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(16, 7, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)

    x = tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=4)(x)

    x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="nano_amc_cnn")
    return model


def train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset | None = None,
    save_path: Path = Path("results/model_1d_base.h5"),
    epochs: int = 30,
) -> tf.keras.callbacks.History:
    """Compile and fit the CNN, then save the best checkpoint."""

    model = build_model()
    param_count = model.count_params()
    logger.info("Model has %d parameters (budget: 50000)", param_count)
    if param_count > 50000:
        logger.warning("Parameter budget exceeded! %d > 50000", param_count)
    
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    if test_ds is not None:
        logger.info("Evaluating on test set")
        test_loss, test_acc = model.evaluate(test_ds)
        logger.info("Test accuracy: %.4f", test_acc)
    
    return history


def main() -> None:
    """Prepare data from the loader and kick off training."""

    try:
        data = load_iq_dataset(Path("data/radio_ml_2018_01A.h5"))
        samples, labels = filter_samples(data, mods=["BPSK", "QPSK"], snrs=[0, 2, 4])
        train_ds, val_ds, test_ds = build_tf_datasets(samples, labels)

        logger.info("Starting training on datasets")
        train_model(train_ds, val_ds, test_ds=test_ds, save_path=Path("results/model_1d_base.h5"))
        logger.info("Training complete; model saved to results/model_1d_base.h5")
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()