"""Training script for the lightweight 1D CNN AMC model."""

from __future__ import annotations

import logging
from pathlib import Path

from typing import Tuple

import tensorflow as tf

from config import (
    DATASET_PATH,
    TARGET_MODS,
    TARGET_SNRS,
    RESULTS_DIR,
    MODEL_PATH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
)
from data_loader_1d import load_and_prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_1d_cnn")


def build_model(input_shape: Tuple[int, int] = (1024, 2), num_classes: int = 8) -> tf.keras.Model:
    """Define the lightweight 1D CNN for AMC.

    Architecture matches the README specification:
    - Conv1D(16, 7, relu) + BatchNorm + MaxPool(4)
    - Conv1D(32, 5, relu) + BatchNorm + MaxPool(4)
    - Conv1D(64, 3, relu) + BatchNorm
    - GlobalAveragePooling1D
    - Dense(32, relu)
    - Dense(num_classes, softmax)

    Target: <50k parameters for FPGA synthesis.
    """
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
    num_classes: int = 8,
    save_path: Path = MODEL_PATH,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> tf.keras.callbacks.History:
    """Compile, train, and save the CNN model.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Optional test dataset for final evaluation
        num_classes: Number of output classes
        save_path: Path to save the best model
        epochs: Maximum training epochs
        learning_rate: Adam optimizer learning rate
        patience: Early stopping patience

    Returns:
        Training history
    """
    # Ensure output directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(num_classes=num_classes)

    # Log parameter count and check budget
    param_count = model.count_params()
    logger.info("Model has %d parameters (budget: 50,000)", param_count)
    if param_count > 50000:
        logger.warning("Parameter budget exceeded! %d > 50,000", param_count)

    # Print model summary
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(save_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    logger.info("Starting training: epochs=%d, lr=%f, patience=%d", epochs, learning_rate, patience)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Evaluate on test set
    if test_ds is not None:
        logger.info("Evaluating on test set...")
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        logger.info("Test loss: %.4f, Test accuracy: %.4f", test_loss, test_acc)

    return history


def main() -> None:
    """Load data and train the model."""
    try:
        logger.info("Loading dataset from %s", DATASET_PATH)
        logger.info("Target modulations: %s", TARGET_MODS)
        logger.info("Target SNRs: %s", TARGET_SNRS)

        train_ds, val_ds, test_ds, class_names = load_and_prepare_data(
            file_path=DATASET_PATH,
            target_mods=TARGET_MODS,
            target_snrs=TARGET_SNRS,
            batch_size=BATCH_SIZE,
        )

        logger.info("Classes: %s", class_names)
        num_classes = len(class_names)

        train_model(
            train_ds,
            val_ds,
            test_ds=test_ds,
            num_classes=num_classes,
            save_path=MODEL_PATH,
        )

        logger.info("Training complete. Model saved to %s", MODEL_PATH)

    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.info("Download RadioML 2018.01A from https://www.deepsig.ai/datasets")
        raise
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
