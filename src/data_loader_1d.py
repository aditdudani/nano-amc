"""Data loading utilities for the FPGA AMC pipeline.

Handles RadioML 2018.01A HDF5 format with one-hot encoded labels.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

# Suppress TensorFlow C++ logs before importing tf
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO and WARNING

import h5py
import numpy as np
import tensorflow as tf

from config import (
    DATASET_PATH,
    TARGET_MODS,
    TARGET_SNRS,
    RADIOML_CLASSES,
    BATCH_SIZE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_loader_1d")


def load_radioml_dataset(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RadioML 2018.01A dataset from HDF5.

    Args:
        file_path: Path to the HDF5 file.

    Returns:
        Tuple of (X, Y_labels, snrs) where:
        - X: Raw I/Q samples [N, 1024, 2]
        - Y_labels: Integer class labels [N] (decoded from one-hot)
        - snrs: SNR values [N]
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    logger.info("Loading HDF5 from %s", file_path)

    with h5py.File(file_path, "r") as hf:
        # X: Raw I/Q data [N, 1024, 2]
        X = hf["X"][:]

        # Y: One-hot encoded labels [N, 24] -> decode to integers
        Y_onehot = hf["Y"][:]
        Y_labels = np.argmax(Y_onehot, axis=1)

        # Z: SNR values [N, 1] -> flatten to [N]
        snrs = hf["Z"][:].flatten()

    logger.info("Loaded: X=%s, labels=%s unique, SNRs=%s",
                X.shape, len(np.unique(Y_labels)), sorted(np.unique(snrs)))

    return X, Y_labels, snrs


def filter_and_prepare(
    X: np.ndarray,
    Y_labels: np.ndarray,
    snrs: np.ndarray,
    target_mods: List[str] = TARGET_MODS,
    target_snrs: List[int] = TARGET_SNRS,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Filter dataset by modulation type and SNR, then normalize.

    Args:
        X: Raw I/Q samples [N, 1024, 2]
        Y_labels: Integer class labels [N] (indices into RADIOML_CLASSES)
        snrs: SNR values [N]
        target_mods: List of modulation names to keep
        target_snrs: List of SNR values to keep

    Returns:
        Tuple of (samples, labels, class_names) where:
        - samples: Normalized I/Q data [M, 1024, 2] in range [-1, 1]
        - labels: Integer labels [M] remapped to [0, num_classes)
        - class_names: Sorted list of class names for label mapping
    """
    target_snrs_set = set(target_snrs)

    # Map modulation names to their indices in RADIOML_CLASSES
    mod_to_idx = {mod: idx for idx, mod in enumerate(RADIOML_CLASSES)}
    target_mod_indices = {mod_to_idx[mod] for mod in target_mods if mod in mod_to_idx}

    if len(target_mod_indices) != len(target_mods):
        missing = set(target_mods) - set(mod_to_idx.keys())
        logger.warning("Modulations not found in RADIOML_CLASSES: %s", missing)

    # Filter samples
    mask = np.array([
        (label_idx in target_mod_indices) and (snr in target_snrs_set)
        for label_idx, snr in zip(Y_labels, snrs)
    ])

    filtered_X = X[mask]
    filtered_labels = Y_labels[mask]

    if len(filtered_X) == 0:
        raise ValueError(f"No samples matched filters: mods={target_mods}, snrs={target_snrs}")

    # Normalize I/Q to [-1, 1] per sample
    samples = []
    for iq in filtered_X:
        iq = iq.astype(np.float32)
        max_val = np.max(np.abs(iq))
        if max_val > 0:
            iq = iq / max_val
        samples.append(iq)
    samples = np.stack(samples)

    # Remap labels to contiguous integers [0, num_classes)
    # Sort class names alphabetically for deterministic ordering
    unique_label_indices = sorted(set(filtered_labels))
    class_names = [RADIOML_CLASSES[idx] for idx in unique_label_indices]
    class_names_sorted = sorted(class_names)

    # Create mapping from original label index to new contiguous index
    original_to_new = {}
    for new_idx, name in enumerate(class_names_sorted):
        original_idx = RADIOML_CLASSES.index(name)
        original_to_new[original_idx] = new_idx

    labels = np.array([original_to_new[l] for l in filtered_labels], dtype=np.int32)

    logger.info("Filtered to %d samples, %d classes: %s",
                len(samples), len(class_names_sorted), class_names_sorted)

    # Validate sample shape
    if samples.shape[1] != 1024:
        logger.warning("Expected 1024 I/Q samples per signal, got %d", samples.shape[1])

    return samples, labels, class_names_sorted


def build_tf_datasets(
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split arrays into train/val/test TensorFlow datasets.

    Args:
        samples: Normalized I/Q data [N, 1024, 2]
        labels: Integer labels [N]
        batch_size: Batch size for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    np.random.seed(seed)

    num_samples = len(samples)
    indices = np.random.permutation(num_samples)

    train_end = int(num_samples * 0.7)
    val_end = train_end + int(num_samples * 0.15)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    logger.info("Split sizes: train=%d, val=%d, test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    def _to_ds(selection: np.ndarray, shuffle: bool = True) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((samples[selection], labels[selection]))
        if shuffle:
            ds = ds.shuffle(len(selection), seed=seed)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return _to_ds(splits["train"]), _to_ds(splits["val"], shuffle=False), _to_ds(splits["test"], shuffle=False)


def load_and_prepare_data(
    file_path: Path = DATASET_PATH,
    target_mods: List[str] = TARGET_MODS,
    target_snrs: List[int] = TARGET_SNRS,
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Convenience function to load, filter, and prepare datasets.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_names)
    """
    X, Y_labels, snrs = load_radioml_dataset(file_path)
    samples, labels, class_names = filter_and_prepare(X, Y_labels, snrs, target_mods, target_snrs)
    train_ds, val_ds, test_ds = build_tf_datasets(samples, labels, batch_size)
    return train_ds, val_ds, test_ds, class_names


def main() -> None:
    """Validate the data pipeline with the configured dataset."""
    try:
        train_ds, val_ds, test_ds, class_names = load_and_prepare_data()

        # Verify shapes
        for batch_x, batch_y in train_ds.take(1):
            logger.info("Batch shape: X=%s, Y=%s", batch_x.shape, batch_y.shape)
            logger.info("X range: [%.3f, %.3f]", float(tf.reduce_min(batch_x)), float(tf.reduce_max(batch_x)))
            logger.info("Y values: %s", batch_y.numpy()[:10])

        logger.info("Class mapping: %s", {i: name for i, name in enumerate(class_names)})
        logger.info("Data pipeline validated successfully")

    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        logger.info("Download RadioML 2018.01A from https://www.deepsig.ai/datasets")
        raise
    except Exception as e:
        logger.error("Data pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
