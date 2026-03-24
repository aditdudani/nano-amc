"""Minimal data handling utilities for the FPGA AMC pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import h5py
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_loader_1d")


def load_iq_dataset(file_path: Path) -> Mapping[str, np.ndarray]:
    """Load the raw RadioML dataset from HDF5 and return every dataset in the file."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    logger.info("Loading HDF5 from %s", file_path)
    with h5py.File(file_path, "r") as reader:
        data = {key: reader[key][()] for key in reader.keys()}
    
    required_keys = {"X", "snrs"}
    mod_keys = {"mods", "modulations"}
    actual_keys = set(data.keys())
    
    if not required_keys.issubset(actual_keys):
        raise KeyError(f"Dataset missing required keys {required_keys}. Found: {actual_keys}")
    
    if not mod_keys & actual_keys:
        raise KeyError(f"Dataset missing modulation key {mod_keys}. Found: {actual_keys}")
    
    logger.info("Loaded dataset with keys: %s (X shape: %s)", sorted(actual_keys), data["X"].shape)
    return data


def filter_samples(data: Mapping[str, np.ndarray], mods: Iterable[str], snrs: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Filter dataset by modulation and SNR, then normalize I/Q to [-1, 1]."""

    mods = set(mods)
    snrs = set(snrs)
    sample_data = data.get("X")
    mod_data = data.get("mods") or data.get("modulations")
    snr_data = data.get("snrs")

    if sample_data is None or mod_data is None or snr_data is None:
        raise KeyError("Dataset missing required X/mods/snrs arrays")

    samples = []
    labels = []

    for idx, (mod, snr) in enumerate(zip(mod_data, snr_data)):
        if mod not in mods or snr not in snrs:
            continue

        iq = sample_data[idx].astype(np.float32)
        iq_max = np.max(np.abs(iq))
        iw_scale = iq_max if iq_max > 0 else 1.0
        iq /= iw_scale
        samples.append(iq)
        labels.append(mod)

    if not samples:
        raise ValueError("No samples matched the requested filters.")

    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[label] for label in labels], dtype=np.int32)

    logger.info("Filtered to %d samples with %d classes: %s", len(samples), len(unique_labels), unique_labels)
    sample_shape = np.stack(samples).shape
    if sample_shape[1] != 1024:
        logger.warning("Expected 1024 I/Q samples per signal, got %d", sample_shape[1])
    
    return np.stack(samples), numeric_labels


def build_tf_datasets(
    samples: np.ndarray, labels: np.ndarray, batch_size: int = 128
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split arrays into train/val/test TensorFlow datasets."""

    num_samples = len(samples)
    indices = np.random.permutation(num_samples)

    train_end = int(num_samples * 0.7)
    val_end = train_end + int(num_samples * 0.15)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    def _to_ds(selection: np.ndarray) -> tf.data.Dataset:
        return (
            tf.data.Dataset.from_tensor_slices((samples[selection], labels[selection]))
            .shuffle(len(selection))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    return _to_ds(splits["train"]), _to_ds(splits["val"]), _to_ds(splits["test"])


def main() -> None:
    """Demonstrate the loader by preparing datasets from placeholder inputs."""

    file_path = Path("data/radio_ml_2018_01A.h5")
    try:
        data = load_iq_dataset(file_path)
        samples, labels = filter_samples(data, mods=["BPSK"], snrs=[0, 2, 4])
        train_ds, val_ds, test_ds = build_tf_datasets(samples, labels)
        logger.info("Data pipeline validated successfully")
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error("Data pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()