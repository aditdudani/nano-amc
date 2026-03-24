"""Helpers that prepare test vectors and wrapper files for the RTL accelerator."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from data_loader_1d import load_iq_dataset, filter_samples, build_tf_datasets


def quantize_to_fixed(samples: np.ndarray, bits: int = 16) -> np.ndarray:
    scale = 2 ** (bits - 1) - 1
    quantized = np.round(np.clip(samples, -1.0, 1.0) * scale).astype(np.int16)
    return quantized


def write_hex_vectors(samples: np.ndarray, output_path: Path) -> None:
    flattened = samples.reshape(-1, 2)
    with open(output_path, "w") as outf:
        for i, (i_sample, q_sample) in enumerate(flattened):
            outf.write(f"{i_sample:04x}{q_sample:04x}\n")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("rtl_testbench")
    
    try:
        data = load_iq_dataset(Path("data/radio_ml_2018_01A.h5"))
        samples, labels = filter_samples(data, mods=["QPSK"], snrs=[8, 10])
        _, val_ds, _ = build_tf_datasets(samples, labels)

        collected = []
        for batch in val_ds.take(3):
            features, _ = batch
            collected.append(features.numpy())

        if not collected:
            raise ValueError("No validation samples found; testbench requires at least 1 sample")
        
        val_samples = np.concatenate(collected, axis=0)
        logger.info("Collected %d validation samples", val_samples.shape[0])
        
        num_vectors = min(16, val_samples.shape[0])
        quantized = quantize_to_fixed(val_samples[:num_vectors])
        
        output_path = Path("fpga_rtl_export/test_vectors/validation.hex")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        write_hex_vectors(quantized, output_path)
        logger.info("Wrote %d test vectors to %s", num_vectors, output_path)
    except Exception as e:
        logger.error("Testbench generation failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()