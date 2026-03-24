"""Convert the trained TensorFlow model to HLS/RTL with hls4ml."""

from __future__ import annotations

import logging
from pathlib import Path

import hls4ml
import tensorflow as tf

from config import (
    MODEL_PATH,
    RTL_EXPORT_DIR,
    HLS_PRECISION,
    HLS_REUSE_FACTOR,
    HLS_STRATEGY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_hls")


def export_model(
    model_path: Path = MODEL_PATH,
    output_dir: Path = RTL_EXPORT_DIR,
    precision: str = HLS_PRECISION,
    reuse_factor: int = HLS_REUSE_FACTOR,
    strategy: str = HLS_STRATEGY,
) -> None:
    """Compile the Keras model into an hls4ml project with fixed-point precision.

    Args:
        model_path: Path to the trained Keras model (.h5)
        output_dir: Directory for HLS/RTL output
        precision: Fixed-point precision (e.g., "ap_fixed<16,6>")
        reuse_factor: Hardware reuse factor for DSP optimization
        strategy: HLS strategy ("Latency" or "Resource")
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(model_path)
    model.summary()

    logger.info("Generating HLS configuration...")
    logger.info("  Precision: %s", precision)
    logger.info("  ReuseFactor: %d", reuse_factor)
    logger.info("  Strategy: %s", strategy)

    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    config["Model"]["Precision"] = precision
    config["Model"]["ReuseFactor"] = reuse_factor
    config["Model"]["Strategy"] = strategy

    logger.info("Converting Keras model to HLS...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir / "hls_project"),
        backend="Vivado",
        part="xc7z020clg400-1",  # PYNQ-Z1/Z2 part number
    )

    logger.info("Compiling HLS model...")
    hls_model.compile()

    logger.info("Running synthesis (csim=False, synth=True, vsynth=True)...")
    logger.info("This may take 5-15 minutes depending on your system.")

    try:
        hls_model.build(csim=False, synth=True, vsynth=True)
        logger.info("Synthesis complete!")

        # Log output locations
        project_dir = output_dir / "hls_project"
        verilog_dir = project_dir / "myproject_prj" / "solution1" / "syn" / "verilog"

        if verilog_dir.exists():
            verilog_files = list(verilog_dir.glob("*.v"))
            logger.info("Generated %d Verilog files in %s", len(verilog_files), verilog_dir)
            for vf in verilog_files[:5]:  # Show first 5
                logger.info("  - %s", vf.name)
            if len(verilog_files) > 5:
                logger.info("  ... and %d more", len(verilog_files) - 5)
        else:
            logger.warning("Verilog output directory not found at %s", verilog_dir)

    except Exception as e:
        logger.error("Synthesis failed: %s", e)
        logger.info("Check Vivado HLS installation and license.")
        raise


def main() -> None:
    """Entry point for HLS export."""
    try:
        export_model()
        logger.info("HLS export successful")
    except FileNotFoundError as e:
        logger.error("Export failed: %s", e)
        logger.info("Train the model first using: python train_1d_cnn.py")
        raise
    except Exception as e:
        logger.error("Export failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
