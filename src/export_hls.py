"""Convert the trained TensorFlow model to HLS/RTL with hls4ml."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Suppress TensorFlow C++ logs before importing tf
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO and WARNING

import hls4ml
import tensorflow as tf

from config import (
    MODEL_PATH,
    RTL_EXPORT_DIR,
    HLS_PRECISION,
    HLS_REUSE_FACTOR,
    HLS_STRATEGY,
    HLS_PART,
    HLS_BACKEND,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_hls")


def export_model(
    model_path: Path = MODEL_PATH,
    output_dir: Path = RTL_EXPORT_DIR,
    precision: str = HLS_PRECISION,
    reuse_factor: int = HLS_REUSE_FACTOR,
    strategy: str = HLS_STRATEGY,
    part: str = HLS_PART,
    backend: str = HLS_BACKEND,
) -> None:
    """Compile the Keras model into an hls4ml project with fixed-point precision.

    Args:
        model_path: Path to the trained Keras model (.h5)
        output_dir: Directory for HLS/RTL output
        precision: Fixed-point precision (e.g., "ap_fixed<16,6>")
        reuse_factor: Hardware reuse factor for DSP optimization
        strategy: HLS strategy ("Latency" or "Resource")
        part: FPGA part number (e.g., xc7z020clg400-1 for PYNQ)
        backend: HLS backend ("Vitis" for 2020.1+, "Vivado" for older)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", model_path)
    # Load with compile=False to avoid Keras 2.x/3.x compatibility issues
    # (optimizer/loss config not needed for HLS export, only architecture + weights)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()

    logger.info("Generating HLS configuration...")
    logger.info("  Precision: %s", precision)
    logger.info("  ReuseFactor: %d", reuse_factor)
    logger.info("  Strategy: %s", strategy)
    logger.info("  Part: %s", part)
    logger.info("  Backend: %s", backend)

    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    config["Model"]["Precision"] = precision
    config["Model"]["ReuseFactor"] = reuse_factor
    config["Model"]["Strategy"] = strategy

    logger.info("Converting Keras model to HLS...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir / "hls_project"),
        backend=backend,
        part=part,
    )

    logger.info("Writing HLS project files...")
    hls_model.write()

    project_dir = output_dir / "hls_project"
    logger.info("HLS C++ project written to: %s", project_dir)

    # Check if we're on Windows and skip compile/build (Unix Makefile incompatible)
    if os.name == 'nt':
        logger.warning("Windows detected - skipping compile() and build() steps")
        logger.info("=" * 60)
        logger.info("NEXT STEPS: Run synthesis manually in Vitis HLS")
        logger.info("=" * 60)
        logger.info("Option 1: Vitis HLS GUI")
        logger.info("  1. Open Vitis HLS 2025.1")
        logger.info("  2. File -> Open Project -> %s", project_dir / "myproject_prj")
        logger.info("  3. Click 'Run C Synthesis' (green play button)")
        logger.info("  4. After synthesis, click 'Export RTL'")
        logger.info("")
        logger.info("Option 2: Vitis HLS TCL (command line)")
        logger.info("  cd %s", project_dir)
        logger.info("  vitis_hls -f build_prj.tcl")
        logger.info("=" * 60)
        return

    # On Linux/WSL, run compile and build normally
    logger.info("Compiling HLS model...")
    hls_model.compile()

    logger.info("Running synthesis (csim=False, synth=True, vsynth=True)...")
    logger.info("This may take 5-15 minutes depending on your system.")

    try:
        hls_model.build(csim=False, synth=True, vsynth=True)
        logger.info("Synthesis complete!")

        # Log output locations
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
