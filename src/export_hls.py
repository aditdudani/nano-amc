"""Convert the trained TensorFlow model to HLS/RTL with hls4ml."""

from __future__ import annotations

from pathlib import Path

import hls4ml
import tensorflow as tf


def export_model(model_path: Path, output_dir: Path) -> None:
    """Compile the Keras model into an hls4ml project with fixed-point precision."""

    import logging
    logger = logging.getLogger("export_hls")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(model_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)
    
    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    config["Model"]["Precision"] = "ap_fixed<16,6>"
    config["Model"]["ReuseFactor"] = 4
    config["Model"]["Strategy"] = "Latency"
    
    logger.info("HLS config: Precision=ap_fixed<16,6>, ReuseFactor=4, Strategy=Latency")

    project = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir)
    logger.info("Running synthesis (csim=False, synth=True, vsynth=True)")
    project.build(csim=False, synth=True, vsynth=True)
    logger.info("Synthesis complete; RTL in %s", output_dir / "syn" or output_dir)


def main() -> None:
    """Entry point that assumes the base model has already been saved."""

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_hls")
    
    try:
        export_model(Path("results/model_1d_base.h5"), Path("fpga_rtl_export"))
        logger.info("HLS export successful")
    except FileNotFoundError as e:
        logger.error("HLS export failed: %s", e)
        raise
    except Exception as e:
        logger.error("HLS export failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()