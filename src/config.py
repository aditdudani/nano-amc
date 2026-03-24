"""Configuration constants for the nano-amc pipeline."""

from pathlib import Path

# Dataset path - update this to point to your RadioML 2018.01A HDF5 file
# Download from: https://www.deepsig.ai/datasets
DATASET_PATH = Path("data/GOLD_XYZ_OSC.0001_1024.hdf5")

# 6 modulation classes (well-separated for high accuracy)
# Dropped: 4ASK (confuses with BPSK), 32QAM (confuses with 16/64QAM)
TARGET_MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "OQPSK"]

# SNR range in dB (6-14 dB for respectable accuracy, avoiding noisy 0-4 dB)
TARGET_SNRS = [6, 8, 10, 12, 14]

# Full 24-class label order from RadioML 2018.01A (for decoding one-hot Y)
RADIOML_CLASSES = [
    "OOK", "4ASK", "8ASK",
    "BPSK", "QPSK", "8PSK",
    "16PSK", "32PSK", "16APSK",
    "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC",
    "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK"
]

# Output paths
RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "model_1d_base.h5"
RTL_EXPORT_DIR = Path("fpga_rtl_export")
TEST_VECTORS_DIR = RTL_EXPORT_DIR / "test_vectors"

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# HLS configuration
HLS_PRECISION = "ap_fixed<16,6>"  # 16 bits total, 6 integer bits, 10 fractional bits
HLS_REUSE_FACTOR = 4
HLS_STRATEGY = "Latency"

# Fixed-point parameters (derived from ap_fixed<16,6>)
FIXED_POINT_TOTAL_BITS = 16
FIXED_POINT_INT_BITS = 6  # includes sign bit
FIXED_POINT_FRAC_BITS = FIXED_POINT_TOTAL_BITS - FIXED_POINT_INT_BITS  # = 10
