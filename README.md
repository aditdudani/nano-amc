# nano-amc

FPGA-based Automatic Modulation Classification for Defense Applications

## Overview

Lightweight 1D CNN for real-time spectrum monitoring and signal classification, deployed on PYNQ Zynq SoC. Built for FPGA Hackathon 2026 Round 1.

**Application Domain:** Defense - Automatic Modulation Classification for spectrum monitoring/threat detection.

**Dataset:** RadioML 2018.01A (public, download from https://www.deepsig.ai/datasets)
- Shape: `[N, 1024, 2]` (1024 I/Q samples, 2 channels)
- 8 modulation classes: BPSK, 4ASK, QPSK, OQPSK, 8PSK, 16QAM, 32QAM, 64QAM
- SNR filter: 0, 2, 4, 6, 8, 10 dB

**Target Hardware:** PYNQ-Z1/Z2 (Zynq xc7z020clg400-1)

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipeline | **Ready** | Loads RadioML 2018.01A, filters, normalizes |
| 1D CNN Model | **Ready** | ~14k params, architecture defined |
| HLS Export | **Ready** | hls4ml config for Vivado HLS |
| RTL Testbench | **Ready** | Verilog TB + hex test vectors |
| **Dataset** | **NEEDED** | Download RadioML 2018.01A HDF5 |

---

## Implementation Plan

### Phase 1: Data Pipeline

**Objective:** Load raw I/Q data directly, bypassing image conversion.

**File:** `src/data_loader_1d.py`

**Tasks:**
1. Load HDF5 data from RadioML 2018.01A dataset
2. Filter by TARGET_MODS and TARGET_SNRS
3. Normalize I/Q values to [-1, 1] range
4. Split: 70% train / 15% val / 15% test
5. Return TensorFlow datasets with shape `[batch, 1024, 2]`

---

### Phase 2: 1D CNN Architecture & Training

**Objective:** Train a hardware-friendly model with <50k parameters.

**File:** `src/train_1d_cnn.py`

**Model Architecture:**
```
Input: (1024, 2)
Conv1D(16, kernel=7, ReLU) + BatchNorm
MaxPooling1D(pool=4)
Conv1D(32, kernel=5, ReLU) + BatchNorm
MaxPooling1D(pool=4)
Conv1D(64, kernel=3, ReLU) + BatchNorm
GlobalAveragePooling1D
Dense(32, ReLU)
Dense(8, Softmax)
```

**Training Config:**
- Epochs: 30 (with EarlyStopping patience=5)
- Optimizer: Adam, lr=0.001
- Loss: SparseCategoricalCrossentropy
- Callbacks: ModelCheckpoint, EarlyStopping

**Output:** `results/model_1d_base.h5`

---

### Phase 3: HLS4ML Export + RTL Generation

**Objective:** Convert trained Keras model to Verilog RTL.

**File:** `src/export_hls.py`

**HLS Configuration:**
```python
config = hls4ml.utils.config_from_keras_model(model, granularity='name')
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 4
config['Model']['Strategy'] = 'Latency'
```

**Build Steps:**
1. Load `model_1d_base.h5`
2. Generate HLS config
3. Convert to HLS model
4. Run synthesis: `hls_model.build(csim=False, synth=True, vsynth=True)`
5. Copy Verilog output to `fpga_rtl_export/`

**File:** `src/rtl_testbench.py`

**Testbench Tasks:**
1. Generate test vectors from validation set (16 samples)
2. Export as hex files for Verilog testbench
3. Create Verilog testbench wrapper with pass/fail checking
4. Output: `fpga_rtl_export/tb_amc_accelerator.v`

---

## Quick Start

```bash
# 1. Install dependencies
cd nano-amc
pip install -r requirements.txt

# 2. Download dataset (manual step)
# Get GOLD_XYZ_OSC.0001_1024.hdf5 from https://www.deepsig.ai/datasets
# Place in: nano-amc/data/GOLD_XYZ_OSC.0001_1024.hdf5

# 3. Run pipeline
cd src
python data_loader_1d.py      # Validate data pipeline
python train_1d_cnn.py        # Train model (~5-10 min)
python export_hls.py          # Generate RTL (requires Vivado HLS)
python rtl_testbench.py       # Generate test vectors
```

---

## Directory Structure

```
nano-amc/
├── README.md
├── requirements.txt
├── data/
│   └── GOLD_XYZ_OSC.0001_1024.hdf5   # RadioML dataset (download required)
├── src/
│   ├── config.py              # Central configuration
│   ├── data_loader_1d.py      # Raw I/Q data pipeline
│   ├── train_1d_cnn.py        # Model definition + training
│   ├── export_hls.py          # hls4ml conversion
│   └── rtl_testbench.py       # Testbench generation
├── results/
│   └── model_1d_base.h5       # Trained model (generated)
└── fpga_rtl_export/
    ├── hls_project/           # hls4ml output (generated)
    ├── tb_amc_accelerator.v   # Verilog testbench (generated)
    └── test_vectors/
        ├── test_inputs.hex    # I/Q test data (generated)
        └── expected_labels.hex # Ground truth labels (generated)
```

---

## Architecture

### Model: nano_amc_cnn

```
Input: (1024, 2)                    # 1024 I/Q samples
    ↓
Conv1D(16, kernel=7, ReLU) + BatchNorm
MaxPooling1D(pool=4)                # → (256, 16)
    ↓
Conv1D(32, kernel=5, ReLU) + BatchNorm
MaxPooling1D(pool=4)                # → (64, 32)
    ↓
Conv1D(64, kernel=3, ReLU) + BatchNorm  # → (64, 64)
    ↓
GlobalAveragePooling1D              # → (64,)
Dense(32, ReLU)                     # → (32,)
Dense(8, Softmax)                   # → (8,) class probabilities
```

**Parameters:** ~14,000 (well under 50k budget)

### Fixed-Point Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Precision | `ap_fixed<16,6>` | 16 bits total, 6 integer bits, 10 fractional bits |
| Range | [-32, +32) | With resolution ~0.001 |
| ReuseFactor | 4 | DSP slice optimization |
| Strategy | Latency | Minimize inference cycles |

---

## Configuration

All parameters are centralized in `src/config.py`:

```python
# Dataset
DATASET_PATH = Path("data/GOLD_XYZ_OSC.0001_1024.hdf5")
TARGET_MODS = ["BPSK", "4ASK", "QPSK", "OQPSK", "8PSK", "16QAM", "32QAM", "64QAM"]
TARGET_SNRS = [0, 2, 4, 6, 8, 10]

# Training
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# HLS
HLS_PRECISION = "ap_fixed<16,6>"
HLS_REUSE_FACTOR = 4
HLS_STRATEGY = "Latency"
```

---

## Development Log

### v0.3 (Current) - Code Audit Fixes

**Critical Fixes:**
- **Central config.py:** All constants now in one place (dataset path, mods, SNRs, HLS params)
- **RadioML 2018.01A format:** Properly handles one-hot `Y` labels and `Z` SNR arrays
- **All 8 modulations:** Training now uses full 8-class set (was only 2)
- **Quantization math:** Fixed ap_fixed<16,6> scaling (2^10 fractional bits, not 2^15)
- **Hex format:** Changed `IIII_QQQQ` → `IIIIQQQQ` for Verilog $readmemh compatibility
- **Expected labels:** Added `expected_labels.hex` for RTL verification
- **Verilog testbench:** Full `tb_amc_accelerator.v` with pass/fail checking

**Minor Fixes:**
- Python 3.8 compatibility (List[str] instead of list[str])
- Removed unused imports
- Random seed for reproducibility (seed=42)

### v0.2 - Initial Scaffolding

- Created `src/` scaffolding with data_loader_1d.py, train_1d_cnn.py, export_hls.py, rtl_testbench.py
- Added results/, fpga_rtl_export/test_vectors/, requirements.txt
- Input validation, error handling, logging in all scripts
- Parameter count verification with 50k budget warning
- Model summary printed during training
- Test set evaluation after training

### v0.1 - Project Setup

- Initial README with implementation plan
- Directory structure defined

---

## Technical Details

### Data Pipeline (`data_loader_1d.py`)

| Function | Description |
|----------|-------------|
| `load_radioml_dataset()` | Load HDF5, decode one-hot Y to integers |
| `filter_and_prepare()` | Filter by mods/SNRs, normalize to [-1,1] |
| `build_tf_datasets()` | 70/15/15 split with batching/prefetch |
| `load_and_prepare_data()` | Convenience wrapper for full pipeline |

### Training (`train_1d_cnn.py`)

- Optimizer: Adam (lr=0.001)
- Loss: SparseCategoricalCrossentropy
- Callbacks: ModelCheckpoint (best val_accuracy), EarlyStopping (patience=5)
- Test set evaluated after training

### HLS Export (`export_hls.py`)

- Backend: Vivado HLS
- Part: xc7z020clg400-1 (PYNQ-Z1/Z2)
- Output: `fpga_rtl_export/hls_project/`

### RTL Testbench (`rtl_testbench.py`)

- 16 test vectors from validation set
- Quantized to ap_fixed<16,6>
- Hex format: 32-bit `IIIIQQQQ` per I/Q pair
- Verilog TB with timeout and pass/fail counting

---

## Next Steps (Round 1 Submission)

### Immediate (Before March 29)

1. **Download Dataset**
   ```bash
   # Download from https://www.deepsig.ai/datasets
   # Place at: nano-amc/data/GOLD_XYZ_OSC.0001_1024.hdf5
   ```

2. **Train Model**
   ```bash
   cd src && python train_1d_cnn.py
   # Expected: val_accuracy > 60%, model saved to results/model_1d_base.h5
   ```

3. **Generate RTL**
   ```bash
   python export_hls.py
   # Requires: Vivado HLS installed and licensed
   # Output: fpga_rtl_export/hls_project/
   ```

4. **Generate Test Vectors**
   ```bash
   python rtl_testbench.py
   # Output: test_inputs.hex, expected_labels.hex, tb_amc_accelerator.v
   ```

5. **Verify in Vivado**
   - Import hls_project into Vivado
   - Run behavioral simulation with tb_amc_accelerator.v
   - Capture waveforms for submission

### For Submission Package

- [ ] Verilog RTL source code (from hls_project/syn/verilog/)
- [ ] Testbench with simulation results
- [ ] Technical report (separate document)
- [ ] 5-10 min video demo

---

## Constraints

- **No changes to deep-amc repo** - all new code in nano-amc only
- **Bypass existing IP** - no image conversion, no SqueezeNet, no adaptive sampling
- **Parameter budget** - <50k parameters (~14k achieved)
- **Fixed-point** - ap_fixed<16,6> (16-bit, 6 integer bits, 10 fractional bits)
- **Input shape** - 1024 I/Q samples per signal

---

## Requirements

```
tensorflow>=2.10
numpy
h5py
hls4ml[profiling]
```

**External:** Vivado HLS (for RTL synthesis)

---

## References

- RadioML 2018.01A Dataset: https://www.deepsig.ai/datasets
- hls4ml Documentation: https://fastmachinelearning.org/hls4ml/
- PYNQ Documentation: https://pynq.readthedocs.io/
