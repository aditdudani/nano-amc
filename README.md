# nano-amc

FPGA-based Automatic Modulation Classification for Defense Applications

## Overview

Lightweight 1D CNN for real-time spectrum monitoring and signal classification, deployed on PYNQ Zynq SoC. Built for FPGA Hackathon 2026 Round 1.

**Application Domain:** Defense - Automatic Modulation Classification for spectrum monitoring/threat detection.

**Dataset:** RadioML 2018.01A (public, download from https://www.deepsig.ai/datasets)
- Shape: `[N, 1024, 2]` (1024 I/Q samples, 2 channels)
- 6 modulation classes: BPSK, QPSK, 8PSK, 16QAM, 64QAM, OQPSK
- SNR range: 6, 8, 10, 12, 14 dB
- **Achieved accuracy: 96.97%** (test set)

**Target Hardware:** PYNQ-Z1/Z2 (Zynq xc7z020clg400-1)

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipeline | **Done** | Loads RadioML 2018.01A, filters, normalizes |
| 1D CNN Model | **Done** | 11,766 params, 96.97% test accuracy |
| Training | **Done** | Trained on server, 26 epochs |
| HLS Export | **Done** | C++ firmware generated in `fpga_rtl_export/hls_project/` |
| C Synthesis | **Pending** | Run in Vitis HLS GUI or via TCL |
| RTL Export | **Pending** | Export IP after synthesis |
| RTL Testbench | **Pending** | Generate test vectors after RTL export |
| Dataset | **Ready** | Uses shared dataset from amc_project |

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
Dense(6, Softmax)
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

### Server/WSL (Training)

```bash
# 1. Clone to server (assumes amc_project already exists)
cd ~/projects
git clone https://github.com/aditdudani/nano-amc.git

# 2. Install dependencies
cd nano-amc
pip install -r requirements.txt

# 3. Run training
cd src
python data_loader_1d.py      # Validate data pipeline
python train_1d_cnn.py        # Train model (~5-10 min)
```

### Windows (HLS Export)

```powershell
# 1. Clone repo to Windows drive
cd C:\Users\<username>
git clone https://github.com/aditdudani/nano-amc.git

# 2. Copy trained model from server/WSL
# Copy results/model_1d_base.h5 to Windows clone

# 3. Install dependencies
cd nano-amc
pip install -r requirements.txt

# 4. Setup Xilinx tools and run HLS export
C:\Xilinx\2025.1\Vitis\settings64.bat
cd src
python export_hls.py          # Generate RTL (5-15 min)
python rtl_testbench.py       # Generate test vectors
```

**Note:** Dataset path is configured to use `../amc_project/data/GOLD_XYZ_OSC.0001_1024.hdf5` (shared with amc_project).

---

## Directory Structure

```
~/projects/
├── amc_project/
│   └── data/
│       └── GOLD_XYZ_OSC.0001_1024.hdf5   # Shared RadioML dataset
└── nano-amc/
    ├── README.md
    ├── requirements.txt
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
Dense(6, Softmax)                   # → (6,) class probabilities
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
# Dataset (shared with amc_project)
DATASET_PATH = Path("../amc_project/data/GOLD_XYZ_OSC.0001_1024.hdf5")
TARGET_MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "OQPSK"]
TARGET_SNRS = [6, 8, 10, 12, 14]

# Training
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# HLS
HLS_PRECISION = "ap_fixed<16,6>"
HLS_REUSE_FACTOR = 4
HLS_STRATEGY = "Latency"
HLS_PART = "xc7z020clg400-1"      # PYNQ (use xc7z020clg484-1 for ZedBoard)
HLS_BACKEND = "Vitis"             # Use "Vitis" for Xilinx 2020.1+
```

---

## Development Log

### v0.6 (Current) - HLS C++ Export Complete

**HLS Export Results:**
- Successfully converted Keras model to HLS C++ using hls4ml
- Output directory: `fpga_rtl_export/hls_project/`
- Generated files:
  - `firmware/myproject.cpp` - Main HLS C++ implementation
  - `firmware/myproject.h` - Header with interface definitions
  - `firmware/weights/` - Quantized weights (ap_fixed<16,6>)
  - `firmware/nnet_utils/` - HLS neural network utilities
  - `build_prj.tcl` - TCL script for Vitis HLS synthesis
- Configuration: `ap_fixed<16,6>`, ReuseFactor=4, Latency strategy
- Target: xc7z020clg400-1 (PYNQ-Z1/Z2)

**Note:** Windows skips automatic synthesis - run manually in Vitis HLS (see Next Steps below).

### v0.5 - Training Complete

**Training Results:**
- **Test accuracy: 96.97%** (far exceeded 85-90% target)
- **Val accuracy: 96.97%**
- Parameters: 11,766 (well under 50k budget)
- Epochs: 26 (early stopped, best at epoch 21)
- Samples: 122,880 (filtered from 2.5M)
- Classes: 16QAM, 64QAM, 8PSK, BPSK, OQPSK, QPSK

**Fixes:**
- **Keras 2.x/3.x compatibility:** Added `compile=False` to model loading in export_hls.py and rtl_testbench.py
- **HLS config:** Added `HLS_PART` and `HLS_BACKEND` to config.py (configurable FPGA part and backend)
- **Vitis backend:** Changed from "Vivado" to "Vitis" for Xilinx 2020.1+ compatibility

**Note:** HLS export requires running from Windows (Xilinx tools have Windows paths).

### v0.4 - Server Deployment

- **Dataset path:** Changed to `../amc_project/data/` to share dataset with amc_project
- **TF log suppression:** Added `TF_CPP_MIN_LOG_LEVEL=2` to all scripts to suppress INFO/WARNING logs
- **Server setup:** Updated Quick Start with clone instructions for `~/projects/` deployment

### v0.3 - Code Audit Fixes

**Critical Fixes:**
- **Central config.py:** All constants now in one place (dataset path, mods, SNRs, HLS params)
- **RadioML 2018.01A format:** Properly handles one-hot `Y` labels and `Z` SNR arrays
- **6 modulations:** Optimized class selection (dropped 4ASK, 32QAM for better accuracy)
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

- Backend: Vitis HLS (Xilinx 2020.1+)
- Part: xc7z020clg400-1 (PYNQ-Z1/Z2)
- Output: `fpga_rtl_export/hls_project/`
- **Note:** Must run from Windows with Xilinx tools on PATH

### RTL Testbench (`rtl_testbench.py`)

- 16 test vectors from validation set
- Quantized to ap_fixed<16,6>
- Hex format: 32-bit `IIIIQQQQ` per I/Q pair
- Verilog TB with timeout and pass/fail counting

---

## Next Steps (Round 1 Submission)

### Immediate (Before March 29)

1. ~~**Train Model**~~ **DONE** ✓
   - Test accuracy: 96.97%
   - Model saved: `results/model_1d_base.h5`

2. ~~**Generate HLS C++ Project**~~ **DONE** ✓
   - HLS firmware generated: `fpga_rtl_export/hls_project/`
   - Configuration: ap_fixed<16,6>, ReuseFactor=4

3. **Run C Synthesis in Vitis HLS** ← **YOU ARE HERE**
   
   **Option A: Vitis HLS GUI**
   ```
   1. Open Vitis HLS 2025.1
   2. File → Open Project → fpga_rtl_export\hls_project\myproject_prj
   3. Click 'Run C Synthesis' (green play button)
   4. Wait for synthesis to complete (~5-15 min)
   5. Review timing/resource utilization report
   6. Click 'Export RTL' to generate Verilog IP
   ```

   **Option B: Command Line (TCL)**
   ```powershell
   cd fpga_rtl_export\hls_project
   vitis_hls -f build_prj.tcl
   ```

4. **Generate Test Vectors**
   ```powershell
   cd src
   python rtl_testbench.py
   # Output: test_inputs.hex, expected_labels.hex, tb_amc_accelerator.v
   ```

5. **Verify in Vivado**
   - Import synthesized IP into Vivado project
   - Run behavioral simulation with testbench
   - Capture waveforms for submission

### For Submission Package

- [x] Trained model with 96.97% accuracy
- [x] HLS C++ project (hls4ml output)
- [ ] Verilog RTL source code (from hls_project/myproject_prj/solution1/syn/verilog/ after synthesis)
- [ ] Resource utilization report (from synthesis)
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

**External:** Vitis HLS 2020.1+ (for RTL synthesis) - Windows recommended

---

## References

- RadioML 2018.01A Dataset: https://www.deepsig.ai/datasets
- hls4ml Documentation: https://fastmachinelearning.org/hls4ml/
- PYNQ Documentation: https://pynq.readthedocs.io/
