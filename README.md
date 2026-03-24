# nano-amc

FPGA-based Automatic Modulation Classification for Defense Applications

## Overview

Lightweight 1D CNN for real-time spectrum monitoring and signal classification, deployed on PYNQ Zynq SoC. Built for FPGA Hackathon 2026 Round 1.

**Application Domain:** Defense - Automatic Modulation Classification for spectrum monitoring/threat detection.

**Dataset:** RadioML 2018.01A (public)
- Shape: `[N, 1024, 2]` (1024 I/Q samples, 2 channels)
- 8 modulation classes: BPSK, 4ASK, QPSK, OQPSK, 8PSK, 16QAM, 32QAM, 64QAM
- SNR filter: 0, 2, 4, 6, 8, 10 dB

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
- Loss: categorical_crossentropy
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
1. Generate test vectors from validation set (10-20 samples)
2. Export as hex files for Verilog testbench
3. Create Verilog testbench wrapper
4. Output: `fpga_rtl_export/tb_amc_accelerator.v`

---

## Directory Structure

```
nano-amc/
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader_1d.py        # Raw I/Q data pipeline
│   ├── train_1d_cnn.py          # Model definition + training
│   ├── export_hls.py            # hls4ml conversion
│   └── rtl_testbench.py         # Testbench generation
├── results/
│   └── model_1d_base.h5         # Trained model
└── fpga_rtl_export/
    ├── *.v                      # Generated Verilog files
    ├── tb_amc_accelerator.v     # Testbench
    └── test_vectors/            # Hex test data
```

---

## Requirements

```
tensorflow>=2.10
numpy
h5py
hls4ml[profiling]
```

---

## Verification

1. **Data Pipeline:** Run `data_loader_1d.py`, verify output shape `[batch, 1024, 2]`
2. **Training:** Run `train_1d_cnn.py`, verify val_accuracy > 60%, model saved
3. **HLS Export:** Run `export_hls.py`, verify Verilog files in `fpga_rtl_export/`
4. **RTL Testbench:** Run Vivado behavioral simulation on `tb_amc_accelerator.v`

---

## Constraints

- **No changes to deep-amc repo** - all new code in nano-amc only
- **Bypass existing IP** - no image conversion, no SqueezeNet, no adaptive sampling
- **Parameter budget** - model must have <50k parameters for FPGA synthesis
- **Fixed-point precision** - ap_fixed<16,6> for hardware efficiency
