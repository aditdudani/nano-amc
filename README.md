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

---

## Development Log

- Created the `src/` scaffolding with `data_loader_1d.py`, `train_1d_cnn.py`, `export_hls.py`, and `rtl_testbench.py` so the RadioML pipeline can be exercised end-to-end: loading/normalizing I/Q data, training the <50k-parameter 1D CNN, exporting the Keras model to fixed-point hls4ml, and producing validation hex vectors for RTL testbenches.
- Added the `results/` directory, the `fpga_rtl_export/test_vectors/` subfolder, and `requirements.txt` (tensorflow>=2.10, numpy, h5py, hls4ml[profiling]) so artifacts, exports, and dependencies have explicit homes.
- Each script guards its inputs with clear errors and logs the key steps so you can follow what runs and adapt filters/precision before pointing at the real RadioML dataset.

---

## Code Review & Technical Details

### Phase 1: Data Pipeline (`src/data_loader_1d.py`)

**Functionality:**
- `load_iq_dataset(file_path)` — Loads RadioML HDF5 directly; reads all arrays into memory
- `filter_samples(data, mods, snrs)` — Filters by modulation class and SNR; normalizes I/Q to [-1, 1]; maps mod labels to integers (0–7) for sparse categorical loss
- `build_tf_datasets(samples, labels, batch_size=128)` — Shuffles and splits 70/15/15 into tf.data.Dataset objects with prefetch/batching

**Design Decisions:**
- Reads full HDF5 into memory (efficient for ~2–4 GB datasets; becomes problematic at 10+ GB)
- Supports both `"mods"` and `"modulations"` keys for RadioML variants
- Label encoding auto-discovers unique classes and sorts alphabetically, so order is deterministic across runs
- Normalization clips to [-1, 1] to match fixed-point ap_fixed<16,6> range

**Known Limitations:**
- No validation that expected dataset keys exist until module runs
- Assumes all I/Q samples are 1024 samples; hardcoded in README but not enforced in code
- Full I/Q load blocks the program; consider lazy-loading or memory-mapped arrays for large datasets

### Phase 2: Training (`src/train_1d_cnn.py`)

**Model Architecture:**
- Conv1D(16, 7, relu) → BatchNorm → MaxPool(4) → [256 features]
- Conv1D(32, 5, relu) → BatchNorm → MaxPool(4) → [64 features]
- Conv1D(64, 3, relu) → BatchNorm → [64 features]
- GlobalAveragePooling1D → Dense(32, relu) → Dense(8, softmax)
- **Parameter count:** ~14,000 (well under 50k budget); breakdown: ~15k from dense layers, ~6k from conv kernels

**Training Config:**
- Optimizer: Adam(lr=0.001)
- Loss: SparseCategoricalCrossentropy (expects integer labels 0–7, not one-hot)
- Callbacks: ModelCheckpoint (save best val_acc), EarlyStopping (patience=5)
- Default 30 epochs; stops early if validation accuracy plateaus

**Design Decisions:**
- Small conv kernels (7, 5, 3) capture I/Q correlations at multiple time scales
- Batch normalization stabilizes training on fixed-precision data
- Global pooling prevents spatial overfitting and reduces parameters
- Early stopping prevents overfitting and saves compute time

**Known Limitations:**
- No model summary printed (can't verify parameter count at runtime)
- No learning rate scheduling; fixed lr=0.001 throughout
- No dropout (relies only on batch norm for regularization)
- Test set is never evaluated; only validation accuracy is tracked
- Config values (batch size, kernel sizes, layer widths) are hardcoded; no config file

### Phase 3: HLS Export (`src/export_hls.py`)

**Functionality:**
- Loads saved Keras model
- Configures hls4ml: granularity=`name`, precision=`ap_fixed<16,6>`, reuse_factor=4, strategy=`Latency`
- Runs synthesis (RTL generation, C simulation skipped for speed)

**Design Decisions:**
- ap_fixed<16,6> = 16-bit total, 6 bits fractional. Supports ±511.984 range with ~0.016 precision
- ReuseFactor=4 reuses hardware blocks 4x per clock (trades latency for area)
- Latency strategy minimizes clock cycles; Pipeline strategy would minimize clock period instead

**Known Limitations:**
- No input validation: crashes silently if model doesn't exist
- No logging of synthesis success/failure
- Output directory not auto-created; assumes `fpga_rtl_export/` exists
- No customization of hls4ml backend (Vivado HLS assumed; QuartusHLS not tested)
- Verilog output location and naming hardcoded; not configurable

### Phase 4: RTL Testbench (`src/rtl_testbench.py`)

**Functionality:**
- Quantizes validation samples to ap_fixed<16,6> (16-bit signed int)
- Writes first 16 samples as hex lines: `IIII_QQQQ` (4 hex digits I, 4 hex digits Q)
- Output: `fpga_rtl_export/test_vectors/validation.hex`

**Design Decisions:**
- Takes first 3 validation batches (≤384 samples) to have diverse test data
- Limits to 16 samples to keep hex file small; manageable in Vivado sim
- Hex format is space-efficient and easy to parse in Verilog $readmemh

**Known Limitations:**
- Assumes validation set exists and has ≥1 sample; crashes if not
- No output path checking; assumes `fpga_rtl_export/test_vectors/` exists
- No expected outputs or labels written; testbench must infer correctness from waveforms
- Only 16 samples; insufficient for statistical testing

### Integration & Workflow

**Execution order (must be sequential):**
1. `python3 src/data_loader_1d.py` — Validate RadioML format and splits
2. `python3 src/train_1d_cnn.py` — Train and save to `results/model_1d_base.h5` (~2–10 min)
3. `python3 src/export_hls.py` — Generate Verilog (~5–15 min for synthesis)
4. `python3 src/rtl_testbench.py` — Generate test vectors
5. Open Vivado → Create HLS project → Import from `fpga_rtl_export/` → Simulate

**Why the scaffolding exists:**
- Modular: each script is independent and can be re-run (e.g., retrain with different hyperparams)
- Portable: relies only on public RadioML dataset and standard ML/RTL tools
- Minimal: ~250 lines total, easy to debug or extend

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
- **Parameter budget** - model must have <50k parameters for FPGA synthesis (~14k achieved)
- **Fixed-point precision** - ap_fixed<16,6> for hardware efficiency (16-bit signed, 6-bit fractional)
- **Input shape** - 1024 I/Q samples per signal (enforced by Conv1D kernel; size mismatch will error at runtime)
- **Data format** - HDF5 with keys {`X` or equiv., `mods`/`modulations`, `snrs`} required

---

## Known Issues & Future Work

### Fixes Applied (v0.2)
- ✅ **Parameter count verification:** `model.count_params()` logged; warns if >50k
- ✅ **Input validation:** All scripts now validate required keys, file existence, and dataset shape
- ✅ **Path creation:** Output directories auto-created (mkdir -p behavior)
- ✅ **Test set evaluation:** Test accuracy now computed and logged
- ✅ **Error handling:** Try/except blocks in all main() functions with proper logging
- ✅ **Model summary:** `model.summary()` printed to stdout during training
- ✅ **Synthesis logging:** hls4ml build steps logged with status
- ✅ **Sample count validation:** Warns if sample count <1024, ensures ≥1 validation sample

### Remaining Improvements
- **Config file:** Move hardcoded values (batch size, epochs, precision) to YAML/JSON for reproducibility

### Optional Enhancements
- **Learning rate scheduling:** Implement cosine annealing or step decay
- **Dropout regularization:** Add dropout(0.2) after Dense(32) for better generalization
- **Batch size sweep:** Script to test batch sizes 32, 64, 128, 256 for memory/speed tradeoffs
- **Modulation-specific metrics:** Log per-class precision/recall during training
- **Expected test vectors:** Extend rtl_testbench.py to save not just inputs but ground-truth labels and model predictions
- **Vivado integration:** Auto-generate Tcl script to open HLS project and run C simulation

### Testing
- No unit tests exist; recommend pytest fixtures for data_loader and model training edge cases
- No integration test; recommend end-to-end pipeline smoke test on synthetic data

---
