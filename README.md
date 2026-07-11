# Lightweight-Design-of-Antenna
## Current Feature Extraction & Multi-Port Antenna Optimization

## Overview

- **run_curExtraction.py** — Current feature extraction via PCA + K-Means. Produces binary feature maps for radiation and ground structures from CST simulation images.
- **run_multiportOptimization.py** — GPU-accelerated genetic algorithm for multi-port antenna tuning. Jointly optimizes S11 and far-field patterns at 1.8 / 2.3 / 2.8 GHz, with real-time convergence visualization.
- **run_ultEvaluation.py** — Optimization result evaluation: compares optimized S11 against MED.s1p reference, and displays 3D antenna models before/after optimization side-by-side.

## Directory Structure

```
AntennaOpt/
├── run_curExtraction.py          # Current extraction entry point
├── run_multiportOptimization.py  # Multi-port optimization entry point
├── run_ultEvaluation.py          # Result evaluation entry point
├── lib/                          # Shared library (3 modules)
│   ├── cutter.py                 # AntennaCutter — PCA+KMeans image processing
│   ├── optimizers.py             # GPU genetic algorithm optimizers
│   └── utils.py                  # .ffs / .s1p I/O, GA real-time plotting
├── data/                         # Input data (user-provided)
│   ├── curimg/                   #   Current distribution images
│   ├── ffdata/                   #   Far-field .ffs files
│   ├── modeldata/                #   3D antenna models (.stl)
│   └── sNpdata/                  #   Touchstone .sNp / .s1p files
├── output/                       # Generated results
│   ├── charadata/                #   Extraction results (.npy)
│   └── MultiportResult/          #   Optimization results, convergence & S11 plots
├── README_CN.md
└── README_EN.md
```

## Dependencies

```bash
pip install numpy matplotlib scikit-rf deap torch opencv-python pillow
```

Optional (3D model viewer):
```bash
pip install pyqt6 vtk
```

## Module 1 — Current Feature Extraction

```bash
python run_curExtraction.py
```

**Pipeline**: PCA (36 phases → 1st principal component) → K-Means binarization → 3-frequency max merge → quadrant symmetrization.

Output: `output/charadata/p_chara.npy` (radiation), `g_chara.npy` (ground).

## Module 2 — Multi-Port Antenna Optimization

```bash
python run_multiportOptimization.py
```

**Cost function**: `cost = 0.6 × cost_S11 + 0.4 × cost_farfield`

- S11 above −11 dB in [1.8, 2.8] GHz → ReLU penalty.
- Far-field: segmented weighted difference averaged over 1.8 / 2.3 / 2.8 GHz.
- Real-time convergence plot updates each generation (min loss + avg loss), auto-saved as `convergence.png`.

**Configurable parameters** (top of `run_multiportOptimization.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 500 | GA population size |
| `n_generations` | 240 | Number of generations |
| `ifs1p` | False | True = S-parameter only (vs reference .s1p) |
| `S11_FREQ_BAND` | (1.8e9, 2.8e9) | S11 target frequency band |
| `S11_THRESHOLD` | −11.0 dB | S11 penalty threshold |

## Module 3 — Result Evaluation

```bash
python run_ultEvaluation.py
```

**Features**:
- Loads optimized solution `result_v1.npy`, computes S11 and plots comparison against `MED.s1p` (saved as `S11_comparison.png`).
- Prints S11 statistics summary in the target band [1.8–2.8] GHz.
- Side-by-side 3D view of the antenna before (`A_Original.stl`) and after (`D_Final.stl`) optimization.

## Data Preparation

- **data/curimg/** — 36 phase-varied current images per frequency (`img_0.png` … `img_35.png`).
- **data/ffdata/ffX.X/** — Per-port far-field .ffs (`ff1.ffs` … `ffN.ffs`).
- **data/ffdata/ori_X.X.ffs` — Reference antenna far-field pattern.
- **data/sNpdata/multiport_sNp.s501p** — Multi-port Z-parameter matrix in Touchstone format.
- **data/sNpdata/MED.s1p** — Reference S11 for evaluation comparison.
- **data/modeldata/A_Original.stl** — Antenna 3D model before optimization.
- **data/modeldata/D_Final.stl** — Antenna 3D model after optimization.
