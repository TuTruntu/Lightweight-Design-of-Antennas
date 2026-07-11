# AntennaOpt — Multi-Port Antenna Optimization & Current Feature Extraction

## Overview

- **run_extraction.py** — Current feature extraction via PCA + K-Means. Produces binary feature maps for radiation and ground structures from CST simulation images.
- **run_optimization.py** — GPU-accelerated genetic algorithm for multi-port antenna tuning. Jointly optimizes S11 and far-field patterns at 1.8 / 2.3 / 2.8 GHz.

## Directory Structure

```
AntennaOpt/
├── run_extraction.py          # Current extraction entry point
├── run_optimization.py        # Multi-port optimization entry point
├── lib/                       # Shared library (3 modules)
│   ├── cutter.py              # AntennaCutter — PCA+KMeans image processing
│   ├── optimizers.py          # GPU genetic algorithm optimizers
│   └── utils.py               # .ffs / .s1p I/O, plotting utilities
├── data/                      # Input data (user-provided)
│   ├── curimg/                #   Current distribution images
│   ├── ffdata/                #   Far-field .ffs files
│   └── sNpdata/               #   Touchstone .sNp / .s1p files
├── output/                    # Generated results
│   ├── charadata/             #   Extraction results (.npy)
│   └── MultiportResult/       #   Optimization results (.npy)
├── README_CN.md
└── README_EN.md
```

## Dependencies

```bash
pip install numpy matplotlib scikit-rf deap torch opencv-python pillow
```

## Module 1 — Current Feature Extraction

```bash
python run_extraction.py
```

**Pipeline**: PCA (36 phases → 1st principal component) → K-Means binarization → 3-frequency max merge → quadrant symmetrization.

Output: `output/charadata/p_chara.npy` (radiation), `g_chara.npy` (ground).

## Module 2 — Multi-Port Antenna Optimization

```bash
python run_optimization.py
```

**Cost function**: `cost = 0.6 × cost_S11 + 0.4 × cost_farfield`

- S11 above -10.5 dB in [1.68, 3.14] GHz → ReLU penalty.
- Far-field: segmented weighted difference averaged over 1.8 / 2.3 / 2.8 GHz.

**Configurable parameters** (top of `run_optimization.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 500 | GA population size |
| `n_generations` | 240 | Number of generations |
| `ifs1p` | False | True = S-parameter only (vs reference .s1p) |

## Data Preparation

- **data/curimg/** — 36 phase-varied current images per frequency (`img_0.png` … `img_35.png`).
- **data/ffdata/ffX.X/** — Per-port far-field .ffs (`ff1.ffs` … `ffN.ffs`).
- **data/ffdata/ori_X.X.ffs** — Reference antenna far-field pattern.
- **data/sNpdata/multiport_sNp.s501p** — Multi-port Z-parameter matrix in Touchstone format.
