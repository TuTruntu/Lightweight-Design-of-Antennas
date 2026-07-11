"""
Multi-Port Antenna Optimisation
-------------------------------
Optimise per-port load impedances (binary open/short) via a GPU-accelerated
genetic algorithm. The cost function jointly considers:

  - S11 < -10 dB across [1.68, 3.14] GHz.
  - Far-field pattern match against reference at 1.8 / 2.3 / 2.8 GHz.

Usage:
    python run_optimization.py
"""

import os
import sys

# Ensure the project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import skrf as rf

from lib.utils import (preget_multiport_phi90,
                       get_ori_phi90,
                       GAProgressPlotter)
from lib.optimizers import (GaOptimizer_GPU_SandF,
                            GaOptimizer_GPU_SvsS,
                            run_EliteGa_gpu)


# ===========================================================================
# 1. Configuration
# ===========================================================================

DATA   = os.path.join(ROOT, 'data')
OUTPUT = os.path.join(ROOT, 'output')

# Output path for best solution
xsavepath = os.path.join(OUTPUT, 'MultiportResult', 'result_v1.npy')

# Convergence plot save path
plot_savepath = os.path.join(OUTPUT, 'MultiportResult', 'convergence.png')

# Set ifs1p=True to use S-vs-S reference optimisation only
ifs1p = False

# Frequency ranges
frqRan  = [1.75e9, 3e9]      # used when ifs1p=True
frqRan1 = [1.8e9,  2.8e9]    # used when ifs1p=False

# File paths
sNp_path           = os.path.join(DATA, 'sNpdata', 'multiport_sNp.s501p')
threshold_sNp_path = os.path.join(DATA, 'sNpdata', 'MED_opt.s1p')

ffFpath1 = os.path.join(DATA, 'ffdata', 'ff1.8') + os.sep
ffFpath2 = os.path.join(DATA, 'ffdata', 'ff2.3') + os.sep
ffFpath3 = os.path.join(DATA, 'ffdata', 'ff2.8') + os.sep

# GA settings
population_size = 500
n_generations   = 240

# ===========================================================================
# 2. Load Z-parameters
# ===========================================================================

network  = rf.Network(sNp_path)
z_params = network.z      # (num_freqs, num_ports, num_ports)
freq     = network.f      # Hz

port_num  = z_params.shape[1]
num_freqs = len(freq)

print(f'Ports: {port_num},  Freq points: {num_freqs},  '
      f'Pop size: {population_size},  Generations: {n_generations}')

# ===========================================================================
# 3. Load far-field data (3 frequencies)
# ===========================================================================

# --- Per-port far-field pre-load ---
# 1.8 GHz
ffpath_list1 = [ffFpath1 + f'ff{i + 1}.ffs' for i in range(port_num)]
(E_Phi_total1, E_Theta_total1,
 phi_90_indices, phi_270_indices,
 _) = preget_multiport_phi90(ffpath_list1)

# 2.3 GHz
ffpath_list2 = [ffFpath2 + f'ff{i + 1}.ffs' for i in range(port_num)]
(E_Phi_total2, E_Theta_total2,
 _, _, _) = preget_multiport_phi90(ffpath_list2)

# 2.8 GHz
ffpath_list3 = [ffFpath3 + f'ff{i + 1}.ffs' for i in range(port_num)]
(E_Phi_total3, E_Theta_total3,
 _, _, _) = preget_multiport_phi90(ffpath_list3)

# --- Reference far-field patterns ---
E_total_dB_ori1, _ = get_ori_phi90(
    os.path.join(DATA, 'ffdata', 'ori_1.8.ffs'))
E_total_dB_ori2, _ = get_ori_phi90(
    os.path.join(DATA, 'ffdata', 'ori_2.3.ffs'))
E_total_dB_ori3, _ = get_ori_phi90(
    os.path.join(DATA, 'ffdata', 'ori_2.8.ffs'))

# --- Frequency indices in Z-parameter array ---
f_idx_1 = np.argmin(np.abs(freq - 1.8e9))
f_idx_2 = np.argmin(np.abs(freq - 2.3e9))
f_idx_3 = np.argmin(np.abs(freq - 2.8e9))

# ===========================================================================
# 4. Genetic optimisation
# ===========================================================================

if __name__ == '__main__':
    print(f'Saving results to: {xsavepath}')

    if ifs1p:
        print('Using: GaOptimizer_GPU_SvsS')
        gpu_opter = GaOptimizer_GPU_SvsS(z_params, freq,
                                         threshold_sNp_path, frqRan)
    else:
        print('Using: GaOptimizer_GPU_SandF (3-frequency far-field)')
        gpu_opter = GaOptimizer_GPU_SandF(
            z_params, freq,
            [f_idx_1, f_idx_2, f_idx_3],
            [E_Phi_total1, E_Phi_total2, E_Phi_total3],
            [E_Theta_total1, E_Theta_total2, E_Theta_total3],
            phi_90_indices, phi_270_indices,
            [E_total_dB_ori1, E_total_dB_ori2, E_total_dB_ori3])

    n_variables = int((port_num - 1) / 4)

    # ---- Setup real-time convergence plotter ----
    plotter = GAProgressPlotter(
        n_generations=n_generations,
        save_path=plot_savepath,
        title=f'GA Optimization — Pop={population_size}, Gen={n_generations}')

    def on_generation(gen, min_cost, avg_cost):
        """Callback invoked after each generation for live plotting."""
        plotter.update(gen, min_cost, avg_cost)

    # ---- Run GA with real-time visualisation ----
    best_solution, logs = run_EliteGa_gpu(
        n_variables,
        gpu_opter=gpu_opter,
        population_size=population_size,
        n_generations=n_generations,
        callback=on_generation)

    # ---- Finalize plot (save + show) ----
    plotter.finalize()

    os.makedirs(os.path.dirname(xsavepath), exist_ok=True)
    np.save(xsavepath, best_solution)

    print('Optimization finished.')
    print(f'Best solution saved to: {xsavepath}')
    print('Best X:', best_solution)
