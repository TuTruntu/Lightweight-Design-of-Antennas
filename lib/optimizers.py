"""
GPU-accelerated genetic-algorithm optimizers for multi-port antenna tuning.

Classes:
    GaOptimizer_GPU_SandF  - Joint S11 + far-field optimization (3-frequency).
    GaOptimizer_GPU_SvsS   - S-parameter-only optimization vs reference .s1p.

Functions:
    run_EliteGa_gpu        - Elitism-preserving GA loop with GPU batch evaluation.
"""

import os
import random

import numpy as np
import torch
from deap import base, creator, tools

from lib.utils import load_threshold_from_s1p

# Fix OpenMP duplicate library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaOptimizer_GPU_SandF:
    """
    GPU-accelerated joint S11 + far-field pattern optimizer.

    Evaluates cost = 0.6 * cost_S11 + 0.4 * cost_farfield where
    far-field is averaged across three target frequencies (1.8 / 2.3 / 2.8 GHz).
    """

    def __init__(self, z_params, freq, f_idx_list,
                 E_Phi_total_list, E_Theta_total_list,
                 phi_90_indices, phi_270_indices,
                 E_total_dB_ori_list):
        """
        Args:
            z_params:             numpy array  (Freq, Port, Port) — complex Z-matrix.
            freq:                 numpy array  (Freq,) — frequency points [Hz].
            f_idx_list:           [f_idx_1, f_idx_2, f_idx_3] indices into freq.
            E_Phi_total_list:     [E_Phi_1.8, E_Phi_2.3, E_Phi_2.8] (Points, Ports).
            E_Theta_total_list:   [E_Theta_1.8, E_Theta_2.3, E_Theta_2.8].
            phi_90_indices:       Indices where Phi ≈ 90 deg.
            phi_270_indices:      Indices where Phi ≈ 270 deg.
            E_total_dB_ori_list:  [ori_1.8, ori_2.3, ori_2.8] reference patterns [dB].
        """
        self.z_params = torch.tensor(z_params, device=device,
                                     dtype=torch.complex64)
        self.freq = torch.tensor(freq, device=device, dtype=torch.float32)

        # Far-field data — one entry per target frequency
        self.E_Phi_total_list = [
            torch.tensor(e, device=device, dtype=torch.complex64)
            for e in E_Phi_total_list
        ]
        self.E_Theta_total_list = [
            torch.tensor(e, device=device, dtype=torch.complex64)
            for e in E_Theta_total_list
        ]

        # Indices shared across all frequencies
        self.phi_90_idx = torch.tensor(phi_90_indices, device=device,
                                       dtype=torch.long)
        self.phi_270_idx = torch.tensor(phi_270_indices, device=device,
                                        dtype=torch.long)

        # Reference far-field patterns (one per frequency)
        self.E_total_dB_ori_list = [
            torch.tensor(e, device=device, dtype=torch.float32)
            for e in E_total_dB_ori_list
        ]

        self.f_idx_list = f_idx_list
        self.n_freq = len(freq)
        self.port_num = self.z_params.shape[1]

        # Precompute S-parameter target frequency band indices [L1, L2]
        f_min, f_max = self.freq.min(), self.freq.max()
        f_len = self.n_freq - 1
        self.L1 = int(round((1.68e9 - f_min.item()) /
                            (f_max.item() - f_min.item()) * f_len))
        self.L2 = int(round((3.14e9 - f_min.item()) /
                            (f_max.item() - f_min.item()) * f_len))
        self.L1 = max(0, min(self.L1, f_len))
        self.L2 = max(0, min(self.L2, f_len))
        if self.L2 <= self.L1:
            self.L2 = self.L1 + 1

    def get_normI_batch(self, x_batch, f_idx):
        """
        Batch-compute normalised current vector I = V / Z at a given frequency.

        x_batch: (Batch, N_vars) — binary switch states (0=short, 1=open).
        f_idx:   index into z_params for the target frequency.

        Returns:
            normI: (Batch, Port, 1) complex tensor.
        """
        batch_size = x_batch.shape[0]

        # Z-matrix at the target frequency, expanded to batch
        z_f = self.z_params[f_idx]
        ZZ = z_f.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Load switches onto diagonal (skip port 1): 1→open(1e9), 0→short(1e-6)
        diag_vals = torch.where(x_batch == 1,
                                torch.tensor(1e9, device=device),
                                torch.tensor(1e-6, device=device)).to(torch.complex64)
        ZZ[:, 1:, 1:] += torch.diag_embed(diag_vals)

        # Reduce: delete row 0 & col 0 → (Batch, P-1, P-1)
        ZZ_reduced = ZZ[:, 1:, 1:]

        # Excitation vector: first column without Z11 → (P-1,)
        z0h = z_f[1:, 0]
        z0h = z0h.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

        # Solve I_remaining = -inv(ZZ_reduced) @ z0h
        i_remaining = torch.linalg.solve(ZZ_reduced, -z0h)

        # Prepend port-1 current (always 1)
        ones = torch.ones((batch_size, 1, 1), device=device,
                          dtype=torch.complex64)
        normI = torch.cat([ones, i_remaining], dim=1)

        return normI

    def getS_batch(self, x_batch):
        """
        Batch-compute S11 across all frequencies.

        x_batch: (Batch, N_vars) — binary switch states.

        Returns:
            S11_all: (Batch, n_freq) complex tensor.
        """
        batch_size = x_batch.shape[0]
        S11_all = torch.zeros((batch_size, self.n_freq),
                              dtype=torch.complex64, device=device)

        # Map binary states to load impedances: 0→1e-6Ω, 1→1e6Ω
        x_mapped = x_batch * 1_000_000.0 + 0.000_001
        x_complex = x_mapped.to(torch.complex64)
        ZL_diag = torch.diag_embed(x_complex)  # (Batch, P-1, P-1)

        for i in range(self.n_freq):
            z_f = self.z_params[i]
            Z00 = z_f[0, 0]
            Z0A = z_f[0, 1:]          # (1, P-1)
            ZA0 = z_f[1:, 0]          # (P-1, 1)
            ZAA = z_f[1:, 1:]         # (P-1, P-1)

            ZAA_batch = ZAA.unsqueeze(0).expand(batch_size, -1, -1)
            ZA0_batch = ZA0.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

            # Solve (ZAA + ZL) @ temp = ZA0
            Mat = ZAA_batch + ZL_diag
            temp = torch.linalg.solve(Mat, ZA0_batch)

            # Back-substitute: Z11 = Z00 - Z0A @ temp
            Z0A_batch = Z0A.unsqueeze(0).expand(batch_size, 1, -1)
            term2 = torch.matmul(Z0A_batch, temp).squeeze()
            Z11_f = Z00 - term2

            S11_all[:, i] = (Z11_f - 50.0) / (Z11_f + 50.0)

        return S11_all

    def calculate_cost(self, population_list):
        """
        Core entry point: compute cost for the entire population.

        Args:
            population_list: list of individuals (each a list of 0/1).

        Returns:
            numpy array of scalar costs, shape (population_size,).
        """
        x_original = torch.tensor(population_list, device=device,
                                  dtype=torch.float32)
        # Replicate each individual's switch states 4 times (4 symmetry sectors)
        x_all = x_original.repeat(1, 4)

        # ---- S-parameter cost ----
        S11_complex = self.getS_batch(x_all)
        S11_dB = 20 * torch.log10(torch.abs(S11_complex) + 1e-12)

        target_S = S11_dB[:, self.L1: self.L2 + 1]
        threshold = -10.5

        diff = target_S - threshold
        penalty_per_point = torch.relu(diff)          # ReLU penalty
        cost_s = torch.sum(penalty_per_point, dim=1)

        # ---- Far-field cost (average over 3 frequencies) ----
        cost_ff = torch.zeros_like(cost_s)

        for freq_idx in range(len(self.f_idx_list)):
            f_idx = self.f_idx_list[freq_idx]
            E_Phi_total = self.E_Phi_total_list[freq_idx]
            E_Theta_total = self.E_Theta_total_list[freq_idx]
            E_total_dB_ori = self.E_total_dB_ori_list[freq_idx]

            normI = self.get_normI_batch(x_all, f_idx)

            # Synthesize far-field: E(Points, Ports) @ I(Batch, Ports, 1)
            E_Phi_90 = E_Phi_total[self.phi_90_idx].unsqueeze(0)
            E_Theta_90 = E_Theta_total[self.phi_90_idx].unsqueeze(0)
            E_Phi_270 = E_Phi_total[self.phi_270_idx].unsqueeze(0)
            E_Theta_270 = E_Theta_total[self.phi_270_idx].unsqueeze(0)

            f_L_Phi = torch.matmul(E_Phi_90, normI)
            f_L_Theta = torch.matmul(E_Theta_90, normI)
            f_R_Phi = torch.matmul(E_Phi_270, normI)
            f_R_Theta = torch.matmul(E_Theta_270, normI)

            mag_L = torch.sqrt(f_L_Theta.abs() ** 2 + f_L_Phi.abs() ** 2)
            mag_R = torch.sqrt(f_R_Theta.abs() ** 2 + f_R_Phi.abs() ** 2)

            dB_L = 20 * torch.log10(mag_L + 1e-12).squeeze(-1)
            dB_R = 20 * torch.log10(mag_R + 1e-12).squeeze(-1)

            # Concatenate left-half + flipped right-half → full pattern
            E_dyn = torch.cat((dB_L, torch.flip(dB_R, dims=[1])), dim=1)

            target = E_total_dB_ori.unsqueeze(0)
            diff_ff = torch.abs(E_dyn - target)

            # ---- Segmented far-field penalty (5 angular regions) ----
            # 0-10   (  0°– 45°): penalize if BELOW reference, weight 5
            m = E_dyn[:, 0:10] < target[:, 0:10]
            cost_ff += torch.sum(m * 5 * diff_ff[:, 0:10], dim=1)

            # 10-19  ( 45°– 90°): penalize if ABOVE reference, weight 2
            m = E_dyn[:, 10:19] > target[:, 10:19]
            cost_ff += torch.sum(m * 2 * diff_ff[:, 10:19], dim=1)

            # 19-56  ( 90°–270°): penalize if ABOVE reference, weight 20
            m = E_dyn[:, 19:56] > target[:, 19:56]
            cost_ff += torch.sum(m * 20 * diff_ff[:, 19:56], dim=1)

            # 56-64  (270°–315°): penalize if ABOVE reference, weight 2
            m = E_dyn[:, 56:64] > target[:, 56:64]
            cost_ff += torch.sum(m * 2 * diff_ff[:, 56:64], dim=1)

            # 64-74  (315°–360°): penalize if BELOW reference, weight 5
            m = E_dyn[:, 64:74] < target[:, 64:74]
            cost_ff += torch.sum(m * 5 * diff_ff[:, 64:74], dim=1)

        cost_ff = cost_ff / len(self.f_idx_list)   # average over 3 frequencies

        cost_total = 0.6 * cost_s + 0.4 * cost_ff

        return cost_total.cpu().numpy()


class GaOptimizer_GPU_SvsS:
    """
    GPU-accelerated S-parameter-only optimizer.

    Compares S11 against a dynamic threshold loaded from a reference .s1p file.
    """

    def __init__(self, z_params, freq, s1p_file_path, frqRan):
        """
        Args:
            z_params:       numpy array (Freq, Port, Port) — complex Z-matrix.
            freq:           numpy array (Freq,) — frequency points [Hz].
            s1p_file_path:  Path to reference .s1p file for dynamic threshold.
            frqRan:         [f_min, f_max] target frequency band [Hz].
        """
        self.n_freq = len(freq)
        self.z_params = torch.tensor(z_params, device=device,
                                     dtype=torch.complex64)
        self.freq = torch.tensor(freq, device=device, dtype=torch.float64)
        self.port_num = self.z_params.shape[1]

        # Precompute S-parameter target frequency band indices [L1, L2]
        f_min, f_max = self.freq.min(), self.freq.max()
        f_len = self.n_freq - 1
        self.L1 = int(round((frqRan[0] - f_min.item()) /
                            (f_max.item() - f_min.item()) * f_len))
        self.L2 = int(round((frqRan[1] - f_min.item()) /
                            (f_max.item() - f_min.item()) * f_len))
        self.L1 = max(0, min(self.L1, f_len))
        self.L2 = max(0, min(self.L2, f_len))
        if self.L2 <= self.L1:
            self.L2 = self.L1 + 1

        # Load dynamic threshold from reference .s1p
        freq_numpy = freq
        full_threshold_numpy = load_threshold_from_s1p(s1p_file_path,
                                                       freq_numpy)
        full_threshold_tensor = torch.tensor(full_threshold_numpy,
                                             device=device, dtype=torch.float64)
        self.dynamic_threshold = full_threshold_tensor[
            self.L1: self.L2 + 1].unsqueeze(0)

    def getS_batch(self, x_batch):
        """Batch-compute S11 across all frequencies (memory-optimised)."""
        batch_size = x_batch.shape[0]
        S11_all = torch.zeros((batch_size, self.n_freq),
                              dtype=torch.complex64, device=device)

        x_mapped = x_batch * 1_000_000.0 + 0.000_001
        x_complex = x_mapped.to(torch.complex64)
        ZL_diag = torch.diag_embed(x_complex)

        for i in range(self.n_freq):
            z_f = self.z_params[i]
            Z00 = z_f[0, 0]
            Z0A = z_f[0, 1:]
            ZA0 = z_f[1:, 0]
            ZAA = z_f[1:, 1:]

            ZAA_batch = ZAA.unsqueeze(0).expand(batch_size, -1, -1)
            ZA0_batch = ZA0.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)

            Mat = ZAA_batch + ZL_diag
            temp = torch.linalg.solve(Mat, ZA0_batch)

            Z0A_batch = Z0A.unsqueeze(0).expand(batch_size, 1, -1)
            term2 = torch.matmul(Z0A_batch, temp).squeeze()
            Z11_f = Z00 - term2

            S11_all[:, i] = (Z11_f - 50.0) / (Z11_f + 50.0)

        return S11_all

    def calculate_cost(self, population_list):
        """
        Compute S-parameter cost vs dynamic threshold.

        Args:
            population_list: list of individuals.

        Returns:
            numpy array of scalar costs.
        """
        x_original = torch.tensor(population_list, device=device,
                                  dtype=torch.float32)
        x_all = x_original.repeat(1, 4)

        S11_complex = self.getS_batch(x_all)
        S11_dB = 20 * torch.log10(torch.abs(S11_complex) + 1e-12)

        target_S = S11_dB[:, self.L1: self.L2 + 1]

        # Difference vs dynamic threshold (with +0.5 dB margin)
        diff = target_S - self.dynamic_threshold
        diff = diff + 0.5

        penalty_per_point = torch.relu(diff)
        cost_s = torch.sum(penalty_per_point, dim=1)

        return cost_s.cpu().numpy()


def run_EliteGa_gpu(n_variables, gpu_opter,
                    population_size=300, n_generations=100,
                    callback=None):
    """
    Run GPU-accelerated genetic algorithm with elitism preservation.

    At each generation the single best individual is cloned and carried
    forward unchanged (elitism), while the remaining (N-1) slots are
    filled through selection, two-point crossover (pc=0.5), and
    bit-flip mutation (pm=0.3, indpb=0.05).

    Args:
        n_variables:      Number of binary optimisation variables.
        gpu_opter:        Optimizer instance with a ``calculate_cost`` method.
        population_size:  GA population size.
        n_generations:    Number of generations to evolve.
        callback:         Optional callable(gen, min_cost, avg_cost) called
                          after each generation for real-time monitoring.

    Returns:
        (best_individual, logbook) — best DEAP individual & stats log.
    """
    # ---- 1. DEAP setup (clear old classes to avoid conflicts) ----
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=n_variables)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ---- 2. Batch evaluation helper ----
    def evaluate_all(population):
        costs = gpu_opter.calculate_cost(population)
        for ind, cost in zip(population, costs):
            ind.fitness.values = (float(cost),)

    # ---- 3. Initial population ----
    pop = toolbox.population(n=population_size)
    evaluate_all(pop)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    logbook = tools.Logbook()

    print(f"Starting GPU GA with Elitism (Pop: {population_size})...")
    print('gen; Pop num; Avg Cost; Min Cost')

    # ---- 4. Generational loop ----
    for gen in range(1, n_generations + 1):

        # A. Elitism — keep the best individual
        best_ind = tools.selBest(pop, 1)[0]
        best_ind_clone = toolbox.clone(best_ind)

        # B. Produce offspring (PopSize - 1)
        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover (pc = 0.5)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation (pm = 0.3, indpb = 0.05)
        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # C. Evaluate only modified individuals
        invalid_ind = [ind for ind in offspring
                       if not ind.fitness.valid]
        if invalid_ind:
            evaluate_all(invalid_ind)

        # D. Merge elite back into population
        pop[:] = [best_ind_clone] + offspring

        # E. Logging
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        print(f"{gen}: {len(pop)} | {record['avg']:.2f} | "
              f"{record['min']:.2f} (Elitism Preserved)")

        # F. Real-time callback (e.g. for live plotting)
        if callback is not None:
            callback(gen, record['min'], record['avg'])

    return tools.selBest(pop, 1)[0], logbook
