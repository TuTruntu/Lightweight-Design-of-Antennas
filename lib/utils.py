"""
Utility functions for far-field file I/O, S1P parsing, and GA plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def load_threshold_from_s1p(file_path, target_freqs_hz):
    """
    Parse a Touchstone .s1p file and interpolate S11 (dB) onto target
    frequency points.

    The file is assumed to use the format:
        # GHz S MA R 50
    with columns:  Freq(GHz)  Mag(linear)  Ang(deg)

    Args:
        file_path:        Path to the .s1p file.
        target_freqs_hz:  numpy array of target frequencies in Hz.

    Returns:
        numpy array of S11 (dB) interpolated at target_freqs_hz.
    """
    s1p_freqs = []
    s1p_s11_db = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('!') or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 2:
            try:
                f_ghz = float(parts[0])
                f_hz = f_ghz * 1e9
                mag = float(parts[1])

                if mag <= 1e-12:
                    db_val = -100.0
                else:
                    db_val = 20 * np.log10(mag)

                s1p_freqs.append(f_hz)
                s1p_s11_db.append(db_val)
            except ValueError:
                continue

    src_x = np.array(s1p_freqs)
    src_y = np.array(s1p_s11_db)

    interpolated_threshold = np.interp(target_freqs_hz, src_x, src_y)
    return interpolated_threshold


def read_ffs(filename):
    """
    Read a CST .ffs far-field source file.

    The parser skips header lines and looks for the '>> Phi, Theta' marker.
    Each data line thereafter is expected to contain:
        Phi  Theta  Re(E_Theta)  Im(E_Theta)  Re(E_Phi)  Im(E_Phi)

    Args:
        filename: Path to the .ffs file.

    Returns:
        numpy array with columns [Phi, Theta, Re(Eth), Im(Eth), Re(Eph), Im(Eph)],
        or None if the file is not found.
    """
    data_lines = []
    header_found = False

    filepath = filename

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    with open(filepath, 'r') as f:
        for line in f:
            if '>> Phi, Theta' in line:
                header_found = True
                continue

            if not header_found:
                continue

            # Skip blank lines and comment lines
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue

            try:
                vals = [float(x) for x in stripped.split()]
                if vals:
                    data_lines.append(vals)
            except ValueError:
                continue

    return np.array(data_lines)


def preget_multiport_phi90(filenames_list):
    """
    Pre-load multi-port far-field data at Phi = 90° and Phi = 270°.

    Reads one .ffs file per port and stacks the complex E-field columns.

    Args:
        filenames_list: List of .ffs file paths (one per port).

    Returns:
        E_Phi_total:            (Points, Ports) complex array.
        E_Theta_total:          (Points, Ports) complex array.
        phi_90_indices:         Indices where Phi ≈ 90°.
        phi_270_indices:        Indices where Phi ≈ 270°.
        theta_at_phi_90_270:    Theta angles at those Phi cuts (concatenated).
    """
    theta_all = None
    phi_all = None
    E_Theta_total = []
    E_Phi_total = []

    for i, fname in enumerate(filenames_list):
        data = read_ffs(fname)
        if data is None:
            raise Warning(f"Error: {fname} not found.")

        # Extract coordinates from the first file only
        if i == 0:
            phi_all = data[:, 0]
            theta_all = data[:, 1]

        # Complex E-field: columns 2+3=Re/Im(Eth), 4+5=Re/Im(Eph)
        E_Theta = data[:, 2] + 1j * data[:, 3]
        E_Phi   = data[:, 4] + 1j * data[:, 5]

        E_Theta_total.append(E_Theta)
        E_Phi_total.append(E_Phi)

    # Stack: (Points, Ports)
    E_Theta_total = np.column_stack(E_Theta_total)
    E_Phi_total   = np.column_stack(E_Phi_total)

    # Locate Phi = 90° (left half of pattern) and Phi = 270° (right half)
    phi_target1 = 90.0
    phi_target2 = 270.0
    phi_90_indices  = np.where(np.abs(phi_all - phi_target1) < 1e-3)[0]
    phi_270_indices = np.where(np.abs(phi_all - phi_target2) < 1e-3)[0]

    if len(phi_90_indices) == 0:
        print(f"Warning: No data found for Phi={phi_target1}. Using all data.")
        phi_90_indices = np.arange(len(phi_all))
    if len(phi_270_indices) == 0:
        print(f"Warning: No data found for Phi={phi_target2}. Using all data.")
        phi_270_indices = np.arange(len(phi_all))

    theta_at_phi_90  = theta_all[phi_90_indices]
    theta_at_phi_270 = theta_all[phi_270_indices]

    theta_at_phi_90_270 = np.concatenate(
        (theta_at_phi_90, theta_at_phi_270 + 180))

    return (E_Phi_total, E_Theta_total,
            phi_90_indices, phi_270_indices,
            theta_at_phi_90_270)


def get_ori_phi90(filenames):
    """
    Read the reference (original) far-field pattern from a single .ffs file
    and extract the Phi = 90° / 270° cut in dB.

    Args:
        filenames: Path to the reference .ffs file.

    Returns:
        E_total_dB:           Full 74-point pattern (left + flipped right) in dB.
        theta_at_phi_90_270:  Corresponding Theta angles.
    """
    data = read_ffs(filenames)

    phi_all   = data[:, 0]
    theta_all = data[:, 1]

    E_Theta = data[:, 2] + 1j * data[:, 3]
    E_Phi   = data[:, 4] + 1j * data[:, 5]

    # Single-column for a single-port reference
    E_Theta_total = np.column_stack(E_Theta)
    E_Phi_total   = np.column_stack(E_Phi)

    phi_target1 = 90.0
    phi_target2 = 270.0
    phi_90_indices  = np.where(np.abs(phi_all - phi_target1) < 1e-3)[0]
    phi_270_indices = np.where(np.abs(phi_all - phi_target2) < 1e-3)[0]

    if len(phi_90_indices) == 0:
        print(f"Warning: No data found for Phi={phi_target1}. Using all data.")
        phi_90_indices = np.arange(len(phi_all))
    if len(phi_270_indices) == 0:
        print(f"Warning: No data found for Phi={phi_target2}. Using all data.")
        phi_270_indices = np.arange(len(phi_all))

    theta_at_phi_90  = theta_all[phi_90_indices]
    theta_at_phi_270 = theta_all[phi_270_indices]

    # Total magnitude
    E_total_magnitude_left = np.sqrt(
        np.abs(E_Phi_total[0, phi_90_indices]) ** 2 +
        np.abs(E_Theta_total[0, phi_90_indices]) ** 2)
    E_total_magnitude_right = np.sqrt(
        np.abs(E_Phi_total[0, phi_270_indices]) ** 2 +
        np.abs(E_Theta_total[0, phi_270_indices]) ** 2)

    # Convert to dB
    E_total_dB_left  = 20 * np.log10(E_total_magnitude_left)
    E_total_dB_right = 20 * np.log10(E_total_magnitude_right)

    # Concatenate: left half + flipped right half → full 74-point pattern
    E_total_dB = np.concatenate(
        (E_total_dB_left, np.flip(E_total_dB_right)))
    theta_at_phi_90_270 = np.concatenate(
        (theta_at_phi_90, theta_at_phi_270 + 180))

    return E_total_dB, theta_at_phi_90_270


class GAProgressPlotter:
    """
    Real-time interactive plotter for tracking GA convergence.

    Updates a live matplotlib figure each generation showing:
      - Minimum cost (loss) per generation
      - Average cost (loss) per generation

    Usage:
        plotter = GAProgressPlotter(n_generations, save_path='output/convergence.png')
        for gen in range(1, n_generations + 1):
            ...
            plotter.update(gen, min_cost, avg_cost)
        plotter.finalize()
    """

    def __init__(self, n_generations, save_path=None, title='GA Optimization Progress'):
        """
        Args:
            n_generations: Total number of generations (for x-axis scaling).
            save_path:     Optional path to save the final figure.
            title:         Plot title.
        """
        self.n_generations = n_generations
        self.save_path = save_path
        self.title = title

        self.generations = []
        self.min_costs = []
        self.avg_costs = []

        # Setup interactive figure
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.fig.suptitle(self.title, fontsize=13, fontweight='bold')

        # Left subplot: linear scale (full range)
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Cost (Loss)')
        self.ax1.set_title('Convergence — Linear Scale')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(0, n_generations)

        self.line_min1, = self.ax1.plot([], [], 'b-', linewidth=2,
                                         label='Min Loss')
        self.line_avg1, = self.ax1.plot([], [], 'r--', linewidth=1.5,
                                         label='Avg Loss')
        self.ax1.legend(loc='upper right')

        # Right subplot: log scale — better for seeing late-stage fine convergence
        self.ax2.set_xlabel('Generation')
        self.ax2.set_ylabel('Cost (Loss) — log scale')
        self.ax2.set_title('Convergence — Log Scale')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(0, n_generations)

        self.line_min2, = self.ax2.plot([], [], 'b-', linewidth=2,
                                         label='Min Loss')
        self.line_avg2, = self.ax2.plot([], [], 'r--', linewidth=1.5,
                                         label='Avg Loss')
        self.ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.pause(0.01)

    def update(self, gen, min_cost, avg_cost):
        """
        Record a new data point and refresh the plot.

        Args:
            gen:      Current generation number (1-indexed).
            min_cost: Minimum cost in the population.
            avg_cost: Average cost in the population.
        """
        self.generations.append(gen)
        self.min_costs.append(min_cost)
        self.avg_costs.append(avg_cost)

        # Update linear-scale plot
        self.line_min1.set_data(self.generations, self.min_costs)
        self.line_avg1.set_data(self.generations, self.avg_costs)
        self.ax1.relim()
        self.ax1.autoscale_view(scaley=True)

        # Update log-scale plot
        self.line_min2.set_data(self.generations, self.min_costs)
        self.line_avg2.set_data(self.generations, self.avg_costs)
        self.ax2.relim()
        self.ax2.autoscale_view(scaley=True)
        self.ax2.set_yscale('log')

        # Dynamic y-axis label on log plot (handle zero/negative costs gracefully)
        if min_cost <= 0:
            self.ax2.set_yscale('linear')

        plt.pause(0.01)

    def finalize(self):
        """Switch to non-interactive mode, save if requested, and display final plot."""
        plt.ioff()

        # Mark the best solution on the linear plot
        if self.min_costs:
            best_gen = np.argmin(self.min_costs)
            best_cost = self.min_costs[best_gen]
            self.ax1.annotate(
                f'Best: {best_cost:.2f}\n(Gen {self.generations[best_gen]})',
                xy=(self.generations[best_gen], best_cost),
                xytext=(self.generations[best_gen] + self.n_generations * 0.05,
                        best_cost * 1.15 if best_cost > 0 else 10),
                arrowprops=dict(arrowstyle='->', color='darkgreen'),
                fontsize=9, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

        plt.tight_layout()

        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.fig.savefig(self.save_path, dpi=150, bbox_inches='tight')
            print(f'Convergence plot saved to: {self.save_path}')

        plt.show(block=True)


def plot_optimization_progress(logbook):
    """
    Plot GA convergence: best and average cost (loss) per generation.

    This is the static (post-hoc) version — use GAProgressPlotter for
    real-time plotting during optimization.

    Args:
        logbook: DEAP Logbook with 'gen', 'min', 'avg' chapters.
    """
    gen = logbook.select("gen")
    min_cost = logbook.select("min")
    avg_cost = logbook.select("avg")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('GA Optimization — Convergence', fontsize=13, fontweight='bold')

    # Linear scale
    ax1.plot(gen, min_cost, 'b-', linewidth=2, label='Min Loss')
    ax1.plot(gen, avg_cost, 'r--', linewidth=2, label='Avg Loss')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Cost (Loss)')
    ax1.set_title('Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate best
    best_idx = np.argmin(min_cost)
    ax1.annotate(f'Best: {min_cost[best_idx]:.2f} (Gen {gen[best_idx]})',
                 xy=(gen[best_idx], min_cost[best_idx]),
                 xytext=(gen[best_idx] + len(gen) * 0.05,
                         min_cost[best_idx] * 1.15 if min_cost[best_idx] > 0 else 10),
                 arrowprops=dict(arrowstyle='->', color='darkgreen'),
                 fontsize=9, color='darkgreen', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    # Log scale
    ax2.plot(gen, min_cost, 'b-', linewidth=2, label='Min Loss')
    ax2.plot(gen, avg_cost, 'r--', linewidth=2, label='Avg Loss')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Cost (Loss) — log scale')
    ax2.set_title('Log Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Fallback to linear if values are non-positive
    if np.min(min_cost) <= 0:
        ax2.set_yscale('linear')

    plt.tight_layout()
    plt.show()
