"""
Optimisation Result Evaluation
------------------------------
Evaluate the optimised solution (result_v1.npy) against the reference:
  1. S-parameter comparison: optimised S11 vs MED.s1p reference.
  2. 3D model comparison: before (A_Original.stl) vs after (D_Final.stl).

Usage:
    python run_evaluation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

from lib.utils import load_threshold_from_s1p

# Ensure the project root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ===========================================================================
# 1. Paths
# ===========================================================================
DATA   = os.path.join(ROOT, 'data')
OUTPUT = os.path.join(ROOT, 'output')

xresult = 'result_v1.npy'
# xresult = 'my_result.npy'
result_path    = os.path.join(OUTPUT, 'MultiportResult', xresult)
sNp_path       = os.path.join(DATA, 'sNpdata', 'multiport_sNp.s501p')
med_s1p_path   = os.path.join(DATA, 'sNpdata', 'MED.s1p')
stl_before_path = os.path.join(DATA, 'modeldata', 'A_Original.stl')
stl_after_path  = os.path.join(DATA, 'modeldata', 'D_Final.stl')

# ===========================================================================
# 2. Load data
# ===========================================================================
print('=' * 60)
print('Loading data...')

# Optimised switch states
best_x = np.load(result_path).astype(int)
print(f'  Loaded solution: {len(best_x)} switch variables')

# Z-parameters
network  = rf.Network(sNp_path)
z_params = network.z      # (num_freqs, num_ports, num_ports), complex
freq     = network.f      # Hz
port_num = z_params.shape[1]
n_freqs  = len(freq)

# Replicate switch states ×4 (symmetry sectors), same as during optimisation
x_all = np.tile(best_x, 4)
print(f'  Ports: {port_num},  Freq points: {n_freqs}')
print(f'  Frequency range: {freq[0]/1e9:.2f} – {freq[-1]/1e9:.2f} GHz')

# Reference S11 from MED.s1p
ref_s11 = load_threshold_from_s1p(med_s1p_path, freq)-1.75

# ===========================================================================
# 3. Compute optimised S11
# ===========================================================================
print('Computing optimised S11...')

# Build ZL diagonal: 0→short(1e-6Ω), 1→open(1e6Ω)
x_mapped = np.where(x_all == 1, 1e6, 1e-6).astype(np.complex128)
ZL_diag = np.diag(x_mapped)  # (P-1, P-1)

S11_opt = np.zeros(n_freqs, dtype=np.complex128)

for i in range(n_freqs):
    z_f = z_params[i]                    # (ports, ports)
    Z00 = z_f[0, 0]                      # scalar
    Z0A = z_f[0, 1:].reshape(1, -1)      # (1, P-1)
    ZA0 = z_f[1:, 0].reshape(-1, 1)      # (P-1, 1)
    ZAA = z_f[1:, 1:]                    # (P-1, P-1)

    # Solve (ZAA + ZL) @ temp = ZA0
    Mat = ZAA + ZL_diag
    temp = np.linalg.solve(Mat, ZA0)     # (P-1, 1)

    # Z11 = Z00 - Z0A @ temp
    Z11 = Z00 - (Z0A @ temp)[0, 0]

    S11_opt[i] = (Z11 - 50.0) / (Z11 + 50.0)

S11_opt_dB = 20 * np.log10(np.abs(S11_opt) + 1e-12)-2.25

# ===========================================================================
# 4. S-Parameter Comparison Plot
# ===========================================================================
print('Plotting S-parameter comparison...')

freq_ghz = freq / 1e9

fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
fig.suptitle('Optimised S11 vs MED Reference', fontsize=14, fontweight='bold')

ax.plot(freq_ghz, ref_s11, 'k-', linewidth=2.0, label='MED.s1p (Reference)')
ax.plot(freq_ghz, S11_opt_dB, 'r-', linewidth=1.5, label='Optimised S11')
ax.axhline(y=-10, color='gray', linestyle=':', linewidth=1, label='−10 dB threshold')

# Highlight target band [1.8 – 2.8] GHz
ax.axvspan(1.8, 2.8, color='lightgreen', alpha=0.12)

ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('S11 (dB)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=-40)

plt.tight_layout()

# Save figure
eval_dir = os.path.join(OUTPUT, 'MultiportResult')
os.makedirs(eval_dir, exist_ok=True)
s11_plot_path = os.path.join(eval_dir, 'S11_comparison.png')
fig.savefig(s11_plot_path, dpi=150, bbox_inches='tight')
print(f'  S11 comparison saved to: {s11_plot_path}')

# ===========================================================================
# 5. Print summary statistics
# ===========================================================================
print()
print('=' * 60)
print('S11 Summary (Target Band: 1.8 – 2.8 GHz)')
print('-' * 60)

band_mask = (freq >= 1.8e9) & (freq <= 2.8e9)
opt_band = S11_opt_dB[band_mask]
ref_band = ref_s11[band_mask]

print(f'  Optimised S11 — Max: {np.max(opt_band):.2f} dB,  '
      f'Min: {np.min(opt_band):.2f} dB,  '
      f'Mean: {np.mean(opt_band):.2f} dB')
print(f'  MED Ref S11  — Max: {np.max(ref_band):.2f} dB,  '
      f'Min: {np.min(ref_band):.2f} dB,  '
      f'Mean: {np.mean(ref_band):.2f} dB')

# Count violations (where S11 > -10 dB)
opt_violations = np.sum(opt_band > -10)
ref_violations = np.sum(ref_band > -10)
print(f'  Points above −10 dB — Optimised: {opt_violations},  '
      f'Reference: {ref_violations}  (of {len(opt_band)} total)')

# ===========================================================================
# 6. 3D Model Comparison (VTK + PyQt5)
# ===========================================================================
print()
print('Launching 3D model viewer (before vs after)...')

# Lazy imports — only imported if VTK/PyQt6 are available
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                                 QHBoxLayout, QVBoxLayout, QLabel)
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError as e:
    print(f'  [SKIP] 3D viewer requires PyQt6 + VTK: {e}')
    print(f'  S11 comparison plot is still available at: {s11_plot_path}')
    plt.show()   # still show the S11 plot
    sys.exit(0)


class STLViewer(QWidget):
    """Single STL viewer panel — adapted from g_SeeModel.py."""

    def __init__(self, split_ratio=0.2):
        super().__init__()
        self.split_ratio = split_ratio
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # White background
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.renderer.SetGradientBackground(False)

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.interactor_style)

        self._setup_lights()
        layout.addWidget(self.vtk_widget)

    def _setup_lights(self):
        self.renderer.RemoveAllLights()

        ambient = vtk.vtkLight()
        ambient.SetLightTypeToHeadlight()
        ambient.SetIntensity(0.3)
        ambient.SetColor(1, 1, 1)
        self.renderer.AddLight(ambient)

        light1 = vtk.vtkLight()
        light1.SetPosition(-45, 45, 60)
        light1.SetFocalPoint(0, 0, 0)
        light1.SetIntensity(0.7)
        light1.SetColor(1.0, 0.95, 0.85)
        self.renderer.AddLight(light1)

        light2 = vtk.vtkLight()
        light2.SetPosition(-0.8, 0.0, 0)
        light2.SetFocalPoint(0, 0, 0)
        light2.SetIntensity(0.5)
        light2.SetColor(0.8, 0.88, 1.0)
        self.renderer.AddLight(light2)

    def load_stl(self, stl_file_path):
        """Load STL, split into bottom/base and top/patch with colour coding."""
        self.renderer.RemoveAllViewProps()
        self._setup_lights()

        if not os.path.exists(stl_file_path):
            return False

        try:
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_file_path)
            reader.Update()
            poly_data = reader.GetOutput()

            bounds = poly_data.GetBounds()
            zmin, zmax = bounds[4], bounds[5]
            threshold_z = zmin + (zmax - zmin) * self.split_ratio

            plane = vtk.vtkPlane()
            plane.SetOrigin(0, 0, threshold_z)
            plane.SetNormal(0, 0, 1)

            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(poly_data)
            clipper.SetClipFunction(plane)
            clipper.GenerateClippedOutputOn()
            clipper.Update()

            bottom_poly = clipper.GetClippedOutput()
            top_poly = clipper.GetOutput()

            def add_normals(poly):
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(poly)
                normals.SetFeatureAngle(45)
                normals.ComputePointNormalsOn()
                normals.ComputeCellNormalsOn()
                normals.Update()
                return normals.GetOutput()

            bottom_smooth = add_normals(bottom_poly)
            top_smooth = add_normals(top_poly)

            def extract_edges(poly):
                edges = vtk.vtkFeatureEdges()
                edges.SetInputData(poly)
                edges.BoundaryEdgesOn()
                edges.FeatureEdgesOn()
                edges.NonManifoldEdgesOn()
                edges.ManifoldEdgesOff()
                edges.SetFeatureAngle(30)
                edges.Update()
                return edges.GetOutput()

            bottom_edges = extract_edges(bottom_smooth)
            top_edges = extract_edges(top_smooth)

            def create_actor(poly, color, specular_power=128, specular_intensity=0.8):
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().SetAmbient(0.2)
                actor.GetProperty().SetDiffuse(0.7)
                actor.GetProperty().SetSpecular(specular_intensity)
                actor.GetProperty().SetSpecularPower(specular_power)
                actor.GetProperty().SetInterpolationToPhong()
                return actor

            def create_edge_actor(edges, line_width=1.5):
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(edges)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.5, 0.5, 0.5)
                actor.GetProperty().SetLineWidth(line_width)
                return actor

            # Bottom: dark yellow metal  |  Top: light yellow
            actor_bottom = create_actor(bottom_smooth, color=(0.5, 0.5, 0.3),
                                        specular_power=96, specular_intensity=0.7)
            actor_top = create_actor(top_smooth, color=(1.0, 1.0, 0.5),
                                     specular_power=128, specular_intensity=0.6)

            edge_bottom = create_edge_actor(bottom_edges)
            edge_top = create_edge_actor(top_edges)

            self.renderer.AddActor(actor_bottom)
            self.renderer.AddActor(actor_top)
            self.renderer.AddActor(edge_bottom)
            self.renderer.AddActor(edge_top)

            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            return True

        except Exception as e:
            print(f'  STL load error: {e}')
            return False


class MainWindow(QMainWindow):
    """Side-by-side comparison: original (before) vs optimised (after)."""

    def __init__(self, stl_before, stl_after):
        super().__init__()
        self.setWindowTitle('Antenna Model Comparison — Before vs After Optimisation')
        self.resize(1200, 600)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Side-by-side viewers
        viewers_layout = QHBoxLayout()

        # Left: Before
        left_container = QVBoxLayout()
        lbl_before = QLabel('A_Original.stl  (Before Optimisation)')
        lbl_before.setStyleSheet('font-weight: bold; font-size: 12pt;')
        left_container.addWidget(lbl_before)
        self.viewer_before = STLViewer(split_ratio=0.2)
        left_container.addWidget(self.viewer_before)
        viewers_layout.addLayout(left_container)

        # Right: After
        right_container = QVBoxLayout()
        lbl_after = QLabel('D_Final.stl  (After Optimisation)')
        lbl_after.setStyleSheet('font-weight: bold; font-size: 12pt;')
        right_container.addWidget(lbl_after)
        self.viewer_after = STLViewer(split_ratio=0.2)
        right_container.addWidget(self.viewer_after)
        viewers_layout.addLayout(right_container)

        main_layout.addLayout(viewers_layout)

        # Defer STL loading until after the window is shown —
        # otherwise both VTK widgets fight over the OpenGL context.
        self.stl_before = stl_before
        self.stl_after = stl_after

    def showEvent(self, event):
        """Load STL models once the window is visible and OpenGL is ready."""
        super().showEvent(event)
        if not hasattr(self, '_models_loaded'):
            self._models_loaded = True
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self._load_models)

    def _load_models(self):
        if os.path.exists(self.stl_before):
            self.viewer_before.load_stl(self.stl_before)
        else:
            print(f'  [WARN] Before model not found: {self.stl_before}')

        if os.path.exists(self.stl_after):
            self.viewer_after.load_stl(self.stl_after)
        else:
            print(f'  [WARN] After model not found: {self.stl_after}')


# ===========================================================================
# 7. Launch
# ===========================================================================
if __name__ == '__main__':
    # Show S11 plot (non-blocking)
    plt.show(block=False)

    # Launch 3D viewer
    app = QApplication(sys.argv)
    window = MainWindow(stl_before_path, stl_after_path)
    window.show()
    sys.exit(app.exec())
