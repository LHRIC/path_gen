"""
Compare three results on the trouble section (ref_less_gates_twist.csv):
  - windowed optimizer, window=8/overlap=4
  - windowed optimizer, window=6/overlap=3
  - global optimizer (spline_optimize_5.py), warm-started from the window=6/overlap=3 result
Same map + instant-radius-vs-distance layout as plot_window_size_comparison.py, extended to 3 layers.

Shows the plots on screen (plt.show()) instead of saving PNGs. Run it, look, close the window when done.
"""
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import spline_optimize_5 as so  # generate_spline, generate_path_t, path_length_fun

GATE_FILE = "ref_less_gates_twist.csv"
DOF_FILE_8_4 = "out_less_gates_twist_windowed_test1/opt_dofs_2026-08-23_01-05-14.npy"
DOF_FILE_6_3 = "out_less_gates_twist_windowed_test1/opt_dofs_2026-08-23_01-07-13.npy"
GLOBAL_OUTPUT_DIR = "out_less_gates_twist_global_test1"

gate_file_path = os.path.join(so.current_path, GATE_FILE)
so.imported_gate_points = so.load_gates_2D(gate_file_path)
so.gate_ct = len(so.imported_gate_points[:, 0])
so.FILENAME = GATE_FILE


def find_global_dof_file():
    """Pick the newest opt_dofs_*.npy written by the global-optimizer run into
    GLOBAL_OUTPUT_DIR. Run spline_optimize_5.py (with the config already pointed
    at ref_less_gates_twist.csv / out_less_gates_twist_global_test1) before this."""
    pattern = os.path.join(GLOBAL_OUTPUT_DIR, "opt_dofs_*.npy")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No opt_dofs_*.npy found in {GLOBAL_OUTPUT_DIR}/. "
            f"Run spline_optimize_5.py first to produce the global-optimizer result."
        )
    return candidates[-1]  # timestamped filenames sort chronologically


def path_xy(dofs):
    x_s, y_s = so.generate_spline(so.imported_gate_points, dofs)
    t = np.linspace(0, so.gate_ct - 1, so.OBJ_PLOT_POINTS_PER_GATE * so.gate_ct)
    return so.generate_path_t(x_s, y_s, t)


def radius_vs_distance(dofs):
    x_s, y_s = so.generate_spline(so.imported_gate_points, dofs)
    total_length, distance_t = so.path_length_fun(x_s, y_s, so.gate_ct, return_path=True)
    distance_t_full = np.concat(([0], distance_t))
    t_uniform = np.linspace(0, so.gate_ct - 1, len(distance_t_full))

    radius_ts = np.linspace(0, so.gate_ct - 1, so.gate_ct * 100)
    x_dot, y_dot = x_s.derivative(1), y_s.derivative(1)
    x_dot2, y_dot2 = x_s.derivative(2), y_s.derivative(2)
    rho = (np.abs(x_dot(radius_ts) * y_dot2(radius_ts) - y_dot(radius_ts) * x_dot2(radius_ts))
           / (x_dot(radius_ts)**2 + y_dot(radius_ts)**2)**1.5)
    radius_array = 1 / rho
    radius_distances = np.interp(radius_ts, t_uniform, distance_t_full)
    return radius_distances, radius_array


global_dof_file = find_global_dof_file()
print(f"Using global-optimizer result: {global_dof_file}")

layers = [
    ("window=8 / overlap=4  (66.8s)", np.load(DOF_FILE_8_4), "tab:blue"),
    ("window=6 / overlap=3  (35.8s)", np.load(DOF_FILE_6_3), "tab:orange"),
    (f"global (warm-started from 6/3)", np.load(global_dof_file), "tab:green"),
]

fig, (ax, rx) = plt.subplots(2, 1, figsize=(19.2, 14))

for gate in so.imported_gate_points:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
for label, dofs, color in layers:
    x, y = path_xy(dofs)
    ax.plot(x, y, '-', color=color, linewidth=1.5, label=label)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend()
ax.set_title("Map: windowed vs global comparison")

for label, dofs, color in layers:
    distances, radius_array = radius_vs_distance(dofs)
    rx.plot(distances, radius_array, '-', color=color, linewidth=1, label=label)
rx.axhline(so.MINIMUM_RADIUS, color='k', linestyle='--', label=f'minimum radius = {so.MINIMUM_RADIUS} m')
rx.set_yscale('log')
rx.set_xlabel('cumulative distance (m)')
rx.set_ylabel('instant radius (m)')
rx.set_title('Instant radius vs distance')
rx.legend()

fig.suptitle(f"{GATE_FILE}: window=8/overlap=4 vs window=6/overlap=3 vs global")

plt.show()
