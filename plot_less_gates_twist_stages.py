"""
Same overlay as plot_optimization_stages.py (map + instant radius vs distance),
but for the trouble section of the gate-reduced 2019 track (ref_less_gates_twist.csv):
initial guess vs. finished windowed result. No global layer -- that optimizer
hasn't been run on this track yet.
"""
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import spline_optimize_5 as so  # generate_spline, generate_path_t, path_length_fun, initial_guess

GATE_FILE = "ref_less_gates_twist.csv"
WINDOWED_DIR = "out_less_gates_twist_windowed_test1"

# Point spline_optimize_5's gate globals at this track instead of its own hardcoded FILENAME.
gate_file_path = os.path.join(so.current_path, GATE_FILE)
so.imported_gate_points = so.load_gates_2D(gate_file_path)
so.gate_ct = len(so.imported_gate_points[:, 0])
so.FILENAME = GATE_FILE

# Pick the latest windowed result by filename (timestamps sort lexicographically).
windowed_files = sorted(glob.glob(os.path.join(WINDOWED_DIR, "opt_dofs_*.npy")))
if not windowed_files:
    raise FileNotFoundError(f"No windowed result found in {WINDOWED_DIR}/ yet.")
WINDOWED_DOF_FILE = windowed_files[-1]


def path_xy(dofs):
    x_s, y_s = so.generate_spline(so.imported_gate_points, dofs)
    t = np.linspace(0, so.gate_ct - 1, so.OBJ_PLOT_POINTS_PER_GATE * so.gate_ct)
    return so.generate_path_t(x_s, y_s, t)


def radius_vs_distance(dofs):
    # Mirrors spline_optimize_5.plot_radius_over_distance().
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


layers = [
    ("initial guess", so.initial_guess(so.imported_gate_points), "tab:blue"),
    ("windowed", np.load(WINDOWED_DOF_FILE), "tab:orange"),
]

fig, (ax, rx) = plt.subplots(2, 1, figsize=(19.2, 14))

# Map plot
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
ax.set_title("Map: path comparison")

# Radius vs distance plot -- same color per layer as the map plot above.
for label, dofs, color in layers:
    distances, radius_array = radius_vs_distance(dofs)
    rx.plot(distances, radius_array, '-', color=color, linewidth=1, label=label)
rx.axhline(so.MINIMUM_RADIUS, color='k', linestyle='--', label=f'minimum radius = {so.MINIMUM_RADIUS} m')
rx.set_yscale('log')
rx.set_xlabel('cumulative distance (m)')
rx.set_ylabel('instant radius (m)')
rx.set_title('Instant radius vs distance')
rx.legend()

fig.suptitle(f"{GATE_FILE}: initial guess vs windowed")

out_path = "outputs/less_gates_twist_stage_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

plt.show()
