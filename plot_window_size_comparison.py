"""
Compare two windowed-optimizer configs on the trouble section (ref_less_gates_twist.csv):
window=8/overlap=4 vs window=6/overlap=3 (both with the fixed 1-gate-back/1-gate-forward
obj-fun eval range). Same map + instant-radius-vs-distance layout as the other stage plots.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import spline_optimize_5 as so  # generate_spline, generate_path_t, path_length_fun

GATE_FILE = "ref_less_gates_twist.csv"
DOF_FILE_8_4 = "out_less_gates_twist_windowed_test1/opt_dofs_2026-08-23_01-05-14.npy"
DOF_FILE_6_3 = "out_less_gates_twist_windowed_test1/opt_dofs_2026-08-23_01-07-13.npy"

gate_file_path = os.path.join(so.current_path, GATE_FILE)
so.imported_gate_points = so.load_gates_2D(gate_file_path)
so.gate_ct = len(so.imported_gate_points[:, 0])
so.FILENAME = GATE_FILE


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


layers = [
    ("window=8 / overlap=4  (66.8s)", np.load(DOF_FILE_8_4), "tab:blue"),
    ("window=6 / overlap=3  (35.8s)", np.load(DOF_FILE_6_3), "tab:orange"),
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
ax.set_title("Map: window/overlap comparison")

for label, dofs, color in layers:
    distances, radius_array = radius_vs_distance(dofs)
    rx.plot(distances, radius_array, '-', color=color, linewidth=1, label=label)
rx.axhline(so.MINIMUM_RADIUS, color='k', linestyle='--', label=f'minimum radius = {so.MINIMUM_RADIUS} m')
rx.set_yscale('log')
rx.set_xlabel('cumulative distance (m)')
rx.set_ylabel('instant radius (m)')
rx.set_title('Instant radius vs distance')
rx.legend()

fig.suptitle(f"{GATE_FILE}: window=8/overlap=4 vs window=6/overlap=3")

out_path = "outputs/window_size_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

#plt.show()
