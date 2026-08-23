"""
Compare all 4 full-window-reoptimization sweep configs on the trouble section
(ref_less_gates_twist.csv): same map + instant-radius-vs-distance layout as
the other stage/comparison plots.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import spline_optimize_5 as so  # generate_spline, generate_path_t, path_length_fun

GATE_FILE = "ref_less_gates_twist.csv"

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
    ("(a) w=10/s=9  500.2s  obj=1.9306", "out_less_gates_twist_w10s9/opt_dofs_2026-08-23_01-33-26.npy", "tab:blue"),
    ("(b) w=6/s=5   229.6s  obj=1.9279", "out_less_gates_twist_w6s5/opt_dofs_2026-08-23_01-37-44.npy", "tab:orange"),
    ("(c) w=8/s=4   496.4s  obj=1.9363", "out_less_gates_twist_w8s4/opt_dofs_2026-08-23_01-46-17.npy", "tab:green"),
    ("(d) w=6/s=3   250.4s  obj=1.9391", "out_less_gates_twist_w6s3/opt_dofs_2026-08-23_01-50-43.npy", "tab:red"),
]
layers = [(label, np.load(path), color) for label, path, color in layers]

# Global optimizer result (warm-started from the (d) windowed run) -- drawn on
# top in black, thin, so it stands out against the desaturated windowed runs.
global_label = "GLOBAL (warm start: d)"
global_dofs = np.load("out_less_gates_twist_global_test1/opt_dofs_2026-08-23_02-14-50.npy")

fig, (ax, rx) = plt.subplots(2, 1, figsize=(19.2, 14))

for gate in so.imported_gate_points:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
for label, dofs, color in layers:
    x, y = path_xy(dofs)
    ax.plot(x, y, '-', color=color, linewidth=2.0, alpha=0.4, label=label)
gx, gy = path_xy(global_dofs)
ax.plot(gx, gy, '-', color='black', linewidth=0.8, label=global_label, zorder=10)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend()
ax.set_title("Map: window-experiment comparison")

for label, dofs, color in layers:
    distances, radius_array = radius_vs_distance(dofs)
    rx.plot(distances, radius_array, '-', color=color, linewidth=1.5, alpha=0.4, label=label)
g_distances, g_radius_array = radius_vs_distance(global_dofs)
rx.plot(g_distances, g_radius_array, '-', color='black', linewidth=0.8, label=global_label, zorder=10)
rx.axhline(so.MINIMUM_RADIUS, color='k', linestyle='--', label=f'minimum radius = {so.MINIMUM_RADIUS} m')
rx.set_yscale('log')
rx.set_xlabel('cumulative distance (m)')
rx.set_ylabel('instant radius (m)')
rx.set_title('Instant radius vs distance')
rx.legend()

fig.suptitle(f"{GATE_FILE}: full-window re-optimization, 4 window/step configs + global")

out_path = "outputs/window_experiments_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

plt.show()
