"""
Overlay three stages of the enduranceTwist path on one map, plus their instant
radius vs. distance (same computation as spline_optimize_5.plot_radius_over_distance),
so the map shape and cornering radius can be compared side by side.
"""
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import spline_optimize_5 as so  # loads enduranceTwist gates; gives generate_spline/generate_path_t

WINDOWED_DOF_FILE = "out_enduranceTwist_windowed_test1/opt_dofs_2026-08-22_04-05-49.npy"
GLOBAL_DIR = "out_enduranceTwist_global_test1"

# Pick the latest global result by filename (timestamps sort lexicographically).
global_files = sorted(glob.glob(os.path.join(GLOBAL_DIR, "opt_dofs_*.npy")))
GLOBAL_DOF_FILE = global_files[-1] if global_files else None


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


def distance_markers(dofs, spacing=50):
    # x,y position every `spacing` meters of cumulative path distance.
    x_s, y_s = so.generate_spline(so.imported_gate_points, dofs)
    total_length, distance_t = so.path_length_fun(x_s, y_s, so.gate_ct, return_path=True)
    distance_t_full = np.concat(([0], distance_t))
    t_uniform = np.linspace(0, so.gate_ct - 1, len(distance_t_full))
    marker_distances = np.arange(0, total_length, spacing)
    marker_times = np.interp(marker_distances, distance_t_full, t_uniform)
    mx, my = so.generate_path_t(x_s, y_s, marker_times)
    return marker_distances, mx, my


layers = [
    ("initial guess", so.initial_guess(so.imported_gate_points), "tab:blue"),
    ("windowed (global warm start)", np.load(WINDOWED_DOF_FILE), "tab:orange"),
]
if GLOBAL_DOF_FILE:
    layers.append(("global optimum", np.load(GLOBAL_DOF_FILE), "tab:red"))
else:
    print(f"No global result found in {GLOBAL_DIR}/; plotting without it.")

fig, (ax, rx) = plt.subplots(2, 1, figsize=(19.2, 14))

# Map plot
for gate in so.imported_gate_points:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
for label, dofs, color in layers:
    x, y = path_xy(dofs)
    ax.plot(x, y, '-', color=color, linewidth=1.5, label=label)

# Distance markers every 50 m, placed along the final (last) layer's path.
marker_distances, mx, my = distance_markers(layers[-1][1], spacing=50)
ax.scatter(mx, my, color='black', zorder=5)
for d, x, y in zip(marker_distances, mx, my):
    ax.annotate(f"{d:.0f}m", (x, y), textcoords="offset points", xytext=(0, 6), ha='center')

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

fig.suptitle(f"{so.FILENAME}: initial guess vs windowed vs global optimum")

#plt.show()

out_path = "outputs/stage_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
