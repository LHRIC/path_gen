"""
Shared helpers for the track_pipeline scripts (1_plot_initial_guess.py,
2_run_pipeline.py, 3_show_result.py). Everything here is a plain function
that takes gate_points/dofs as arguments -- nothing relies on spline_optimize_5's
module-level globals (so.imported_gate_points, so.gate_ct), so windowed and
global results can be handled side by side without one overwriting the other's
state.

Result bundle format: a saved result is ONE .npz with the gate_file name, the
gate_points array, AND the dofs array together -- so a saved result can never
get separated from the gate file it belongs to (the old opt_dofs_*.npy files
were bare arrays; you had to remember/hardcode which GATE_FILE each one came
from, and nothing checked it for you).
"""
import os
import sys

import numpy as np

# spline_optimize_5 (imported as `so` below) figures out its own directory from
# sys.argv[0] to find its default gate CSV. Since these scripts live one folder
# down from path_gen, patch sys.argv[0] before importing so `so.current_path`
# still resolves to the path_gen root (same trick the other track_pipeline
# scripts rely on via this shared import).
PATH_GEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PATH_GEN_DIR not in sys.path:
    sys.path.insert(0, PATH_GEN_DIR)
# spline_optimize_5 also loads its warm-start GUESS_DOF_FILE with a bare
# relative np.load() (relative to the process CWD, not so.current_path), so
# the shim isn't complete without also chdir'ing -- every existing sibling
# script gets this for free by always being run from path_gen root.
os.chdir(PATH_GEN_DIR)
_real_argv0 = sys.argv[0]
sys.argv[0] = os.path.join(PATH_GEN_DIR, "_track_pipeline_shim.py")
import spline_optimize_5 as so  # noqa: E402
sys.argv[0] = _real_argv0


def load_gate_file(gate_file):
    """Load a gate CSV given a path relative to the path_gen root (e.g.
    'endurance_tracks/Michigan-2019-endurance_gates.csv'). Returns the
    (N,5) [gate_index, x1, y1, x2, y2] array."""
    return so.load_gates_2D(os.path.join(PATH_GEN_DIR, gate_file))


# ----- Result save/load -----

def save_result(path, gate_file, gate_points, dofs):
    """Save one optimizer result: which gate file it's for, the gate points
    themselves, and the flat DOF array, all in one .npz."""
    np.savez(path, gate_file=np.array(gate_file), gate_points=gate_points, dofs=dofs)


def load_result(path):
    """Load a result saved by save_result. Returns (gate_file, gate_points, dofs)."""
    with np.load(path) as data:
        gate_file = str(data["gate_file"])
        gate_points = data["gate_points"]
        dofs = data["dofs"]
    return gate_file, gate_points, dofs


# ----- Path / radius sampling (generate_spline etc. take gate_points/dofs as
# plain arguments, so these don't need any global state) -----

def path_xy(gate_points, dofs):
    gate_ct = len(gate_points)
    x_s, y_s = so.generate_spline(gate_points, dofs)
    t = np.linspace(0, gate_ct - 1, so.OBJ_PLOT_POINTS_PER_GATE * gate_ct)
    return so.generate_path_t(x_s, y_s, t)


def radius_vs_distance(gate_points, dofs):
    gate_ct = len(gate_points)
    x_s, y_s = so.generate_spline(gate_points, dofs)
    total_length, distance_t = so.path_length_fun(x_s, y_s, gate_ct, return_path=True)
    distance_t_full = np.concat(([0], distance_t))
    t_uniform = np.linspace(0, gate_ct - 1, len(distance_t_full))

    radius_ts = np.linspace(0, gate_ct - 1, gate_ct * 100)
    x_dot, y_dot = x_s.derivative(1), y_s.derivative(1)
    x_dot2, y_dot2 = x_s.derivative(2), y_s.derivative(2)
    rho = (np.abs(x_dot(radius_ts) * y_dot2(radius_ts) - y_dot(radius_ts) * x_dot2(radius_ts))
           / (x_dot(radius_ts) ** 2 + y_dot(radius_ts) ** 2) ** 1.5)
    radius_array = 1 / rho
    radius_distances = np.interp(radius_ts, t_uniform, distance_t_full)
    return radius_distances, radius_array


def plot_result(gate_points, dofs, title, guess_dofs=None):
    """Map + instant-radius-vs-distance, same layout as the comparison plots
    used earlier this session. If guess_dofs is given (e.g. the raw geometric
    initial guess), it's overlaid as a thin dashed grey line on both subplots
    so you can see how far the optimizer moved it. Returns the figure; caller
    decides whether to plt.show() it or savefig it."""
    import matplotlib.pyplot as plt

    fig, (ax, rx) = plt.subplots(2, 1, figsize=(19.2, 14))

    for gate in gate_points:
        x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
    for gate in gate_points:
        if int(gate[0]) % 10 != 0:
            continue
        x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
        bot_x = x1 if y1 < y2 else x2
        bot_y = min(y1, y2)
        ax.annotate(int(gate[0]), xy=(bot_x, bot_y), ha='center', va='top',
                    xytext=(0, -4), textcoords='offset points')
    if guess_dofs is not None:
        gx, gy = path_xy(gate_points, guess_dofs)
        ax.plot(gx, gy, '--', color='grey', linewidth=1, alpha=0.7, label='initial guess')
    x, y = path_xy(gate_points, dofs)
    ax.plot(x, y, 'r-', linewidth=1.5, label='result')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title("Map")
    if guess_dofs is not None:
        ax.legend()

    if guess_dofs is not None:
        g_distances, g_radius_array = radius_vs_distance(gate_points, guess_dofs)
        rx.plot(g_distances, g_radius_array, '--', color='grey', linewidth=1, alpha=0.7, label='initial guess')
    distances, radius_array = radius_vs_distance(gate_points, dofs)
    rx.plot(distances, radius_array, 'r-', linewidth=1, label='result')
    rx.axhline(so.MINIMUM_RADIUS, color='k', linestyle='--',
               label=f'minimum radius = {so.MINIMUM_RADIUS} m')
    rx.set_yscale('log')
    rx.set_xlabel('cumulative distance (m)')
    rx.set_ylabel('instant radius (m)')
    rx.set_title('Instant radius vs distance')
    rx.legend()

    fig.suptitle(title)
    return fig


# ----- CSV export -----
# End-of-track handling (no closed loop, so the real path just stops instead
# of wrapping): normal evenly-spaced sampling stops at the last point that's
# a whole number of `spacing` steps into the path -- call that point1, the
# "last untruncated point". There's usually a leftover bit of real path
# after point1, shorter than `spacing`, ending at point2 (the final gate's
# crossing point). Instead of dropping that leftover, keep placing points
# every `spacing` along the straight line through point1 and point2,
# continuing past point2, until we're at least 1m past point1 -- so the
# track always slightly overruns the real finish rather than stopping short,
# and every segment (real or extended) stays exactly `spacing` apart.
def export_csv(gate_points, dofs, spacing, out_path):
    gate_ct = len(gate_points)
    x_s, y_s = so.generate_spline(gate_points, dofs)
    total_length, distance_t = so.path_length_fun(x_s, y_s, gate_ct, return_path=True)
    distance_t_full = np.concat(([0], distance_t))
    t_uniform = np.linspace(0, gate_ct - 1, len(distance_t_full))

    n_real = int(total_length // spacing) + 1  # points fully within the real path
    sample_distances = np.arange(n_real) * spacing
    sample_t = np.interp(sample_distances, distance_t_full, t_uniform)
    x_pts, y_pts = so.generate_path_t(x_s, y_s, sample_t)

    point1 = np.array([x_pts[-1], y_pts[-1]])
    point2 = np.array([float(x_s(gate_ct - 1)), float(y_s(gate_ct - 1))])
    direction = point2 - point1
    direction_len = np.linalg.norm(direction)
    if direction_len > 1e-9:
        direction = direction / direction_len
        n_extra = 1
        while n_extra * spacing < 1.0:  # keep going until >= 1m past point1
            n_extra += 1
        steps = np.arange(1, n_extra + 1)
        ext_pts = point1 + direction * (spacing * steps[:, None])
        sample_distances = np.concatenate([sample_distances, sample_distances[-1] + spacing * steps])
        x_pts = np.concatenate([x_pts, ext_pts[:, 0]])
        y_pts = np.concatenate([y_pts, ext_pts[:, 1]])

    n_points = len(sample_distances)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    header = "distance_m,x_m,y_m"
    data = np.column_stack([sample_distances, x_pts, y_pts])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.4f")
    print(f"Saved CSV ({n_points} points, spacing={spacing} m): {out_path}")
    return out_path
