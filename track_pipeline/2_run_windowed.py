"""
Windowed sweep only (window=6, step=5 -- the fast/good config from this
session's window-size experiments on the trouble section: 229.6s,
obj=1.9279, best of the 4 configs tried). Runs unattended, no plt.show().

Saves a result (.npz), a map/radius plot (.png, with the raw geometric
initial guess overlaid so you can see what the optimizer actually changed),
and a CSV. This is the fast, reviewable checkpoint -- run 3_run_global.py
afterward, once you've looked at this, to attempt the (much slower) global
refine on top of it.

get_gate_dofs/set_window_dofs/solve_window copied from
spline_optimize_windowed.py, parametrized on gate_points/gate_ct (function
arguments) instead of reading so.imported_gate_points/so.gate_ct as globals.
"""
import os
import time
import math
from datetime import datetime

import numpy as np
from scipy.optimize import Bounds, minimize

import pipeline_lib as pl

so = pl.so

GATE_FILE = "endurance_tracks/Michigan-2019-endurance_gates.csv"
GATE_FILE = "autox_tracks/Michigan-2026-autox_gates.csv"
OUTPUT_DIR = "track_pipeline/outputs"
CSV_SPACING = 0.5  # meters -- change and re-run 5_export_csv.py to regenerate without re-optimizing

WINDOW_SIZE = 6
WINDOW_STEP = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

gate_points = pl.load_gate_file(GATE_FILE)
gate_ct = len(gate_points)
print(f"Loaded {GATE_FILE}: {gate_ct} gates")

so.OBJ_FUN_INTEGRATION_EPSABS = 1e-2  # documented setting for autoX/endurance-size tracks
so.CONE_SPACING = 0.30  # meters -- clearance to cones, was 0.2
so.TRACK_WIDTH = 1.22  # meters, was 1.25

# Gate-position DOF bounds (trackwidth/clearance based), computed for this
# track's real gate lengths -- same formula spline_optimize_5/_windowed use.
gate_position_dof_lb = np.zeros(gate_ct)
gate_position_dof_ub = np.zeros(gate_ct)
for i in range(gate_ct):
    num, x1, y1, x2, y2 = gate_points[i, :]
    gate_length = math.dist([x1, y1], [x2, y2])
    normalized_offset = (so.TRACK_WIDTH / 2 + so.CONE_SPACING) / gate_length
    gate_position_dof_lb[i] = normalized_offset
    gate_position_dof_ub[i] = 1 - normalized_offset


def get_gate_dofs(dofs, i):
    """[position, D1x, D1y, D2x, D2y] for global gate index i."""
    return np.array([dofs[i], dofs[gate_ct + i], dofs[gate_ct * 2 + i],
                      dofs[gate_ct * 3 + i], dofs[gate_ct * 4 + i]])


def set_window_dofs(dofs, start, window_len, local_dofs):
    dofs[start:start + window_len] = local_dofs[0:window_len]
    dofs[gate_ct + start:gate_ct + start + window_len] = local_dofs[window_len:window_len * 2]
    dofs[gate_ct * 2 + start:gate_ct * 2 + start + window_len] = local_dofs[window_len * 2:window_len * 3]
    dofs[gate_ct * 3 + start:gate_ct * 3 + start + window_len] = local_dofs[window_len * 3:window_len * 4]
    dofs[gate_ct * 4 + start:gate_ct * 4 + start + window_len] = local_dofs[window_len * 4:window_len * 5]


def solve_window(dofs, start, end):
    """Optimize ALL gates in [start, end) together. One gate before `start`
    and one after `end` are read in (frozen, objective-only context)."""
    window_len = end - start
    has_pre = start > 0
    has_post = end < gate_ct
    pre_offset = 1 if has_pre else 0
    obj_len = window_len + pre_offset + (1 if has_post else 0)

    slice_start = start - 1 if has_pre else start
    slice_end = end + 1 if has_post else end
    gate_slice = gate_points[slice_start:slice_end]

    fixed_locals, fixed_globals = [], []
    if has_pre:
        fixed_locals.append(0)
        fixed_globals.append(start - 1)
    if has_post:
        fixed_locals.append(obj_len - 1)
        fixed_globals.append(end)
    fixed_dofs_list = [get_gate_dofs(dofs, g) for g in fixed_globals]

    free_idx = list(range(pre_offset, pre_offset + window_len))
    n_free = window_len

    def assemble(free_vec):
        local = np.zeros(obj_len * 5)
        for local_i, fd in zip(fixed_locals, fixed_dofs_list):
            local[local_i] = fd[0]
            local[obj_len + local_i] = fd[1]
            local[obj_len * 2 + local_i] = fd[2]
            local[obj_len * 3 + local_i] = fd[3]
            local[obj_len * 4 + local_i] = fd[4]
        for k, i in enumerate(free_idx):
            local[i] = free_vec[k]
            local[obj_len + i] = free_vec[n_free + k]
            local[obj_len * 2 + i] = free_vec[2 * n_free + k]
            local[obj_len * 3 + i] = free_vec[3 * n_free + k]
            local[obj_len * 4 + i] = free_vec[4 * n_free + k]
        return local

    def objective(free_vec):
        local_dofs = assemble(free_vec)
        x_s, y_s = so.generate_spline(gate_slice, local_dofs)
        value, _ = so.curvature_obj_evaluation(x_s, y_s, [0, obj_len - 1])
        return value

    global_free_gates = list(range(start, end))
    x0_blocks = [dofs[global_free_gates],
                 dofs[[gate_ct + g for g in global_free_gates]],
                 dofs[[gate_ct * 2 + g for g in global_free_gates]],
                 dofs[[gate_ct * 3 + g for g in global_free_gates]],
                 dofs[[gate_ct * 4 + g for g in global_free_gates]]]
    lb_blocks = [gate_position_dof_lb[global_free_gates],
                 [-80] * n_free, [-80] * n_free, [-20] * n_free, [-20] * n_free]
    ub_blocks = [gate_position_dof_ub[global_free_gates],
                 [80] * n_free, [80] * n_free, [20] * n_free, [20] * n_free]
    x0 = np.concatenate(x0_blocks)
    lb = np.concatenate(lb_blocks)
    ub = np.concatenate(ub_blocks)

    res = minimize(objective, x0, method=so.OPT_METHOD, bounds=Bounds(lb, ub),
                    options={'maxiter': 100, 'ftol': 1e-4, 'maxfun': so.OPT_MAXFUN})

    full_local = assemble(res.x)
    committed = np.concatenate([
        full_local[pre_offset:pre_offset + window_len],
        full_local[obj_len + pre_offset:obj_len + pre_offset + window_len],
        full_local[obj_len * 2 + pre_offset:obj_len * 2 + pre_offset + window_len],
        full_local[obj_len * 3 + pre_offset:obj_len * 3 + pre_offset + window_len],
        full_local[obj_len * 4 + pre_offset:obj_len * 4 + pre_offset + window_len],
    ])
    return committed, res.fun


print(f"\n===== Windowed sweep (window={WINDOW_SIZE}, step={WINDOW_STEP}) =====")
guess_dofs = so.initial_guess(gate_points)
sweep_start = time.perf_counter()
dofs = guess_dofs.copy()
start = 0
window_num = 0
while start < gate_ct - 1:
    end = start + WINDOW_SIZE
    if end >= gate_ct:
        end = gate_ct
        start = max(0, end - WINDOW_SIZE)
    window_num += 1
    t0 = time.perf_counter()
    local_dofs, obj_val = solve_window(dofs, start, end)
    set_window_dofs(dofs, start, end - start, local_dofs)
    print(f"window {window_num}: gates [{start},{end}) obj={obj_val:.6f} "
          f"time={time.perf_counter() - t0:.1f}s", flush=True)
    if end == gate_ct:
        break
    start += WINDOW_STEP
sweep_runtime = time.perf_counter() - sweep_start
print(f"Windowed sweep done: {sweep_runtime:.1f}s over {window_num} windows")

stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
windowed_result_path = os.path.join(OUTPUT_DIR, f"windowed_{stamp}.npz")
pl.save_result(windowed_result_path, GATE_FILE, gate_points, dofs)
fig = pl.plot_result(gate_points, dofs, f"Windowed (w={WINDOW_SIZE}/s={WINDOW_STEP}), {sweep_runtime:.0f}s",
                      guess_dofs=guess_dofs)
fig.savefig(os.path.join(OUTPUT_DIR, f"windowed_{stamp}.png"), dpi=150)
pl.export_csv(gate_points, dofs, CSV_SPACING, os.path.join(OUTPUT_DIR, f"windowed_{stamp}.csv"))
print(f"\nWindowed checkpoint saved: {windowed_result_path}")
print(f"Plot: {os.path.join(OUTPUT_DIR, f'windowed_{stamp}.png')}")
print(f"To view interactively: python track_pipeline\\4_show_result.py {windowed_result_path}")
print(f"To run the global refine on this: edit WINDOWED_RESULT_PATH in 3_run_global.py to this path, then run it.")
