"""
Global optimization (spline_optimize_5's own unconstrained D1/D2 bounds, same
settings as the successful global test-section run), warm-started from a
saved windowed result. This is the expensive, possibly-multi-hour step for a
~140-gate track. Runs unattended, no plt.show().

Usage: set WINDOWED_RESULT_PATH below to the .npz that 2_run_windowed.py
printed, then run this script.
"""
import os
import time
from datetime import datetime

import numpy as np
from scipy.optimize import Bounds, minimize

import pipeline_lib as pl

so = pl.so

WINDOWED_RESULT_PATH = "track_pipeline/outputs/windowed_CHANGE_ME.npz"  # set this to 2_run_windowed.py's output
OUTPUT_DIR = "track_pipeline/outputs"
CSV_SPACING = 0.5  # meters -- change and re-run 5_export_csv.py to regenerate without re-optimizing

os.makedirs(OUTPUT_DIR, exist_ok=True)

gate_file, gate_points, windowed_dofs = pl.load_result(WINDOWED_RESULT_PATH)
gate_ct = len(gate_points)
print(f"Loaded windowed result: {WINDOWED_RESULT_PATH} (gate file: {gate_file}, {gate_ct} gates)")

so.OBJ_FUN_INTEGRATION_EPSABS = 1e-2
so.CONE_SPACING = 0.30  # meters -- clearance to cones, was 0.2 (must match whatever 2_run_windowed.py used)
so.TRACK_WIDTH = 1.22  # meters, was 1.25 (must match whatever 2_run_windowed.py used)
so.imported_gate_points = gate_points
so.gate_ct = gate_ct

guess_dofs = so.initial_guess(gate_points)  # for the "how far from scratch" overlay on the final plot

import math
gate_position_dof_lb = np.zeros(gate_ct)
gate_position_dof_ub = np.zeros(gate_ct)
for i in range(gate_ct):
    num, x1, y1, x2, y2 = gate_points[i, :]
    gate_length = math.dist([x1, y1], [x2, y2])
    normalized_offset = (so.TRACK_WIDTH / 2 + so.CONE_SPACING) / gate_length
    gate_position_dof_lb[i] = normalized_offset
    gate_position_dof_ub[i] = 1 - normalized_offset

d1_dof_lb = [-80] * gate_ct * 2
d1_dof_ub = [80] * gate_ct * 2
d2_dof_lb = [-20] * gate_ct * 2
d2_dof_ub = [20] * gate_ct * 2
lb = np.concatenate([gate_position_dof_lb, d1_dof_lb, d2_dof_lb])
ub = np.concatenate([gate_position_dof_ub, d1_dof_ub, d2_dof_ub])

print("\n===== Global optimization (warm-started from windowed result) =====")
global_start = time.perf_counter()
res = minimize(so.raw_obj_fun, windowed_dofs, method=so.OPT_METHOD, bounds=Bounds(lb, ub),
                options={'maxiter': so.OPT_MAXITER, 'ftol': so.OPT_FTOL,
                         'maxfun': so.OPT_MAXFUN, 'disp': True})
global_runtime = time.perf_counter() - global_start
print(res)
print(f"Global optimization runtime: {global_runtime:.1f}s")

global_dofs = res.x
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
global_result_path = os.path.join(OUTPUT_DIR, f"global_{stamp}.npz")
pl.save_result(global_result_path, gate_file, gate_points, global_dofs)
fig = pl.plot_result(gate_points, global_dofs, f"Global (warm start: windowed), {global_runtime:.0f}s",
                      guess_dofs=guess_dofs)
fig.savefig(os.path.join(OUTPUT_DIR, f"global_{stamp}.png"), dpi=150)
pl.export_csv(gate_points, global_dofs, CSV_SPACING, os.path.join(OUTPUT_DIR, f"global_{stamp}.csv"))
print(f"\nGlobal result saved: {global_result_path}")
print(f"To view interactively: python track_pipeline\\4_show_result.py {global_result_path}")
