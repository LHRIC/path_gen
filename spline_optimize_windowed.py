"""
Windowed / local curvature optimization for large tracks (e.g. endurance).

spline_optimize_5.py optimizes ALL gate DOFs (position, D1, D2) at once across
the whole track. For endurance (~200 gates -> ~1000 DOFs) that takes hours,
because every objective-function evaluation rebuilds a spline over the whole
track, and every gradient estimate needs ~1000 extra evaluations of that.

This script instead sweeps a small window of WINDOW_SIZE gates along the
track. Each window optimizes ALL of its own gates -- nothing inside the
window is frozen. This works because the spline here (BPoly.from_derivatives
with position+D1+D2 given at every gate) only needs the two DOFs bracketing a
segment to build that segment's shape -- a segment never depends on gates
further away. So the full-track objective is exactly the sum of each
segment's own local integral, and solving it window by window is a fast,
close approximation of solving everything at once.

Window layout (gates [start, end), window_len = end - start):
  - Every gate in [start, end) is FREE (being optimized), every time a window
    touches it -- including gates an earlier, overlapping window already
    solved. WINDOW_STEP < WINDOW_SIZE means consecutive windows overlap and
    the shared gates get re-optimized with better information (what's ahead
    of them), instead of being permanently frozen after one pass. Set
    WINDOW_STEP = WINDOW_SIZE for the old non-overlapping behavior.
  - One gate before `start` and one gate after `end` are also read in
    (frozen, not yet re-solved) and included in the objective only -- not as
    free variables -- so the objective sees the segments the window's first
    and last free gates actually connect to. Neither exists at the very
    first/last window, since there's nothing there.
  - The last window of the sweep is pulled back (its start shifted earlier)
    so it's still exactly WINDOW_SIZE gates, rather than being left short
    when (gate_ct - start) doesn't divide evenly by WINDOW_STEP. This means
    the final step's overlap with the previous window can be larger than
    WINDOW_STEP.

Runs unattended: no plt.show() windows. Results are saved straight to a PNG
map plot and an opt_dofs_*.npy (same flat format spline_optimize_5 writes, so
unpack_dofs.py can load and re-plot it later) in OUTPUT_DIR.
"""

import os
import math
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, minimize

import spline_optimize_5 as so  # reuses generate_spline, curvature_obj_evaluation, bounds machinery

# ----- Track used for this run (edit to switch tracks) -----
#GATE_FILE = "ref_gates_autoX.csv"
#GATE_FILE = "ref_gates_endurance.csv"
#GATE_FILE = "ref_gates_enduranceTwist.csv"  # gates 104-170, the loop-prone stretch (re-cut after CAD re-spacing)
#GATE_FILE = "ref_gates_gates.csv"  # 6-gate initial test track
GATE_FILE = "ref_less_gates_twist.csv"  # gates 64-102 of ref_less_gates.csv -- same loop-prone stretch, mapped onto the gate-reduced 2019 track
#OUTPUT_DIR = "out_enduranceTwist_windowed_test1"
#OUTPUT_DIR = "out_endurance_windowed_test1"
#OUTPUT_DIR = "out_autoX_windowed_test1"
#OUTPUT_DIR = "out_gates_windowed_test1"
OUTPUT_DIR = "out_less_gates_twist_w6s3"
# -------------------------------------------------------------

# spline_optimize_5.py loads its own hardcoded FILENAME (endurance) at import.
# Override it here with GATE_FILE, and recompute the guess/bounds arrays that
# depend on gate count -- same pattern unpack_dofs.py uses to swap tracks.
gate_file_path = os.path.join(so.current_path, GATE_FILE)
so.imported_gate_points = so.load_gates_2D(gate_file_path)
so.gate_ct = len(so.imported_gate_points[:, 0])
so.FILENAME = GATE_FILE
so.guess_dofs = so.initial_guess(so.imported_gate_points)

gate_position_dof_lb = np.zeros(so.gate_ct)
gate_position_dof_ub = np.zeros(so.gate_ct)
for i in range(so.gate_ct):
    num, x1, y1, x2, y2 = so.imported_gate_points[i, :]
    gate_length = math.dist([x1, y1], [x2, y2])
    normalized_offset = (so.TRACK_WIDTH / 2 + so.CONE_SPACING) / gate_length
    gate_position_dof_lb[i] = normalized_offset
    gate_position_dof_ub[i] = 1 - normalized_offset
so.gate_position_dof_lb = gate_position_dof_lb
so.gate_position_dof_ub = gate_position_dof_ub

# so.py's own comment says epsabs is normally loosened to 1e-2 for autoX/endurance
# -size tracks (it ships at 1e-6, tuned for small tracks). Confirmed by testing: at
# 1e-6 a single 10-gate window's objective eval took ~2000 quad calls (~0.1s each),
# i.e. minutes per window. 1e-2 is the documented setting for this track size.
so.OBJ_FUN_INTEGRATION_EPSABS = 1e-2

# so.py's ftol (1e-7) chases 6+ digits of precision, which on a small ~50-DOF
# window means dozens of extra finite-difference gradient rounds for no visible
# path change. Loosened for speed -- this is a fast experimental sweep, not the
# final full-track optimizer.
WINDOW_OPT_MAXITER = 100
WINDOW_OPT_FTOL = 1e-4

WINDOW_SIZE = 6
WINDOW_STEP = 3  # how far the window slides each pass; overlap = WINDOW_SIZE - WINDOW_STEP

# Experiment: leave D2 (2nd derivative / curvature) DOFs at their initial guess
# (0, see initial_guess()) instead of letting the optimizer touch them. Position
# and D1 (tangent) stay free either way.
OPTIMIZE_D2 = True

gate_ct = so.gate_ct

# Checkpointing: this run can be killed and re-launched partway through (e.g. to
# restart it in the foreground) without losing progress. Every finished window
# saves the full-track dofs + which gate to resume from here; a matching file
# on disk at startup means "pick up where the last run left off" instead of
# starting the sweep over from the geometric guess.
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.npz")
if os.path.exists(CHECKPOINT_PATH):
    with np.load(CHECKPOINT_PATH) as checkpoint:  # 'with' closes the file handle,
        dofs = checkpoint["dofs"]                 # otherwise Windows won't let the
        resume_start = int(checkpoint["next_start"])  # end-of-sweep os.remove delete it
    print(f"Resuming from checkpoint: next window starts at gate {resume_start}")
else:
    dofs = so.guess_dofs.copy()  # full-track DOF array, filled in window by window
    resume_start = 0


def get_gate_dofs(dofs, i):
    """[position, D1x, D1y, D2x, D2y] for global gate index i."""
    return np.array([dofs[i], dofs[gate_ct + i], dofs[gate_ct * 2 + i],
                      dofs[gate_ct * 3 + i], dofs[gate_ct * 4 + i]])


def set_window_dofs(dofs, start, window_len, local_dofs):
    """Write a solved window's dofs back into the full-track array at global
    gates [start, start+window_len)."""
    dofs[start:start + window_len] = local_dofs[0:window_len]
    dofs[gate_ct + start:gate_ct + start + window_len] = local_dofs[window_len:window_len * 2]
    dofs[gate_ct * 2 + start:gate_ct * 2 + start + window_len] = local_dofs[window_len * 2:window_len * 3]
    dofs[gate_ct * 3 + start:gate_ct * 3 + start + window_len] = local_dofs[window_len * 3:window_len * 4]
    dofs[gate_ct * 4 + start:gate_ct * 4 + start + window_len] = local_dofs[window_len * 4:window_len * 5]


def solve_window(start, end):
    """Optimize ALL gates in [start, end) together -- nothing in the window is
    frozen, even if an earlier, overlapping window already solved some of
    them. One gate before `start` and one gate after `end` are also read in
    (frozen, for the objective only), unless this is the first/last window
    and there's nothing there to read.
    """
    window_len = end - start
    has_pre = start > 0
    has_post = end < gate_ct
    pre_offset = 1 if has_pre else 0
    obj_len = window_len + pre_offset + (1 if has_post else 0)

    slice_start = start - 1 if has_pre else start
    slice_end = end + 1 if has_post else end
    gate_slice = so.imported_gate_points[slice_start:slice_end]

    fixed_locals = []
    fixed_globals = []
    if has_pre:
        fixed_locals.append(0)
        fixed_globals.append(start - 1)
    if has_post:
        fixed_locals.append(obj_len - 1)
        fixed_globals.append(end)
    fixed_dofs_list = [get_gate_dofs(dofs, g) for g in fixed_globals]

    free_idx = list(range(pre_offset, pre_offset + window_len))  # every window gate is free
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
            if OPTIMIZE_D2:
                local[obj_len * 3 + i] = free_vec[3 * n_free + k]
                local[obj_len * 4 + i] = free_vec[4 * n_free + k]
            else:
                # D2 stays at whatever it already is (the initial guess, or a
                # previous window's answer if this gate was touched before).
                fd = get_gate_dofs(dofs, start + (i - pre_offset))
                local[obj_len * 3 + i] = fd[3]
                local[obj_len * 4 + i] = fd[4]
        return local

    def objective(free_vec):
        local_dofs = assemble(free_vec)
        x_s, y_s = so.generate_spline(gate_slice, local_dofs)
        value, _ = so.curvature_obj_evaluation(x_s, y_s, [0, obj_len - 1])
        return value

    # Initial guess and bounds for the free gates, pulled from the same
    # geometric guess / bound arrays spline_optimize_5 already built -- if a
    # gate was solved by an earlier overlapping window, this is that answer,
    # not the original geometric guess. D2 is only included here when it's
    # actually a free DOF (OPTIMIZE_D2).
    global_free_gates = list(range(start, end))
    x0_blocks = [dofs[global_free_gates],
                 dofs[[gate_ct + g for g in global_free_gates]],
                 dofs[[gate_ct * 2 + g for g in global_free_gates]]]
    lb_blocks = [so.gate_position_dof_lb[global_free_gates], [-40] * n_free, [-40] * n_free]
    ub_blocks = [so.gate_position_dof_ub[global_free_gates], [40] * n_free, [40] * n_free]
    if OPTIMIZE_D2:
        x0_blocks += [dofs[[gate_ct * 3 + g for g in global_free_gates]],
                      dofs[[gate_ct * 4 + g for g in global_free_gates]]]
        lb_blocks += [[-10] * n_free, [-10] * n_free]
        ub_blocks += [[10] * n_free, [10] * n_free]
    x0 = np.concatenate(x0_blocks)
    lb = np.concatenate(lb_blocks)
    ub = np.concatenate(ub_blocks)

    res = minimize(objective, x0, method=so.OPT_METHOD, bounds=Bounds(lb, ub),
                    options={'maxiter': WINDOW_OPT_MAXITER, 'ftol': WINDOW_OPT_FTOL, 'maxfun': so.OPT_MAXFUN})

    # Commit the window_len gates this window owns -- the pre/post context
    # gates (if any) were only borrowed to shape the objective; they aren't
    # written back here (the pre gate keeps its previous answer, the post
    # gate gets properly solved when its own window comes around).
    full_local = assemble(res.x)
    committed = np.concatenate([
        full_local[pre_offset:pre_offset + window_len],
        full_local[obj_len + pre_offset:obj_len + pre_offset + window_len],
        full_local[obj_len * 2 + pre_offset:obj_len * 2 + pre_offset + window_len],
        full_local[obj_len * 3 + pre_offset:obj_len * 3 + pre_offset + window_len],
        full_local[obj_len * 4 + pre_offset:obj_len * 4 + pre_offset + window_len],
    ])
    return committed, res.fun


# ----- Sweep the whole track -----
sweep_start_time = time.perf_counter()
start = resume_start
window_num = start // WINDOW_STEP  # roughly which window we're resuming at, for the printed count
while start < gate_ct - 1:
    end = start + WINDOW_SIZE
    if end >= gate_ct:
        # Last window of the sweep: pull it back so it's still a full
        # WINDOW_SIZE gates instead of being left short.
        end = gate_ct
        start = max(0, end - WINDOW_SIZE)
    window_num += 1
    t0 = time.perf_counter()
    local_dofs, obj_val = solve_window(start, end)
    set_window_dofs(dofs, start, end - start, local_dofs)
    print(f"window {window_num}: gates [{start},{end}) obj={obj_val:.6f} "
          f"time={time.perf_counter() - t0:.1f}s", flush=True)
    if end == gate_ct:
        np.savez(CHECKPOINT_PATH, dofs=dofs, next_start=end)  # so a crash before the
        start = end                                           # final save can still resume
        break
    start += WINDOW_STEP
    np.savez(CHECKPOINT_PATH, dofs=dofs, next_start=start)

# Sweep finished -- the checkpoint is no longer needed (a fresh run of this
# script should start over, not silently resume a finished/old sweep).
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

sweep_runtime = time.perf_counter() - sweep_start_time
print(f"\nTotal sweep runtime: {sweep_runtime:.1f} s over {window_num} windows")

# ----- Build the final full-track path from the swept DOFs and report stats -----
x_spline, y_spline = so.generate_spline(so.imported_gate_points, dofs)
total_length, distance_t = so.path_length_fun(x_spline, y_spline, gate_ct, return_path=True)
full_obj_value, _ = so.curvature_obj_evaluation(x_spline, y_spline, [0, gate_ct - 1])
print(f"Total path length: {total_length:.2f} m")
print(f"Full-track objective value (comparable to spline_optimize_5's res.fun): {full_obj_value:.6f}")

# ----- Save results (no plt.show() -- this runs unattended) -----
os.makedirs(OUTPUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
np.save(os.path.join(OUTPUT_DIR, f"opt_dofs_{stamp}.npy"), dofs)

gate_x_pts, gate_y_pts = so.generate_gate_cross_points(so.imported_gate_points, dofs)
t_smooth = np.linspace(0, gate_ct - 1, so.OBJ_PLOT_POINTS_PER_GATE * gate_ct)
x_smooth, y_smooth = so.generate_path_t(x_spline, y_spline, t_smooth)

fig, ax = plt.subplots()
for gate in so.imported_gate_points:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    ax.plot([x1, x2], [y1, y2], 'b-')
ax.plot(x_smooth, y_smooth, 'r-')
ax.scatter(gate_x_pts, gate_y_pts, s=15, c='r', zorder=3)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.suptitle(f"Windowed optimization: {so.FILENAME}, window={WINDOW_SIZE}, "
             f"length={total_length:.1f} m, {sweep_runtime:.0f}s")
fig.savefig(os.path.join(OUTPUT_DIR, f"map_plot_{stamp}.png"))
plt.close(fig)

print(f"\nSaved DOFs and map plot to {OUTPUT_DIR}/ (stamp {stamp})")
