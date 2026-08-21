import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from datetime import datetime

from scipy.interpolate import BPoly
from scipy.optimize import Bounds
from scipy.optimize import minimize

from scipy.integrate import quad
from matplotlib.backends.backend_pdf import PdfPages

# Make every figure 1920x1080 px (19.2 x 10.8 in at 100 dpi) for both on-screen
# display and saved/PDF output.
plt.rcParams['figure.figsize'] = (19.2, 10.8)
plt.rcParams['figure.dpi'] = 100

from timer import make_timer
from gate_csv_parser import load_gates_2D

PI = math.pi
tick = make_timer()

# ----- Run configuration (edit these) -----
#PATHGEN PARAMS
PATHGEN_POINT_COUNT = 18 # Number of points used when sampling the final path for plotting
OBJ_PLOT_POINTS_PER_GATE = 20 # Resolution of the smooth obj-fun spline line on the map plot

PATH_GEN_OVERRIDE = True # IF TRUE, OVERRIDES ABOVE VALUES AS MULTIPLE OF GATE COUNT
PATHGEN_POINT_MULT = 3 #affects runtime slightly slightly

TRACK_WIDTH = 1.25 #meters
CONE_SPACING = 0.2 #meters - 100mm clearance to cones
MINIMUM_RADIUS = 2.7 #meters vehicle centerline w/ 9 deg slip geometric

#TRACK USED
FILENAME = "ref_gates_gates.csv"
FILENAME = "ref_gates_2_gates.csv"
FILENAME = "ref_gates_autoX.csv"
FILENAME = "ref_gates_endurance.csv"

# Optimization parameters
OPT_METHOD = 'L-BFGS-B'
#OPT_METHOD = 'trust-constr'
OPT_MAXITER = 20
OPT_FTOL = 1e-7
OPT_MAXFUN = 500*200*5

# Guess DOF override (warm-start): if USE_DOF_GUESS is True, the geometric guess
# is replaced with the DOFs loaded from GUESS_DOF_FILE and the optimizer starts
# from there. The file must come from the same gate CSV as FILENAME (same gate
# count) or it will be rejected.
USE_DOF_GUESS = False
GUESS_DOF_FILE = "out_autoX_test1/opt_dofs_2026-05-30_23-23-53.npy"

# Objective Function Integration
OBJ_FUN_INTEGRATION_EPSABS = 1.0e-5 #I use 1.0 e-3 for small tracks, e-2 for autoX and endurance tho
OBJ_FUN_INTEGRATION_EPSREL = 1.0e-5 #default, haven't fucked with

#OUTPUT FILE
OUTPUT_DIR = "outputs"
OUTPUT_DIR = "out_autoX_test1"
OUTPUT_DIR = "out_endurance_test1"
SAVE_DOF_HISTORY = False # if True, also save the per-iteration DOF history with the report
PLOT_SPLINE_DEBUG = False # show spline debug plots
PLOT_PATH_DEBUG = False # show path debug + equidistant fit plots
PROFILE_OBJ_FUN = True # cProfile one curvature_obj_evaluation to see where the time goes
# -------------------------------------------

# Maximize the current figure window, then show it. Falls back to a normal
# show() if the active matplotlib backend doesn't support maximizing.
def show_maximized():
    try:
        mgr = plt.get_current_fig_manager()
        mgr.window.state('zoomed')  # Windows / TkAgg
    except Exception:
        try:
            plt.get_current_fig_manager().window.showMaximized()  # Qt backends
        except Exception:
            pass
    plt.show()

# Figures are collected here as they are created so the report can save them all.
report_figures = []

current_path = os.path.dirname(sys.argv[0])

gate_file_path = os.path.join(current_path, FILENAME)
imported_gate_points = load_gates_2D(gate_file_path)

gate_ct = len(imported_gate_points[:,0])
if PATH_GEN_OVERRIDE:
    PATHGEN_POINT_COUNT = PATHGEN_POINT_MULT*gate_ct

#DOF Guesses
guess_gate_position_dofs = np.ones(gate_ct)*0.5

#maked D1 guess paralle to prev. and next gates
gate_midpoints = np.zeros((gate_ct,2))
for i in range(gate_ct):
    num, x1, y1, x2, y2 = imported_gate_points[i,:]
    gate_midpoints[i,:] = [(x1+x2)/2, (y1+y2)/2]

slope_vectors = np.diff(gate_midpoints, axis =0)
start_vector = np.array([[slope_vectors[0,0],slope_vectors[0,1]]])
pre_slope_vectors = np.concat((start_vector, slope_vectors))
end_vector = np.array([[slope_vectors[-1,0],slope_vectors[-1,1]]])
post_slope_vectors = np.concat((slope_vectors,end_vector))
combined_slope_vectors = pre_slope_vectors + post_slope_vectors

guess_d1_dofs = combined_slope_vectors.flatten('F') / 2
guess_d2_dofs = np.zeros(gate_ct*2)

guess_dofs = np.concat((guess_gate_position_dofs, guess_d1_dofs, guess_d2_dofs))

# Optional warm-start: replace the geometric guess with DOFs loaded from a file.
if USE_DOF_GUESS:
    from unpack_dofs import load_dof_guess
    guess_dofs = load_dof_guess(GUESS_DOF_FILE, expected_len=gate_ct * 5)

#DOF Bounds
#gate position bounded by trackwidth and clearance
#find length of gate, scale 0-1 bound to eliminate half trackwidth + clerance at each end
gate_lengths = np.zeros(gate_ct)
gate_position_dof_lb = np.zeros(gate_ct)
gate_position_dof_ub = np.zeros(gate_ct)
for i in range(gate_ct):
    num, x1, y1, x2, y2 = imported_gate_points[i,:]
    gate_lengths[i] = math.dist([x1,y1],[x2,y2])

    normalized_offset = (TRACK_WIDTH/2 + CONE_SPACING) / gate_lengths[i]
    gate_position_dof_lb[i] = 0 + normalized_offset
    gate_position_dof_ub[i] = 1 - normalized_offset

d1_dof_lb = [-40]*gate_ct*2
d1_dof_ub = [40]*gate_ct*2
d2_dof_lb = [-1.2]*gate_ct*2
d2_dof_ub = [1.2]*gate_ct*2

#d1_dof_lb = [-np.inf]*gate_ct*2
#d1_dof_ub = [np.inf]*gate_ct*2
#d2_dof_lb = [-np.inf]*gate_ct*2
#d2_dof_ub = [np.inf]*gate_ct*2

lb = np.concat([gate_position_dof_lb, d1_dof_lb, d2_dof_lb])
ub = np.concat([gate_position_dof_ub, d1_dof_ub, d2_dof_ub])
dof_bounds = Bounds(lb,ub)

def generate_gate_cross_points(gate_points, dofs):
    gate_ct = len(gate_points[:,0])
    gate_dofs = dofs[:gate_ct] #one dof per gate

    x = np.zeros(gate_ct)
    y = np.zeros(gate_ct)
    for i in range(gate_ct):
        x1, y1, x2, y2 = gate_points[i, 1:5]
        scale = gate_dofs[i]
        x[i] = np.interp(scale, [0,1], [x1,x2])
        y[i] = np.interp(scale, [0,1], [y1,y2])
    
    return x,y

def generate_spline(gate_points, dofs):
    gate_ct = len(gate_points[:,0])
    x,y = generate_gate_cross_points(gate_points, dofs)
    xy = np.concat((x,y))
    xy_d1 = dofs[gate_ct:gate_ct*3] #x and y per gate
    xy_d2 = dofs[gate_ct*3:]

    spline_BCs = np.transpose(np.array([
        xy,
        xy_d1,
        xy_d2,
    ]))

    x_BCs, y_BCs = np.split(spline_BCs,2)

    x_s = BPoly.from_derivatives(np.arange(gate_ct),x_BCs)
    y_s = BPoly.from_derivatives(np.arange(gate_ct),y_BCs)

    return x_s, y_s

def generate_path_t(x_s, y_s, t):
    x_path_pts, y_path_pts = x_s(t),y_s(t)
    return x_path_pts, y_path_pts

def path_length_fun(x_s, y_s,gate_ct, return_path=False):
    points = 1000
    times = np.linspace(0,gate_ct-1,points)
    x_pts, y_pts = generate_path_t(x_s,y_s,times)
    x_diffs, y_diffs = np.diff(x_pts), np.diff(y_pts)
    distances = np.hypot(x_diffs,y_diffs)
    path = np.cumsum(distances)
    total_length = path[-1]
    

    if return_path == False:
        return total_length
    else:
        return total_length, path

def curvature_obj_evaluation(x_s, y_s, time):
    #needs equally spaced points
    x_dot = x_s.derivative(1)
    y_dot = y_s.derivative(1)
    x_dot2 = x_s.derivative(2)
    y_dot2 = y_s.derivative(2)

    # Signed curvature: no abs() so the integrand stays smooth (abs creates a
    # kink at every inflection point, which makes adaptive quad subdivide hard).
    # The sign cancels in rho**2; abs is applied only at the radius line below.
    def rho(t):
        return ((x_dot(t)*y_dot2(t) - y_dot(t)*x_dot2(t)) / (x_dot(t)**2 + y_dot(t)**2)**1.5)
    rho_norm_calls = [0]  # diagnostic: how many times quad evaluates the integrand
    def rho_normalized(t):
        rho_norm_calls[0] += 1
        return rho(t)**2 * (np.hypot(x_dot(t), y_dot(t))) #scales according to velocity at the current point

    fun = quad(rho_normalized, time[0], time[-1], epsrel=OBJ_FUN_INTEGRATION_EPSREL, epsabs=OBJ_FUN_INTEGRATION_EPSABS)
    print(f"\nObj funtion: {fun}   (rho_normalized calls: {rho_norm_calls[0]}) \n")

    ts = np.linspace(time[0], time[-1], gate_ct*PATHGEN_POINT_MULT)
    radius_array = 1 / np.abs(rho(ts))  # abs keeps radius physically positive

    return fun[0], radius_array

# Optimizer progress tracking.
# obj_eval_count counts EVERY call to raw_obj_fun (this includes the extra calls
# the optimizer makes to estimate the gradient by finite differences).
# optimizer_iter_count counts accepted optimizer steps (callback invocations).
obj_eval_count = 0
optimizer_iter_count = 0

# History arrays, one entry appended per optimizer iteration (in save_iteration).
obj_value_history = []   # objective value at each iteration
dof_history = []         # DOF vector at each iteration

def raw_obj_fun(dofs):
    global obj_eval_count
    tick("obj fun start")
    x_spline, y_spline = generate_spline(imported_gate_points, dofs)
    tick("make spline from dofs")

    # quad integrates over the full t-domain; it only reads the interval endpoints.
    value,radius_array = curvature_obj_evaluation(x_spline,y_spline,[0, gate_ct-1])
    tick("ran objective function")

    obj_eval_count += 1
    print(f"Total objective function evaluations (incl. gradient probes): {obj_eval_count}")

    return value

#copy of obj fun evaluation but for radius
def radius_constraint(dofs):
    x_spline, y_spline = generate_spline(imported_gate_points, dofs)
    tick("make spline from dofs")

    value,radius_array = curvature_obj_evaluation(x_spline,y_spline,[0, gate_ct-1])
    return radius_array

# Save the path at each SLSQP major iteration for later overlay plotting.
iteration_paths = []
def save_iteration(current_dofs):
    global optimizer_iter_count
    optimizer_iter_count += 1
    print(f"Optimizer iteration (accepted steps): {optimizer_iter_count}")

    # Record the objective value and DOFs at this iteration. The value is
    # recomputed from current_dofs (one extra eval) so it reflects this exact
    # accepted point rather than the optimizer's last gradient probe.
    obj_value_history.append(raw_obj_fun(current_dofs))
    dof_history.append(np.array(current_dofs))

    x_s, y_s = generate_spline(imported_gate_points, current_dofs)
    t = np.linspace(0, gate_ct-1, 300)
    iteration_paths.append((x_s(t), y_s(t)))

# The optimization run is guarded by __name__ == "__main__" so that other files
# (e.g. the DOF unpacker) can import this module's functions and plots WITHOUT
# triggering a full optimization or report. The optimized-result globals
# (opt_dofs, x_spline, total_length, ...) are only created when run directly;
# the unpacker recomputes and injects them from a loaded DOF file instead.
if __name__ == "__main__":
    # Save the initial guess as iteration 0.
    save_iteration(guess_dofs)

    opt_start_time = time.perf_counter()
    res = minimize(raw_obj_fun, guess_dofs, method=OPT_METHOD, bounds = dof_bounds,
                    callback=save_iteration,
                    options={'maxiter': OPT_MAXITER, 'ftol': OPT_FTOL, 'disp': True, 'maxfun': OPT_MAXFUN}) #

    opt_runtime = time.perf_counter() - opt_start_time
    print(res)
    print(f"Total optimization runtime: {opt_runtime:.3f} s")

    # Extract optimized results
    opt_dofs = res.x
    gate_x_pts, gate_y_pts = generate_gate_cross_points(imported_gate_points, opt_dofs)
    x_spline, y_spline = generate_spline(imported_gate_points, opt_dofs)
    total_length, distance_t = path_length_fun(x_spline, y_spline, gate_ct, return_path=True)
    distance_t_full = np.concat(([0], distance_t))
    t_uniform = np.linspace(0, gate_ct-1, len(distance_t_full))
    equidistant_times = np.interp(np.linspace(0, total_length, PATHGEN_POINT_COUNT), distance_t_full, t_uniform)
    x_path_pts, y_path_pts = generate_path_t(x_spline, y_spline, equidistant_times)

def plot_spline_debug():
    #SPLINE DEBUG
    times = np.arange(gate_ct)
    t = np.linspace(0,times[-1],PATHGEN_POINT_COUNT)
    x_path_pts, y_path_pts = generate_path_t(x_spline, y_spline, t)
    fig, ax = plt.subplots()
    ax.scatter(gate_x_pts,gate_y_pts)
    ax.plot(x_path_pts,y_path_pts)
    ax.scatter(x_path_pts,y_path_pts)
    fig.suptitle("Spline debug: path and gate cross points")
    report_figures.append(fig)
    show_maximized()

    fig, bx = plt.subplots()
    bx.scatter(times,gate_x_pts)
    bx.scatter(times,gate_y_pts)
    bx.plot(t,x_spline(t))
    bx.plot(t,y_spline(t))
    fig.suptitle("Spline debug: x(t) and y(t)")
    report_figures.append(fig)
    show_maximized()

def plot_map_plots():
    #MAP PLOT
    # Obj-fun spline: a smooth high-resolution sampling of the optimized path (red line).
    obj_times = np.linspace(0, gate_ct-1, OBJ_PLOT_POINTS_PER_GATE*gate_ct)
    x_obj_pts, y_obj_pts = generate_path_t(x_spline, y_spline, obj_times)

    # Path gen points: the coarser PATHGEN_POINT_COUNT sampling (translucent grey).
    x_pathgen_pts, y_pathgen_pts = generate_path_t(x_spline, y_spline,
                                                   np.clip(equidistant_times, 0, gate_ct-1))

    fig, ex = plt.subplots()
    ex.plot(x_pathgen_pts, y_pathgen_pts, '-', color='grey', alpha=0.4, zorder=1)
    ex.scatter(x_pathgen_pts, y_pathgen_pts, s=30, c='grey', alpha=0.4, zorder=1)
    ex.plot(x_obj_pts, y_obj_pts, 'r-', zorder=2)
    for gate in imported_gate_points:
        x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
        ex.plot([x1, x2], [y1, y2], 'b-')
        bot_x = x1 if y1 < y2 else x2
        bot_y = min(y1, y2)
        ex.annotate(int(gate[0]), xy=(bot_x, bot_y), ha='center', va='top',
                    xytext=(0, -4), textcoords='offset points')

    for gate, label in [(imported_gate_points[0], 'Start'), (imported_gate_points[-1], 'End')]:
        x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
        top_x = x1 if y1 > y2 else x2
        top_y = max(y1, y2)
        ex.text(top_x, top_y + 1, label, ha='center')

    ex.set_aspect('equal', adjustable='datalim')
    ex.margins(0.1)
    ex.set_xlabel('x (m)')
    ex.set_ylabel('y (m)')
    fig.suptitle("Map plot: optimized path through gates")
    report_figures.append(fig)
    show_maximized()

    #ITERATION OVERLAY PLOT
    #Shows every saved iteration path. Early iterations fade, final iteration solid.
    fig, ox = plt.subplots()
    n_iter = len(iteration_paths)
    cmap = plt.get_cmap('viridis')
    for i, (xp, yp) in enumerate(iteration_paths):
        # Colors progress from start (purple) to end (yellow); final path drawn thicker.
        color = cmap(i / max(n_iter - 1, 1))
        alpha = 0.3 + 0.7 * (i / max(n_iter - 1, 1))
        lw = 2.0 if i == n_iter - 1 else 1.0
        label = f"iter {i}" if (i == 0 or i == n_iter - 1) else None
        ox.plot(xp, yp, color=color, alpha=alpha, linewidth=lw, label=label)

    for gate in imported_gate_points:
        x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
        ox.plot([x1, x2], [y1, y2], 'b-', alpha=0.6)

    ox.set_aspect('equal', adjustable='datalim')
    ox.margins(0.1)
    ox.set_xlabel('x (m)')
    ox.set_ylabel('y (m)')
    ox.set_title(f'Optimization progression ({n_iter} iterations)')
    ox.legend()
    fig.suptitle("Optimization progression")
    report_figures.append(fig)
    show_maximized()

    # 5 equally spaced iterations (including first and last) in distinct colors.
    # If fewer than 5 iterations exist, just plot all of them.
    fig, sx = plt.subplots()
    n_sample = min(5, n_iter)
    sample_idx = np.unique(np.linspace(0, n_iter - 1, n_sample).astype(int))
    sample_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for c, i in enumerate(sample_idx):
        xp, yp = iteration_paths[i]
        sx.plot(xp, yp, color=sample_colors[c], linewidth=1.5, label=f"iter {i}")

    for gate in imported_gate_points:
        x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
        sx.plot([x1, x2], [y1, y2], 'b-', alpha=0.6)

    sx.set_aspect('equal', adjustable='datalim')
    sx.margins(0.1)
    sx.set_xlabel('x (m)')
    sx.set_ylabel('y (m)')
    sx.set_title(f'{len(sample_idx)} equally spaced iterations')
    sx.legend()
    fig.suptitle("Equally spaced iterations")
    report_figures.append(fig)
    show_maximized()

def plot_path_debug():
    #PATH DEBUG
    print(f"Total path length: {total_length}")
    print(f"Optimized DOFs:\n{opt_dofs}")
    print(f"  gate position dofs: {opt_dofs[:gate_ct]}")
    print(f"  d1 dofs: {opt_dofs[gate_ct:gate_ct*3]}")
    print(f"  d2 dofs: {opt_dofs[gate_ct*3:]}")

    distance_t_plot = np.concat(([0], distance_t))
    distance_times = np.linspace(0, gate_ct-1, len(distance_t_plot))
    fig, (cx, dx) = plt.subplots(1, 2)
    cx.plot(distance_times, distance_t_plot)
    cx.set_xlabel("gate number")
    cx.set_ylabel("cumulative distance")
    dx.plot(distance_t_plot, distance_times)
    dx.set_xlabel("cumulative distance")
    dx.set_ylabel("gate number")
    fig.suptitle("Path debug: cumulative distance vs gate number")
    report_figures.append(fig)
    show_maximized()

    #EQUIDISTANT FIT
    fig, pre_e = plt.subplots()
    pre_e.plot(distance_t_full, t_uniform)
    pre_e.set_xlabel('cumulative distance (m)')
    pre_e.set_ylabel('t (gate number)')
    fig.suptitle("Equidistant fit: distance to t mapping")
    report_figures.append(fig)
    show_maximized()


def plot_radius_over_distance():
    # Instant radius plotted against cumulative distance along the optimized path.
    # Sample densely (independent of the optimizer's PATHGEN_POINT_MULT) so the
    # radius curve is high-resolution. Compute curvature rho directly from the
    # already-optimized splines, mirroring rho() in curvature_obj_evaluation.
    RADIUS_PLOT_POINTS = gate_ct * 100   # ~33x denser than the optimizer sampling
    radius_ts = np.linspace(0, gate_ct - 1, RADIUS_PLOT_POINTS)

    x_dot = x_spline.derivative(1)
    y_dot = y_spline.derivative(1)
    x_dot2 = x_spline.derivative(2)
    y_dot2 = y_spline.derivative(2)
    rho = (np.abs(x_dot(radius_ts) * y_dot2(radius_ts) - y_dot(radius_ts) * x_dot2(radius_ts))
           / (x_dot(radius_ts)**2 + y_dot(radius_ts)**2)**1.5)
    radius_array = 1 / rho

    # Map sample t values to real cumulative distance using the existing
    # distance<->t data (t_uniform -> distance_t_full).
    radius_distances = np.interp(radius_ts, t_uniform, distance_t_full)

    fig, rx = plt.subplots()
    rx.plot(radius_distances, radius_array, '-', linewidth=1)
    rx.axhline(MINIMUM_RADIUS, color='r', linestyle='--',
               label=f'minimum radius = {MINIMUM_RADIUS} m')
    rx.set_yscale('log')
    rx.set_xlabel('cumulative distance (m)')
    rx.set_ylabel('instant radius (m)')
    rx.set_title('Instant radius along path')
    rx.legend()
    fig.suptitle("Instant radius vs distance")
    report_figures.append(fig)
    show_maximized()


def plot_obj_over_iterations():
    # Objective function value at each optimizer iteration (accepted step).
    fig, ix = plt.subplots()
    iters = np.arange(len(obj_value_history))
    ix.plot(iters, obj_value_history, '-o', markersize=4)
    ix.set_xlabel('optimizer iteration')
    ix.set_ylabel('objective function value')
    ix.set_title('Objective value over iterations')
    fig.suptitle("Objective value over iterations")
    report_figures.append(fig)
    show_maximized()


if __name__ == "__main__":
    plot_map_plots()
    plot_obj_over_iterations()
    plot_radius_over_distance()

    # Spline debug runs after the three full-track plots.
    if PLOT_SPLINE_DEBUG:
        plot_spline_debug()

    if PLOT_PATH_DEBUG:
        plot_path_debug()


def measure_obj_step_times(dofs):
    # Re-run the objective function once, timing each step individually.
    # Returns a dict of {step name: seconds}. Mirrors raw_obj_fun's steps.
    times = {}

    t0 = time.perf_counter()
    generate_gate_cross_points(imported_gate_points, dofs)
    x_spline, y_spline = generate_spline(imported_gate_points, dofs)
    times["make spline from dofs"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    path_length_fun(x_spline, y_spline, gate_ct, return_path=True)
    times["path length function"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    curvature_obj_evaluation(x_spline, y_spline, [0, gate_ct-1])
    times["curvature objective function"] = time.perf_counter() - t0

    return times


def profile_obj_fun(dofs):
    # cProfile a single curvature_obj_evaluation so you can see, line by line in
    # the printed table, whether time is spent in quad, in the BPoly spline
    # evaluations (called by rho/rho_normalized), or elsewhere. The
    # "(rho_normalized calls: N)" print above tells you how many integrand
    # evaluations quad needed; the profile tells you how long each took.
    import cProfile, pstats
    x_spline, y_spline = generate_spline(imported_gate_points, dofs)

    pr = cProfile.Profile()
    pr.enable()
    curvature_obj_evaluation(x_spline, y_spline, [0, gate_ct-1])
    pr.disable()
    print("\n----- cProfile: one curvature_obj_evaluation -----")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    if PROFILE_OBJ_FUN:
        profile_obj_fun(opt_dofs)


def write_report():
    # Write a single self-contained, timestamped PDF report into OUTPUT_DIR.
    # Page 1 is the first map plot; page 2 is the text summary; remaining figures follow.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = os.path.join(OUTPUT_DIR, f"track_gen_report_{stamp}.pdf")

    # Save the optimized spline input DOFs as a plain flat array (same order
    # generate_spline expects), sharing the report's timestamp.
    dofs_filename = f"opt_dofs_{stamp}.npy"
    np.save(os.path.join(OUTPUT_DIR, dofs_filename), opt_dofs)

    # Optionally save the per-iteration DOF history (one row per iteration) as a
    # 2D array. Off by default; controlled by the SAVE_DOF_HISTORY global.
    dof_history_filename = None
    if SAVE_DOF_HISTORY:
        dof_history_filename = f"dof_history_{stamp}.npy"
        np.save(os.path.join(OUTPUT_DIR, dof_history_filename), np.array(dof_history))

    # Re-measure per-step timings once (a single extra objective evaluation).
    step_times = measure_obj_step_times(opt_dofs)

    step_lines = "\n".join(
        f"    {name}: {dt*1000:.2f} ms" for name, dt in step_times.items()
    )

    text = (
        f"Track Generation Report\n"
        f"{'='*40}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Gate file: {FILENAME}\n"
        f"Gate count: {gate_ct}\n"
        f"Saved DOFs file: {dofs_filename}\n"
        f"Saved DOF history file: {dof_history_filename if dof_history_filename else 'not saved'}\n"
        f"\n"
        f"Optimization parameters\n"
        f"    method: {OPT_METHOD}\n"
        f"    maxiter: {OPT_MAXITER}\n"
        f"    ftol: {OPT_FTOL}\n"
        f"\n"
        f"Point counts\n"
        f"    path generation points: {PATHGEN_POINT_COUNT}\n"
        f"\n"
        f"Results\n"
        f"    total optimization time: {opt_runtime:.3f} s\n"
        f"    total objective function evaluations: {obj_eval_count}\n"
        f"    optimizer iterations (accepted steps): {optimizer_iter_count}\n"
        f"    total path length: {total_length:.4f} m\n"
        f"\n"  # blank line reserved for the bold objective value (drawn below)
        f"\n"
        f"Per-step time estimate (one measured objective evaluation)\n"
        f"{step_lines}\n"
    )

    # The total objective value is rendered separately so it can be bold.
    obj_line = f"    total objective function value: {res.fun:.6f}"
    # Line number of the reserved blank line (0-based), so the bold overlay
    # lands inside the Results section, just under the path length line.
    lines = text.split("\n")
    obj_line_index = lines.index("    total path length: " + f"{total_length:.4f} m") + 1

    with PdfPages(pdf_path) as pdf:
        # Page 1: the first map plot.
        if report_figures:
            pdf.savefig(report_figures[0])

        # Page 2: text summary.
        text_fig = plt.figure(figsize=(8.5, 11))
        text_top = 0.95
        font_size = 10
        text_fig.text(0.07, text_top, text, va='top', ha='left',
                      family='monospace', fontsize=font_size)
        # Overlay the objective value as a bold line at the reserved blank line.
        # Lines advance by 1.2 * font_size points; figure is 11 in = 792 pt tall.
        line_step = 1.2 * font_size / 792
        text_fig.text(0.07, text_top - obj_line_index * line_step, obj_line,
                      va='top', ha='left', family='monospace',
                      fontsize=font_size, fontweight='bold')
        text_fig.gca().axis('off')
        pdf.savefig(text_fig)
        plt.close(text_fig)

        # Remaining plots, one per page.
        for f in report_figures[1:]:
            pdf.savefig(f)

    print(f"Report written to {pdf_path}")


# Single report call at the end of the run.
if __name__ == "__main__":
    write_report()
