import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from datetime import datetime

from scipy.interpolate import BPoly
from scipy.interpolate import CubicSpline
from scipy.optimize import Bounds
from scipy.optimize import minimize
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
OBJ_POINT_COUNT = 42 # Number of equidistant points used inside the curvature objective function
PATHGEN_POINT_COUNT = 18 # Number of points used when sampling the final path for plotting

PATH_GEN_OVERRIDE = True # IF TRUE, OVERRIDES ABOVE VALUES AS MULTIPLE OF GATE COUNT
OBJ_POINT_MULT = 7
PATHGEN_POINT_MULT = 3

TRACK_WIDTH = 1.25 #meters
CONE_SPACING = 0.1 #meters - 100mm clearance to cones
MINIMUM_RADIUS = 2.7 #meters vehicle centerline w/ 9 deg slip geometric

#TRACK USED
FILENAME = "ref_gates_gates.csv"
FILENAME = "ref_gates_autoX.csv"
#FILENAME = "ref_gates_endurance.csv"

# Optimization parameters
OPT_METHOD = 'L-BFGS-B'
OPT_MAXITER = 1000
OPT_FTOL = 1e-7


#OUTPUT
OUTPUT_DIR = "outputs" #
PLOT_SPLINE_DEBUG = False # show spline debug plots
PLOT_PATH_DEBUG = False # show path debug + equidistant fit plots
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

gate_file_path = f"{current_path}//{FILENAME}"
imported_gate_points = load_gates_2D(gate_file_path)

gate_ct = len(imported_gate_points[:,0])
if PATH_GEN_OVERRIDE:
    OBJ_POINT_COUNT = OBJ_POINT_MULT*gate_ct
    PATHGEN_POINT_COUNT = PATHGEN_POINT_MULT*gate_ct


#DOF Guesses
guess_gate_position_dofs = np.ones(gate_ct)*0.5

# Seed d1 from finite differences between gate centers.
# Layout: [x_d1 at all gates, then y_d1 at all gates] to match np.split in generate_spline.
_gate_cx = (imported_gate_points[:,1] + imported_gate_points[:,3]) / 2
_gate_cy = (imported_gate_points[:,2] + imported_gate_points[:,4]) / 2
_d1x = np.gradient(_gate_cx)  # tangent in x per unit-t (one t-unit per gate)
_d1y = np.gradient(_gate_cy)
guess_d1_dofs = np.concat((_d1x, _d1y))
guess_d2_dofs = np.zeros(gate_ct*2)

guess_dofs = np.concat((guess_gate_position_dofs, guess_d1_dofs, guess_d2_dofs))

#DOF Bounds
gate_position_dof_lb = [0]*gate_ct
gate_position_dof_ub = [1]*gate_ct
d1_dof_lb = [-np.inf]*gate_ct*2
d1_dof_ub = [np.inf]*gate_ct*2
d2_dof_lb = [-np.inf]*gate_ct*2
d2_dof_ub = [np.inf]*gate_ct*2

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

def curvature_obj_function(x_s, y_s, time):
    #needs equally spaced points
    x_dot = x_s.derivative(1)
    y_dot = y_s.derivative(1) 
    x_dot2 = x_s.derivative(2)
    y_dot2 = y_s.derivative(2)

    sum = 0
    for i, t in enumerate(time):
        rho =  abs(x_dot(t)*y_dot2(t) - y_dot(t)*x_dot2(t)) / (x_dot(t)**2 + y_dot(t)**2)**1.5
        sum += rho**2
    print(sum)
    return sum

def raw_obj_fun(dofs):
    tick("obj fun start")
    gate_x_pts,gate_y_pts = generate_gate_cross_points(imported_gate_points, dofs)
    x_spline, y_spline = generate_spline(imported_gate_points, dofs)
    tick("make spline from dofs")

    total_length, distance_t = path_length_fun(x_spline, y_spline, gate_ct, return_path=True)
    tick("path function")

    point_count = OBJ_POINT_COUNT
    distance_t_full = np.concat(([0], distance_t))
    dist_fit = CubicSpline(distance_t_full, np.linspace(0, gate_ct-1, len(distance_t_full)))
    equidistant_times = dist_fit(np.linspace(0, total_length, point_count))
    tick("equidistant times generated")

    value = curvature_obj_function(x_spline,y_spline,equidistant_times)
    tick("ran objective function")
    return value

# Save the path at each SLSQP major iteration for later overlay plotting.
iteration_paths = []
def save_iteration(current_dofs):
    x_s, y_s = generate_spline(imported_gate_points, current_dofs)
    t = np.linspace(0, gate_ct-1, 300)
    iteration_paths.append((x_s(t), y_s(t)))

# Save the initial guess as iteration 0.
save_iteration(guess_dofs)

opt_start_time = time.perf_counter()
res = minimize(raw_obj_fun, guess_dofs, method=OPT_METHOD, bounds = dof_bounds,
               callback=save_iteration,
               options={'maxiter': OPT_MAXITER, 'ftol': OPT_FTOL, 'disp': True}) # 'maxfun':50000,
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
    # Obj-fun spline points: OBJ_POINT_COUNT points equidistant in arc length,
    # using the same distance->t mapping that raw_obj_fun uses (red).
    obj_times = np.interp(np.linspace(0, total_length, OBJ_POINT_COUNT),
                          distance_t_full, t_uniform)
    x_obj_pts, y_obj_pts = generate_path_t(x_spline, y_spline,
                                           np.clip(obj_times, 0, gate_ct-1))

    # Path gen points: the coarser PATHGEN_POINT_COUNT sampling (translucent grey).
    x_pathgen_pts, y_pathgen_pts = generate_path_t(x_spline, y_spline,
                                                   np.clip(equidistant_times, 0, gate_ct-1))

    fig, ex = plt.subplots()
    ex.plot(x_pathgen_pts, y_pathgen_pts, '-', color='grey', alpha=0.4, zorder=1)
    ex.scatter(x_pathgen_pts, y_pathgen_pts, s=30, c='grey', alpha=0.4, zorder=1)
    ex.plot(x_obj_pts, y_obj_pts, 'r-', zorder=2)
    ex.scatter(x_obj_pts, y_obj_pts, s=5, c='r', zorder=3)
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


plot_map_plots()

# Spline debug runs after the three full-track plots.
if PLOT_SPLINE_DEBUG:
    plot_spline_debug()

if PLOT_PATH_DEBUG:
    plot_path_debug()

'''
TANGENT PLOT

dx_dt = x_spline(equidistant_times_clamped, 1)
dy_dt = y_spline(equidistant_times_clamped, 1)

fig, fx = plt.subplots()
fx.plot(x_path_pts, y_path_pts, 'r-', alpha=0.2)
for gate in imported_gate_points:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    fx.plot([x1, x2], [y1, y2], 'b-')
mag = np.hypot(dx_dt, dy_dt)
fx.quiver(x_path_pts, y_path_pts, dx_dt/mag, dy_dt/mag, scale=1.2, angles='xy', scale_units='xy')
fx.set_aspect('equal', adjustable='datalim')
fx.margins(0.1)
fx.set_xlabel('x (m)')
fx.set_ylabel('y (m)')
show_maximized()

'''


def measure_obj_step_times(dofs):
    # Re-run the objective function once, timing each step individually.
    # Returns a dict of {step name: seconds}. Mirrors raw_obj_fun's steps.
    times = {}

    t0 = time.perf_counter()
    generate_gate_cross_points(imported_gate_points, dofs)
    x_spline, y_spline = generate_spline(imported_gate_points, dofs)
    times["make spline from dofs"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    total_length, distance_t = path_length_fun(x_spline, y_spline, gate_ct, return_path=True)
    times["path length function"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    distance_t_full = np.concat(([0], distance_t))
    dist_fit = CubicSpline(distance_t_full, np.linspace(0, gate_ct-1, len(distance_t_full)))
    equidistant_times = dist_fit(np.linspace(0, total_length, OBJ_POINT_COUNT))
    times["equidistant times generated"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    curvature_obj_function(x_spline, y_spline, equidistant_times)
    times["curvature objective function"] = time.perf_counter() - t0

    return times


def write_report():
    # Write a single self-contained, timestamped PDF report into OUTPUT_DIR.
    # Page 1 is the first map plot; page 2 is the text summary; remaining figures follow.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = os.path.join(OUTPUT_DIR, f"track_gen_report_{stamp}.pdf")

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
        f"\n"
        f"Optimization parameters\n"
        f"    method: {OPT_METHOD}\n"
        f"    maxiter: {OPT_MAXITER}\n"
        f"    ftol: {OPT_FTOL}\n"
        f"\n"
        f"Point counts\n"
        f"    objective function points: {OBJ_POINT_COUNT}\n"
        f"    path generation points: {PATHGEN_POINT_COUNT}\n"
        f"\n"
        f"Results\n"
        f"    total optimization time: {opt_runtime:.3f} s\n"
        f"    total path length: {total_length:.4f} m\n"
        f"\n"
        f"Per-step time estimate (one measured objective evaluation)\n"
        f"{step_lines}\n"
    )

    with PdfPages(pdf_path) as pdf:
        # Page 1: the first map plot.
        if report_figures:
            pdf.savefig(report_figures[0])

        # Page 2: text summary.
        text_fig = plt.figure(figsize=(8.5, 11))
        text_fig.text(0.07, 0.95, text, va='top', ha='left',
                      family='monospace', fontsize=10)
        text_fig.gca().axis('off')
        pdf.savefig(text_fig)
        plt.close(text_fig)

        # Remaining plots, one per page.
        for f in report_figures[1:]:
            pdf.savefig(f)

    print(f"Report written to {pdf_path}")


# Single report call at the end of the run.
write_report()
