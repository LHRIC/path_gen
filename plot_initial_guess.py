"""
Plot the raw geometric initial guess for a gate file -- no optimization at all.
Used to eyeball the guess for loops/self-intersections before spending time
running the windowed or global optimizer on it.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import spline_optimize_5 as so  # generate_spline, generate_path_t, initial_guess

GATE_FILE = "ref_less_gates.csv"

gate_file_path = os.path.join(so.current_path, GATE_FILE)
gate_points = so.load_gates_2D(gate_file_path)
gate_ct = len(gate_points[:, 0])
print(f"Loaded {GATE_FILE}: {gate_ct} gates")

guess_dofs = so.initial_guess(gate_points)
x_s, y_s = so.generate_spline(gate_points, guess_dofs)
t = np.linspace(0, gate_ct - 1, so.OBJ_PLOT_POINTS_PER_GATE * gate_ct)
x, y = so.generate_path_t(x_s, y_s, t)


def cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def find_self_intersections(x, y):
    """Check every pair of non-adjacent path segments for a crossing (standard
    orientation-test method). Skips segments that share an endpoint (i, i+1),
    since those always "touch" there without it being a real self-intersection.
    Returns a list of (ix, iy) crossing points."""
    n = len(x) - 1  # number of segments
    hits = []
    for i in range(n):
        ax1, ay1, ax2, ay2 = x[i], y[i], x[i + 1], y[i + 1]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # first/last segment share the track's start==end gate
            bx1, by1, bx2, by2 = x[j], y[j], x[j + 1], y[j + 1]
            d1 = cross(bx1, by1, bx2, by2, ax1, ay1)
            d2 = cross(bx1, by1, bx2, by2, ax2, ay2)
            d3 = cross(ax1, ay1, ax2, ay2, bx1, by1)
            d4 = cross(ax1, ay1, ax2, ay2, bx2, by2)
            if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
                denom = cross(0, 0, ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1)
                t_hit = cross(0, 0, bx1 - ax1, by1 - ay1, bx2 - bx1, by2 - by1) / denom
                hits.append((ax1 + t_hit * (ax2 - ax1), ay1 + t_hit * (ay2 - ay1)))
    return hits


loops = find_self_intersections(x, y)
print(f"Self-intersections found: {len(loops)}")
for ix, iy in loops:
    print(f"  loop at x={ix:.1f}, y={iy:.1f}")

fig, ax = plt.subplots()
for gate in gate_points:
    x1, y1, x2, y2 = gate[1], gate[2], gate[3], gate[4]
    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
ax.plot(x, y, 'b-', linewidth=1.5)
if loops:
    lx, ly = zip(*loops)
    ax.scatter(lx, ly, color='red', marker='x', s=100, zorder=5, label=f'{len(loops)} self-intersection(s)')
    ax.legend()
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
fig.suptitle(f"{GATE_FILE}: initial guess ({gate_ct} gates) -- loop check")

out_path = "outputs/less_gates_guess_check.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

#plt.show()
