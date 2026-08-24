"""
STOP HERE FIRST. Plots the raw geometric initial guess for GATE_FILE -- no
optimization at all -- and checks it for self-intersections (loops / gate
misordering). Run this and eyeball the plot before running 2_run_pipeline.py;
that script runs unattended for hours with no further stops, so this is the
one chance to catch a bad gate file before committing to it.

Same loop-check logic as the existing plot_initial_guess.py, pointed at the
real track and switched to an interactive window instead of a saved PNG.
"""
import numpy as np
import matplotlib.pyplot as plt

import pipeline_lib as pl

GATE_FILE = "endurance_tracks/Michigan-2019-endurance_gates.csv"  # the cleaned-up 2019 endurance track

gate_points = pl.load_gate_file(GATE_FILE)
gate_ct = len(gate_points)
print(f"Loaded {GATE_FILE}: {gate_ct} gates")

guess_dofs = pl.so.initial_guess(gate_points)
x, y = pl.path_xy(gate_points, guess_dofs)


def cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def find_self_intersections(x, y):
    """Check every pair of non-adjacent path segments for a crossing (standard
    orientation-test method). Skips segments that share an endpoint."""
    n = len(x) - 1
    hits = []
    for i in range(n):
        ax1, ay1, ax2, ay2 = x[i], y[i], x[i + 1], y[i + 1]
        for j in range(i + 2, n):
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

plt.show()
