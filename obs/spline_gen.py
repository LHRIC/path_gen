import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from gate_csv_parser import load_gates_2D

from scipy.interpolate import BPoly
from scipy.interpolate import CubicSpline
from scipy.optimize import Bounds

PI = math.pi

#n = len(x)
#t = np.linspace(0,n-1,50)
#times = np.linspace(0,n-1,n)

#x_spline = BPoly.from_derivatives(times,x_derivatives)
#y_spline = BPoly.from_derivatives(times,y_derivatives)


current_path = os.path.dirname(sys.argv[0])
filename = "ref_gates_gates.csv"
#filename = "ref_gates_autoX.csv"
#filename = "ref_gates_endurance.csv"
gate_file_path = f"{current_path}//{filename}"
imported_gate_points = load_gates_2D(gate_file_path)

gate_ct = len(imported_gate_points[:,0])

#DOF Guesses
guess_gate_position_dofs = np.ones(gate_ct)*0.5
guess_d1_dofs = np.ones(gate_ct*2)*1e-8
guess_d2_dofs = np.ones(gate_ct*2)*1e-8

guess_dofs = np.concat((guess_gate_position_dofs, guess_d1_dofs, guess_d2_dofs))

#DOF Bounds
gate_position_dof_lb = [0]*gate_ct
gate_position_dof_ub = [1]*gate_ct
d1_dof_lb = [None]*gate_ct*2
d1_dof_ub = [None]*gate_ct*2
d2_dof_lb = [None]*gate_ct*2
d2_dof_ub = [None]*gate_ct*2

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
    points = 100
    path_length_delta = np.inf
    total_length = 0

    while path_length_delta > 2:
        print(points)
        prev_length = total_length
        times = np.linspace(0,gate_ct-1,points)
        x_pts, y_pts = generate_path_t(x_s,y_s,times)
        x_diffs, y_diffs = np.diff(x_pts), np.diff(y_pts)
        distances = np.hypot(x_diffs,y_diffs)
        path = np.cumsum(distances)
        total_length = path[-1]
        path_length_delta = abs(prev_length - total_length)
        points *= 2

    if return_path == False:
        return total_length
    else:
        return total_length, path

def curvature_obj_function(x_s, y_s, time):
    #assumes equally spaced points
    x_dot = x_s.derivative(1)
    y_dot = y_s.derivative(1) 
    x_dot2 = x_s.derivative(2)
    y_dot2 = y_s.derivative(2)

    sum = 0
    for i, t in enumerate(time):
        rho =  abs(x_dot(t)*y_dot2(t) - y_dot(t)*x_dot2(t)) / (x_dot(t)**2 + y_dot(t)**2)**1.5
        sum += rho

    return sum

gate_x_pts,gate_y_pts = generate_gate_cross_points(imported_gate_points, guess_dofs)
x_spline, y_spline = generate_spline(imported_gate_points, guess_dofs)
total_length, distance_t = path_length_fun(x_spline, y_spline, gate_ct, return_path=True)

point_count = 500
distance_t_full = np.concat(([0], distance_t))
dist_fit = CubicSpline(distance_t_full, np.linspace(0, gate_ct-1, len(distance_t_full)))
equidistant_times = dist_fit(np.linspace(0, total_length, point_count))

print("curvature sum:")
print(curvature_obj_function(x_spline,y_spline,equidistant_times))



'''
SPLINE DEBUG
times = np.arange(gate_ct)
t = np.linspace(0,times[-1],500)
x_path_pts, y_path_pts = generate_path_t(x_spline, y_spline, t)
fig, ax = plt.subplots()
ax.scatter(gate_x_pts,gate_y_pts)
ax.plot(x_path_pts,y_path_pts)
ax.scatter(x_path_pts,y_path_pts)
plt.show()


fig, bx = plt.subplots()
bx.scatter(times,gate_x_pts)
bx.scatter(times,gate_y_pts)
bx.plot(t,x_spline(t))
bx.plot(t,y_spline(t))
plt.show()
'''

'''
PATH DEBUG
'''
print(f"Total path length: {total_length}")

distance_t = np.concat(([0], distance_t))
distance_times = np.linspace(0, gate_ct-1, len(distance_t))
fig, (cx, dx) = plt.subplots(1, 2)
cx.plot(distance_times, distance_t)
cx.set_xlabel("gate number")
cx.set_ylabel("cumulative distance")
dx.plot(distance_t, distance_times)
dx.set_xlabel("cumulative distance")
dx.set_ylabel("gate number")
plt.show()

'''
EQUIDISTANT FIT
'''
fig, pre_e = plt.subplots()
pre_e.plot(distance_t_full, dist_fit(distance_t_full))
pre_e.set_xlabel('cumulative distance (m)')
pre_e.set_ylabel('t (gate number)')
plt.show()

'''
MAP PLOT
'''
equidistant_times_clamped = np.clip(equidistant_times, 0, gate_ct-1)
x_path_pts, y_path_pts = generate_path_t(x_spline, y_spline, equidistant_times_clamped)

fig, ex = plt.subplots()
ex.plot(x_path_pts, y_path_pts, 'r-')
ex.scatter(x_path_pts, y_path_pts, s=5, c='r')
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
plt.show()

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
plt.show()

'''