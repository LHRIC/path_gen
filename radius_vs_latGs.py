import numpy as np
import math
import matplotlib.pyplot as plt


velocity = np.linspace(0,30,100) #m/s
radius = np.zeros(len(velocity))
w = np.zeros(len(velocity))
ay = np.zeros(len(velocity))
total_g = np.zeros(len(velocity))
F_normal_arr = np.zeros(len(velocity))

vehicle_mass = 220 + 68 #kg

CLA = 3.6
CDA = 1.3

min_radius = 2.6 # meters
skidpad_event_radius = 15.25 /2 #

trackwidth = 1.24 #m
skidpad_real_radius = skidpad_event_radius + trackwidth/2 + 0.1 #100 mm spacing 

def main():
    for i, v in enumerate(velocity):
        F_normal_arr[i] = normal_force(v)

    for i, v in enumerate(velocity):
        F_n = F_normal_arr[i]
        drag = dyn_pressure(v) * CDA

        roll_resist = F_n * 0.025
        friction_drag = 500 #Newtons
        F_long = drag + roll_resist + friction_drag

        mu_long = F_long / F_n
        mu_lat = math.sqrt(mu_max(F_n)**2 - mu_long**2)

        F_lat = mu_lat * F_n

        ay_ms2 = F_lat / vehicle_mass

        ss_radius = v**2 / ay_ms2
        ss_w = v / ss_radius
        radius[i] = ss_radius
        w[i] = ss_w
        ay[i] = ay_ms2
        total_g[i] = math.sqrt(ay_ms2**2 + (F_long / vehicle_mass)**2) / 9.81



    fig, ax1 = plt.subplots()

    mask = radius >= min_radius
    ax1.plot(radius, w, color='tab:blue')
    ax1.set_ylim(0, w[mask].max() * 1.1)
    ax1.set_xlabel('Radius (m)')
    ax1.set_ylabel('Yaw Rate (rad/s)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(radius, ay / 9.81, color='tab:red')
    ax2.plot(radius, total_g, color='tab:red', linestyle='--')
    ax2.set_ylabel('Lateral Acceleration (G)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 3)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(radius, velocity, color='tab:green')
    ax3.set_ylabel('Velocity (m/s)', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    ax1.set_xlim(min_radius - 1, radius.max())
    ax1.axvline(min_radius, color='gray', linestyle='-', label=f'Min radius ({min_radius} m)')
    ax1.axvline(skidpad_real_radius, color='gray', linestyle='--', label=f'Skidpad radius ({skidpad_real_radius:.2f} m)')
    ax1.legend(loc='lower right')

    plt.title('Yaw Rate and Lateral Acceleration vs Corner Radius')
    plt.tight_layout()
    plt.show()

def dyn_pressure(v):
    rho = 1.2250 #kg/m^3
    return 0.5*rho*v**2

def normal_force(v):
    downforce = dyn_pressure(v) * CLA
    return vehicle_mass * 9.81 + downforce

def mu_max(F_normal):
    scale = 0.7
    coeffs = [-1.526946e-11, 2.198102e-08, -5.515655e-04, 2.714861e+00]  # deg3 poly fit from tire_model/R25B_12psi.tir
    return scale * np.polyval(coeffs, F_normal/3)

if __name__ == "__main__":
    main()

