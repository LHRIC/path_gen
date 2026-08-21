import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(__file__))
from mf_52 import MF52
from tirParse import tirParse

tir_params = tirParse(os.path.join(os.path.dirname(__file__), "R25B_12psi.tir"))
tire = MF52(tir_params, fxScaleFactor=1.0, fyScaleFactor=1.0, mzScaleFactor=1.0)

sa_sweep = np.linspace(math.radians(0.1), math.radians(10), 50)
fz_range = np.linspace(200, 2000, 40)

peak_mu = []
for fz in fz_range:
    fy_vals = [tire.Fy(fz, sa, 0) for sa in sa_sweep]
    peak_mu.append(max(abs(f) for f in fy_vals) / fz)

peak_mu = np.array(peak_mu)

# fit increasing degree until residual is small
for deg in [1, 2, 3]:
    coeffs = np.polyfit(fz_range, peak_mu, deg)
    residual = np.max(np.abs(np.polyval(coeffs, fz_range) - peak_mu))
    print(f"deg {deg}: max residual = {residual:.4f}  coeffs = {np.array2string(coeffs, precision=6)}")

# mu vs SA at several loads
fig, ax_sa = plt.subplots()
for fz in [200, 400, 800, 1200]:
    fy_vals = [tire.Fy(fz, sa, 0) / fz for sa in sa_sweep]
    ax_sa.plot(np.rad2deg(sa_sweep), fy_vals, label=f'Fz={fz}N')
ax_sa.set_xlabel('Slip Angle (deg)')
ax_sa.set_ylabel('mu (Fy/Fz)')
ax_sa.legend()
plt.title('mu vs SA at several loads')
plt.tight_layout()
plt.show()

# peak mu vs Fz with poly fit
coeffs3 = np.polyfit(fz_range, peak_mu, 3)
fig, ax = plt.subplots()
ax.plot(fz_range, peak_mu, label='MF52 peak mu')
ax.plot(fz_range, np.polyval(coeffs3, fz_range), '--', label='poly fit deg 3')
ax.set_xlabel('Normal Load Fz (N)')
ax.set_ylabel('Peak mu (Fy/Fz)')
ax.legend()
plt.title('Peak mu vs Normal Load - R25B 12psi')
plt.tight_layout()
plt.show()
