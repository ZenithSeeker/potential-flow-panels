"""NACA 0021: surface pressure compared with XFOIL, and the flow field at 10 degrees."""

import numpy as np
import matplotlib.pyplot as plt

import panelflow as pf
from panelflow.geometry import naca4
from panelflow.plotting import plot_speed_field
from panelflow.solver import induced_field_wind_axes, rotate

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, alpha_deg, ref in zip(axs, (0, 10), ("data/naca0021_zero_alpha.txt", "data/naca0021_alpha_10.txt")):
    alpha = np.radians(alpha_deg)
    geom = naca4("0021", 200)
    _, gamma = pf.solve(geom, alpha, kutta=True)
    cp = pf.surface_cp(geom, gamma, alpha=alpha)
    cl = pf.kutta_joukowski_cl(geom, gamma)
    cl = 0.0 if abs(cl) < 1e-9 else cl     # avoid printing -0.000

    x_ref, cp_ref = np.loadtxt(ref, unpack=True)
    ax.plot(x_ref, cp_ref, "-", color="tab:red", lw=1.2, label="XFOIL")
    ax.plot([p.xc for p in geom], cp, "o", ms=3, mfc="none", color="k", label="panelflow, 200 panels")
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("$C_p$")
    ax.set_title(f"NACA 0021, α = {alpha_deg}°   ($C_l$ = {cl:.3f})")
    ax.grid(alpha=0.3)
    ax.legend()
    print(f"alpha = {alpha_deg:2d} deg: Cl = {cl:.4f}")
fig.tight_layout()
fig.savefig("docs/naca0021_cp.png", dpi=150, bbox_inches="tight")

# flow field at 10 degrees, drawn in wind axes so the free stream is horizontal
alpha = np.radians(10)
geom = naca4("0021", 200)
_, gamma = pf.solve(geom, alpha, kutta=True)
X, Y = pf.make_grid((-0.6, 1.6), (-0.55, 0.45), N=250)
U, V = induced_field_wind_axes(geom, X, Y, alpha, gamma=gamma)
xs, ys = pf.outline(geom)
ax = plot_speed_field(X, Y, U, V, geom, body_outline=rotate(xs, ys, -alpha), levels=np.linspace(0, 2.0, 81))
ax.set_title("NACA 0021 at α = 10°: speed and streamlines")
plt.savefig("docs/naca0021_field.png", dpi=150, bbox_inches="tight")
plt.show()
