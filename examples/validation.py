"""Check the solver against exact potential-flow solutions.

1. Cylinder surface pressure vs the exact Cp = 1 - 4 sin^2(theta)
2. Joukowski aerofoil lift vs the exact conformal-mapping result, symmetric and cambered
3. Convergence of the lift error with panel count
"""

import numpy as np
import matplotlib.pyplot as plt

import panelflow as pf
from panelflow import geometry as g

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

# ---------- cylinder
geom = g.cylinder(32)
pf.solve(geom, 0.0)
cp = pf.surface_cp(geom)
theta = np.degrees(np.arctan2([p.yc for p in geom], [p.xc for p in geom])) % 360
t = np.linspace(0, 360, 400)
axs[0].plot(t, 1 - 4 * np.sin(np.radians(t)) ** 2, "k-", lw=1, label="exact")
axs[0].plot(theta, cp, "o", mfc="none", label="32 panels")
axs[0].set_xlabel("θ [deg]")
axs[0].set_ylabel("$C_p$")
axs[0].set_title("Cylinder surface pressure")
axs[0].legend()

# ---------- Joukowski lift curves
alphas = np.radians(np.linspace(-4, 12, 9))
for y0, colour in ((0.0, "tab:blue"), (0.1, "tab:orange")):
    a_fine = np.radians(np.linspace(-4, 12, 100))
    axs[1].plot(np.degrees(a_fine), g.joukowski_exact_cl(a_fine, -0.1, y0), "-", color=colour, lw=1,
                label=f"exact, {'cambered' if y0 else 'symmetric'}")
    cls = []
    for a in alphas:
        geom = g.joukowski(400, -0.1, y0)
        _, gamma = pf.solve(geom, a, kutta=True)
        cls.append(pf.kutta_joukowski_cl(geom, gamma, c=pf.chord(geom)))
    axs[1].plot(np.degrees(alphas), cls, "o", mfc="none", color=colour, label="400 panels")
axs[1].set_xlabel("α [deg]")
axs[1].set_ylabel("$C_l$")
axs[1].set_title("Joukowski aerofoils: lift")
axs[1].legend(fontsize=9)

# ---------- convergence
alpha = np.radians(5)
exact = g.joukowski_exact_cl(alpha, -0.1, 0.1)
ns = np.array([25, 50, 100, 200, 400, 800, 1600])
err_kj, err_p = [], []
for n in ns:
    geom = g.joukowski(n, -0.1, 0.1)
    _, gamma = pf.solve(geom, alpha, kutta=True)
    c = pf.chord(geom)
    err_kj.append(abs(pf.kutta_joukowski_cl(geom, gamma, c=c) / exact - 1))
    err_p.append(abs(pf.integrate_forces(geom, pf.surface_cp(geom, gamma, alpha=alpha), alpha, c)[0] / exact - 1))
order = np.polyfit(np.log(ns[2:]), np.log(err_kj[2:]), 1)[0]
axs[2].loglog(ns, np.array(err_kj) * 100, "o-", label=f"circulation (slope {order:.2f})")
axs[2].loglog(ns, np.array(err_p) * 100, "s-", label="surface pressure integration")
axs[2].set_xlabel("Panels")
axs[2].set_ylabel("|$C_l$ error| [%]")
axs[2].set_title("Cambered Joukowski at α = 5°: convergence")
axs[2].legend()
print("lift error (circulation) by panel count:", dict(zip(ns.tolist(), np.round(np.array(err_kj) * 100, 2))))

for ax in axs:
    ax.grid(alpha=0.3, which="both")
axs[0].invert_yaxis()
fig.tight_layout()
fig.savefig("docs/validation.png", dpi=150, bbox_inches="tight")
plt.show()
