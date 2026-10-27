"""Potential flow past a cow.

Any closed line drawing can be panelled: geometry.from_image traces the silhouette, and the
panel method does the rest. Non-lifting (no Kutta condition), since a cow has no sharp trailing edge.
"""

import matplotlib.pyplot as plt

import panelflow as pf
from panelflow.geometry import from_image
from panelflow.plotting import plot_speed_field

geom = from_image("data/cow.png", n_panels=400)
pf.solve(geom, alpha=0.0)

X, Y = pf.make_grid((-8, 8), (-4.5, 4.5), N=220)
U, V = pf.induced_field(geom, X, Y)

ax = plot_speed_field(X, Y, U, V, geom)
ax.set_title("Potential flow past a cow (400 panels)")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("docs/cow.png", dpi=150, bbox_inches="tight")
plt.show()
