import numpy as np
import matplotlib.pyplot as plt

from .panel import outline

STREAMLINE_DENSITY = 0.9


def _figsize(X, Y, width=7.0):
    """Fixed width, height matched to the plotted domain so equal-aspect axes fill the figure."""
    aspect = (np.max(Y) - np.min(Y)) / (np.max(X) - np.min(X))
    return width, width * aspect


def plot_streamlines(X, Y, U, V, geom, body_outline=None, ax=None):
    """Streamlines around the body. Pass body_outline=(xs, ys) to draw a rotated body for wind-axes plots."""
    xs, ys = body_outline if body_outline is not None else outline(geom)
    if ax is None:
        _, ax = plt.subplots(figsize=_figsize(X, Y))
    ax.streamplot(X, Y, U, V, color="0.3", density=STREAMLINE_DENSITY, linewidth=0.8)
    ax.fill(xs, ys, color="0.85", edgecolor="k", linewidth=1.2, zorder=5)
    ax.set_aspect("equal")
    return ax


def plot_speed_field(X, Y, U, V, geom, U_inf=1.0, levels=None, body_outline=None, ax=None):
    """Filled contours of local speed / free-stream speed, with streamlines on top."""
    xs, ys = body_outline if body_outline is not None else outline(geom)
    speed = np.hypot(U, V) / U_inf
    if levels is None:
        levels = np.linspace(0, 1.6, 81)
    if ax is None:
        fig, ax = plt.subplots(figsize=_figsize(X, Y))
    cs = ax.contourf(X, Y, speed, levels=levels, cmap="viridis", extend="max")
    ax.figure.colorbar(cs, ax=ax, label=r"$|V| / U_\infty$", shrink=0.8,
                       ticks=np.linspace(levels[0], levels[-1], 5))
    ax.streamplot(X, Y, U, V, color="w", density=STREAMLINE_DENSITY, linewidth=0.5, arrowsize=0.6)
    ax.fill(xs, ys, color="0.85", edgecolor="k", linewidth=1.0, zorder=5)
    ax.set_aspect("equal")
    return ax
