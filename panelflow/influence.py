"""Velocity induced by a single straight panel of constant source or vortex strength.

Both functions work in the panel's own frame (x' along the panel from (x1, y1), y' normal to it),
use the closed-form integrals for a constant-strength line element, then rotate back to global axes.
X and Y can be scalars or arrays of any shape.
"""

import numpy as np


def _to_panel_frame(X, Y, x1, y1, x2, y2):
    theta = np.arctan2(y2 - y1, x2 - x1)
    c, s = np.cos(theta), np.sin(theta)
    g_to_p = np.array([[c, s], [-s, c]])
    xp, yp = np.einsum("ij,j...->i...", g_to_p, np.array([X - x1, Y - y1]))
    length = np.hypot(x2 - x1, y2 - y1)
    return xp, yp, length, g_to_p


def _to_global(u_p, v_p, g_to_p):
    # rotation matrix, so its inverse is its transpose
    return np.einsum("ij,j...->i...", g_to_p.T, np.array([u_p, v_p]))


def source_panel_velocity(sigma, X, Y, x1, y1, x2, y2):
    """(u, v) induced at (X, Y) by a source panel of strength sigma per unit length."""
    xp, yp, L, g_to_p = _to_panel_frame(X, Y, x1, y1, x2, y2)
    with np.errstate(divide="ignore", invalid="ignore"):   # yp = 0 on the panel itself; the solver handles that limit
        u_p = sigma / (4 * np.pi) * np.log((xp ** 2 + yp ** 2) / ((xp - L) ** 2 + yp ** 2))
        v_p = sigma / (2 * np.pi) * (np.arctan(xp / yp) - np.arctan((xp - L) / yp))
    return _to_global(u_p, v_p, g_to_p)


def vortex_panel_velocity(gamma, X, Y, x1, y1, x2, y2):
    """(u, v) induced at (X, Y) by a vortex panel of strength gamma (positive clockwise)."""
    xp, yp, L, g_to_p = _to_panel_frame(X, Y, x1, y1, x2, y2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_p = gamma / (2 * np.pi) * (np.arctan(xp / yp) - np.arctan((xp - L) / yp))
        v_p = -gamma / (4 * np.pi) * np.log((xp ** 2 + yp ** 2) / ((xp - L) ** 2 + yp ** 2))
    return _to_global(u_p, v_p, g_to_p)
