"""Surface pressure and force coefficients from a solved panel geometry."""

import numpy as np

from .solver import induced_field


def field_cp(U, V, U_inf=1.0):
    """Incompressible Bernoulli: Cp = 1 - (V / V_inf)^2."""
    return 1 - (U ** 2 + V ** 2) / U_inf ** 2


def surface_velocity(geom, gamma=0.0, U_inf=1.0, alpha=0.0, eps=1e-6):
    """Velocity at each control point, evaluated a hair outside the body along the normal.

    Exactly on the panel the self-induced terms are discontinuous, so step just off it.
    """
    pts = np.array([[p.xc, p.yc] + eps * p.normal() for p in geom])
    return induced_field(geom, pts[:, 0], pts[:, 1], gamma, U_inf, alpha)


def surface_cp(geom, gamma=0.0, U_inf=1.0, alpha=0.0):
    u, v = surface_velocity(geom, gamma, U_inf, alpha)
    return field_cp(u, v, U_inf)


def integrate_forces(geom, Cp, alpha=0.0, c=1.0):
    """(Cl, Cd) from integrating -Cp * n over the surface, normalised by reference length c.

    In inviscid 2D flow Cd should come out ~0 (d'Alembert's paradox): a useful check on the solution.
    """
    lengths = np.array([p.length() for p in geom])
    normals = np.array([p.normal() for p in geom])
    Cx, Cy = np.sum(-Cp[:, None] * normals * lengths[:, None], axis=0) / c
    Cd = Cx * np.cos(alpha) + Cy * np.sin(alpha)
    Cl = Cy * np.cos(alpha) - Cx * np.sin(alpha)
    return Cl, Cd


def kutta_joukowski_cl(geom, gamma, U_inf=1.0, c=1.0):
    """Lift from the total circulation, Cl = 2 Gamma / (U c). Gamma > 0 (clockwise) gives lift."""
    circulation = gamma * sum(p.length() for p in geom)
    return 2 * circulation / (U_inf * c)


def chord(geom):
    xs = [p.x1 for p in geom]
    return max(xs) - min(xs)
