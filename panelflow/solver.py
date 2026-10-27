"""Hess–Smith panel method.

Each panel carries its own constant source strength sigma_j, and every panel shares one constant
vortex strength gamma. The unknowns are found from:

- flow tangency (zero normal velocity) at every control point: N equations
- the Kutta condition: equal and opposite tangential velocity on the first and last panels, i.e.
  the flow leaves the trailing edge smoothly: 1 equation (only for lifting bodies)

Without the Kutta condition this reduces to a pure source method, which is right for non-lifting
bodies like a cylinder.
"""

import numpy as np

from .influence import source_panel_velocity, vortex_panel_velocity


def _control_points(geom):
    return np.array([p.xc for p in geom]), np.array([p.yc for p in geom])


def assemble_system(geom, alpha, U_inf=1.0, kutta=False):
    """Influence matrix A and right-hand side b. Size N, or N+1 with the Kutta condition."""
    N = len(geom)
    size = N + 1 if kutta else N
    A = np.zeros((size, size))
    b = np.zeros(size)

    xc, yc = _control_points(geom)
    normals = np.array([p.normal() for p in geom])
    tangents = np.array([p.tangential() for p in geom])

    for j, panel in enumerate(geom):
        # velocity at every control point from unit-strength panel j, in one vectorised call
        u_s, v_s = source_panel_velocity(1.0, xc, yc, panel.x1, panel.y1, panel.x2, panel.y2)
        A[:N, j] = u_s * normals[:, 0] + v_s * normals[:, 1]
        # a source panel pushes flow straight out of its own control point at sigma/2
        A[j, j] = 0.5

        if kutta:
            u_v, v_v = vortex_panel_velocity(1.0, xc, yc, panel.x1, panel.y1, panel.x2, panel.y2)
            vn = u_v * normals[:, 0] + v_v * normals[:, 1]
            vn[j] = 0.0                     # a vortex panel induces no normal velocity on itself
            A[:N, N] += vn

            A[N, j] = (u_s[0] * tangents[0, 0] + v_s[0] * tangents[0, 1]
                       + u_s[-1] * tangents[-1, 0] + v_s[-1] * tangents[-1, 1])

            # The tangential self-term is discontinuous exactly at the control point, so evaluating
            # it there gives a sign set by round-off. Use the analytic limit (gamma/2) instead.
            vt_first = 0.5 if j == 0 else u_v[0] * tangents[0, 0] + v_v[0] * tangents[0, 1]
            vt_last = 0.5 if j == N - 1 else u_v[-1] * tangents[-1, 0] + v_v[-1] * tangents[-1, 1]
            A[N, N] += vt_first + vt_last

    thetas = np.array([p.theta() for p in geom])
    b[:N] = -U_inf * np.sin(alpha - thetas)
    if kutta:
        b[N] = -U_inf * (np.cos(alpha - thetas[0]) + np.cos(alpha - thetas[-1]))
    return A, b


def solve(geom, alpha, U_inf=1.0, kutta=False):
    """Solve for panel strengths and store sigma on each panel. Returns (sigma, gamma)."""
    A, b = assemble_system(geom, alpha, U_inf, kutta)
    sol = np.linalg.solve(A, b)
    sigma = sol[:len(geom)]
    gamma = sol[-1] if kutta else 0.0
    for panel, s in zip(geom, sigma):
        panel.sigma = s
    return sigma, gamma


def induced_field(geom, X, Y, gamma=0.0, U_inf=1.0, alpha=0.0):
    """Total velocity (free stream + all panels) at points X, Y of any shape."""
    U = np.full(np.shape(X), U_inf * np.cos(alpha), dtype=float)
    V = np.full(np.shape(X), U_inf * np.sin(alpha), dtype=float)
    for p in geom:
        u, v = source_panel_velocity(p.sigma, X, Y, p.x1, p.y1, p.x2, p.y2)
        U += u
        V += v
        if gamma:
            u, v = vortex_panel_velocity(gamma, X, Y, p.x1, p.y1, p.x2, p.y2)
            U += u
            V += v
    return U, V


def make_grid(xlim, ylim, N=100):
    x = np.linspace(*xlim, N)
    y = np.linspace(*ylim, N)
    return np.meshgrid(x, y)


def rotate(x, y, angle):
    """Rotate point(s) or vector(s) counter-clockwise by angle (rad)."""
    c, s = np.cos(angle), np.sin(angle)
    x, y = np.asarray(x), np.asarray(y)
    return c * x - s * y, s * x + c * y


def induced_field_wind_axes(geom, X_wind, Y_wind, alpha, gamma=0.0, U_inf=1.0):
    """Velocity on a grid given in wind axes (free stream horizontal), for nicer plots at incidence.

    The solution lives in body axes, so query there and rotate the result back.
    """
    X_body, Y_body = rotate(X_wind, Y_wind, alpha)
    U_body, V_body = induced_field(geom, X_body, Y_body, gamma, U_inf, alpha)
    return rotate(U_body, V_body, -alpha)
