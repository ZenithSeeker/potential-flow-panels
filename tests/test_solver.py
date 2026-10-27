"""Whole-solver checks against exact potential-flow results."""

import numpy as np
import pytest

import panelflow as pf
from panelflow import geometry as g


@pytest.mark.parametrize("n", [16, 64, 256])
def test_cylinder_matches_exact_surface_pressure(n):
    geom = g.cylinder(n)
    pf.solve(geom, 0.0)
    cp = pf.surface_cp(geom)
    theta = np.array([np.arctan2(p.yc, p.xc) for p in geom])
    assert np.allclose(cp, 1 - 4 * np.sin(theta) ** 2, atol=1e-4)


def test_cylinder_has_no_net_force():
    # d'Alembert's paradox
    geom = g.cylinder(64)
    pf.solve(geom, np.radians(20))
    Cl, Cd = pf.integrate_forces(geom, pf.surface_cp(geom, alpha=np.radians(20)), np.radians(20), c=2.0)
    assert abs(Cl) < 1e-10 and abs(Cd) < 1e-10


def test_net_source_strength_tends_to_zero():
    # a closed body neither creates nor swallows fluid; discretely that only holds in the limit
    leak = []
    for n in (60, 120, 240):
        geom = g.naca4("2412", n)
        sigma, _ = pf.solve(geom, np.radians(4), kutta=True)
        q = sigma * [p.length() for p in geom]
        leak.append(abs(q.sum()) / np.abs(q).sum())
    assert leak[-1] < 1e-3
    assert leak[0] > leak[1] > leak[2]


def test_no_flow_through_surface():
    geom = g.naca4("0012", 100)
    alpha = np.radians(6)
    _, gamma = pf.solve(geom, alpha, kutta=True)
    u, v = pf.induced_field(geom, *np.array([[p.xc, p.yc] + 1e-7 * p.normal() for p in geom]).T, gamma=gamma, alpha=alpha)
    normal_flow = u * [p.normal()[0] for p in geom] + v * [p.normal()[1] for p in geom]
    assert np.max(np.abs(normal_flow)) < 1e-5


@pytest.mark.parametrize("y0", [0.0, 0.1])
@pytest.mark.parametrize("alpha_deg", [0, 4, 8])
def test_joukowski_lift_converges_to_exact(y0, alpha_deg):
    alpha = np.radians(alpha_deg)
    exact = g.joukowski_exact_cl(alpha, x0=-0.1, y0=y0)
    geom = g.joukowski(600, x0=-0.1, y0=y0)
    _, gamma = pf.solve(geom, alpha, kutta=True)
    cl = pf.kutta_joukowski_cl(geom, gamma, c=pf.chord(geom))
    assert cl == pytest.approx(exact, rel=0.015, abs=1e-6)


def test_lift_error_shrinks_with_panel_count():
    alpha = np.radians(5)
    exact = g.joukowski_exact_cl(alpha, x0=-0.1, y0=0.1)
    errors = []
    for n in (50, 100, 200, 400):
        geom = g.joukowski(n, x0=-0.1, y0=0.1)
        _, gamma = pf.solve(geom, alpha, kutta=True)
        errors.append(abs(pf.kutta_joukowski_cl(geom, gamma, c=pf.chord(geom)) / exact - 1))
    assert all(a > b for a, b in zip(errors, errors[1:]))


def test_symmetric_aerofoil_at_zero_incidence_has_no_lift():
    geom = g.naca4("0021", 160)
    _, gamma = pf.solve(geom, 0.0, kutta=True)
    assert abs(gamma) < 1e-10


def test_camber_gives_lift_at_zero_incidence():
    geom = g.naca4("4412", 160)
    _, gamma = pf.solve(geom, 0.0, kutta=True)
    assert pf.kutta_joukowski_cl(geom, gamma) > 0.3


def test_lift_slope_close_to_thin_aerofoil_theory():
    # a thin symmetric section should approach 2*pi per radian (thickness raises it slightly)
    geom = g.naca4("0006", 200)
    cls = []
    for a in (0.0, np.radians(2)):
        _, gamma = pf.solve(geom, a, kutta=True)
        cls.append(pf.kutta_joukowski_cl(geom, gamma))
    slope = (cls[1] - cls[0]) / np.radians(2)
    assert 2 * np.pi < slope < 1.1 * 2 * np.pi


def test_image_tracer_produces_clockwise_closed_body():
    geom = g.from_image("data/cow.png", n_panels=200)
    assert len(geom) == 200
    xs, ys = np.array(pf.outline(geom))
    assert np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1]) < 0       # negative signed area = clockwise
    assert np.isclose(geom[-1].x2, geom[0].x1) and np.isclose(geom[-1].y2, geom[0].y1)
