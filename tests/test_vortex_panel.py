import numpy as np
import pytest

from panelflow.influence import vortex_panel_velocity

PANEL = dict(x1=-4.0, y1=0.0, x2=4.0, y2=0.0)


def panel_frame():
    theta = np.arctan2(PANEL["y2"] - PANEL["y1"], PANEL["x2"] - PANEL["x1"])
    tangent = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    length = np.hypot(PANEL["x2"] - PANEL["x1"], PANEL["y2"] - PANEL["y1"])
    midpoint = np.array([(PANEL["x1"] + PANEL["x2"]) / 2, (PANEL["y1"] + PANEL["y2"]) / 2])
    return tangent, normal, length, midpoint


def test_zero_strength_gives_zero_velocity():
    u, v = vortex_panel_velocity(0.0, 3.0, 3.0, **PANEL)
    assert u == 0.0
    assert v == 0.0


def test_self_induced_velocity_at_control_point():
    # Just above a panel's own control point, the TANGENTIAL
    # component jumps to gamma/2 (mirrors the source panel's normal-component jump),
    # while the NORMAL component self-induces exactly 0.
    tangent, normal, _, midpoint = panel_frame()
    px, py = midpoint + 1e-6 * normal
    u, v = vortex_panel_velocity(1.0, px, py, **PANEL)
    vel = np.array([u, v])
    assert np.dot(vel, tangent) == pytest.approx(0.5, abs=1e-3)
    assert np.dot(vel, normal) == pytest.approx(0.0, abs=1e-6)


def test_clockwise_circulation_on_perpendicular_bisector():
    # Along the panel's own perpendicular bisector, the normal component is
    # exactly 0 for any distance, and the tangential component has the
    # closed form gamma/pi * atan(L / (2r)) - derived directly from the
    # vortex-sheet integral, independent of any particular
    # implementation. Sign follows "gamma > 0 = clockwise".
    tangent, normal, length, midpoint = panel_frame()
    gamma = 1.0
    for r in (3.0, 10.0, 100.0):
        px, py = midpoint + r * normal
        u, v = vortex_panel_velocity(gamma, px, py, **PANEL)
        vel = np.array([u, v])
        expected_tangential = gamma / np.pi * np.arctan(length / (2 * r))
        assert np.dot(vel, tangent) == pytest.approx(expected_tangential, rel=1e-6)
        assert np.dot(vel, normal) == pytest.approx(0.0, abs=1e-9)


def test_flow_is_rotational_not_radial_far_from_panel():
    # Far away, a vortex panel should look like a point vortex: velocity
    # nearly perpendicular to the radius vector from the panel midpoint,
    # i.e. negligible radial component relative to speed.
    _, _, length, midpoint = panel_frame()
    r = 20.0 * length
    for angle_deg in range(0, 360, 15):
        angle = np.radians(angle_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        px, py = midpoint + r * direction
        u, v = vortex_panel_velocity(1.0, px, py, **PANEL)
        speed = np.hypot(u, v)
        radial = u * direction[0] + v * direction[1]
        assert speed > 0
        assert abs(radial) / speed < 0.01, f"too radial at angle={angle_deg}"


def test_antisymmetric_about_panel():
    # Opposite of the source panel's symmetry: for a vortex panel the TANGENTIAL
    # component is odd across the panel (flips sign) and the NORMAL
    # component is even (unchanged).
    tangent, normal, _, midpoint = panel_frame()
    for d in (1.0, 4.0):
        p1 = midpoint + d * normal
        p2 = midpoint - d * normal
        vel1 = np.array(vortex_panel_velocity(1.0, p1[0], p1[1], **PANEL))
        vel2 = np.array(vortex_panel_velocity(1.0, p2[0], p2[1], **PANEL))
        assert np.dot(vel1, tangent) == pytest.approx(-np.dot(vel2, tangent), rel=1e-6)
        assert np.dot(vel1, normal) == pytest.approx(np.dot(vel2, normal), rel=1e-6)


def test_linear_in_gamma():
    px, py = -3.0, 6.0
    u1, v1 = vortex_panel_velocity(1.0, px, py, **PANEL)
    u2, v2 = vortex_panel_velocity(2.5, px, py, **PANEL)
    assert u2 == pytest.approx(2.5 * u1)
    assert v2 == pytest.approx(2.5 * v1)


def test_vectorized_matches_scalar_calls():
    X = np.array([[-7.0, -2.0, 6.0], [4.0, -8.0, 9.0]])
    Y = np.array([[-6.0, 5.0, -3.0], [9.0, 2.0, -7.0]])

    U, V = vortex_panel_velocity(1.0, X, Y, **PANEL)
    assert U.shape == X.shape
    assert V.shape == Y.shape

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u_s, v_s = vortex_panel_velocity(1.0, X[i, j], Y[i, j], **PANEL)
            assert U[i, j] == pytest.approx(u_s)
            assert V[i, j] == pytest.approx(v_s)
