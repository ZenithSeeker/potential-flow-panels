import numpy as np
import pytest

from panelflow.influence import source_panel_velocity

PANEL = dict(x1=0.0, y1=4.0, x2=5.0, y2=1.0)


def panel_frame():
    theta = np.arctan2(PANEL["y2"] - PANEL["y1"], PANEL["x2"] - PANEL["x1"])
    tangent = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    length = np.hypot(PANEL["x2"] - PANEL["x1"], PANEL["y2"] - PANEL["y1"])
    midpoint = np.array([(PANEL["x1"] + PANEL["x2"]) / 2, (PANEL["y1"] + PANEL["y2"]) / 2])
    return tangent, normal, length, midpoint


def test_zero_strength_gives_zero_velocity():
    u, v = source_panel_velocity(0.0, 3.0, 3.0, **PANEL)
    assert u == 0.0
    assert v == 0.0


def test_self_induced_velocity_at_control_point():
    # Just above a panel's own control point, the
    # induced velocity should approach sigma/2 along the panel normal.
    tangent, normal, _, midpoint = panel_frame()
    px, py = midpoint + 1e-6 * normal
    u, v = source_panel_velocity(1.0, px, py, **PANEL)
    vel = np.array([u, v])
    assert np.dot(vel, normal) == pytest.approx(0.5, abs=1e-3)
    assert np.dot(vel, tangent) == pytest.approx(0.0, abs=1e-6)


def test_flow_points_away_from_panel_in_every_direction():
    # Regression test: an earlier atan2-based implementation produced
    # inward flow on roughly half the angular sweep.
    _, _, _, midpoint = panel_frame()
    for angle_deg in range(0, 360, 15):
        angle = np.radians(angle_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        for r in (3.0, 8.0):
            px, py = midpoint + r * direction
            u, v = source_panel_velocity(1.0, px, py, **PANEL)
            radial = u * direction[0] + v * direction[1]
            assert radial > 0, f"inward flow at angle={angle_deg}, r={r}"


def test_negative_sigma_is_a_sink():
    _, normal, _, midpoint = panel_frame()
    px, py = midpoint + 5.0 * normal
    u, v = source_panel_velocity(-1.0, px, py, **PANEL)
    assert np.dot([u, v], normal) < 0


def test_symmetric_about_panel():
    tangent, normal, _, midpoint = panel_frame()
    for d in (1.0, 4.0):
        p1 = midpoint + d * normal
        p2 = midpoint - d * normal
        vel1 = np.array(source_panel_velocity(1.0, p1[0], p1[1], **PANEL))
        vel2 = np.array(source_panel_velocity(1.0, p2[0], p2[1], **PANEL))
        # normal component flips sign across the panel, tangential doesn't
        assert np.dot(vel1, normal) == pytest.approx(-np.dot(vel2, normal), rel=1e-6)
        assert np.dot(vel1, tangent) == pytest.approx(np.dot(vel2, tangent), rel=1e-6)


def test_far_field_matches_point_source():
    # Far away, a finite source panel should look like a point source of
    # total strength sigma * L sitting at the panel midpoint.
    _, normal, length, midpoint = panel_frame()
    sigma = 1.0
    r = 500.0 * length
    px, py = midpoint + r * normal
    u, v = source_panel_velocity(sigma, px, py, **PANEL)
    speed = np.hypot(u, v)
    expected = sigma * length / (2 * np.pi * r)
    assert speed == pytest.approx(expected, rel=1e-3)


def test_linear_in_sigma():
    px, py = -3.0, 6.0
    u1, v1 = source_panel_velocity(1.0, px, py, **PANEL)
    u2, v2 = source_panel_velocity(2.5, px, py, **PANEL)
    assert u2 == pytest.approx(2.5 * u1)
    assert v2 == pytest.approx(2.5 * v1)


def test_vectorized_matches_scalar_calls():
    # Regression test: g_to_p @ pos used to break (or silently misbehave)
    # once X, Y stopped being scalars.
    X = np.array([[-7.0, -2.0, 6.0], [4.0, -8.0, 9.0]])
    Y = np.array([[-6.0, 5.0, -3.0], [9.0, 2.0, -7.0]])

    U, V = source_panel_velocity(1.0, X, Y, **PANEL)
    assert U.shape == X.shape
    assert V.shape == Y.shape

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u_s, v_s = source_panel_velocity(1.0, X[i, j], Y[i, j], **PANEL)
            assert U[i, j] == pytest.approx(u_s)
            assert V[i, j] == pytest.approx(v_s)
