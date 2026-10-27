import numpy as np


class Panel:
    """Straight panel from (x1, y1) to (x2, y2), with its control point at the midpoint."""

    def __init__(self, x1, y1, x2, y2, sigma=0.0):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.xc, self.yc = (x1 + x2) / 2, (y1 + y2) / 2
        self.sigma = sigma

    def theta(self):
        return np.arctan2(self.y2 - self.y1, self.x2 - self.x1)

    def length(self):
        return np.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def normal(self):
        # outward for a body panelled clockwise
        theta = self.theta()
        return np.array([-np.sin(theta), np.cos(theta)])

    def tangential(self):
        theta = self.theta()
        return np.array([np.cos(theta), np.sin(theta)])


def panels_from_vertices(xs, ys, closed=False):
    """Panels joining consecutive vertices. Bodies must run clockwise, starting at the trailing edge.

    closed=True joins the last vertex back to the first; otherwise the caller supplies the
    trailing-edge point twice (first and last), as for aerofoils.
    """
    n = len(xs) if closed else len(xs) - 1
    return [Panel(xs[i], ys[i], xs[(i + 1) % len(xs)], ys[(i + 1) % len(ys)]) for i in range(n)]


def outline(geom):
    """Closed (xs, ys) of the body, for plotting."""
    xs = [p.x1 for p in geom] + [geom[0].x1]
    ys = [p.y1 for p in geom] + [geom[0].y1]
    return xs, ys
