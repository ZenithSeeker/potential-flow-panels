"""Panel geometries. Every body is panelled clockwise, starting at the trailing edge (or angle 0)."""

import numpy as np
from PIL import Image
from scipy import ndimage

from .panel import panels_from_vertices


# ---------- cylinder
def cylinder(n_panels=64, R=1.0):
    """Circle of radius R, offset by half a panel so a panel face (not a vertex) sits at angle 0."""
    theta = np.pi / n_panels - np.linspace(0, 2 * np.pi, n_panels + 1)
    return panels_from_vertices(R * np.cos(theta), R * np.sin(theta))


# ---------- NACA 4-digit
def naca4(code="0012", n_panels=160, c=1.0):
    """NACA 4-digit section with cosine-spaced stations and a closed trailing edge."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = np.linspace(0, np.pi, n_panels // 2 + 1)
    x = 0.5 * (1 - np.cos(beta))         # clusters points at the leading and trailing edges

    # -0.1036 (rather than -0.1015) closes the trailing edge exactly
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0:
        fwd = x < p
        yc[fwd] = m / p ** 2 * (2 * p * x[fwd] - x[fwd] ** 2)
        yc[~fwd] = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x[~fwd] - x[~fwd] ** 2)
        dyc[fwd] = 2 * m / p ** 2 * (p - x[fwd])
        dyc[~fwd] = 2 * m / (1 - p) ** 2 * (p - x[~fwd])
    th = np.arctan(dyc)

    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)

    # lower surface TE -> LE, then upper LE -> TE
    xs = np.concatenate([xl[::-1], xu[1:]]) * c
    ys = np.concatenate([yl[::-1], yu[1:]]) * c
    return panels_from_vertices(xs, ys)


# ---------- Joukowski aerofoil
def joukowski_vertices(n_panels=100, x0=-0.1, y0=0.0, a=1.0):
    """Map a circle through z = zeta + a^2/zeta.

    The circle is centred at (x0, y0) and passes through zeta = a, which becomes the cusped trailing
    edge. x0 < 0 sets thickness, y0 > 0 sets camber.
    """
    R = np.hypot(a - x0, y0)
    beta0 = np.arctan2(-y0, a - x0)      # angle on the circle of the trailing-edge point
    beta = beta0 - np.linspace(0, 2 * np.pi, n_panels + 1)
    zeta = (x0 + R * np.cos(beta)) + 1j * (y0 + R * np.sin(beta))
    z = zeta + a ** 2 / zeta
    return z.real, z.imag


def joukowski(n_panels=100, x0=-0.1, y0=0.0, a=1.0):
    return panels_from_vertices(*joukowski_vertices(n_panels, x0, y0, a))


def joukowski_exact_cl(alpha, x0=-0.1, y0=0.0, a=1.0):
    """Exact lift coefficient from conformal mapping, using the aerofoil's actual chord."""
    R = np.hypot(a - x0, y0)
    beta = np.arcsin(y0 / R)             # zero-lift angle offset due to camber
    xs, _ = joukowski_vertices(2000, x0, y0, a)
    c = xs.max() - xs.min()
    return 8 * np.pi * R * np.sin(alpha + beta) / c


# ---------- any shape from a drawing
def _silhouette_mask(image_path, ink_threshold=200, close_size=5, open_size=3):
    """Largest connected, interior-filled silhouette in the image."""
    img = np.array(Image.open(image_path).convert("L"))
    ink = img < ink_threshold

    closed = ndimage.binary_closing(ink, structure=np.ones((close_size, close_size)))
    filled = ndimage.binary_fill_holes(closed)
    solid = ndimage.binary_opening(filled, structure=np.ones((open_size, open_size)))

    labeled, n = ndimage.label(solid)
    if n == 0:
        raise ValueError(f"no shape found in {image_path}, check ink_threshold")
    sizes = ndimage.sum(solid, labeled, range(1, n + 1))
    return labeled == (np.argmax(sizes) + 1)


def _outer_contour(mask):
    """The mask's outer boundary as (n, 2) points, borrowed from matplotlib's contouring."""
    import matplotlib.pyplot as plt

    rows, cols = mask.shape
    C, R = np.meshgrid(np.arange(cols), np.arange(rows))
    fig, ax = plt.subplots()
    cs = ax.contour(C, R, mask.astype(float), levels=[0.5])
    plt.close(fig)
    polygons = [poly for path in cs.get_paths() for poly in path.to_polygons()]
    if not polygons:
        raise ValueError("silhouette has no boundary to trace")
    return max(polygons, key=len)


def _smooth_closed(v, window):
    """Moving average around a closed (periodic) curve."""
    if window <= 0:
        return v
    n = len(v)
    padded = np.concatenate([v[-window:], v, v[:window]])
    kernel = np.ones(2 * window + 1) / (2 * window + 1)
    return np.convolve(padded, kernel, mode="same")[window:window + n]


def _resample_by_arclength(x, y, n):
    x_c, y_c = np.r_[x, x[0]], np.r_[y, y[0]]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_c), np.diff(y_c)))])
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.interp(s_new, s, x_c), np.interp(s_new, s, y_c)


def from_image(image_path, n_panels=350, ink_threshold=200, close_size=5, smooth_window=10, size=10.0):
    """Panel the outline of a line drawing.

    Threshold to ink, close small gaps, fill the interior, trace the outer contour, smooth it,
    resample to evenly spaced points, flip to y-up, centre, scale so the largest dimension is
    `size`, and orient clockwise.
    """
    mask = _silhouette_mask(image_path, ink_threshold, close_size)
    poly = _outer_contour(mask)
    x, y = poly[:-1, 0], poly[:-1, 1]

    x, y = _smooth_closed(x, smooth_window), _smooth_closed(y, smooth_window)
    x, y = _resample_by_arclength(x, y, n_panels)

    y = mask.shape[0] - y                # image rows increase downward
    x, y = x - x.mean(), y - y.mean()
    scale = size / (x.max() - x.min())
    x, y = x * scale, y * scale

    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0:   # positive area = anticlockwise
        x, y = x[::-1], y[::-1]

    # start at the rightmost point, so the "trailing edge" for the Kutta condition is at the back
    k = np.argmax(x)
    x, y = np.roll(x, -k), np.roll(y, -k)
    return panels_from_vertices(x, y, closed=True)
