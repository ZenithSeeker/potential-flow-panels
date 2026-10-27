"""panelflow: a 2D Hess–Smith panel method for potential flow around arbitrary bodies."""

from . import geometry, influence, pressure, solver
from .panel import Panel, outline, panels_from_vertices
from .pressure import chord, integrate_forces, kutta_joukowski_cl, surface_cp
from .solver import induced_field, make_grid, solve
