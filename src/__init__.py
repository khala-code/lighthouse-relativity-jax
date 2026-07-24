"""
Lighthouse Relativity Simulation Source Package
"""

from .topology import create_klein_grid, pad_klein_bc, laplacian_klein, klein_bottle_3d
from .operators import init_state_fields, dbi_radical, extended_bloch_rhs, soft_clamp_state
from .solver import run_simulation
from .visualization import plot_field_and_defects

__all__ = [
    "create_klein_grid",
    "pad_klein_bc",
    "laplacian_klein",
    "klein_bottle_3d",
    "init_state_fields",
    "dbi_radical",
    "extended_bloch_rhs",
    "soft_clamp_state",
    "run_simulation",
    "plot_field_and_defects",
]