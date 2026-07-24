"""
Lighthouse Relativity Engine
============================
Native 3D OZJ Scale-Space Manifold Engine.
"""

from .topology import (
    create_klein_grid,
    laplacian_klein,
    klein_bottle_3d,
)
from .operators import (
    extended_bloch_rhs,
    soft_clamp_state,
    create_quartz_lattice_with_al_impurities,
    compute_physical_larmor_field,
    solve_dynamic_gr_poisson_metric,
)
from .solver import (
    run_simulation,
    load_pre_wound_topology_via_action,
)

__all__ = [
    "create_klein_grid",
    "laplacian_klein",
    "klein_bottle_3d",
    "extended_bloch_rhs",
    "soft_clamp_state",
    "create_quartz_lattice_with_al_impurities",
    "compute_physical_larmor_field",
    "solve_dynamic_gr_poisson_metric",
    "run_simulation",
    "load_pre_wound_topology_via_action",
]