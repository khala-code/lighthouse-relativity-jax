"""
Lighthouse Relativity Engine
============================
Native 3D OZJ Scale-Space Manifold Engine with Levain Compiler & CPT-V Octant Architecture.
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
    load_topology_from_frame_zero,
    load_topology_from_frame_zero,
)
from .baker import (
    compile_simulation,
    LevainBaker,
    FrameZeroBundle,
    DefectState,
    OZJNode,
)
from .cpt_v import (
    CPTVOctantEngine,
)

__all__ = [
    # Topology
    "create_klein_grid",
    "laplacian_klein",
    "klein_bottle_3d",
    # Operators
    "extended_bloch_rhs",
    "soft_clamp_state",
    "create_quartz_lattice_with_al_impurities",
    "compute_physical_larmor_field",
    "solve_dynamic_gr_poisson_metric",
    # Solver
    "run_simulation",
    "load_topology_from_frame_zero",
    # Levain Baker
    "compile_simulation",
    "LevainBaker",
    "FrameZeroBundle",
    "DefectState",
    "OZJNode",
    # CPT-V Octant Engine
    "CPTVOctantEngine",
]