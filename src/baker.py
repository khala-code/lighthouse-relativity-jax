"""
src/baker.py
Levain Euclidean Compiler & Frame Zero Generation Engine
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, NamedTuple

class OZJNode(NamedTuple):
    local_omega: int
    local_zeta_phase: float
    local_j_invariant: float

class DefectState(NamedTuple):
    z_coordinate: jnp.ndarray  # Complex phase-space coordinate Z = q + i*p
    octant_span_mask: int
    ozj_node: OZJNode

class FrameZeroBundle(NamedTuple):
    spatial_grid: Tuple[jnp.ndarray, jnp.ndarray]
    void_density_field: jnp.ndarray
    complex_phase_field: jnp.ndarray
    t_axis_current: float
    defects: list[DefectState]
    seam_config: Dict[str, Any]

class LevainBaker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sim_config = config.get("simulation", {})
        self.resolution = self.sim_config.get("resolution", [512, 512])
        self.seam_operator = self.sim_config.get("seam_operator", {})
        
        init_config = config.get("initial_conditions", {})
        self.t_axis_start = float(init_config.get("t_axis_start", 163.0))
        self.raw_defects = init_config.get("defect_clusters", [])

    def _generate_mesh(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generates the foundational 2D spatial coordinate grid centered at (0, 0)."""
        nx, ny = self.resolution
        Lx = self.sim_config.get("domain_extents", [6.283185, 6.283185, 10.0])[0]
        Ly = self.sim_config.get("domain_extents", [6.283185, 6.283185, 10.0])[1]
        
        x = jnp.linspace(-Lx / 2.0, Lx / 2.0, nx)
        y = jnp.linspace(-Ly / 2.0, Ly / 2.0, ny)
        return jnp.meshgrid(x, y, indexing='ij')

    def _mexican_hat_wavelet(self, X: jnp.ndarray, Y: jnp.ndarray, center: jnp.ndarray, sigma: float = 16.0) -> jnp.ndarray:
        """Computes a deterministic radial Mexican hat (Laplacian-of-Gaussian) template."""
        r_sq = (X - center[0].real)**2 + (Y - center[1].real)**2
        term = 1.0 - (r_sq / (sigma**2))
        return term * jnp.exp(-r_sq / (2.0 * sigma**2))

    def bake_frame_zero(self) -> FrameZeroBundle:
        """Compiles the declarative configuration into executable Frame Zero tensors."""
        X, Y = self._generate_mesh()
        
        # Initialize base Void Density field (Pi_V) and complex phase field (s)
        void_density = jnp.zeros(self.resolution, dtype=jnp.float64)
        complex_phase = jnp.zeros(self.resolution, dtype=jnp.complex128)
        
        compiled_defects = []
        
        for defect_def in self.raw_defects:
            # Parse unified complex phase-space coordinates Z = q + i*p safely
            z_raw = defect_def.get("complex_phase_space_z", [256.0 + 0.0j, 256.0 + 0.0j])
            
            def parse_z(val):
                if isinstance(val, complex):
                    return val
                if isinstance(val, (int, float)):
                    return complex(val, 0.0)
                if isinstance(val, str):
                    return complex(val.replace(" ", ""))
                return complex(val)

            z_coord = jnp.array([parse_z(z_raw[0]), parse_z(z_raw[1])], dtype=jnp.complex128)
            
            # Parse nested OZJ node attributes
            ozj_def = defect_def.get("nested_ozj_node", {})
            ozj_node = OZJNode(
                local_omega=int(ozj_def.get("local_omega", 1)),
                local_zeta_phase=float(ozj_def.get("local_zeta_phase", 0.5)),
                local_j_invariant=float(ozj_def.get("local_j_invariant", 1728.0))
            )
            
            # Stamp deterministic Mexican hat template onto the fields
            spatial_centroid = jnp.array([z_coord[0].real, z_coord[1].real])
            wavelet = self._mexican_hat_wavelet(X, Y, spatial_centroid)
            
            void_density += wavelet * jnp.abs(ozj_node.local_j_invariant / 1728.0)
            complex_phase += wavelet * jnp.exp(1j * z_coord[0].imag) # Momentum maps to initial phase angle
            
            compiled_defects.append(DefectState(
                z_coordinate=z_coord,
                octant_span_mask=int(defect_def.get("octant_span_mask", 0x01), 16) if isinstance(defect_def.get("octant_span_mask", 0x01), str) else int(defect_def.get("octant_span_mask", 0x01)),
                ozj_node=ozj_node
            ))

        return FrameZeroBundle(
            spatial_grid=(X, Y),
            void_density_field=void_density,
            complex_phase_field=complex_phase,
            t_axis_current=self.t_axis_start,
            defects=compiled_defects,
            seam_config=self.seam_operator
        )

def compile_simulation(config: Dict[str, Any]) -> FrameZeroBundle:
    """Public entry point for Levain compilation."""
    baker = LevainBaker(config)
    return baker.baker_frame_zero() if hasattr(baker, 'baker_frame_zero') else baker.bake_frame_zero()