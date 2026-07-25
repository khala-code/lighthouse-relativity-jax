"""
Lighthouse Relativity: Solver Module (JAX)
=========================================
Strict 3D scan loop across the OZJ Scale-Space Manifold.
Incorporates prime-locked T-axis indexing, CPT-V octant transformations, and zero control-flow branching.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

from src.operators import extended_bloch_rhs, soft_clamp_state, solve_dynamic_gr_poisson_metric
from src.cpt_v import CPTVOctantEngine


def load_topology_from_frame_zero(
    grid: Dict[str, Any],
    frame_zero_bundle: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Loads pre-wound topological state directly from Levain's FrameZeroBundle compiler output."""
    Nz = grid['Nz']
    
    # Extract complex phase field compiled by Levain baker and project into 3D scale layers
    base_phase = frame_zero_bundle.complex_phase_field
    
    # Broadcast across Nz scale layers with interlayer phase modulation
    s_layered = jnp.stack([
        jnp.real(base_phase),
        jnp.imag(base_phase),
        jnp.sqrt(jnp.maximum(1.0 - (jnp.abs(base_phase)**2), 0.05)) * jnp.ones_like(base_phase)
    ], axis=-1)
    
    # If Nz > 1, expand along the vertical scale axis
    if Nz > 1:
        s_layered = jnp.repeat(s_layered[:, :, jnp.newaxis, :], Nz, axis=2)
        k_indices = jnp.arange(Nz).reshape(1, 1, Nz, 1)
        s_layered = s_layered * jnp.cos(k_indices * jnp.pi / Nz)

    norm = jnp.linalg.norm(s_layered, axis=-1, keepdims=True)
    s_layered = s_layered / jnp.maximum(norm, 1e-8)

    u_drive_layered = jnp.zeros_like(s_layered)
    return s_layered, u_drive_layered


def run_simulation(
    grid: Dict[str, Any],
    s_init: jnp.ndarray,
    u: jnp.ndarray,
    Pi_V: jnp.ndarray,
    omega_larmor_field: jnp.ndarray,
    dt: float = 0.003,
    num_steps: int = 2000,
    Xi: float = 0.8,
    T1: float = 30.0,
    T2: float = 3.0,
    alpha: float = 0.3,
    H0: float = 0.0,
    omega_meta: float = 0.0,
    D_z: float = 0.05,
    lambda_scale: float = 0.02,
    noise_std: float = 0.005,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    w_larmor: float = 0.0,
    f_triad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    B1: float = 0.0,
    t_axis_start: float = 163.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pure 3D JAX scan loop integrating CPT-V octant transformations and mirror spiral interference."""
    X, Y = grid['X'], grid['Y']

    def step_fn(carry, step_idx):
        s_current, current_key = carry
        
        # Prime-locked temporal index progression along the T-axis
        t_step = t_axis_start + (step_idx * dt)

        # 1. Evaluate dynamic metric and extract complex phase proxy for octant classification
        Pi_V_dynamic = solve_dynamic_gr_poisson_metric(s_current, grid)
        Pi_V_eff = Pi_V + Pi_V_dynamic
        z_proxy = s_current[..., 0] + 1j * s_current[..., 1]
        positions = jnp.stack([X, Y, jnp.zeros_like(X)], axis=-1)

        # 2. CPT-V Octant Classification & Retrocausal Reflection Mapping
        octant_indices = CPTVOctantEngine.classify_octants(positions, z_proxy, Pi_V_eff)
        _, transformed_z, transformed_void = CPTVOctantEngine.apply_cptv_transformation(positions, z_proxy, Pi_V_eff)

        # 3. Prime-locked mirror spiral interference weighted by CPT-V sector parity
        prime_distance = jnp.abs(jnp.sin(t_step * jnp.pi / 7.0))
        octant_parity_modulation = jnp.mean(jnp.cos(octant_indices * jnp.pi / 4.0))
        mirror_interference_weight = 0.5 * (1.0 - jnp.cos(prime_distance * jnp.pi)) * jnp.abs(octant_parity_modulation)

        a_t = 1.0 + H0 * t_step
        H_t = H0 / jnp.maximum(a_t, 1e-4)
        omega_meta_t = omega_meta + (0.002 * mirror_interference_weight * jnp.mean(jnp.real(transformed_z)))
        noise_std_t = noise_std * (1.0 - 0.2 * mirror_interference_weight)

        current_key, subkey = jax.random.split(current_key)
        noise = noise_std_t * jax.random.normal(subkey, shape=s_current.shape)

        ds_dt = extended_bloch_rhs(
            s=s_current,
            u=u,
            Pi_V=Pi_V_eff,
            omega_larmor_field=omega_larmor_field,
            grid=grid,
            t_step=t_step,
            a_t=a_t,
            H_t=H_t,
            omega_meta_t=omega_meta_t,
            w_larmor=w_larmor,
            Xi=Xi,
            T1=T1,
            T2=T2,
            alpha=alpha,
            D_z=D_z,
            lambda_scale=lambda_scale,
            f_triad=f_triad,
            B1=B1,
        )

        s_next = soft_clamp_state(s_current + (ds_dt + noise) * dt)
        return (s_next, current_key), s_next

    initial_carry = (s_init, key)
    step_indices = jnp.arange(num_steps)

    print(f"⚡ Running CPT-V coupled 3D simulation grid (T0={t_axis_start}, {num_steps} steps, dt={dt})...")
    (final_carry, _), trajectory = jax.lax.scan(step_fn, initial_carry, step_indices)
    return trajectory, final_carry[0]