"""
Lighthouse Relativity: Solver Module (JAX)
=========================================
Strict 3D scan loop across the OZJ Scale-Space Manifold.
100% branchless execution path from configuration to JIT scan.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

from src.operators import extended_bloch_rhs, soft_clamp_state, void_density_rhs, momentum_flow_rhs

def load_pre_wound_topology_via_action(
    grid: Dict[str, Any],
    action_preset_name: str = "flat_vacuum",
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Action-driven initial state loader.
    Bakes the Void Density (Pi_V) and Momentum Flow (v_flow) natively.
    """
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    X, Y = grid['X'], grid['Y']

    # Initialize default blank fields
    s_layered = jnp.zeros((Nx, Ny, Nz, 3)).at[..., 2].set(1.0)
    u_drive_layered = jnp.zeros_like(s_layered)
    Pi_V_layered = jnp.zeros((Nx, Ny, Nz))
    v_flow_layered = jnp.zeros((Nx, Ny, Nz, 3))

    if action_preset_name == "binary_merger_initial":
        r0 = 1.5
        x1, y1 =  r0, 0.0
        x2, y2 = -r0, 0.0

        r1_sq = (X - x1)**2 + (Y - y1)**2
        r2_sq = (X - x2)**2 + (Y - y2)**2
        sigma = 0.8  

        # 1. Prebake Spin Topology (Existing Lighthouse Defect Logic)
        phi1 = jnp.arctan2(Y - y1, X - x1)
        phi2 = jnp.arctan2(Y - y2, X - x2) + jnp.pi

        w1 = jnp.exp(-r1_sq / (2.0 * sigma**2))
        w2 = jnp.exp(-r2_sq / (2.0 * sigma**2))

        z_combined = (w1 * jnp.exp(1j * phi1) + w2 * jnp.exp(1j * phi2)) / jnp.maximum(w1 + w2, 1e-8)
        z_mag = jnp.abs(z_combined)
        z_safe = jnp.where(z_mag > 1e-5, z_combined / z_mag, 1.0 + 0j)

        amplitude_scale = 0.95
        sx = amplitude_scale * jnp.real(z_safe)
        sy = amplitude_scale * jnp.imag(z_safe)
        sz = jnp.sqrt(jnp.maximum(1.0 - (sx**2 + sy**2), 0.01))

        s_layered = jnp.stack([sx, sy, sz], axis=-1)
        norm = jnp.linalg.norm(s_layered, axis=-1, keepdims=True)
        s_layered = s_layered / jnp.maximum(norm, 1e-8)

        # 2. Prebake Initial Void Density (Pi_V)
        # X and Y are already (Nx, Ny, Nz), so Pi_V naturally evaluates to (Nx, Ny, Nz)
        void_peak = 18.0
        Pi_V_layered = void_peak * (jnp.exp(-r1_sq / (2.0 * 0.45**2)) + jnp.exp(-r2_sq / (2.0 * 0.45**2)))

        # 3. Prebake Momentum Flow Field (v_flow)
        omega = 2.0         
        lambda_in = 0.1     
        r_sq = X**2 + Y**2
        envelope = jnp.exp(-r_sq / (2.0 * 3.0**2)) 

        # vx, vy, and vz automatically evaluate to shape (Nx, Ny, Nz)
        vx = (-omega * Y - lambda_in * X) * envelope
        vy = ( omega * X - lambda_in * Y) * envelope
        vz = jnp.zeros_like(X)

        # Stack directly on the last axis to yield (Nx, Ny, Nz, 3)
        v_flow_layered = jnp.stack([vx, vy, vz], axis=-1)

    elif action_preset_name in ["early_universe_primordial", "full_cosmic_evolution"]:
        key_x, key_y, key_z = jax.random.split(key, 3)
        sx = jax.random.normal(key_x, shape=(Nx, Ny, Nz))
        sy = jax.random.normal(key_y, shape=(Nx, Ny, Nz))
        sz = jax.random.normal(key_z, shape=(Nx, Ny, Nz))

        s_layered = jnp.stack([sx, sy, sz], axis=-1)
        norm = jnp.linalg.norm(s_layered, axis=-1, keepdims=True)
        s_layered = s_layered / jnp.maximum(norm, 1e-8)

    return s_layered, u_drive_layered, Pi_V_layered, v_flow_layered


def run_simulation(
    grid: Dict[str, Any],
    s_init: jnp.ndarray,
    u: jnp.ndarray,
    Pi_V: jnp.ndarray,
    v_flow: jnp.ndarray,
    omega_larmor_field: jnp.ndarray,  # Required, non-optional
    dt: float = 0.003,
    num_steps: int = 2000,
    L_patch: float = 0.5,
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
    is_full_evolution: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pure 3D JAX scan loop with zero runtime conditional branching."""
    w_full_evo = 1.0 if is_full_evolution else 0.0

    def step_fn(carry, step_idx):
        s_current, Pi_V_current, v_flow_current, current_key = carry
        t_step = step_idx * dt

        tau = step_idx / jnp.maximum(float(num_steps), 1.0)

        w_inf = 0.5 * (jnp.tanh((tau - 0.15) / 0.05) - jnp.tanh((tau - 0.45) / 0.05))
        w_late = 0.5 * (1.0 + jnp.tanh((tau - 0.45) / 0.05))

        a_inf = jnp.exp(H0 * jnp.maximum(tau - 0.15, 0.0) * 8.0)
        a_late = a_inf * (1.0 + 0.5 * (tau - 0.45))
        a_t_evo = (1.0 - w_inf - w_late) * 1.0 + w_inf * a_inf + w_late * a_late
        a_t_std = 1.0 + H0 * t_step
        a_t = w_full_evo * a_t_evo + (1.0 - w_full_evo) * a_t_std

        H_t_evo = w_inf * H0 + w_late * (0.15 / jnp.maximum(a_t, 1e-4))
        H_t_std = H0 / jnp.maximum(a_t_std, 1e-4)
        H_t = w_full_evo * H_t_evo + (1.0 - w_full_evo) * H_t_std
        
        dPi_dt = void_density_rhs(
            Pi_V=Pi_V_current, 
            v_flow=v_flow, 
            grid=grid, 
            nu_diffusion=0.02
        )
        Pi_V_next = Pi_V_current + dPi_dt * dt

        dv_dt = momentum_flow_rhs(
            v_flow=v_flow_current, 
            Pi_V=Pi_V_current, 
            grid=grid, 
            nu_viscosity=0.02, 
            kappa_drag=0.15   # Tunes how strongly the mass pulls the flow
        )
        v_flow_next = v_flow_current + dv_dt * dt

        omega_meta_t = w_full_evo * (omega_meta * (1.0 - w_late) + 0.002 * w_late) + (1.0 - w_full_evo) * omega_meta
        noise_std_t = w_full_evo * (noise_std * (1.0 - 0.8 * w_late)) + (1.0 - w_full_evo) * noise_std

        current_key, subkey = jax.random.split(current_key)
        noise = noise_std_t * jax.random.normal(subkey, shape=s_current.shape)

        ds_dt = extended_bloch_rhs(
            s=s_current,
            u=u,
            Pi_V=Pi_V_next,
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
        return (s_next, Pi_V_next, v_flow_next, current_key), (s_next, Pi_V_next, v_flow_next)

    initial_carry = (s_init, Pi_V, v_flow, key)
    step_indices = jnp.arange(num_steps)

    print(f"⚡ Running pure 3D simulation grid ({num_steps} steps, dt={dt})...")
    (final_carry, trajectory_tuple) = jax.lax.scan(step_fn, initial_carry, step_indices)
    return trajectory_tuple, final_carry[0]