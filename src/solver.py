"""
Lighthouse Relativity: Solver Module (JAX)
=========================================
Strict 3D scan loop across the OZJ Scale-Space Manifold.
100% branchless execution path from configuration to JIT scan.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

from src.operators import extended_bloch_rhs, soft_clamp_state


def load_pre_wound_topology_via_action(
    grid: Dict[str, Any],
    action_preset_name: str = "flat_vacuum",
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Action-driven initial state loader for pre-winding topological defects."""
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    X, Y = grid['X'], grid['Y']

    if action_preset_name == "binary_merger_initial":
        r0 = 1.5
        x1, y1 =  r0, 0.0
        x2, y2 = -r0, 0.0

        r1_sq = (X - x1)**2 + (Y - y1)**2
        r2_sq = (X - x2)**2 + (Y - y2)**2
        sigma = 0.5

        phi1 = jnp.arctan2(Y - y1, X - x1)
        phi2 = jnp.arctan2(Y - y2, X - x2) + jnp.pi

        w1 = jnp.exp(-r1_sq / (2.0 * sigma**2))
        w2 = jnp.exp(-r2_sq / (2.0 * sigma**2))

        sx = w1 * jnp.cos(phi1) + w2 * jnp.cos(phi2)
        sy = w1 * jnp.sin(phi1) + w2 * jnp.sin(phi2)
        sz = jnp.sqrt(jnp.maximum(1.0 - (sx**2 + sy**2), 0.05))

        s_layered = jnp.stack([sx, sy, sz], axis=-1)
        norm = jnp.linalg.norm(s_layered, axis=-1, keepdims=True)
        s_layered = s_layered / jnp.maximum(norm, 1e-8)

    elif action_preset_name in ["early_universe_primordial", "full_cosmic_evolution"]:
        key_x, key_y, key_z = jax.random.split(key, 3)
        sx = jax.random.normal(key_x, shape=(Nx, Ny, Nz))
        sy = jax.random.normal(key_y, shape=(Nx, Ny, Nz))
        sz = jax.random.normal(key_z, shape=(Nx, Ny, Nz))

        s_layered = jnp.stack([sx, sy, sz], axis=-1)
        norm = jnp.linalg.norm(s_layered, axis=-1, keepdims=True)
        s_layered = s_layered / jnp.maximum(norm, 1e-8)

    else:
        s_layered = jnp.zeros((Nx, Ny, Nz, 3))
        s_layered = s_layered.at[..., 2].set(1.0)

    u_drive_layered = jnp.zeros_like(s_layered)
    return s_layered, u_drive_layered


def run_simulation(
    grid: Dict[str, Any],
    s_init: jnp.ndarray,
    u: jnp.ndarray,
    Pi_V: jnp.ndarray,
    omega_larmor_field: jnp.ndarray,  # Required, non-optional
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
    is_full_evolution: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pure 3D JAX scan loop with zero runtime conditional branching."""
    w_full_evo = 1.0 if is_full_evolution else 0.0

    def step_fn(carry, step_idx):
        s_current, current_key = carry
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

        omega_meta_t = w_full_evo * (omega_meta * (1.0 - w_late) + 0.002 * w_late) + (1.0 - w_full_evo) * omega_meta
        noise_std_t = w_full_evo * (noise_std * (1.0 - 0.8 * w_late)) + (1.0 - w_full_evo) * noise_std

        current_key, subkey = jax.random.split(current_key)
        noise = noise_std_t * jax.random.normal(subkey, shape=s_current.shape)

        ds_dt = extended_bloch_rhs(
            s=s_current,
            u=u,
            Pi_V=Pi_V,
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

    print(f"⚡ Running pure 3D simulation grid ({num_steps} steps, dt={dt})...")
    (final_carry, _), trajectory = jax.lax.scan(step_fn, initial_carry, step_indices)
    return trajectory, final_carry[0]