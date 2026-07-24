"""
Lighthouse Relativity: Solver Module (JAX)
=========================================
Strict 3D scan loop across the OZJ Scale-Space Manifold.
Executes multi-epoch cosmic evolution: Big Bang -> Inflation -> Recombination -> Cosmic Web.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any, Optional

from src.operators import extended_bloch_rhs, soft_clamp_state


def load_pre_wound_topology_via_action(
    grid: Dict[str, Any],
    action_preset_name: str = "flat_vacuum",
    key: jax.random.PRNGKey = jax.random.PRNGKey(0)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Action-driven initial state loader for the 3D OZJ tensor grid."""
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']

    if action_preset_name in ["early_universe_primordial", "full_cosmic_evolution"]:
        # Random isotropic spin directions across all 3D cells (Hot Big Bang Entropy)
        key_x, key_y, key_z = jax.random.split(key, 3)
        sx = jax.random.normal(key_x, shape=(Nx, Ny, Nz))
        sy = jax.random.normal(key_y, shape=(Nx, Ny, Nz))
        sz = jax.random.normal(key_z, shape=(Nx, Ny, Nz))

        s_layered = jnp.stack([sx, sy, sz], axis=-1)
        norm = jnp.linalg.norm(s_layered, axis=-1, keepdims=True)
        s_layered = s_layered / jnp.maximum(norm, 1e-8)

    elif action_preset_name == "pre_wound_cosmic":
        K = grid['K']
        s_layered = jnp.stack([
            0.5 * jnp.cos(2.0 * jnp.pi * K / Nz),
            0.5 * jnp.sin(2.0 * jnp.pi * K / Nz),
            jnp.sqrt(1.0 - 0.5**2) * jnp.ones_like(K)
        ], axis=-1)

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
    dt: float = 0.003,
    num_steps: int = 5000,
    Xi: float = 0.85,
    T1: float = 50.0,
    T2: float = 8.0,
    alpha: float = 0.2,
    H0: float = 1.2,
    omega_meta: float = 0.1,
    void_peak: float = 0.0,
    void_sigma: float = 1.0,
    noise_std: float = 0.04,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    omega_larmor_field: Optional[jnp.ndarray] = None,
    w_larmor: float = 0.0,
    f_triad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    B1: float = 0.0,
    is_merger: bool = False,
    is_full_evolution: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """3D JAX scan loop with multi-epoch cosmological parameter profiles."""
    X, Y = grid['X'], grid['Y']
    w_merger = 1.0 if is_merger else 0.0
    w_full_evo = 1.0 if is_full_evolution else 0.0

    def step_fn(carry, step_idx):
        s_current, current_key = carry
        t_step = step_idx * dt

        # Normalize total step fraction tau in [0, 1]
        tau = step_idx / jnp.maximum(float(num_steps), 1.0)

        # Smooth Epoch Weights using Hyperbolic Tangents (100% Branchless)
        # Epoch 1: Primordial (tau < 0.15)
        # Epoch 2: Inflation (0.15 <= tau <= 0.45)
        # Epoch 3: Cosmic Web / Classical GR (tau > 0.45)
        w_inf = 0.5 * (jnp.tanh((tau - 0.15) / 0.05) - jnp.tanh((tau - 0.45) / 0.05))
        w_late = 0.5 * (1.0 + jnp.tanh((tau - 0.45) / 0.05))

        # Dynamic Cosmological Parameter Profiles
        # Scale factor exponential during inflation, transitioning to power-law expansion
        a_inf = jnp.exp(H0 * jnp.maximum(tau - 0.15, 0.0) * 8.0)
        a_late = a_inf * (1.0 + 0.5 * (tau - 0.45))
        a_t_evo = (1.0 - w_inf - w_late) * 1.0 + w_inf * a_inf + w_late * a_late
        a_t_std = 1.0 + H0 * t_step
        a_t = w_full_evo * a_t_evo + (1.0 - w_full_evo) * a_t_std

        # Dynamic Hubble Parameter H(t)
        H_t_evo = w_inf * H0 + w_late * (0.15 / jnp.maximum(a_t, 1e-4))
        H_t_std = H0 / jnp.maximum(a_t_std, 1e-4)
        H_t = w_full_evo * H_t_evo + (1.0 - w_full_evo) * H_t_std

        # Meta-Clock Frequency Decay (High in Early Universe -> Locks in Classical GR Era)
        omega_meta_t = w_full_evo * (omega_meta * (1.0 - w_late) + 0.002 * w_late) + (1.0 - w_full_evo) * omega_meta

        # Thermal Fluctuation Decay (Dilutes post-inflation)
        noise_std_t = w_full_evo * (noise_std * (1.0 - 0.8 * w_late)) + (1.0 - w_full_evo) * noise_std

        current_key, subkey = jax.random.split(current_key)
        noise = noise_std_t * jax.random.normal(subkey, shape=s_current.shape)

        # Binary Inspiral Trajectories
        inspiral_rate = 2.0 / num_steps
        r_t = 1.5 * jnp.exp(-inspiral_rate * step_idx)
        theta_t = 0.05 * step_idx
        x1, y1 =  r_t * jnp.cos(theta_t),  r_t * jnp.sin(theta_t)
        x2, y2 = -r_t * jnp.cos(theta_t), -r_t * jnp.sin(theta_t)

        Pi1 = void_peak * jnp.exp(-(((X - x1)**2 + (Y - y1)**2) / (2.0 * (void_sigma**2))))
        Pi2 = void_peak * jnp.exp(-(((X - x2)**2 + (Y - y2)**2) / (2.0 * (void_sigma**2))))
        Pi_merger = Pi1 + Pi2
        Pi_V_dynamic = w_merger * Pi_merger + (1.0 - w_merger) * Pi_V

        ds_dt = extended_bloch_rhs(
            s=s_current,
            u=u,
            Pi_V=Pi_V_dynamic,
            grid=grid,
            t_step=t_step,
            a_t=a_t,
            H_t=H_t,
            omega_meta_t=omega_meta_t,
            omega_larmor_field=omega_larmor_field,
            w_larmor=w_larmor,
            Xi=Xi,
            T1=T1,
            T2=T2,
            alpha=alpha,
            f_triad=f_triad,
            B1=B1,
        )

        s_next = soft_clamp_state(s_current + (ds_dt + noise) * dt)
        return (s_next, current_key), s_next

    initial_carry = (s_init, key)
    step_indices = jnp.arange(num_steps)

    print(f"⚡ Running pure 3D multi-epoch simulation ({num_steps} steps, dt={dt})...")
    (final_carry, _), trajectory = jax.lax.scan(step_fn, initial_carry, step_indices)
    return trajectory, final_carry[0]