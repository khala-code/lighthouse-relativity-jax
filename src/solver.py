"""
Lighthouse Relativity: Simulation Solver Module (JAX)
===================================================
Executes the branchless, time-dependent integration of the Extended Bloch equations
across the non-orientable Klein bottle mesh using `jax.lax.scan`.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

from src.operators import extended_bloch_rhs, soft_clamp_state


def run_simulation(
    grid: Dict[str, Any],
    s_init: jnp.ndarray,
    u: jnp.ndarray,
    Pi_V: jnp.ndarray,
    dt: float = 0.01,
    num_steps: int = 1000,
    Xi: float = 0.5,
    T1: float = 10.0,
    T2: float = 2.0,
    alpha: float = 0.1,
    noise_std: float = 0.02,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Runs the full time-stepping simulation using `jax.lax.scan` for maximum compiled hardware speed.
    
    Parameters
    ----------
    grid : dict
        Grid metadata containing spatial steps dx, dy.
    s_init : jnp.ndarray
        Initial spin field with shape (Nx, Ny, 3).
    u : jnp.ndarray
        External driving field with shape (Nx, Ny, 3).
    Pi_V : jnp.ndarray
        Void Density field with shape (Nx, Ny).
    dt : float
        Integration time step.
    num_steps : int
        Total number of simulation steps.
    Xi : float
        Non-linear torque coupling parameter.
    T1, T2 : float
        Longitudinal and transverse relaxation parameters.
    alpha : float
        DBI saturation coefficient.
    noise_std : float
        Standard deviation of the stochastic driver eta(t).
    key : jax.random.PRNGKey
        PRNG key for stochastic integration.
        
    Returns
    -------
    trajectory : jnp.ndarray
        Time history of the spin field array with shape (num_steps, Nx, Ny, 3).
    final_state : jnp.ndarray
        Final spin field state with shape (Nx, Ny, 3).
    """
    dx, dy = grid['dx'], grid['dy']

    # Define the single-step integration function for jax.lax.scan
    def step_fn(carry, step_idx):
        s_current, current_key = carry

        # Split PRNG key for stochastic driver eta(t)
        current_key, subkey = jax.random.split(current_key)

        # Stochastic driving noise term eta(t) adhering to Fluctuation-Dissipation theorem
        noise = noise_std * jax.random.normal(subkey, shape=s_current.shape)

        # Compute Right-Hand Side ds/dt
        ds_dt = extended_bloch_rhs(
            s=s_current,
            u=u,
            Pi_V=Pi_V,
            dx=dx,
            dy=dy,
            Xi=Xi,
            T1=T1,
            T2=T2,
            alpha=alpha,
        )

        # Heun/Euler integration update with noise
        s_next = s_current + (ds_dt + noise) * dt

        # Apply branchless soft-clamping / unit sphere normalization
        s_next = soft_clamp_state(s_next)

        return (s_next, current_key), s_next

    # Execute the compiled loop natively across GPU/CPU hardware
    initial_carry = (s_init, key)
    step_indices = jnp.arange(num_steps)

    (final_carry, _), trajectory = jax.lax.scan(step_fn, initial_carry, step_indices)
    final_state = final_carry[0]

    return trajectory, final_state