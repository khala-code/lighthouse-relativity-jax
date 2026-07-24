"""
Lighthouse Relativity: Operators & Field Dynamics Module (JAX)
============================================================
Implements the branchless continuous operators for Lighthouse Relativity:
  1. Initial state allocation (s0, drive field u, localized Void Density Pi_V).
  2. DBI radical soft-clamping for Void Density saturation: 1 / sqrt(1 + alpha * Pi_V^2).
  3. Non-linear Extended Bloch RHS differential operators:
     ds/dt = s x Omega_eff - Damping(T1, T2) + Xi * (s . u)(s x u) + noise
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

from src.topology import laplacian_klein


def init_state_fields(grid: Dict[str, Any], key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Initializes state vectors s(x, y), driving field u(x, y), and Void Density Pi_V(x, y)
    on the Klein bottle mesh.
    
    Parameters
    ----------
    grid : dict
        Grid metadata from create_klein_grid.
    key : jax.random.PRNGKey
        PRNG key for stochastic initialization.
        
    Returns
    -------
    state_0 : jnp.ndarray
        Initial spin vector field with shape (Nx, Ny, 3).
    drive_u : jnp.ndarray
        External drive vector field with shape (Nx, Ny, 3).
    void_density : jnp.ndarray
        Scalar Void Density field Pi_V with shape (Nx, Ny).
    """
    X, Y = grid['X'], grid['Y']
    Nx, Ny = grid['Nx'], grid['Ny']
    
    # Split PRNG keys for vector initialization
    k1, k2 = jax.random.split(key)
    
    # 1. Base Spin Vector Field s_0(x, y): Initialized near ground state (s_z ~ 1.0) with slight phase tilt
    theta = 0.1 * jnp.sin(X) * jnp.cos(Y)
    phi = X
    
    sx_0 = jnp.sin(theta) * jnp.cos(phi)
    sy_0 = jnp.sin(theta) * jnp.sin(phi)
    sz_0 = jnp.cos(theta)
    
    state_0 = jnp.stack([sx_0, sy_0, sz_0], axis=-1)
    
    # Add minor random noise perturbation to initialize non-orientable mode excitations
    state_0 = state_0 + 0.01 * jax.random.normal(k1, shape=state_0.shape)
    # Normalize state vector to unit sphere |s| = 1
    state_0 = state_0 / jnp.linalg.norm(state_0, axis=-1, keepdims=True)
    
    # 2. Drive Vector Field u(x, y): Transverse rotating pump along x-axis
    ux = jnp.cos(X)
    uy = jnp.sin(Y)
    uz = jnp.zeros_like(X)
    drive_u = jnp.stack([ux, uy, uz], axis=-1)
    drive_u = drive_u / jnp.linalg.norm(drive_u, axis=-1, keepdims=True)
    
    # 3. Void Density Field Pi_V(x, y): Concentrated near the Klein neck domain wall (y = Ly/2)
    y_center = grid['Ly'] / 2.0
    void_density = 2.5 * jnp.exp(-((Y - y_center) ** 2) / 0.5) + 0.1 * jax.random.uniform(k2, shape=(Nx, Ny))
    
    return state_0, drive_u, void_density


@jax.jit
def dbi_radical(Pi_V: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """
    Computes the continuous Dirac-Born-Infeld (DBI) saturation radical:
        gamma_D(Pi_V) = 1.0 / sqrt(1 + alpha * Pi_V^2)
        
    Prevents infinite void density accumulation and regulates friction coefficients.
    """
    return 1.0 / jnp.sqrt(1.0 + alpha * (Pi_V ** 2))


@jax.jit
def extended_bloch_rhs(
    s: jnp.ndarray,
    u: jnp.ndarray,
    Pi_V: jnp.ndarray,
    dx: float,
    dy: float,
    Xi: float = 0.5,
    T1: float = 10.0,
    T2: float = 2.0,
    alpha: float = 0.1,
    s0_z: float = 1.0,
    D_spatial: float = 0.05,
) -> jnp.ndarray:
    """
    Evaluates the continuous, branchless Right-Hand Side (ds/dt) of the Extended Bloch Equation:
    
        ds/dt = s x Omega_eff - Damping(T1, T2) + Xi * (s . u)(s x u) + D * grad^2(s)
        
    Parameters
    ----------
    s : jnp.ndarray
        Current spin field array with shape (Nx, Ny, 3).
    u : jnp.ndarray
        External drive field array with shape (Nx, Ny, 3).
    Pi_V : jnp.ndarray
        Void Density field with shape (Nx, Ny).
    dx, dy : float
        Grid spacing along x and y dimensions.
    Xi : float
        Non-linear torque coupling parameter.
    T1, T2 : float
        Longitudinal and transverse relaxation times.
    alpha : float
        DBI radical scaling parameter.
    s0_z : float
        Equilibrium longitudinal relaxation state (ground state z = 1.0).
    D_spatial : float
        Spatial spin-diffusion coefficient along the topology.
        
    Returns
    -------
    ds_dt : jnp.ndarray
        Rate of change array with shape (Nx, Ny, 3).
    """
    # 1. DBI Friction Modulation Factor
    gamma_dbi = dbi_radical(Pi_V, alpha)
    # Modulate effective damping times near high Void Density regions
    T1_eff = T1 / gamma_dbi[..., None]
    T2_eff = T2 / gamma_dbi[..., None]

    # 2. Effective Precession Frequency Vector Omega_eff
    # Base precession field pointing along z-axis plus localized Void pressure
    Omega_z = 1.0 + 0.2 * Pi_V
    Omega_eff = jnp.stack([jnp.zeros_like(Omega_z), jnp.zeros_like(Omega_z), Omega_z], axis=-1)

    # Precession torque: s x Omega_eff
    precession = jnp.cross(s, Omega_eff)

    # 3. Transverse (T2) and Longitudinal (T1) Dissipation
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    damping_x = sx / T2_eff[..., 0]
    damping_y = sy / T2_eff[..., 0]
    damping_z = (sz - s0_z) / T1_eff[..., 0]
    damping = jnp.stack([damping_x, damping_y, damping_z], axis=-1)

    # 4. Non-Linear Alignment Torque: Xi * (s . u) * (s x u)
    dot_su = jnp.sum(s * u, axis=-1, keepdims=True)
    cross_su = jnp.cross(s, u)
    nonlinear_torque = Xi * dot_su * cross_su

    # 5. Spatial Diffusion across Klein bottle topology (grad^2 s)
    # Applies spatial coupling between neighboring grid points with non-orientable BCs
    spatial_diff_x = laplacian_klein(s[..., 0], dx, dy)
    spatial_diff_y = laplacian_klein(s[..., 1], dx, dy)
    spatial_diff_z = laplacian_klein(s[..., 2], dx, dy)
    spatial_diffusion = D_spatial * jnp.stack([spatial_diff_x, spatial_diff_y, spatial_diff_z], axis=-1)

    # Combined continuous derivative
    ds_dt = precession - damping + nonlinear_torque + spatial_diffusion

    return ds_dt


@jax.jit
def soft_clamp_state(s: jnp.ndarray) -> jnp.ndarray:
    """
    Branchless norm normalization / soft-clamping using jnp.tanh to maintain unit length
    differentiably across non-linear iterations.
    """
    norm = jnp.linalg.norm(s, axis=-1, keepdims=True)
    # Smooth, branchless unit sphere projection
    return s / jnp.maximum(norm, 1e-8)