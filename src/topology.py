"""
Lighthouse Relativity: Topology Module (JAX)
=========================================
100% 3D OZJ Scale-Space Manifold. No 2D fallbacks.
Crossing the non-orientable y-neck twists vector parity AND shifts scale layer k.
Pre-computes spatial Fourier wavenumber grids for dynamic spectral GR solvers.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple


def create_klein_grid(
    Nx: int = 64, 
    Ny: int = 64, 
    Nz: int = 8, 
    Lx: float = 6.283185, 
    Ly: float = 6.283185, 
    Lz: float = 10.0
) -> Dict[str, Any]:
    """Creates a 3D mesh grid centered at domain centroid (0, 0, 0)."""
    dx, dy = Lx / Nx, Ly / Ny
    dz = Lz / Nz
    
    x = jnp.linspace(-Lx / 2.0, Lx / 2.0, Nx)
    y = jnp.linspace(-Ly / 2.0, Ly / 2.0, Ny)
    k = jnp.arange(Nz)
    
    X, Y, K = jnp.meshgrid(x, y, k, indexing='ij')

    # Pre-compute Fourier wavenumber grid K_sq_spatial for spectral Poisson GR metric solver
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(Nz, d=dz)
    Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
    K_sq_spatial = Kx**2 + Ky**2 + Kz**2
    
    return {
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'X': X, 'Y': Y, 'K': K,
        'x': x, 'y': y, 'k': k,
        'K_sq_spatial': K_sq_spatial,
    }


@jax.jit
def apply_non_orientable_neck_twist(field: jnp.ndarray) -> jnp.ndarray:
    """Applies P_y parity twist [1.0, -1.0, 1.0] across trailing vector dimension."""
    parity_vector = jnp.array([1.0, -1.0, 1.0])
    return field * parity_vector


@jax.jit
def laplacian_klein(s: jnp.ndarray, grid: Dict[str, Any]) -> jnp.ndarray:
    """
    Branchless 3D Laplacian for stacked OZJ Scale-Space tensors.
    Dynamically handles both 4D vector fields (spin) and 3D scalar fields (Void Density).
    """
    dx, dy, dz = grid['dx'], grid['dy'], grid['dz']

    # 1. Orientable X rolls and Scale-depth Z rolls
    s_xp = jnp.roll(s, shift=1, axis=0)
    s_xm = jnp.roll(s, shift=-1, axis=0)

    s_zp = jnp.roll(s, shift=1, axis=2)
    s_zm = jnp.roll(s, shift=-1, axis=2)

    # 2. Non-Orientable Neck Crossing (Y Roll + Z Layer Shift + Parity Twist)
    s_yp = jnp.roll(s, shift=1, axis=1)
    s_ym = jnp.roll(s, shift=-1, axis=1)

    mask_y_min = jnp.zeros_like(s)
    # Use Ellipsis (...) to handle any number of trailing dimensions automatically
    mask_y_min = mask_y_min.at[:, 0, ...].set(1.0)

    mask_y_max = jnp.zeros_like(s)
    mask_y_max = mask_y_max.at[:, -1, ...].set(1.0)

    # Shift layer index k on neck boundary crossing
    rolled_zm = jnp.roll(s, shift=(1, -1), axis=(1, 2))
    rolled_zp = jnp.roll(s, shift=(-1, 1), axis=(1, 2))

    # Only apply the vector parity twist if it is a 4D vector field
    if s.ndim == 4:
        s_yp_twisted_zm = apply_non_orientable_neck_twist(rolled_zm)
        s_ym_twisted_zp = apply_non_orientable_neck_twist(rolled_zp)
    else:
        # Scalar fields (like Pi_V) cross the neck without vector inversion
        s_yp_twisted_zm = rolled_zm
        s_ym_twisted_zp = rolled_zp

    s_yp_final = jnp.where(mask_y_min > 0.5, s_yp_twisted_zm, s_yp)
    s_ym_final = jnp.where(mask_y_max > 0.5, s_ym_twisted_zp, s_ym)

    # 3. Finite Difference Sum
    lap_x = (s_xp - 2.0 * s + s_xm) / (dx ** 2)
    lap_y = (s_yp_final - 2.0 * s + s_ym_final) / (dy ** 2)
    lap_z = (s_zp - 2.0 * s + s_zm) / (dz ** 2)

    return lap_x + lap_y + lap_z


def klein_bottle_3d(X: jnp.ndarray, Y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Parametric Figure-8 immersion mapping into R^3."""
    u = X
    v = Y
    r = 4.0 * (1.0 - jnp.cos(u) / 2.0)

    x_3d = jnp.where(
        u < jnp.pi,
        6.0 * jnp.cos(u) * (1.0 + jnp.sin(u)) + r * jnp.cos(u) * jnp.cos(v),
        6.0 * jnp.cos(u) * (1.0 + jnp.sin(u)) + r * jnp.cos(v + jnp.pi)
    )
    y_3d = jnp.where(
        u < jnp.pi,
        16.0 * jnp.sin(u) + r * jnp.sin(u) * jnp.cos(v),
        16.0 * jnp.sin(u)
    )
    z_3d = r * jnp.sin(v)

    return x_3d, y_3d, z_3d