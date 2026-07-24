"""
Lighthouse Relativity: Topology Module (JAX)
============================================
Defines the non-orientable Klein bottle coordinate grid, 3D parametric surface
embeddings, and non-orientable boundary condition wrappers (pad_klein_bc)
implementing the fundamental topological identification:

    a * b * a^(-1) * b = 1
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any


def klein_bottle_3d(X: jnp.ndarray, Y: jnp.ndarray, r: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Parametric 3D immersion of the Klein bottle (Figure-8 parametrization).
    
    Maps 2D grid coordinates (X, Y) in [0, 2pi] x [0, 2pi] into 3D Cartesian coordinates
    (X_3d, Y_3d, Z_3d) for surface visualization and metric distance calculations.
    """
    # u in [0, 2pi], v in [0, 2pi]
    u = X
    v = Y
    
    # Figure-8 immersion equations
    r_u = r * (1.16 + jnp.cos(u))
    
    X_3d = r_u * jnp.cos(v)
    Y_3d = r_u * jnp.sin(v)
    Z_3d = r * jnp.sin(u) * jnp.sin(v / 2.0) + r * jnp.sin(2.0 * u) * jnp.cos(v / 2.0)
    
    return X_3d, Y_3d, Z_3d


def create_klein_grid(Nx: int = 128, Ny: int = 128, Lx: float = 2.0 * jnp.pi, Ly: float = 2.0 * jnp.pi) -> Dict[str, Any]:
    """
    Initializes the spatial coordinate grid and topological metadata for the Klein bottle.
    
    Parameters
    ----------
    Nx, Ny : int
        Number of grid points along x (orientable) and y (non-orientable) axes.
    Lx, Ly : float
        Domain lengths along x and y dimensions.
        
    Returns
    -------
    dict
        Dictionary containing 1D coordinates, 2D meshgrids, spatial steps (dx, dy),
        and 3D parametric embedding coordinates.
    """
    dx = Lx / Nx
    dy = Ly / Ny
    
    x = jnp.linspace(0.0, Lx - dx, Nx)
    y = jnp.linspace(0.0, Ly - dy, Ny)
    
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # Generate 3D surface coordinates for visualization/metric calculation
    X_3d, Y_3d, Z_3d = klein_bottle_3d(X, Y)
    
    grid = {
        'Nx': Nx,
        'Ny': Ny,
        'Lx': Lx,
        'Ly': Ly,
        'dx': dx,
        'dy': dy,
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'X_3d': X_3d,
        'Y_3d': Y_3d,
        'Z_3d': Z_3d,
    }
    
    return grid


@jax.jit
def pad_klein_bc(field: jnp.ndarray) -> jnp.ndarray:
    """
    Applies non-orientable Klein bottle boundary conditions by padding a 2D (or trailing 2D) array
    with 1 ghost cell on all sides.
    
    Topological Identifications:
      - x-axis (orientable loop): Periodic boundary condition
            field[x + Lx, y] = field[x, y]
      - y-axis (non-orientable loop / Klein neck): Parity reversal twist
            field[x, y + Ly] = field[Lx - x, y]
            
    Parameters
    ----------
    field : jnp.ndarray
        Field array with spatial shape (Nx, Ny) or (..., Nx, Ny).
        
    Returns
    -------
    jnp.ndarray
        Padded field array with shape (..., Nx + 2, Ny + 2).
    """
    # Extract spatial dimensions (assumes last two axes are Nx, Ny)
    Nx, Ny = field.shape[-2], field.shape[-1]
    
    # 1. Handle x-axis periodic wrapping (left/right padding)
    left_ghost = field[..., -1:, :]   # x = Nx-1 wraps to x_pad = 0
    right_ghost = field[..., :1, :]   # x = 0 wraps to x_pad = Nx+1
    field_x_padded = jnp.concatenate([left_ghost, field, right_ghost], axis=-2)
    
    # 2. Handle y-axis non-orientable wrapping with x-parity flip (bottom/top padding)
    # Bottom ghost row (y = -1): wraps from y = Ny-1 with x flipped along spatial axis -2
    bottom_ghost = jnp.flip(field_x_padded[..., :, -1:], axis=-2)
    
    # Top ghost row (y = Ny): wraps from y = 0 with x flipped along spatial axis -2
    top_ghost = jnp.flip(field_x_padded[..., :, :1], axis=-2)
    
    # Concatenate along y-axis (axis -1)
    full_padded_field = jnp.concatenate([bottom_ghost, field_x_padded, top_ghost], axis=-1)
    
    return full_padded_field


@jax.jit
def laplacian_klein(field: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Computes the 2D spatial Laplacian operator (grad^2 field) on a Klein bottle topology
    using a 5-point finite-difference stencil with non-orientable boundary padding.
    """
    padded = pad_klein_bc(field)
    
    # Center, Left, Right, Bottom, Top interior slices
    center = padded[..., 1:-1, 1:-1]
    left   = padded[..., :-2, 1:-1]
    right  = padded[..., 2:, 1:-1]
    bottom = padded[..., 1:-1, :-2]
    top    = padded[..., 1:-1, 2:]
    
    d2_dx2 = (left - 2.0 * center + right) / (dx ** 2)
    d2_dy2 = (bottom - 2.0 * center + top) / (dy ** 2)
    
    return d2_dx2 + d2_dy2