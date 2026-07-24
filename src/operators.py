"""
Lighthouse Relativity: Operators Module (JAX)
===========================================
Native 3D Extended Bloch differential operator with zero control-flow branching.
Supports dynamic time-varying scale factors a(t), Hubble drag H(t), and spectral GR.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any, Optional

from src.topology import laplacian_klein

GAMMA_29SI = -53.19e6
GAMMA_27AL = +69.76e6


@jax.jit
def dbi_radical(Pi_V: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    return 1.0 / jnp.sqrt(1.0 + alpha * (Pi_V ** 2))


@jax.jit
def compute_topological_charge_density_jax(s: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Evaluates local topological charge density q = s . (ds/dx x ds/dy) / 4pi across 3D tensor."""
    ds_dx = jnp.gradient(s, dx, axis=0)
    ds_dy = jnp.gradient(s, dy, axis=1)
    cross = jnp.cross(ds_dx, ds_dy)
    dot = jnp.sum(s * cross, axis=-1)
    return dot / (4.0 * jnp.pi)


@jax.jit
def solve_dynamic_gr_poisson_metric(
    s: jnp.ndarray,
    grid: Dict[str, Any],
    eta_grav: float = 5.0,
    m_screening: float = 0.2
) -> jnp.ndarray:
    """
    Branchless Spectral Poisson Solver: Solves (Del^2 - m^2) Pi_V = -eta * |q|^2 in Fourier Space.
    """
    dx, dy = grid['dx'], grid['dy']

    q = compute_topological_charge_density_jax(s, dx, dy)
    rho_energy = q ** 2

    K_sq = grid['K_sq_spatial'] + (m_screening ** 2)

    rho_ft = jnp.fft.fftn(rho_energy)
    Pi_V_ft = (eta_grav * rho_ft) / K_sq
    Pi_V_dynamic = jnp.real(jnp.fft.ifftn(Pi_V_ft))

    return Pi_V_dynamic


@jax.jit
def create_quartz_lattice_with_al_impurities(
    grid: Dict[str, Any],
    key: jax.random.PRNGKey,
    n_lattice_waves: int = 4,
    pi_lattice_amp: float = 1.0,
    n_al_impurities: int = 8,
    al_peak_amp: float = 3.5,
    al_sigma: float = 0.15,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    X, Y = grid['X'], grid['Y']
    Lx, Ly = grid['Lx'], grid['Ly']

    lattice = pi_lattice_amp * (
        jnp.cos(2.0 * jnp.pi * n_lattice_waves * X / Lx) +
        jnp.cos(2.0 * jnp.pi * n_lattice_waves * Y / Ly)
    )

    key_x, key_y = jax.random.split(key)
    al_x = jax.random.uniform(key_x, shape=(n_al_impurities,), minval=-Lx/2.0, maxval=Lx/2.0)
    al_y = jax.random.uniform(key_y, shape=(n_al_impurities,), minval=-Ly/2.0, maxval=Ly/2.0)
    al_positions = jnp.stack([al_x, al_y], axis=-1)

    def body_fn(i, current_pi):
        dx = X - al_positions[i, 0]
        dy = Y - al_positions[i, 1]
        r2 = dx**2 + dy**2
        spike = al_peak_amp * jnp.exp(-r2 / (2.0 * (al_sigma**2)))
        return current_pi + spike

    impurity_field = jax.lax.fori_loop(0, n_al_impurities, body_fn, jnp.zeros_like(X))
    return 1.0 + lattice + impurity_field, al_positions


@jax.jit
def compute_physical_larmor_field(
    grid: Dict[str, Any],
    al_positions: jnp.ndarray,
    B0: float = 50.0e-6,
    al_sigma: float = 0.15,
) -> jnp.ndarray:
    X, Y = grid['X'], grid['Y']
    omega_si = -GAMMA_29SI * B0
    omega_al = -GAMMA_27AL * B0
    omega_field = jnp.full_like(X, omega_si)

    def body_fn(i, current_omega):
        dx = X - al_positions[i, 0]
        dy = Y - al_positions[i, 1]
        r2 = dx**2 + dy**2
        weight = jnp.exp(-r2 / (2.0 * (al_sigma**2)))
        return current_omega + weight * (omega_al - omega_si)

    return jax.lax.fori_loop(0, al_positions.shape[0], body_fn, omega_field)


@jax.jit
def extended_bloch_rhs(
    s: jnp.ndarray,
    u: jnp.ndarray,
    Pi_V: jnp.ndarray,
    grid: Dict[str, Any],
    t_step: float = 0.0,
    a_t: float = 1.0,
    H_t: float = 0.0,
    omega_meta_t: float = 0.0,
    omega_larmor_field: Optional[jnp.ndarray] = None,
    w_larmor: float = 0.0,
    Xi: float = 0.5,
    T1: float = 10.0,
    T2: float = 2.0,
    alpha: float = 0.1,
    s0_z: float = 1.0,
    D_spatial: float = 0.05,
    f_triad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    B1: float = 0.0,
    kappa_grav: float = 0.2,
    dynamic_gr: bool = True,
) -> jnp.ndarray:
    """Evaluates RHS natively with time-varying scale factor a(t) and Hubble drag H(t)."""
    dx, dy = grid['dx'], grid['dy']

    if omega_larmor_field is None:
        omega_larmor_field = jnp.zeros_like(Pi_V)

    # 1. Dynamic Poisson GR Metric
    Pi_V_gr = solve_dynamic_gr_poisson_metric(s, grid, eta_grav=8.0, m_screening=0.15)
    w_gr = 1.0 if dynamic_gr else 0.0
    Pi_V_eff = w_gr * Pi_V_gr + (1.0 - w_gr) * Pi_V

    # 2. Dynamic Meta-Clock Phase
    phi_meta = omega_meta_t * t_step
    gamma_euclidean = 0.5 * (1.0 + jnp.cos(phi_meta))
    gamma_lorentzian = 1.0 - gamma_euclidean

    # 3. Dynamic Gravitational Torque Field
    dPi_dx = jnp.gradient(Pi_V_eff, dx, axis=0)
    dPi_dy = jnp.gradient(Pi_V_eff, dy, axis=1)
    u_grav = -kappa_grav * jnp.stack([dPi_dx, dPi_dy, jnp.zeros_like(Pi_V_eff)], axis=-1)

    # 4. Transverse Driving Fields
    f_vec = jnp.array(f_triad)
    u_ac = (B1 / 3.0) * jnp.sum(jnp.sin(2.0 * jnp.pi * f_vec * t_step))
    u_eff = u + u_grav + jnp.stack([
        u_ac * jnp.ones_like(Pi_V_eff),
        jnp.zeros_like(Pi_V_eff),
        jnp.zeros_like(Pi_V_eff)
    ], axis=-1)

    # 5. Macro Order Parameter
    phi_macro = jnp.mean(s, axis=(0, 1, 2), keepdims=True)
    u_total = u_eff + phi_macro

    # 6. DBI Saturation & Relaxation
    gamma_dbi = dbi_radical(Pi_V_eff, alpha)
    T1_eff = (T1 / gamma_dbi[..., None]) / jnp.maximum(gamma_euclidean, 0.1)
    T2_eff = (T2 / gamma_dbi[..., None]) / jnp.maximum(gamma_euclidean, 0.1)

    # 7. Precession
    omega_base = w_larmor * omega_larmor_field + (1.0 - w_larmor) * (1.0 + 0.2 * Pi_V_eff)
    Omega_z = omega_base * (1.0 + 0.5 * gamma_lorentzian)
    Omega_eff = jnp.stack([jnp.zeros_like(Omega_z), jnp.zeros_like(Omega_z), Omega_z], axis=-1)
    precession = jnp.cross(s, Omega_eff)

    # 8. Dissipation & Hubble Friction Drag (-2 * H(t) * s)
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    damping = jnp.stack([
        sx / T2_eff[..., 0],
        sy / T2_eff[..., 0],
        (sz - s0_z) / T1_eff[..., 0]
    ], axis=-1) + (2.0 * H_t * s)

    # 9. Non-Linear Scale Torque & Dilated Spatial Diffusion (1 / a(t)^2)
    dot_su = jnp.sum(s * u_total, axis=-1, keepdims=True)
    cross_su = jnp.cross(s, u_total)
    nonlinear_torque = Xi * dot_su * cross_su

    spatial_diffusion = (D_spatial / (a_t ** 2)) * laplacian_klein(s, grid)

    return precession - damping + nonlinear_torque + spatial_diffusion


@jax.jit
def soft_clamp_state(s: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(s, axis=-1, keepdims=True)
    return s / jnp.maximum(norm, 1e-8)