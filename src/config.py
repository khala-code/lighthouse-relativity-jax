"""
Lighthouse Relativity: Simulation Configuration Module
======================================================
Stores SimConfig dataclass and preset configurations for cosmological,
gravitational collapse, binary merger, and solid-state resonance runs.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Callable


@dataclass
class SimConfig:
    Nx: int = 64
    Ny: int = 64
    Nz: int = 8
    Lx: float = 6.283185
    Ly: float = 6.283185
    Lz: float = 10.0
    dt: float = 0.003
    num_steps: int = 5000
    Xi: float = 0.85
    T1: float = 50.0
    T2: float = 8.0
    alpha: float = 0.2
    omega_meta: float = 0.1
    H0: float = 1.2
    void_peak: float = 0.0
    void_sigma: float = 1.0
    noise_std: float = 0.04
    D_spatial: float = 0.08
    kappa_grav: float = 0.45
    B0: float = 50.0e-6
    B1: float = 0.0
    f_triad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    seed: int = 42


def get_full_cosmic_evolution_config() -> SimConfig:
    """Full Cosmic Evolution: Big Bang -> Inflation -> Recombination -> Cosmic Web (5,000 steps)."""
    return SimConfig(
        Nx=64,
        Ny=64,
        Nz=8,
        void_peak=0.0,
        void_sigma=1.0,
        alpha=0.18,
        T1=60.0,
        T2=10.0,
        Xi=0.92,
        H0=1.5,            # Inflation expansion rate amplitude
        omega_meta=0.15,   # High initial meta-clock frequency
        kappa_grav=0.5,    # Strong post-inflation gravitational coupling
        D_spatial=0.08,
        num_steps=5000,    # Long run budget to capture all 3 cosmic epochs
        dt=0.003,
        noise_std=0.045    # Quantum fluctuation baseline
    )


def get_cosmic_web_config() -> SimConfig:
    return SimConfig(
        Nx=64, Ny=64, Nz=8,
        void_peak=0.0, void_sigma=1.0,
        alpha=0.15, T1=60.0, T2=12.0,
        Xi=0.95, H0=0.8, omega_meta=0.005,
        kappa_grav=0.5, D_spatial=0.08,
        num_steps=2500, dt=0.004, noise_std=0.035
    )


def get_classical_gr_config() -> SimConfig:
    return SimConfig(
        Nx=64, Ny=64, Nz=8,
        void_peak=0.0, void_sigma=1.0,
        alpha=0.2, T1=50.0, T2=10.0,
        Xi=0.85, H0=0.2, omega_meta=0.01,
        kappa_grav=0.4, num_steps=2000, dt=0.005, noise_std=0.02
    )


def get_early_universe_config() -> SimConfig:
    return SimConfig(
        Nx=64, Ny=64, Nz=8,
        void_peak=0.2, void_sigma=3.0,
        alpha=0.1, T1=50.0, T2=15.0,
        Xi=0.9, H0=1.5, omega_meta=0.12,
        num_steps=2000, dt=0.005, noise_std=0.04
    )


def get_stellar_collapse_config() -> SimConfig:
    return SimConfig(
        Nx=64, Ny=64, Nz=8,
        void_peak=20.0, void_sigma=0.6,
        alpha=0.3, T1=40.0, T2=4.0, H0=0.0,
        num_steps=2000, dt=0.005, noise_std=0.005
    )


def get_binary_merger_config() -> SimConfig:
    return SimConfig(
        Nx=64, Ny=64, Nz=8,
        void_peak=18.0, void_sigma=0.45,
        alpha=0.3, T1=30.0, T2=3.0, H0=0.0,
        Xi=0.8, num_steps=2000, dt=0.005, noise_std=0.005
    )


def get_quartz_larmor_config() -> SimConfig:
    return SimConfig(
        Nx=64, Ny=64, Nz=8,
        void_peak=3.5, void_sigma=0.15,
        alpha=0.1, T1=10.0, T2=2.0, H0=0.0,
        B0=50.0e-6, B1=0.01,
        f_triad=(555.0, 423.0, 132.0),
        num_steps=2000, dt=0.005, noise_std=0.001
    )


def get_standard_config() -> SimConfig:
    return SimConfig()


CONFIG_PRESETS: Dict[str, Callable[[], SimConfig]] = {
    "full_cosmic_evolution": get_full_cosmic_evolution_config,
    "cosmic_web": get_cosmic_web_config,
    "classical_gr": get_classical_gr_config,
    "early_universe": get_early_universe_config,
    "stellar_collapse": get_stellar_collapse_config,
    "binary_merger": get_binary_merger_config,
    "quartz_larmor": get_quartz_larmor_config,
    "standard": get_standard_config,
}