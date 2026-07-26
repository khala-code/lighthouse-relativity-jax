"""
Lighthouse Relativity: Main Engine Entrypoint
============================================
Runs 3D OZJ Scale-Space Kleinion simulations with upstream gauge and larmor initialization.
"""

import os
import argparse
import jax
import jax.numpy as jnp

from src.config import CONFIG_PRESETS, SimConfig
from src.topology import create_klein_grid
from src.operators import create_quartz_lattice_with_al_impurities, compute_physical_larmor_field
from src.solver import run_simulation, load_pre_wound_topology_via_action
from src.visualization import render_trajectory_frames, plot_field_and_defects


def main():
    parser = argparse.ArgumentParser(description="Lighthouse Relativity 3D OZJ Simulation Engine")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="binary_merger",
        choices=list(CONFIG_PRESETS.keys()),
        help="Simulation preset config name"
    )
    parser.add_argument("--steps", type=int, default=None, help="Override number of simulation time steps")
    parser.add_argument("--dt", type=float, default=None, help="Override integration time step size")
    parser.add_argument("--stride", type=int, default=20, help="Frame rendering stride for video animation")
    parser.add_argument("--vdb", action="store_true", help="Export 3D volumetric OpenVDB dataset (.vdb) for Blender")
    args = parser.parse_args()

    print(f"🚀 Initializing 3D OZJ Engine with preset: '{args.config}'")
    cfg_fn = CONFIG_PRESETS[args.config]
    cfg: SimConfig = cfg_fn()

    num_steps = args.steps if args.steps is not None else cfg.num_steps
    dt = args.dt if args.dt is not None else cfg.dt

    grid = create_klein_grid(
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        Nz=cfg.Nz,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        Lz=cfg.Lz
    )
    key = jax.random.PRNGKey(cfg.seed)

    if args.config == "binary_merger":
        action_preset = "binary_merger_initial"
    elif args.config in ["early_universe", "full_cosmic_evolution", "cosmic_web"]:
        action_preset = "full_cosmic_evolution"
    else:
        action_preset = "flat_vacuum"

    s_0, drive_u, Pi_V, v_flow = load_pre_wound_topology_via_action(grid=grid, action_preset_name=action_preset, key=key)

    X, Y = grid['X'], grid['Y']

    if args.config == "binary_merger":
        r0 = 1.5
        x1, y1 =  r0, 0.0
        x2, y2 = -r0, 0.0
        Pi1 = cfg.void_peak * jnp.exp(-(((X - x1)**2 + (Y - y1)**2) / (2.0 * (cfg.void_sigma**2))))
        Pi2 = cfg.void_peak * jnp.exp(-(((X - x2)**2 + (Y - y2)**2) / (2.0 * (cfg.void_sigma**2))))
        void_density = Pi1 + Pi2
    else:
        void_density = cfg.void_peak * jnp.exp(-((X**2 + Y**2) / (2.0 * (cfg.void_sigma**2))))

    # Upfront Larmor Field Default Setup (Zero-branching upstream)
    if args.config == "quartz_larmor":
        key, lattice_key = jax.random.split(key)
        pi_lattice, al_positions = create_quartz_lattice_with_al_impurities(
            grid=grid,
            key=lattice_key,
            n_lattice_waves=4,
            pi_lattice_amp=1.0,
            n_al_impurities=8,
            al_peak_amp=cfg.void_peak,
            al_sigma=cfg.void_sigma,
        )
        void_density = pi_lattice
        omega_larmor_field = compute_physical_larmor_field(grid, al_positions, B0=cfg.B0, al_sigma=cfg.void_sigma)
        w_larmor = 1.0
    else:
        omega_larmor_field = jnp.zeros_like(void_density)
        w_larmor = 0.0

    is_full_evo = (args.config == "full_cosmic_evolution")

    trajectory_tuple, final_state = run_simulation(
        grid=grid,
        s_init=s_0,
        u=drive_u,
        Pi_V=void_density,
        v_flow=v_flow,
        omega_larmor_field=omega_larmor_field,
        dt=dt,
        num_steps=num_steps,
        L_patch=cfg.L_patch,
        Xi=cfg.Xi,
        T1=cfg.T1,
        T2=cfg.T2,
        alpha=cfg.alpha,
        H0=cfg.H0,
        omega_meta=cfg.omega_meta,
        D_z=cfg.D_z,
        lambda_scale=cfg.lambda_scale,
        noise_std=cfg.noise_std,
        key=key,
        w_larmor=w_larmor,
        f_triad=cfg.f_triad,
        B1=cfg.B1,
        is_full_evolution=is_full_evo,
    )

    s_history, Pi_V_history, v_flow_history = trajectory_tuple

    diag_filename = f"lighthouse_relaxation_{args.config}.png"
    plot_field_and_defects(
        grid=grid,
        trajectory=s_history,
        output_filename=diag_filename,
        dt=dt,
        H0=cfg.H0
    )
    
    output_dir = f"frames_{args.config}"
    vdb_output_dir = f"vdb_exports_{args.config}"
    print(f"🎨 Rendering output frames to '{output_dir}/'...")
    render_trajectory_frames(
        grid=grid,
        trajectory_tuple=trajectory_tuple,
        frame_stride=20,
        output_dir=output_dir,
        vdb_output_dir=vdb_output_dir,
        export_vdb=args.vdb,
        dt=dt,
        H0=cfg.H0
    )

    print(f"✨ Simulation completed successfully for '{args.config}'!")


if __name__ == "__main__":
    main()