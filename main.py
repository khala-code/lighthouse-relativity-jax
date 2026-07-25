"""
Lighthouse Relativity: Main Orchestrator (JAX)
==============================================
Loads YAML configurations, compiles Frame Zero via Levain, 
initializes the 3D Klein topology, runs the branchless JAX solver, 
and renders visualization frames & relaxation metrics.
"""

import os
import numpy as np
import argparse
import yaml
import jax
import jax.numpy as jnp
from pathlib import Path

from src.baker import compile_simulation
from src.topology import create_klein_grid
from src.solver import load_topology_from_frame_zero, run_simulation
from src.visualization import render_trajectory_frames, plot_field_and_defects

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Lighthouse Relativity Simulation Engine")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="configs/cosmic_evolution.yaml",
        help="Path to the YAML simulation configuration file."
    )
    parser.add_argument("--steps", type=int, default=None, help="Override number of simulation time steps")
    parser.add_argument("--dt", type=float, default=None, help="Override integration time step size")
    parser.add_argument("--stride", type=int, default=20, help="Frame rendering stride for video animation")
    parser.add_argument(
        "--export-vdb", 
        action="store_true", 
        help="Export 3D volumetric data fields as .pyopenvdb files for Blender rendering."
    )
    args = parser.parse_args()

    print(f"📄 Loading configuration from {args.config}...")
    config = load_config(args.config)
    sim_meta = config.get("simulation", {})
    solver_params = sim_meta.get("solver_parameters", {})

    num_steps = args.steps if args.steps is not None else solver_params.get("num_steps", 2000)
    dt = args.dt if args.dt is not None else solver_params.get("dt", 0.003)
    H0 = solver_params.get("h0", 0.0)

    # 1. Compile Frame Zero via Levain Euclidean Compiler
    print("🔬 Compiling Frame Zero via Levain Baker...")
    frame_zero = compile_simulation(config)

    # 2. Extract Grid Parameters & Build 3D Klein Manifold
    resolution = sim_meta.get("resolution", [64, 64])
    nz_layers = sim_meta.get("nz_layers", 8)
    extents = sim_meta.get("domain_extents", [6.283185, 6.283185, 10.0])
    
    print(f"🌐 Building 3D Klein Grid ({resolution[0]}x{resolution[1]}x{nz_layers})...")
    grid = create_klein_grid(
        Nx=resolution[0],
        Ny=resolution[1],
        Nz=nz_layers,
        Lx=extents[0],
        Ly=extents[1],
        Lz=extents[2],
        seam_config=sim_meta.get("seam_operator", {})
    )

    # 3. Load Pre-Wound Topology from Frame Zero Bundle
    print("🌀 Initializing scale-space state from Frame Zero bundle...")
    s_init, u_drive = load_topology_from_frame_zero(grid, frame_zero)
    
    Pi_V_init = frame_zero.void_density_field
    if nz_layers > 1:
        Pi_V_init = jnp.repeat(Pi_V_init[:, :, jnp.newaxis], nz_layers, axis=2)
    
    omega_larmor_field = jnp.ones_like(Pi_V_init) * 50.0e-6

    # 4. Extract Prime-Locked T-Axis Start
    t_axis_start = config.get("initial_conditions", {}).get("t_axis_start", 163.0)

    # 5. Execute Solver Scan Loop
    sim_name = sim_meta.get("name", "simulation")
    print(f"🚀 Launching simulation microgame: {sim_name}...")
    trajectory, final_state = run_simulation(
        grid=grid,
        s_init=s_init,
        u=u_drive,
        Pi_V=Pi_V_init,
        omega_larmor_field=omega_larmor_field,
        dt=dt,
        num_steps=num_steps,
        Xi=solver_params.get("xi", 0.8),
        T1=solver_params.get("t1", 30.0),
        T2=solver_params.get("t2", 3.0),
        alpha=solver_params.get("alpha", 0.3),
        H0=H0,
        omega_meta=solver_params.get("omega_meta", 0.0),
        D_z=solver_params.get("d_z", 0.05),
        lambda_scale=solver_params.get("lambda_scale", 0.02),
        noise_std=solver_params.get("noise_std", 0.005),
        t_axis_start=t_axis_start
    )

    diag_filename = f"lighthouse_relaxation_{sim_name}.png"
    plot_field_and_defects(
        grid=grid,
        trajectory=jnp.array(trajectory),
        output_filename=diag_filename,
        dt=dt,
        H0=H0
    )

    # 6. Render Visualization Frames & Diagnostic Plots
    output_dir = f"frames_{sim_name}"
    print(f"🎨 Rendering output frames to '{output_dir}/'...")
    render_trajectory_frames(
        grid=grid,
        trajectory=jnp.array(trajectory),
        frame_stride=args.stride,
        output_dir=output_dir,
        n_scale_layers=nz_layers,
        dt=dt,
        H0=H0,
        save_vdb=args.export_vdb
    )

    print(f"✨ Simulation and visualization completed successfully for '{sim_name}'!")
    print(f"\n🎬 To compile frames into an MP4, run this FFmpeg command:")
    print(f"ffmpeg -framerate 30 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 {sim_name}_animation.mp4\n")


if __name__ == "__main__":
    main()