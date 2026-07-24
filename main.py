"""
Lighthouse Relativity Simulation Engine (JAX)
=============================================
Entry point for running branchless, non-orientable topological field simulations
on a Klein bottle mesh.
"""

import jax
import jax.numpy as jnp
import time

# Imports from local src modules (to be implemented next)
from src.topology import create_klein_grid
from src.operators import init_state_fields
from src.solver import run_simulation
from src.visualization import plot_field_and_defects


def main():
    print("⚡ Initializing Lighthouse Relativity JAX Engine...")

    # -------------------------------------------------------------
    # 1. Simulation Hyperparameters & Physical Constants
    # -------------------------------------------------------------
    # Grid parameters
    Nx, Ny = 128, 128                      # Spatial resolution for the Klein bottle mesh
    Lx, Ly = 2.0 * jnp.pi, 2.0 * jnp.pi    # Spatial domain extents

    # Time-stepping parameters
    dt = 0.01                              # Integration step size
    num_steps = 1000                       # Total time steps

    # Extended Bloch & DBI Parameters
    Xi = 0.5                               # Non-linear alignment torque coupling: Xi(s . u)(s x u)
    T1 = 10.0                              # Longitudinal relaxation time (energy)
    T2 = 2.0                               # Transverse dephasing time (phase coherence)
    alpha = 0.1                            # DBI saturation radical parameter for Void Density
    noise_std = 0.02                       # Stochastic driver amplitude eta(t)

    print(f"   Grid Resolution: {Nx}x{Ny}")
    print(f"   Timesteps:       {num_steps} (dt = {dt})")
    print(f"   JAX Device:      {jax.devices()[0].device_kind.upper()}")

    # -------------------------------------------------------------
    # 2. Grid & Topology Setup
    # -------------------------------------------------------------
    grid = create_klein_grid(Nx, Ny, Lx, Ly)

    # -------------------------------------------------------------
    # 3. Initial Field Allocation
    # -------------------------------------------------------------
    key = jax.random.PRNGKey(42)
    state_0, drive_u, void_density = init_state_fields(grid, key)

    # -------------------------------------------------------------
    # 4. Run Branchless JAX Simulation Loop
    # -------------------------------------------------------------
    print("\n🚀 Compiling JAX pipeline (XLA) and stepping through manifold...")
    start_time = time.time()

    trajectory, final_state = run_simulation(
        grid=grid,
        s_init=state_0,
        u=drive_u,
        Pi_V=void_density,
        dt=dt,
        num_steps=num_steps,
        Xi=Xi,
        T1=T1,
        T2=T2,
        alpha=alpha,
        noise_std=noise_std,
        key=key,
    )

    # Block until JAX finishes asynchronous GPU/CPU computation for accurate timing
    final_state.block_until_ready()
    elapsed_time = time.time() - start_time
    print(f"✅ Simulation complete in {elapsed_time:.3f} seconds!")

    # -------------------------------------------------------------
    # 5. Visualizing Fields & Defect Pinning
    # -------------------------------------------------------------
    print("\n📊 Rendering field trajectory and defect pinning analysis...")
    plot_field_and_defects(grid, trajectory, output_filename="lighthouse_defect_plot.png")
    print("✨ Output saved to 'lighthouse_defect_plot.png'.")


if __name__ == "__main__":
    main()