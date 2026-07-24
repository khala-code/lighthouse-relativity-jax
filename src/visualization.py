"""
Lighthouse Relativity: Visualization Module
===========================================
Renders 2D spatial domain heatmaps, 3D parametric Klein bottle surface
projections, and defect trajectory time-series.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any


def plot_field_and_defects(
    grid: Dict[str, Any],
    trajectory: jnp.ndarray,
    output_filename: str = "lighthouse_defect_plot.png"
) -> None:
    """
    Renders a 3-panel figure showing:
      1. 2D spatial heatmap of the final spin z-component s_z(x, y).
      2. 3D parametric projection of the Klein bottle surface mapped with field states.
      3. Time-evolution of the mean transverse excitation <|s_xy|> demonstrating defect relaxation.

    Parameters
    ----------
    grid : dict
        Grid metadata containing spatial meshes and 3D surface coordinates.
    trajectory : jnp.ndarray
        Simulation history array with shape (num_steps, Nx, Ny, 3).
    output_filename : str
        Target file path for saving the figure.
    """
    # Convert JAX device arrays to NumPy for Matplotlib rendering
    X = np.array(grid['X'])
    Y = np.array(grid['Y'])
    X_3d = np.array(grid['X_3d'])
    Y_3d = np.array(grid['Y_3d'])
    Z_3d = np.array(grid['Z_3d'])

    final_state = np.array(trajectory[-1])  # Shape: (Nx, Ny, 3)
    sz_final = final_state[..., 2]

    # Calculate global transverse excitation metric over time: mean sqrt(sx^2 + sy^2)
    transverse_excitation = np.array(
        jnp.mean(jnp.sqrt(trajectory[..., 0]**2 + trajectory[..., 1]**2), axis=(-2, -1))
    )

    fig = plt.figure(figsize=(16, 5))

    # -------------------------------------------------------------
    # Subplot 1: 2D Domain Heatmap (s_z component)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.pcolormesh(X, Y, sz_final, cmap='twilight_shifted', shading='auto')
    ax1.set_title("Final Spin State $s_z(x, y)$ on Klein Domain")
    ax1.set_xlabel("x (Orientable Loop)")
    ax1.set_ylabel("y (Non-Orientable Loop / Neck)")
    ax1.axhline(grid['Ly'] / 2.0, color='red', linestyle='--', alpha=0.8, label='Klein Neck Domain Wall')
    ax1.legend(loc='upper right')
    fig.colorbar(im, ax=ax1, label="$s_z$")

    # -------------------------------------------------------------
    # Subplot 2: 3D Parametric Klein Bottle Surface Mapping
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    norm_color = (sz_final - sz_final.min()) / (sz_final.max() - sz_final.min() + 1e-8)
    surf = ax2.plot_surface(
        X_3d, Y_3d, Z_3d,
        facecolors=plt.cm.twilight_shifted(norm_color),
        rstride=1, cstride=1,
        antialiased=True, alpha=0.9
    )
    ax2.set_title("3D Klein Bottle Holographic Projection")
    ax2.set_axis_off()

    # -------------------------------------------------------------
    # Subplot 3: Transverse Excitation & Defect Trajectory
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    time_steps = np.arange(len(transverse_excitation))
    ax3.plot(time_steps, transverse_excitation, color='crimson', linewidth=2)
    ax3.set_title("Transverse Excitation / Defect Relaxation")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel(r"Mean Transverse Spin $\langle |s_{xy}| \rangle$")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()