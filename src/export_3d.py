"""
Lighthouse Relativity: 3D Volumetric VTK Export Module
=====================================================
Exports 4D/3D OZJ scale-space tensor data into standard VTK (.vtk) legacy ASCII files
compatible with ParaView, VisIt, and Blender (via VTK importers).
"""

import os
import numpy as np
from typing import Dict, Any


def export_trajectory_to_vtk_volume(
    grid: Dict[str, Any],
    trajectory: np.ndarray,
    output_filename: str = "kleinion_volume.vtk",
    n_scale_layers: int = 8,
    frame_idx: int = -1
) -> str:
    """
    Exports a single 3D frame state s(Nx, Ny, Nz, 3) from the trajectory tensor
    to a legacy ASCII VTK Structured Points dataset.
    """
    s_3d = np.array(trajectory[frame_idx])  # Shape: (Nx, Ny, Nz, 3)

    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    Lx, Ly, Lz = grid['Lx'], grid['Ly'], grid['Lz']
    dx, dy, dz = grid['dx'], grid['dy'], grid['dz']

    sx, sy, sz = s_3d[..., 0], s_3d[..., 1], s_3d[..., 2]
    s_transverse = np.sqrt(sx**2 + sy**2)

    with open(output_filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Lighthouse Relativity 3D OZJ Volumetric State\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        f.write(f"ORIGIN {-Lx/2.0:.6f} {-Ly/2.0:.6f} {0.0:.6f}\n")
        f.write(f"SPACING {dx:.6f} {dy:.6f} {dz:.6f}\n")

        n_points = Nx * Ny * Nz
        f.write(f"POINT_DATA {n_points}\n")

        # 1. Transverse Excitation Field Scalar
        f.write("SCALARS transverse_excitation float 1\n")
        f.write("LOOKUP_TABLE default\n")
        trans_flat = np.transpose(s_transverse, (2, 1, 0)).flatten()
        for val in trans_flat:
            f.write(f"{val:.6f}\n")

        # 2. Longitudinal Sz Component Scalar
        f.write("\nSCALARS sz_longitudinal float 1\n")
        f.write("LOOKUP_TABLE default\n")
        sz_flat = np.transpose(sz, (2, 1, 0)).flatten()
        for val in sz_flat:
            f.write(f"{val:.6f}\n")

        # 3. Full 3D Spin Vector Field
        f.write("\nVECTORS spin_vector float\n")
        s_vec_transposed = np.transpose(s_3d, (2, 1, 0, 3)).reshape(-1, 3)
        for vec in s_vec_transposed:
            f.write(f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}\n")

    print(f"📦 Exported volumetric VTK dataset to '{output_filename}'.")
    return output_filename