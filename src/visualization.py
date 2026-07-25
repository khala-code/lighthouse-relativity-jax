"""
Lighthouse Relativity: Visualization Module (3D OZJ Tensor Stack)
=================================================================
Visualizes 4D trajectory tensors with dynamic cosmological metric expansion a(t).
Includes diagnostic relaxation tracking for transverse excitations and defect decay curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from typing import Dict, Any

# Optional import check for openvdb
try:
    import openvdb as vdb
    HAS_VDB = True
except ImportError:
    HAS_VDB = False

def save_vdb_frame(q_3d_data, vdb_frame_counter, output_dir="vdb_exports"):
    """Exports a 3D NumPy/JAX array to an OpenVDB file for Blender."""
    if not HAS_VDB:
        print("⚠️ OpenVDB is not installed. Run 'conda install -c conda-forge openvdb' to enable VDB exports.")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"frame_{vdb_frame_counter:04d}.vdb")
    
    # Create float grid and populate using the correct OpenVDB copy method
    grid = vdb.FloatGrid()
    grid.copyFromArray(np.ascontiguousarray(np.real(q_3d_data), dtype=np.float32))
    
    # Use FOG_VOLUME for density/gas fields (or assign string directly: grid.gridClass = 'fog volume')
    grid.gridClass = vdb.GridClass.FOG_VOLUME
    grid.name = "density"
    
    vdb.write(file_path, grids=[grid])

def compute_topological_charge_density(s_2d: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Computes instant 2D topological charge density q(x,y) = s . (ds/dx x ds/dy) / 4pi."""
    ds_dx = np.gradient(s_2d, dx, axis=0)
    ds_dy = np.gradient(s_2d, dy, axis=1)
    cross = np.cross(ds_dx, ds_dy)
    dot = np.sum(s_2d * cross, axis=-1)
    return dot / (4.0 * np.pi)


def embedded_klein_bottle_3d(X_2d: np.ndarray, Y_2d: np.ndarray, scale_a: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parametric Figure-8 immersion mapping scaled by cosmological factor a(t)."""
    u = X_2d
    v = Y_2d

    r = 4.0 * (1.0 - np.cos(u) / 2.0)

    x_3d = np.where(
        u < np.pi,
        6.0 * np.cos(u) * (1.0 + np.sin(u)) + r * np.cos(u) * np.cos(v),
        6.0 * np.cos(u) * (1.0 + np.sin(u)) + r * np.cos(v + np.pi)
    )

    y_3d = np.where(
        u < np.pi,
        16.0 * np.sin(u) + r * np.sin(u) * np.cos(v),
        16.0 * np.sin(u)
    )

    z_3d = r * np.sin(v)

    return scale_a * x_3d, scale_a * y_3d, scale_a * z_3d


def render_trajectory_frames(
    grid: Dict[str, Any],
    trajectory: np.ndarray,
    frame_stride: int = 20,
    output_dir: str = "frames",
    n_scale_layers: int = 8,
    dt: float = 0.003,
    H0: float = 0.0,
    save_vdb: bool = False,
) -> None:
    """Renders 4D OZJ tensor trajectories with dynamic Klein bottle inflation."""
    os.makedirs(output_dir, exist_ok=True)
    num_frames = len(trajectory)

    Lx, Ly = grid['Lx'], grid['Ly']
    dx, dy = grid['dx'], grid['dy']
    Nz = grid.get('Nz', n_scale_layers)

    x_1d = np.array(grid['x'])
    y_1d = np.array(grid['y'])
    X_2d, Y_2d = np.meshgrid(x_1d, y_1d, indexing='ij')

    z_layer_spacing = 1.8
    z_max = (Nz / 2.0) * z_layer_spacing + 2.0

    print(f"🎬 Exporting expanding 4D OZJ frames to '{output_dir}/' (stride={frame_stride}, H0={H0})...")
    
    # Initialize a strict sequential counter for VDB exports
    vdb_frame_counter = 0

    frame_idx = 0
    for t in range(0, num_frames, frame_stride):
        s_3d = np.array(trajectory[t])
        t_step = t * dt
        a_t = 1.0 + H0 * t_step

        X_scaled = X_2d * a_t
        Y_scaled = Y_2d * a_t
        x_3d_exp, y_3d_exp, z_3d_exp = embedded_klein_bottle_3d(X_2d, Y_2d, scale_a=a_t)

        fig = plt.figure(figsize=(18, 8), dpi=110)
        fig.patch.set_facecolor('#0a0a10')

        # 1. 3D OZJ Scale-Space Stack
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_facecolor('#0a0a10')

        layer_indices = np.linspace(-(Nz - 1) / 2.0, (Nz - 1) / 2.0, Nz)

        # Extract spatial dimensions directly from the 2D mesh grid shape
        Nx, Ny = X_2d.shape
        q_3d = np.zeros((Nx, Ny, Nz))
        
        for k in range(Nz):
            s_k = s_3d[:, :, k, :]
            q_3d[:, :, k] = compute_topological_charge_density(s_k, dx * a_t, dy * a_t)
            
        # Apply a 3D Gaussian filter to let scale-space feedback bleed across layers visually
        q_3d_smooth = gaussian_filter(q_3d, sigma=(1.0, 1.0, 0.5))

        for k in range(Nz):
            s_k = s_3d[:, :, k, :]
            sx_k = np.real(s_k[..., 0])
            sy_k = np.real(s_k[..., 1])

            z_baseline = layer_indices[k] * z_layer_spacing

            # Drive displacement using the cross-coupled 3D field 
            q_k_coupled = q_3d_smooth[:, :, k]
            funnel_displacement = 2.5 * np.sign(q_k_coupled) * np.log1p(10.0 * np.abs(q_k_coupled))
            Z_layer = z_baseline + funnel_displacement

            phase_angle = np.arctan2(sy_k, sx_k)
            norm_phase = (phase_angle + np.pi) / (2.0 * np.pi)
            colors = cm.inferno(norm_phase)

            alpha_val = 0.85 - 0.05 * abs(layer_indices[k])

            ax1.plot_surface(
                np.real(X_scaled), np.real(Y_scaled), np.real(Z_layer),
                facecolors=colors,
                rstride=2, cstride=2,
                linewidth=0,
                edgecolor='none',
                alpha=alpha_val,
                antialiased=True
            )
        
        # VDB Export (Strict Sequential Naming)
        if save_vdb:
            save_vdb_frame(q_3d_smooth, vdb_frame_counter)
            vdb_frame_counter += 1

        ax1.set_title(f"3D Scale Hierarchy | Scale Factor a(t) = {a_t:.2f}", color='white', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xlabel("X - Comoving", color='white')
        ax1.set_ylabel("Y - Comoving", color='white')
        ax1.set_zlabel("Scale Layer ($Z$)", color='white')
        max_bound = (Lx / 2.0) * (1.0 + H0 * (num_frames * dt))
        ax1.set_xlim(-max_bound, max_bound)
        ax1.set_ylim(-max_bound, max_bound)
        ax1.set_zlim(-z_max, z_max)
        ax1.tick_params(colors='white')
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.view_init(elev=22, azim=-50 + (t / num_frames) * 40)

        # 2. Expanding Holographic Klein Bottle Horizon
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_facecolor('#0a0a10')

        k_mid = Nz // 2
        s_mid = s_3d[:, :, k_mid, :]
        phase_mid = np.arctan2(np.real(s_mid[..., 1]), np.real(s_mid[..., 0]))
        norm_phase_mid = (phase_mid + np.pi) / (2.0 * np.pi)
        colors_klein = cm.inferno(norm_phase_mid)

        ax2.plot_surface(
            np.real(x_3d_exp), np.real(y_3d_exp), np.real(z_3d_exp),
            facecolors=colors_klein,
            rstride=2, cstride=2,
            linewidth=0,
            edgecolor='none',
            alpha=0.95,
            antialiased=True
        )

        ax2.set_title(f"Expanding Klein Horizon | Step {t}/{num_frames}", color='white', fontsize=13, fontweight='bold', pad=15)
        ax2.axis('off')
        ax2.view_init(elev=25, azim=-45 + (t / num_frames) * 60)

        plt.tight_layout()

        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        plt.savefig(frame_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        frame_idx += 1

    print(f"✅ Exported {frame_idx} expanding manifold frames.")


def plot_field_and_defects(
    grid: Dict[str, Any],
    trajectory: np.ndarray,
    output_filename: str = "lighthouse_relaxation_metrics.png",
    dt: float = 0.003,
    H0: float = 0.0
) -> None:
    """Renders diagnostic chart tracking Transverse Excitation Energy, Defect Density RMS, and Longitudinal Relaxation."""
    num_steps = len(trajectory)
    t_axis = np.arange(num_steps) * dt
    dx, dy = grid['dx'], grid['dy']
    Nz = grid.get('Nz', 8)

    print("📊 Computing Transverse Excitation & Defect Relaxation diagnostics...")

    sx_all = np.real(trajectory[..., 0])
    sy_all = np.real(trajectory[..., 1])
    sz_all = np.real(trajectory[..., 2])

    e_transverse = np.mean(sx_all**2 + sy_all**2, axis=(1, 2, 3))
    sz_mean = np.mean(sz_all, axis=(1, 2, 3))

    sample_stride = max(1, num_steps // 200)
    t_samples = np.arange(0, num_steps, sample_stride)
    q_rms_list = []

    for t_idx in t_samples:
        s_t = trajectory[t_idx]
        q_layers = []
        for k in range(Nz):
            s_k = s_t[:, :, k, :]
            q_k = compute_topological_charge_density(s_k, dx, dy)
            q_layers.append(q_k)
        q_rms = np.sqrt(np.mean(np.array(q_layers)**2))
        q_rms_list.append(q_rms)

    q_rms_arr = np.array(q_rms_list)
    t_sampled_axis = t_samples * dt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=120)
    fig.patch.set_facecolor('#0a0a10')

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#12121c')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='#444466')
        for spine in ax.spines.values():
            spine.set_color('#444466')

    ax1.plot(t_axis, e_transverse, color='#00e5ff', linewidth=2.0, label=r"Transverse Energy $\langle s_x^2 + s_y^2 \rangle$")
    ax1.set_ylabel("Excitation Energy", color='white', fontsize=11, fontweight='bold')
    ax1.set_title("Lighthouse Relativity Engine: Cosmological Metric & Defect Relaxation", color='white', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc="upper right", facecolor='#0a0a10', edgecolor='white', labelcolor='white')

    ax2.plot(t_sampled_axis, q_rms_arr, color='#ff2a6d', linewidth=2.0, label=r"Defect Activity $Q_{\mathrm{RMS}}(t)$")
    ax2.set_ylabel("Defect Density ($Q_{\mathrm{RMS}}$)", color='white', fontsize=11, fontweight='bold')
    ax2.legend(loc="upper right", facecolor='#0a0a10', edgecolor='white', labelcolor='white')

    ax3.plot(t_axis, sz_mean, color='#00ff66', linewidth=2.0, label=r"Longitudinal Alignment $\langle s_z \rangle$")
    ax3.set_xlabel("Cosmological Time ($t$)", color='white', fontsize=11, fontweight='bold')
    ax3.set_ylabel("Order Parameter ($S_z$)", color='white', fontsize=11, fontweight='bold')
    ax3.legend(loc="lower right", facecolor='#0a0a10', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"📈 Relaxation diagnostic graph saved to '{output_filename}'.")