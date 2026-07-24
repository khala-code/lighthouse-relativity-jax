## Simulation Engine Free Parameters

Below is a reference of the physical and numerical free parameters exposed in `lighthouse-relativity-jax`.

---

### 1. Extended Bloch Engine & Non-Linear Dynamics

* **$\Xi$ (`Xi`, default: `0.5`):** Non-linear alignment torque coupling strength. Controls how aggressively the state vector snaps into alignment with the driving field $\mathbf{u}$ via $\Xi (\mathbf{s} \cdot \mathbf{u})(\mathbf{s} \times \mathbf{u})$.
* **$T_1$ (`T1`, default: `10.0`):** Longitudinal relaxation time. Dictates how fast energy dissipates and $s_z$ returns to the equilibrium ground state $s_{0,z}$.
* **$T_2$ (`T2`, default: `2.0`):** Transverse dephasing time. Dictates how quickly phase coherence ($s_x, s_y$) decays.
* **$D_{\text{spatial}}$ (`D_spatial`, default: `0.05`):** Spatial diffusion coefficient. Controls the spin-wave coupling strength across neighboring grid points on the Klein bottle mesh ($\nabla^2 \mathbf{s}$).
* **$s_{0,z}$ (`s0_z`, default: `1.0`):** Ground state z-axis alignment target.

---

### 2. Void Density & DBI Metric Soft-Clamping

* **$\alpha$ (`alpha`, default: `0.1`):** Dirac-Born-Infeld (DBI) saturation radical parameter. Controls how aggressively high Void Density soft-clamps effective relaxation times via:
  $$\gamma_D = \frac{1}{\sqrt{1 + \alpha \Pi_V^2}}$$
* **$\Pi_V(x,y)$ (`void_density` spatial profile):**
  * **Peak Amplitude** (default: `2.5`): Controls localized void pressure at the neck.
  * **Spatial Width $\sigma$** (default: `0.5`): Gaussian spread of the domain wall centered at $y = L_y / 2$.
  * **Base Noise Floor** (default: `0.1`): Background void density floor.

---

### 3. Stochastic & Drive Fields

* **$\sigma_\eta$ (`noise_std`, default: `0.02`):** Standard deviation of the stochastic background driver $\boldsymbol{\eta}(t)$ (thermal/quantum fluctuations adhering to fluctuation-dissipation).
* **$\mathbf{u}(x,y)$ (`drive_u` spatial geometry):** Vector field structure and polarization of the external transverse pump.

---

### 4. Spacetime Grid & Integration Parameters

* **$N_x, N_y$ (`Nx`, `Ny`, default: `128, 128`):** Spatial mesh resolution.
* **$L_x, L_y$ (`Lx`, `Ly`, default: `2\pi, 2\pi`):** Physical domain lengths along the orientable ($x$) and non-orientable ($y$) axes.
* **$\Delta t$ (`dt`, default: `0.01`):** Time-step increment for integration.
* **$N_{\text{steps}}$ (`num_steps`, default: `1000`):** Total duration of the simulation trajectory.

---

# Example Command-Line Configurations

### 1. Black Hole Regime (Default)
Simulates strongly warped spacetime with a massive singularity.
```bash
python main.py --config black_hole
```

### 2. Early Universe Regime
Simulates the inflationary epoch with high vacuum energy and rapid expansion.
```bash
python main.py --config early_universe
```

### 3. General Relativity Era
Simulates the current epoch of GR
```bash
python main.py --config gr_era
```