"""Compare source function values at specific (k, tau) against CLASS reference.

Focuses on source_T1 (ISW dipole) which is ~7% too small at l=30.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("Devices:", jax.devices(), flush=True)

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve

p = CosmoParams()
pp = PrecisionParams.planck_cl()
bg = background_solve(p, pp)
th = thermodynamics_solve(p, pp, bg)
pt = perturbations_solve(p, pp, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

tau_star = float(th.tau_star)
tau_0 = float(bg.conformal_age)
chi_star = tau_0 - tau_star

# For l~30, the relevant k is k ~ l / chi_star ~ 30/14000 ~ 0.002
# Find the nearest k-mode
k_target = 30.0 / chi_star
k_idx = int(jnp.argmin(jnp.abs(pt.k_grid - k_target)))
k_val = float(pt.k_grid[k_idx])

# Also check a mode near the first acoustic peak
k_peak = 100.0 / chi_star
k_idx2 = int(jnp.argmin(jnp.abs(pt.k_grid - k_peak)))
k_val2 = float(pt.k_grid[k_idx2])

print(f"\nchi_star = {chi_star:.1f} Mpc, tau_star = {tau_star:.1f} Mpc, tau_0 = {tau_0:.1f} Mpc")
print(f"k_target (l=30) = {k_target:.5f} Mpc^-1, nearest k = {k_val:.5f} (idx={k_idx})")
print(f"k_target (l=100) = {k_peak:.5f} Mpc^-1, nearest k = {k_val2:.5f} (idx={k_idx2})")

# Extract source functions near recombination
tau_grid = pt.tau_grid
mask = (tau_grid > tau_star - 200) & (tau_grid < tau_star + 200)
tau_sel = tau_grid[mask]

# Source T0, T1, T2 at k_target
source_T0_sel = pt.source_T0[k_idx, mask]
source_T1_sel = pt.source_T1[k_idx, mask]
source_T2_sel = pt.source_T2[k_idx, mask]

# Source components
source_SW_sel = pt.source_SW[k_idx, mask]
source_ISW_vis_sel = pt.source_ISW_vis[k_idx, mask]
source_ISW_fs_sel = pt.source_ISW_fs[k_idx, mask]
source_Dopp_sel = pt.source_Doppler[k_idx, mask]

# Print peak values
print(f"\n=== Source function peaks at k={k_val:.5f} (l~30) ===")
print(f"  source_T0 peak:  {float(jnp.max(jnp.abs(source_T0_sel))):.6e}")
print(f"  source_T1 peak:  {float(jnp.max(jnp.abs(source_T1_sel))):.6e}")
print(f"  source_T2 peak:  {float(jnp.max(jnp.abs(source_T2_sel))):.6e}")
print(f"  source_T2/8 peak: {float(jnp.max(jnp.abs(source_T2_sel/8))):.6e}")
print(f"  source_SW peak:  {float(jnp.max(jnp.abs(source_SW_sel))):.6e}")
print(f"  source_ISW_vis:  {float(jnp.max(jnp.abs(source_ISW_vis_sel))):.6e}")
print(f"  source_ISW_fs:   {float(jnp.max(jnp.abs(source_ISW_fs_sel))):.6e}")
print(f"  source_Dopp:     {float(jnp.max(jnp.abs(source_Dopp_sel))):.6e}")

# Ratio of T1 to T0 peak
ratio = float(jnp.max(jnp.abs(source_T1_sel))) / float(jnp.max(jnp.abs(source_T0_sel)))
print(f"  |T1/T0| at peak: {ratio:.4f}")

# Compute transfer integrals at l=30 for each source
from jaxclass.bessel import spherical_jl_backward
chi_grid = tau_0 - tau_grid

l = 30
x = k_val * chi_grid
jl = spherical_jl_backward(l, x)
jl_p1 = spherical_jl_backward(l + 1, x)
x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
jl_prime = (l / x_safe) * jl - jl_p1
jl_pp = (l * (l - 1) / (x_safe * x_safe) - 1.0) * jl + (2.0 / x_safe) * jl_p1
radial_T2 = 0.5 * (3.0 * jl_pp + jl)

dtau = jnp.diff(tau_grid)
dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

T_l_T0 = float(jnp.sum(pt.source_T0[k_idx, :] * jl * dtau_mid))
T_l_T1 = float(jnp.sum(pt.source_T1[k_idx, :] * jl_prime * dtau_mid))
T_l_T2 = float(jnp.sum((pt.source_T2[k_idx, :] / 8.0) * radial_T2 * dtau_mid))
T_l_full = T_l_T0 + T_l_T1 + T_l_T2

print(f"\n=== Transfer function T_l at l={l}, k={k_val:.5f} ===")
print(f"  T_l^T0  = {T_l_T0:+.6e}")
print(f"  T_l^T1  = {T_l_T1:+.6e}")
print(f"  T_l^T2  = {T_l_T2:+.6e}")
print(f"  T_l^full = {T_l_full:+.6e}")
print(f"  T1/T0 ratio: {T_l_T1/T_l_T0:+.4f}")
print(f"  T2/T0 ratio: {T_l_T2/T_l_T0:+.4f}")

# Also print at l=100
l2 = 100
x2 = k_val2 * chi_grid
jl2 = spherical_jl_backward(l2, x2)
jl2_p1 = spherical_jl_backward(l2 + 1, x2)
x2_safe = jnp.where(jnp.abs(x2) < 1e-30, 1e-30, x2)
jl2_prime = (l2 / x2_safe) * jl2 - jl2_p1
jl2_pp = (l2 * (l2 - 1) / (x2_safe * x2_safe) - 1.0) * jl2 + (2.0 / x2_safe) * jl2_p1
radial2_T2 = 0.5 * (3.0 * jl2_pp + jl2)

T_l2_T0 = float(jnp.sum(pt.source_T0[k_idx2, :] * jl2 * dtau_mid))
T_l2_T1 = float(jnp.sum(pt.source_T1[k_idx2, :] * jl2_prime * dtau_mid))
T_l2_T2 = float(jnp.sum((pt.source_T2[k_idx2, :] / 8.0) * radial2_T2 * dtau_mid))

print(f"\n=== Transfer function T_l at l={l2}, k={k_val2:.5f} ===")
print(f"  T_l^T0  = {T_l2_T0:+.6e}")
print(f"  T_l^T1  = {T_l2_T1:+.6e}")
print(f"  T_l^T2  = {T_l2_T2:+.6e}")
print(f"  T1/T0 ratio: {T_l2_T1/T_l2_T0:+.4f}")

print("\nDone!", flush=True)
