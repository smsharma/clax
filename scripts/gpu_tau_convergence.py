"""Test tau-grid convergence for T_l at specific (l,k).

Interpolates source functions to finer tau grids and checks if T_l changes.
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
from jaxclass.harmonic import compute_cl_tt_interp
from jaxclass.interpolation import CubicSpline
from jaxclass.bessel import spherical_jl_backward

p = CosmoParams()
pp = PrecisionParams.planck_cl()
bg = background_solve(p, pp)
th = thermodynamics_solve(p, pp, bg)
pt = perturbations_solve(p, pp, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

tau_0 = float(bg.conformal_age)
tau_star = float(th.tau_star)
ref = np.load('reference_data/lcdm_fiducial/cls.npz')

tau_grid = pt.tau_grid
chi_grid = tau_0 - tau_grid
dtau = jnp.diff(tau_grid)
dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

# Standard computation at l=30
l_test = [20, 30, 50, 100]

# 1. Original grid (5000 pts)
print("\n=== Original tau grid (5000 pts) ===")
cl_orig = compute_cl_tt_interp(pt, p, bg, l_test, n_k_fine=3000, tt_mode="T0+T1+T2")
for i, l in enumerate(l_test):
    err = (float(cl_orig[i]) / ref['tt'][l] - 1) * 100
    print(f"  l={l:4d}: err={err:+.3f}%")

# 2. Interpolate to 2x finer tau grid and recompute
# Create fine tau grid: interleave midpoints
tau_np = np.array(tau_grid)
tau_fine = np.sort(np.concatenate([tau_np, (tau_np[:-1] + tau_np[1:])/2]))
tau_fine = jnp.array(tau_fine)
print(f"\n=== Fine tau grid ({len(tau_fine)} pts) ===")

chi_fine = tau_0 - tau_fine
dtau_f = jnp.diff(tau_fine)
dtau_mid_f = jnp.concatenate([dtau_f[:1] / 2, (dtau_f[:-1] + dtau_f[1:]) / 2, dtau_f[-1:] / 2])

# Interpolate sources to fine tau grid
def interp_tau(source, tau_old, tau_new):
    def interp_single_k(src_k):
        return CubicSpline(tau_old, src_k).evaluate(tau_new)
    return jax.vmap(interp_single_k)(source)

# Use the fine k-grid method: interpolate sources to fine k AND fine tau
log_k_coarse = jnp.log(pt.k_grid)
log_k_fine = jnp.linspace(log_k_coarse[0], log_k_coarse[-1], 3000)
k_fine = jnp.exp(log_k_fine)

# Step 1: Interpolate to fine k
from jaxclass.harmonic import _interp_sources_to_fine_k
fine_k_sources = _interp_sources_to_fine_k(
    [pt.source_T0, pt.source_T1, pt.source_T2],
    log_k_coarse, log_k_fine
)
source_T0_fk = fine_k_sources[0]
source_T1_fk = fine_k_sources[1]
source_T2_fk = fine_k_sources[2]

# Step 2: Interpolate to fine tau
source_T0_ff = interp_tau(source_T0_fk, tau_grid, tau_fine)
source_T1_ff = interp_tau(source_T1_fk, tau_grid, tau_fine)
source_T2_ff = interp_tau(source_T2_fk, tau_grid, tau_fine)

# Step 3: Compute T_l on fine tau grid
from jaxclass.harmonic import _exact_transfer_tt, _cl_k_integral
from jaxclass.primordial import primordial_scalar_pk

print("Computing T_l on fine tau grid...")
for l in l_test:
    T_l_fine = _exact_transfer_tt(
        source_T0_ff, tau_fine, k_fine, chi_fine, dtau_mid_f, l,
        source_T1=source_T1_ff, source_T2=source_T2_ff, mode="T0+T1+T2"
    )
    cl_fine = _cl_k_integral(T_l_fine, k_fine, p, k_interp_factor=1)
    err = (float(cl_fine) / ref['tt'][l] - 1) * 100
    print(f"  l={l:4d}: err={err:+.3f}%")

print("\nDone!", flush=True)
