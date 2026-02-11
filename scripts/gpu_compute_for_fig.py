"""Compute jaxCLASS spectra on GPU, save to .npz for local plotting."""
import sys
sys.path.insert(0, '.')
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp
from jaxclass.primordial import primordial_scalar_pk

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()

# ── P(k) ──
print("Computing P(k)...", flush=True)
prec_pk = PrecisionParams()
bg = background_solve(params, prec_pk)
th = thermodynamics_solve(params, prec_pk, bg)
pt_pk = perturbations_solve(params, prec_pk, bg, th)
k_pk = np.array(pt_pk.k_grid)
P_R = np.array(primordial_scalar_pk(pt_pk.k_grid, params))
delta_m = np.array(pt_pk.delta_m[:, -1])
pk_jax = (2 * np.pi**2 / k_pk**3) * P_R * delta_m**2

# ── C_l with source interpolation ──
print("Computing C_l (planck_cl preset)...", flush=True)
prec_cl = PrecisionParams.planck_cl()
bg_cl = background_solve(params, prec_cl)
th_cl = thermodynamics_solve(params, prec_cl, bg_cl)
pt_cl = perturbations_solve(params, prec_cl, bg_cl, th_cl)
print(f"  n_k={len(pt_cl.k_grid)}, n_tau={len(pt_cl.tau_grid)}", flush=True)

# Dense l sampling
ells = list(range(2, 31)) + list(range(35, 101, 5)) + list(range(120, 501, 20)) + list(range(550, 2001, 50))
print(f"  Computing {len(ells)} multipoles with source interpolation...", flush=True)

cl_tt = np.array(compute_cl_tt_interp(pt_cl, params, bg_cl, ells, n_k_fine=3000))
print("  TT done", flush=True)
cl_ee = np.array(compute_cl_ee_interp(pt_cl, params, bg_cl, ells, n_k_fine=3000))
print("  EE done", flush=True)
cl_te = np.array(compute_cl_te_interp(pt_cl, params, bg_cl, ells, n_k_fine=3000))
print("  TE done", flush=True)

# Save
np.savez('figures/jaxclass_spectra.npz',
         k_pk=k_pk, pk_jax=pk_jax,
         ells=np.array(ells), cl_tt=cl_tt, cl_ee=cl_ee, cl_te=cl_te)
print("Saved to figures/jaxclass_spectra.npz", flush=True)
