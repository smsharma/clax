"""Compute jaxCLASS spectra with planck_cl preset and save for comparison figure."""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp
from jaxclass.transfer import compute_pk_from_perturbations
from jaxclass.primordial import primordial_scalar_pk

print(f"Devices: {jax.devices()}", flush=True)

params = CosmoParams()
prec = PrecisionParams.planck_cl()
N_K_FINE = 10000

# ell grid
ells = list(range(2, 31)) + list(range(35, 101, 5)) + list(range(120, 501, 20)) + list(range(550, 2001, 50))
print(f"n_ells = {len(ells)}, ell_min = {ells[0]}, ell_max = {ells[-1]}", flush=True)

# Pipeline
print("Running background...", flush=True)
bg = background_solve(params, prec)

print("Running thermodynamics...", flush=True)
th = thermodynamics_solve(params, prec, bg)

print("Running perturbations...", flush=True)
t0 = time.time()
pt = perturbations_solve(params, prec, bg, th)
print(f"  Done in {time.time()-t0:.1f}s, n_k={len(pt.k_grid)}", flush=True)

# P(k) at the perturbation k_grid
print("Computing P(k)...", flush=True)
delta_m_z0 = pt.delta_m[:, -1]
P_R = primordial_scalar_pk(pt.k_grid, params)
pk_jax = 2.0 * jnp.pi**2 / pt.k_grid**3 * P_R * delta_m_z0**2

# C_l spectra
print(f"Computing C_l^TT (n_k_fine={N_K_FINE})...", flush=True)
t0 = time.time()
cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=N_K_FINE)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

print(f"Computing C_l^EE (n_k_fine={N_K_FINE})...", flush=True)
t0 = time.time()
cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=N_K_FINE)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

print(f"Computing C_l^TE (n_k_fine={N_K_FINE})...", flush=True)
t0 = time.time()
cl_te = compute_cl_te_interp(pt, params, bg, ells, n_k_fine=N_K_FINE)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# Save
outpath = 'figures/jaxclass_spectra_h100.npz'
np.savez(outpath,
    k_pk=np.array(pt.k_grid),
    pk_jax=np.array(pk_jax),
    ells=np.array(ells),
    cl_tt=np.array(cl_tt),
    cl_ee=np.array(cl_ee),
    cl_te=np.array(cl_te),
)
print(f"\nSaved to {outpath}", flush=True)
print(f"  k_pk: {len(pt.k_grid)} points, [{float(pt.k_grid[0]):.2e}, {float(pt.k_grid[-1]):.2e}]")
print(f"  pk_jax: [{float(pk_jax[0]):.2e}, {float(pk_jax.max()):.2e}]")
print(f"  ells: {len(ells)} points, [{ells[0]}, {ells[-1]}]")
print(f"  cl_tt[0]={float(cl_tt[0]):.4e}, cl_ee[0]={float(cl_ee[0]):.4e}")
print("Done!", flush=True)
