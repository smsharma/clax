"""Decompose TT C_l error into SW+Doppler vs ISW contributions.

Computes C_l from source_SW + source_Doppler only (no ISW) and compares
against CLASS with temperature contributions = 'tsw,dop'.
Then the ISW contribution is the difference.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import (compute_cl_tt_interp, _interp_sources_to_fine_k,
                                _exact_transfer_tt, _cl_k_integral)

params = CosmoParams()
prec = PrecisionParams.planck_cl()

# Generate CLASS references
from classy import Class
base_params = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.0544,
    'N_ncdm': 1, 'm_ncdm': 0.06, 'T_ncdm': 0.71611, 'N_ur': 2.0328,
    'output': 'tCl,pCl', 'l_max_scalars': 2500,
    'recombination': 'RECFAST', 'tol_background_integration': 1e-12, 'tol_ncdm': 1e-10,
}

# CLASS full
cosmo = Class()
cosmo.set(base_params)
cosmo.compute()
cl_full_class = cosmo.raw_cl(2500)['tt']
cosmo.struct_cleanup(); cosmo.empty()

# CLASS no-ISW (SW + Doppler only)
cosmo = Class()
p = dict(base_params)
p['temperature contributions'] = 'tsw,dop'
cosmo.set(p)
cosmo.compute()
cl_noISW_class = cosmo.raw_cl(2500)['tt']
cosmo.struct_cleanup(); cosmo.empty()

print("CLASS references generated.", flush=True)

N_K_FINE = 10000
ells = [20, 100, 200, 300, 400, 500, 600, 700, 1000]

t0 = time.time()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"Pipeline: {time.time()-t0:.0f}s", flush=True)

# Full TT (default T0+T1+T2)
cl_tt_full = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=N_K_FINE)

# No-ISW: compute C_l from source_SW + source_Doppler only
# source_T0_noISW = source_SW + source_Doppler (no ISW_vis, no ISW_fs)
source_noISW = pt.source_SW + pt.source_Doppler
tau_0 = float(bg.conformal_age)
chi_grid = tau_0 - pt.tau_grid
dtau = jnp.diff(pt.tau_grid)
dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

log_k_coarse = jnp.log(pt.k_grid)
log_k_fine = jnp.linspace(log_k_coarse[0], log_k_coarse[-1], N_K_FINE)
k_fine = jnp.exp(log_k_fine)

# Interpolate noISW source to fine k
fine_sources = _interp_sources_to_fine_k([source_noISW], log_k_coarse, log_k_fine)
source_noISW_fine = fine_sources[0]

# Also interpolate T2 for T0+T2 mode (SW+Doppler+T2)
fine_T2 = _interp_sources_to_fine_k([pt.source_T2], log_k_coarse, log_k_fine)
source_T2_fine = fine_T2[0]

cls_noISW = []
for l in ells:
    # T0 mode: just SW+Doppler (no T1 ISW, no T2)
    T_l = _exact_transfer_tt(source_noISW_fine, pt.tau_grid, k_fine, chi_grid, dtau_mid, l,
                              mode="T0")  # T0-only uses just j_l radial
    cl = _cl_k_integral(T_l, k_fine, params, k_interp_factor=1)
    cls_noISW.append(float(cl))

print(f"\n{'l':>6} {'TT full err%':>14} {'noISW err%':>12} {'ISW err (residual)':>18}")
print("-" * 55)
for i, ell in enumerate(ells):
    idx = int(ell)
    full_err = (float(cl_tt_full[i]) - cl_full_class[idx]) / abs(cl_full_class[idx]) * 100
    noISW_err = (cls_noISW[i] - cl_noISW_class[idx]) / abs(cl_noISW_class[idx]) * 100 if abs(cl_noISW_class[idx]) > 1e-30 else 0

    # ISW contribution error: full - noISW for both us and CLASS
    our_ISW = float(cl_tt_full[i]) - cls_noISW[i]
    cls_ISW = cl_full_class[idx] - cl_noISW_class[idx]
    isw_err = (our_ISW - cls_ISW) / abs(cl_full_class[idx]) * 100 if abs(cls_ISW) > 1e-30 else 0

    mk = lambda e: " ***" if abs(e) < 0.1 else (" **" if abs(e) < 0.25 else "")
    print(f"{ell:6d} {full_err:+14.4f}{mk(full_err)} {noISW_err:+12.4f}{mk(noISW_err)} {isw_err:+18.4f}")

print(f"\nTotal: {time.time()-t0:.0f}s")
