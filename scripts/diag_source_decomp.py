#!/usr/bin/env python3
"""Diagnostic: decompose C_l^TT into SW, ISW_vis, ISW_fs, Doppler.

Single perturbation solve, then computes C_l from each source subterm.
This isolates which source component is responsible for the accuracy gap.

Usage:
    cd /path/to/jaxclass && PYTHONPATH=. python scripts/diag_source_decomp.py
"""
import hashlib
import os
import sys
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.bessel import spherical_jl_backward
from jaxclass.interpolation import CubicSpline
from jaxclass.primordial import primordial_scalar_pk

# Preflight
for fname in ["perturbations.py", "harmonic.py"]:
    fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "jaxclass", fname)
    with open(fpath, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"{fname} sha256[:16]: {sha}", flush=True)

# Config — use 40 k/decade l_max=25 (fast, well-characterized)
params = CosmoParams()
prec = PrecisionParams(
    pt_k_max_cl=0.25, pt_k_per_decade=40, pt_tau_n_points=3000,
    pt_l_max_g=25, pt_l_max_pol_g=25, pt_l_max_ur=25,
    pt_ode_rtol=1e-5, pt_ode_atol=1e-10, ode_max_steps=65536,
)
n_k = int(jnp.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade)
print(f"Config: {n_k} k-modes, l_max={prec.pt_l_max_g}", flush=True)

# Solve
t0 = time.time()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"Solve: {time.time()-t0:.0f}s", flush=True)

# Verify source decomposition consistency
print("\nVerifying: source_T0 == SW + ISW_vis + ISW_fs + Doppler", flush=True)
recon = pt.source_SW + pt.source_ISW_vis + pt.source_ISW_fs + pt.source_Doppler
max_diff = float(jnp.max(jnp.abs(pt.source_T0 - recon)))
max_val = float(jnp.max(jnp.abs(pt.source_T0)))
print(f"  max |T0 - sum_of_parts| = {max_diff:.2e} (max |T0| = {max_val:.2e})", flush=True)

# Load reference
ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "reference_data", "lcdm_fiducial", "cls.npz")
cls_ref = np.load(ref_path, allow_pickle=True)
tt_ref = cls_ref["tt"]

# Helper: compute C_l from a source array using j_l radial
tau_grid = pt.tau_grid
k_grid = pt.k_grid
tau_0 = float(bg.conformal_age)
chi_grid = tau_0 - tau_grid
dtau = jnp.diff(tau_grid)
dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

def cl_from_source(source, l_int, k_interp_factor=3):
    """C_l = 4pi int dlnk P_R |T_l|^2, T_l = int dtau source*j_l."""
    def transfer_k(ik):
        k = k_grid[ik]
        x = k * chi_grid
        jl = spherical_jl_backward(l_int, x)
        return jnp.sum(source[ik, :] * jl * dtau_mid)

    T_l = jax.vmap(transfer_k)(jnp.arange(len(k_grid)))
    log_k = jnp.log(k_grid)

    if k_interp_factor > 1:
        n_fine = len(k_grid) * k_interp_factor
        log_k_fine = jnp.linspace(log_k[0], log_k[-1], n_fine)
        k_fine = jnp.exp(log_k_fine)
        T_l_fine = CubicSpline(log_k, T_l).evaluate(log_k_fine)
        P_R_fine = primordial_scalar_pk(k_fine, params)
        integrand = P_R_fine * T_l_fine**2
        dlnk = jnp.diff(log_k_fine)
    else:
        P_R = primordial_scalar_pk(k_grid, params)
        integrand = P_R * T_l**2
        dlnk = jnp.diff(log_k)

    return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

# Main diagnostic: C_l from each source subterm
print(f"\n=== C_l^TT SOURCE DECOMPOSITION (auto-spectra, unsigned) ===", flush=True)
print(f"{'l':>4} | {'total T0':>10} | {'SW':>10} | {'ISW_vis':>10} | {'ISW_fs':>10} | {'Doppler':>10} | {'CLASS':>10}", flush=True)
print("-" * 80, flush=True)

for l in [10, 50, 100, 200]:
    cl_total = float(cl_from_source(pt.source_T0, l))
    cl_sw = float(cl_from_source(pt.source_SW, l))
    cl_isw_v = float(cl_from_source(pt.source_ISW_vis, l))
    cl_isw_f = float(cl_from_source(pt.source_ISW_fs, l))
    cl_dop = float(cl_from_source(pt.source_Doppler, l))
    cl_class = float(tt_ref[l])

    print(f"{l:4d} | {cl_total:10.3e} | {cl_sw:10.3e} | {cl_isw_v:10.3e} | {cl_isw_f:10.3e} | {cl_dop:10.3e} | {cl_class:10.3e}", flush=True)

# Also show ratios to CLASS
print(f"\n=== RATIOS TO CLASS (total C_l / CLASS) ===", flush=True)
print(f"{'l':>4} | {'total':>8} | {'err':>6}", flush=True)
print("-" * 25, flush=True)
for l in [10, 50, 100, 200]:
    cl_total = float(cl_from_source(pt.source_T0, l))
    cl_class = float(tt_ref[l])
    ratio = cl_total / cl_class
    print(f"{l:4d} | {ratio:8.4f} | {abs(ratio-1)*100:5.1f}%", flush=True)

# Cross-check: does sum of sub-C_l equal total? (No — cross-terms matter!)
print(f"\n=== NOTE: sub-C_l don't sum to total (cross-terms!) ===", flush=True)
print("The sub-C_l above are |T_l^{sub}|^2, but total C_l = |sum T_l^{sub}|^2", flush=True)
print("The individual auto-spectra show which TERMS have the most power.", flush=True)

print("\nDone", flush=True)
