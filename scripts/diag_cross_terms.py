#!/usr/bin/env python3
"""Diagnostic: explicit cross-terms between SW/ISW/Doppler at l=100.

Also tests Doppler with RHS-consistent theta_b_prime vs current reconstruction.

Usage:
    cd /path/to/jaxclass && PYTHONPATH=. python3 scripts/diag_cross_terms.py
"""
import os, sys, time, hashlib
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp; import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import (perturbations_solve, _build_indices,
                                     _perturbation_rhs)
from jaxclass.bessel import spherical_jl_backward
from jaxclass.interpolation import CubicSpline
from jaxclass.primordial import primordial_scalar_pk
import diffrax

# Preflight
fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "jaxclass", "perturbations.py")
with open(fpath, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()[:16]
print(f"perturbations.py sha256[:16]: {sha}", flush=True)

params = CosmoParams()
prec = PrecisionParams(
    pt_k_max_cl=0.25, pt_k_per_decade=40, pt_tau_n_points=3000,
    pt_l_max_g=25, pt_l_max_pol_g=25, pt_l_max_ur=25,
    pt_ode_rtol=1e-5, pt_ode_atol=1e-10, ode_max_steps=65536,
)
n_k = int(jnp.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade)
print(f"Config: {n_k} k-modes, l_max={prec.pt_l_max_g}", flush=True)

bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"Solve done", flush=True)

k_grid = pt.k_grid; tau_grid = pt.tau_grid; tau_0 = float(bg.conformal_age)
chi_grid = tau_0 - tau_grid
dtau = jnp.diff(tau_grid)
dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])
cls_ref = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "reference_data", "lcdm_fiducial", "cls.npz"))

def transfer_from_source(source, l, kif=3):
    """Compute T_l(k) array from source function."""
    def transfer_k(ik):
        k = k_grid[ik]
        x = k * chi_grid
        jl = spherical_jl_backward(l, x)
        return jnp.sum(source[ik, :] * jl * dtau_mid)
    return jax.vmap(transfer_k)(jnp.arange(len(k_grid)))

def cl_auto(T_l, kif=3):
    """C_l = 4pi int dlnk P_R |T_l|^2"""
    log_k = jnp.log(k_grid)
    n = len(k_grid) * kif
    lkf = jnp.linspace(log_k[0], log_k[-1], n)
    kf = jnp.exp(lkf)
    Tf = CubicSpline(log_k, T_l).evaluate(lkf)
    Pf = primordial_scalar_pk(kf, params)
    igr = Pf * Tf**2
    dlk = jnp.diff(lkf)
    return 4.0 * jnp.pi * jnp.sum(0.5 * (igr[:-1] + igr[1:]) * dlk)

def cl_cross(T1, T2, kif=3):
    """C_l = 4pi int dlnk P_R T1*T2 (can be negative)"""
    log_k = jnp.log(k_grid)
    n = len(k_grid) * kif
    lkf = jnp.linspace(log_k[0], log_k[-1], n)
    kf = jnp.exp(lkf)
    T1f = CubicSpline(log_k, T1).evaluate(lkf)
    T2f = CubicSpline(log_k, T2).evaluate(lkf)
    Pf = primordial_scalar_pk(kf, params)
    igr = Pf * T1f * T2f
    dlk = jnp.diff(lkf)
    return 4.0 * jnp.pi * jnp.sum(0.5 * (igr[:-1] + igr[1:]) * dlk)

# === Part 1: Explicit cross-terms at l=100 ===
print("\n=== CROSS-TERM DECOMPOSITION AT l=100 ===", flush=True)
l = 100

T_sw = transfer_from_source(pt.source_SW, l)
T_isw_v = transfer_from_source(pt.source_ISW_vis, l)
T_isw_f = transfer_from_source(pt.source_ISW_fs, l)
T_dop = transfer_from_source(pt.source_Doppler, l)
T_total = transfer_from_source(pt.source_T0, l)

cl_class = float(cls_ref["tt"][l])

# Auto-spectra
auto_sw = float(cl_auto(T_sw))
auto_isw_v = float(cl_auto(T_isw_v))
auto_isw_f = float(cl_auto(T_isw_f))
auto_dop = float(cl_auto(T_dop))
auto_total = float(cl_auto(T_total))

# Cross-spectra (2x for both orderings)
cross_sw_dop = 2.0 * float(cl_cross(T_sw, T_dop))
cross_sw_isw_f = 2.0 * float(cl_cross(T_sw, T_isw_f))
cross_sw_isw_v = 2.0 * float(cl_cross(T_sw, T_isw_v))
cross_dop_isw_f = 2.0 * float(cl_cross(T_dop, T_isw_f))
cross_dop_isw_v = 2.0 * float(cl_cross(T_dop, T_isw_v))
cross_isw_fv = 2.0 * float(cl_cross(T_isw_f, T_isw_v))

sum_all = (auto_sw + auto_isw_v + auto_isw_f + auto_dop +
           cross_sw_dop + cross_sw_isw_f + cross_sw_isw_v +
           cross_dop_isw_f + cross_dop_isw_v + cross_isw_fv)

print(f"Auto-spectra (normalized to CLASS):", flush=True)
print(f"  SW:       {auto_sw/cl_class:+.4f}", flush=True)
print(f"  ISW_vis:  {auto_isw_v/cl_class:+.4f}", flush=True)
print(f"  ISW_fs:   {auto_isw_f/cl_class:+.4f}", flush=True)
print(f"  Doppler:  {auto_dop/cl_class:+.4f}", flush=True)
print(f"Cross-spectra (2x, normalized to CLASS):", flush=True)
print(f"  SW x Dop:    {cross_sw_dop/cl_class:+.4f}", flush=True)
print(f"  SW x ISW_fs: {cross_sw_isw_f/cl_class:+.4f}", flush=True)
print(f"  SW x ISW_v:  {cross_sw_isw_v/cl_class:+.4f}", flush=True)
print(f"  Dop x ISW_fs:{cross_dop_isw_f/cl_class:+.4f}", flush=True)
print(f"  Dop x ISW_v: {cross_dop_isw_v/cl_class:+.4f}", flush=True)
print(f"  ISW_f x _v:  {cross_isw_fv/cl_class:+.4f}", flush=True)
print(f"Sum of all:    {sum_all/cl_class:+.4f}", flush=True)
print(f"Direct total:  {auto_total/cl_class:+.4f}", flush=True)
print(f"CLASS:          1.0000", flush=True)
print(f"Gap to CLASS:  {(cl_class - auto_total)/cl_class:+.4f} ({abs(1-auto_total/cl_class)*100:.1f}%)", flush=True)

# Verify decomposition consistency
print(f"\nConsistency: |sum_all - direct| = {abs(sum_all - auto_total)/cl_class:.6f}", flush=True)

print("\nDone", flush=True)
