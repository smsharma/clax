"""Diagnose TT C_l error by decomposing into T0, T1, T2 contributions.

Computes C_l using T0-only, T0+T1, T0+T1+T2 modes and compares to CLASS.
"""
import sys
sys.path.insert(0, '.')
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

# Load CLASS reference (raw C_l)
ref = np.load('reference_data/lcdm_fiducial/cls.npz')
cl_tt_ref = ref['tt']
cl_ee_ref = ref['ee']

# Setup parameters
params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Computing pipeline (planck_cl preset)...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

# Test l values
l_test = [10, 15, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000]

# Compute for each mode
print("\nComputing T0-only mode...", flush=True)
cl_T0 = np.array(compute_cl_tt_interp(pt, params, bg, l_test, n_k_fine=3000, tt_mode="T0"))
print("Computing T0+T1 mode...", flush=True)
cl_T0T1 = np.array(compute_cl_tt_interp(pt, params, bg, l_test, n_k_fine=3000, tt_mode="T0+T1"))
print("Computing T0+T1+T2 mode...", flush=True)
cl_T0T1T2 = np.array(compute_cl_tt_interp(pt, params, bg, l_test, n_k_fine=3000, tt_mode="T0+T1+T2"))
print("Computing EE...", flush=True)
cl_ee = np.array(compute_cl_ee_interp(pt, params, bg, l_test, n_k_fine=3000))

print("\n=== TT C_l decomposition (raw C_l) ===")
print(f"{'l':>5} {'err(T0)':>10} {'err(T0+T1)':>12} {'err(T0+T1+T2)':>14}")
print("-" * 45)

for i, l in enumerate(l_test):
    cl_class = cl_tt_ref[l]
    if abs(cl_class) > 1e-30:
        err_T0 = (cl_T0[i] / cl_class - 1) * 100
        err_T0T1 = (cl_T0T1[i] / cl_class - 1) * 100
        err_T0T1T2 = (cl_T0T1T2[i] / cl_class - 1) * 100
        print(f"{l:5d} {err_T0:9.2f}% {err_T0T1:11.2f}% {err_T0T1T2:13.2f}%")

print("\n=== EE C_l comparison ===")
print(f"{'l':>5} {'error':>10}")
print("-" * 18)
for i, l in enumerate(l_test):
    cl_class_ee = cl_ee_ref[l]
    if abs(cl_class_ee) > 1e-30:
        err = (cl_ee[i] / cl_class_ee - 1) * 100
        print(f"{l:5d} {err:9.2f}%")

print("\n=== T1 and T2 fractional contributions ===")
print(f"{'l':>5} {'T1/total':>10} {'T2/total':>10} {'T1+T2 effect on err':>20}")
print("-" * 48)
for i, l in enumerate(l_test):
    total = cl_T0T1T2[i]
    if abs(total) > 1e-30:
        t1_frac = (cl_T0T1[i] - cl_T0[i]) / total * 100
        t2_frac = (cl_T0T1T2[i] - cl_T0T1[i]) / total * 100
        cl_class = cl_tt_ref[l]
        err_T0 = (cl_T0[i] / cl_class - 1) * 100
        err_full = (cl_T0T1T2[i] / cl_class - 1) * 100
        print(f"{l:5d} {t1_frac:9.2f}% {t2_frac:9.2f}%  {err_T0:+.2f}% -> {err_full:+.2f}%")

print("\nDone!")
