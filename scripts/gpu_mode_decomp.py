"""Decompose TT C_l into T0, T1, T2 contributions and compare with CLASS.

This diagnostic measures:
1. C_l^TT from T0 only, T0+T1, T0+T1+T2
2. Per-l error compared to CLASS reference
3. ISW vs SW decomposition

Run on GPU: python scripts/gpu_mode_decomp.py
"""
import sys
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

print("JAX devices:", jax.devices())

from jaxclass import params, background, thermodynamics, perturbations, harmonic

# --- Setup ---
p = params.CosmoParams()
pp = params.PrecisionParams.planck_cl()
print(f"Preset: planck_cl (l_max_g={pp.pt_l_max_g}, k_max={pp.pt_k_max_cl}, "
      f"n_k_per_decade={pp.pt_k_per_decade}, n_tau={pp.pt_tau_n_points})")

bg = background.background_solve(p)
th = thermodynamics.thermo_solve(p, bg, pp)
print(f"tau_star = {float(th.tau_star):.2f} Mpc")

# Solve perturbations
pt = perturbations.perturbations_solve(p, bg, th, pp)
k_grid = pt.k_grid
print(f"k-grid: {len(k_grid)} modes, [{float(k_grid[0]):.5f}, {float(k_grid[-1]):.4f}] Mpc^-1")

# --- Reference CLASS C_l ---
ref = np.load("reference_data/lcdm_fiducial/cls.npz")
ell_ref = ref['ell']
cl_tt_ref = ref['tt']
cl_ee_ref = ref['ee']

# --- Test l values ---
l_test = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000]

# --- Compute T_l(k) for each mode ---
print("\n=== MODE DECOMPOSITION (T0 vs T0+T1 vs T0+T1+T2) ===")
print(f"{'l':>5} {'T0_only':>12} {'T0+T1':>12} {'T0+T1+T2':>12} {'CLASS':>12} | {'T0_err%':>8} {'T0T1_err%':>8} {'full_err%':>8} | {'T1_frac%':>8} {'T2_frac%':>8}")

for l in l_test:
    if l > 2500:
        continue

    # Compute with different modes
    cl_T0 = harmonic.compute_cl_tt_interp(
        pt, p, bg, [l], n_k_fine=3000, tt_mode="T0")[0]
    cl_T0T1 = harmonic.compute_cl_tt_interp(
        pt, p, bg, [l], n_k_fine=3000, tt_mode="T0+T1")[0]
    cl_full = harmonic.compute_cl_tt_interp(
        pt, p, bg, [l], n_k_fine=3000, tt_mode="T0+T1+T2")[0]

    cl_ref = cl_tt_ref[l]

    # Errors
    err_T0 = (float(cl_T0) / cl_ref - 1.0) * 100 if cl_ref != 0 else 0
    err_T0T1 = (float(cl_T0T1) / cl_ref - 1.0) * 100 if cl_ref != 0 else 0
    err_full = (float(cl_full) / cl_ref - 1.0) * 100 if cl_ref != 0 else 0

    # Fractional contribution of T1 and T2
    T1_frac = (float(cl_T0T1) - float(cl_T0)) / float(cl_full) * 100 if float(cl_full) != 0 else 0
    T2_frac = (float(cl_full) - float(cl_T0T1)) / float(cl_full) * 100 if float(cl_full) != 0 else 0

    print(f"{l:>5} {float(cl_T0):>12.4e} {float(cl_T0T1):>12.4e} {float(cl_full):>12.4e} {cl_ref:>12.4e} | "
          f"{err_T0:>+8.2f} {err_T0T1:>+8.2f} {err_full:>+8.2f} | {T1_frac:>+8.2f} {T2_frac:>+8.2f}")

# --- Also test EE for sanity ---
print("\n=== EE CHECK ===")
print(f"{'l':>5} {'jaxCLASS':>12} {'CLASS':>12} {'error%':>8}")
for l in [20, 50, 100, 200, 500, 1000]:
    if l > 2500:
        continue
    cl_ee = harmonic.compute_cl_ee_interp(
        pt, p, bg, [l], n_k_fine=3000)[0]
    cl_ref_ee = cl_ee_ref[l]
    err = (float(cl_ee) / cl_ref_ee - 1.0) * 100 if cl_ref_ee != 0 else 0
    print(f"{l:>5} {float(cl_ee):>12.4e} {cl_ref_ee:>12.4e} {err:>+8.2f}")

print("\nDone.")
