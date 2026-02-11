"""Separate T1 vs T2 contributions to TT C_l.

Tests: T0+T1 (without T2) and T0+T2 (without T1) to see which
correction is off.
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

p = CosmoParams()
pp = PrecisionParams.planck_cl()
bg = background_solve(p, pp)
th = thermodynamics_solve(p, pp, bg)
pt = perturbations_solve(p, pp, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

ref = np.load('reference_data/lcdm_fiducial/cls.npz')

l_test = [10, 20, 30, 50, 100, 200]

modes = {
    "T0": "T0",
    "T0+T1": "T0+T1",
    "T0+T2": "T0+T2",  # Won't work with standard mode strings; test manually
    "T0+T1+T2": "T0+T1+T2",
}

# Compute for each mode
results = {}
for name, mode in [("T0", "T0"), ("T0+T1", "T0+T1"), ("T0+T1+T2", "T0+T1+T2")]:
    print(f"Computing {name}...", flush=True)
    cl = compute_cl_tt_interp(pt, p, bg, l_test, n_k_fine=3000, tt_mode=mode)
    results[name] = cl

# Also test T0-T1+T2 (sign flip on T1) to check T1 sign
print("Computing T0-T1+T2 (T1 sign flip test)...", flush=True)
cl_signflip = compute_cl_tt_interp(pt, p, bg, l_test, n_k_fine=3000, tt_mode="T0-T1+T2")
results["T0-T1+T2"] = cl_signflip

print("\n" + "="*90)
print("TT MODE SEPARATION")
print("="*90)
print(f"{'l':>5} | {'T0_err%':>9} {'T0T1_err%':>10} {'T0T1T2_err%':>11} {'T0-T1+T2_err%':>13}")
print("-"*90)

for i, l in enumerate(l_test):
    cl_ref = ref['tt'][l]
    errs = {}
    for name in ["T0", "T0+T1", "T0+T1+T2", "T0-T1+T2"]:
        errs[name] = (float(results[name][i]) / cl_ref - 1) * 100 if cl_ref != 0 else 0

    print(f"{l:>5} | {errs['T0']:>+9.3f} {errs['T0+T1']:>+10.3f} {errs['T0+T1+T2']:>+11.3f} {errs['T0-T1+T2']:>+13.3f}")

# Decompose contributions
print("\n" + "="*90)
print("CONTRIBUTION DECOMPOSITION (as % of CLASS C_l)")
print("="*90)
print(f"{'l':>5} | {'T0':>9} {'T1':>9} {'T2':>9} {'T0+T1+T2':>11} {'CLASS':>9}")
print("-"*90)

for i, l in enumerate(l_test):
    cl_ref = ref['tt'][l]
    t0 = float(results["T0"][i])
    t0t1 = float(results["T0+T1"][i])
    t0t1t2 = float(results["T0+T1+T2"][i])

    # Note: these are C_l, not T_l, so contributions aren't simply additive
    # T_l = T_l^0 + T_l^1 + T_l^2, and C_l = int P_R |T_l|^2
    # C_l^{T0+T1} = C_l^{T0} + 2*cross(T0,T1) + C_l^{T1}
    # So: cross(T0,T1) ~ (C_l^{T0+T1} - C_l^{T0}) / 2  (if C_l^{T1} << cross)

    t1_contrib = (t0t1 - t0) / cl_ref * 100  # Includes 2*cross(T0,T1) + C_l^{T1}
    t2_contrib = (t0t1t2 - t0t1) / cl_ref * 100  # Includes 2*cross(T0+T1,T2) + C_l^{T2}

    print(f"{l:>5} | {t0/cl_ref*100:>+9.3f} {t1_contrib:>+9.3f} {t2_contrib:>+9.3f} {t0t1t2/cl_ref*100:>+11.3f} {100.0:>9.3f}")

print("\nDone!", flush=True)
