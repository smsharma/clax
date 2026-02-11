"""Decompose TT C_l into T0-only, T0+T1+T2 contributions vs CLASS.

Also test: does the error come from T0 (SW+ISW+Doppler) or T1/T2 (ISW dipole + quadrupole)?
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

l_test = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000]

print("\n=== TT MODE DECOMPOSITION ===")
print(f"{'l':>5} | {'T0_only':>10} {'T0T1T2':>10} {'CLASS':>10} | {'T0_err%':>9} {'full_err%':>9} | {'T1T2_frac%':>10}")

for l in l_test:
    cl_T0 = float(compute_cl_tt_interp(pt, p, bg, [l], n_k_fine=3000, tt_mode="T0")[0])
    cl_full = float(compute_cl_tt_interp(pt, p, bg, [l], n_k_fine=3000, tt_mode="T0+T1+T2")[0])
    cl_ref = ref['tt'][l]

    err_T0 = (cl_T0 / cl_ref - 1.0) * 100 if cl_ref != 0 else 0
    err_full = (cl_full / cl_ref - 1.0) * 100 if cl_ref != 0 else 0
    T1T2_frac = (cl_full - cl_T0) / abs(cl_ref) * 100 if cl_ref != 0 else 0

    print(f"{l:>5} | {cl_T0:>10.4e} {cl_full:>10.4e} {cl_ref:>10.4e} | {err_T0:>+9.3f} {err_full:>+9.3f} | {T1T2_frac:>+10.3f}", flush=True)

print("\nDone.", flush=True)
