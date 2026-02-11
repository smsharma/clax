"""Compare IBP vs non-IBP TT C_l to isolate Doppler contribution error.

If IBP and non-IBP agree, the error is in common terms (SW, ISW, radial functions).
If they differ, the error is in IBP-specific quantities (g', theta_b').
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

l_test = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500]

# Compute T0+T1+T2 (IBP form, standard)
print("\n=== T0+T1+T2 (IBP, standard) ===")
cl_ibp = compute_cl_tt_interp(pt, p, bg, l_test, n_k_fine=3000, tt_mode="T0+T1+T2")

# Compute T0-only (IBP, monopole only — no T1/T2 radial contributions)
print("=== T0-only (IBP) ===")
cl_t0 = compute_cl_tt_interp(pt, p, bg, l_test, n_k_fine=3000, tt_mode="T0")

# Compute with non-IBP Doppler (if available)
# The nonIBP mode uses source_T0_noDopp*j_l + source_Doppler_nonIBP*j_l'
# This avoids g' and theta_b' entirely
try:
    print("=== nonIBP (no g'/theta_b' dependence) ===")
    cl_nonibp = compute_cl_tt_interp(pt, p, bg, l_test, n_k_fine=3000, tt_mode="nonIBP")
except Exception as e:
    print(f"nonIBP not available in interp method: {e}")
    cl_nonibp = None

print("\n" + "="*80)
print(f"{'l':>5} | {'IBP':>10} {'T0only':>10} {'CLASS':>10} | {'IBP_err%':>9} {'T0_err%':>9} {'T1T2_frac%':>10}")
print("-"*80)

for i, l in enumerate(l_test):
    cl_ref = ref['tt'][l]
    cl_ibp_val = float(cl_ibp[i])
    cl_t0_val = float(cl_t0[i])

    err_ibp = (cl_ibp_val / cl_ref - 1) * 100 if cl_ref != 0 else 0
    err_t0 = (cl_t0_val / cl_ref - 1) * 100 if cl_ref != 0 else 0
    t1t2_frac = (cl_ibp_val - cl_t0_val) / abs(cl_ref) * 100 if cl_ref != 0 else 0

    nonibp_str = ""
    if cl_nonibp is not None:
        cl_ni = float(cl_nonibp[i])
        err_ni = (cl_ni / cl_ref - 1) * 100 if cl_ref != 0 else 0
        nonibp_str = f" | nonIBP_err={err_ni:+.3f}%"

    print(f"{l:>5} | {cl_ibp_val:>10.4e} {cl_t0_val:>10.4e} {cl_ref:>10.4e} | {err_ibp:>+9.3f} {err_t0:>+9.3f} {t1t2_frac:>+10.3f}{nonibp_str}")

print("\nDone!", flush=True)
