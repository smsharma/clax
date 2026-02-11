"""Test RSA source substitution effect on TT C_l."""
import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

print(f"JAX device: {jax.devices()}", flush=True)

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

# Load CLASS reference
ref = np.load('reference_data/lcdm_fiducial/cls.npz')

params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Solving background + thermo...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)

print("Solving perturbations (with RSA source substitution)...", flush=True)
pt = perturbations_solve(params, prec, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

l_values = [20, 30, 50, 100, 200, 300, 500, 700, 1000]

print("\nComputing TT + EE...", flush=True)
cl_tt = compute_cl_tt_interp(pt, params, bg, l_values, n_k_fine=3000)
cl_ee = compute_cl_ee_interp(pt, params, bg, l_values, n_k_fine=3000)

print(f"\nC_l^TT (with RSA source substitution):")
for i, l in enumerate(l_values):
    err = (float(cl_tt[i]) - ref['tt'][l]) / ref['tt'][l] * 100
    sub = " ***" if abs(err) < 1.0 else ""
    print(f"  l={l:5d}: err={err:+.3f}%{sub}")

print(f"\nC_l^EE (with RSA source substitution):")
for i, l in enumerate(l_values):
    err = (float(cl_ee[i]) - ref['ee'][l]) / ref['ee'][l] * 100
    sub = " ***" if abs(err) < 1.0 else ""
    print(f"  l={l:5d}: err={err:+.3f}%{sub}")

print("\nDone!", flush=True)
