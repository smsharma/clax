"""Decompose TT into T0, T1, T2 contributions and compare each against CLASS.

This helps identify which transfer type has the error.
"""
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
from jaxclass.harmonic import compute_cl_tt_interp

# Load CLASS reference
ref = np.load('reference_data/lcdm_fiducial/cls.npz')

params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Solving...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

ells = [20, 50, 100, 200, 300, 500, 700, 1000]

# Test different TT modes to see which contributes the error
modes = ["T0", "T0+T1", "T0+T1+T2"]

for mode in modes:
    print(f"\nTT mode: {mode}", flush=True)
    cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000, tt_mode=mode)
    for i, ell in enumerate(ells):
        cl_class = ref['tt'][ell]
        err = (float(cl_tt[i]) - cl_class) / cl_class * 100
        sub = " ***" if abs(err) < 1.0 else ""
        print(f"  l={ell:5d}: err={err:+.3f}%{sub}")

# Also test with more tau points to rule out integration error
print("\n\nTesting tau-point sensitivity (T0+T1+T2 mode):", flush=True)
for n_tau in [5000, 10000]:
    prec2 = PrecisionParams(
        pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=n_tau,
        pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
        pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
    )
    pt2 = perturbations_solve(params, prec2, bg, th)
    cl_tt2 = compute_cl_tt_interp(pt2, params, bg, ells, n_k_fine=3000)
    print(f"\nn_tau={n_tau}:")
    for i, ell in enumerate(ells):
        cl_class = ref['tt'][ell]
        err = (float(cl_tt2[i]) - cl_class) / cl_class * 100
        sub = " ***" if abs(err) < 1.0 else ""
        print(f"  l={ell:5d}: err={err:+.3f}%{sub}")

print("\nDone.", flush=True)
