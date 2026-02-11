"""Scan l_max to see if higher hierarchy truncation fixes TT high-l error.

Tests l_max = 50, 100, 150 with planck_cl-like settings.
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
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

# Load CLASS reference
ref = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref['ell']

params = CosmoParams()
l_values = [20, 50, 100, 200, 300, 500, 700, 1000]

# Only compute bg/th once (they don't depend on l_max)
prec0 = PrecisionParams.planck_cl()
bg = background_solve(params, prec0)
th = thermodynamics_solve(params, prec0, bg)

for l_max in [50, 100, 150]:
    print(f"\n{'='*60}", flush=True)
    print(f"Testing l_max = {l_max}", flush=True)
    print(f"{'='*60}", flush=True)

    prec = PrecisionParams(
        pt_k_max_cl=1.0,
        pt_k_per_decade=60,
        pt_tau_n_points=5000,
        pt_l_max_g=l_max,
        pt_l_max_pol_g=min(l_max, 50),
        pt_l_max_ur=min(l_max, 50),
        pt_ode_rtol=1e-6,
        pt_ode_atol=1e-11,
        ode_max_steps=131072,
    )

    pt = perturbations_solve(params, prec, bg, th)
    print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

    # Compute C_l^TT
    cl_tt = compute_cl_tt_interp(pt, params, bg, l_values, n_k_fine=3000)
    cl_ee = compute_cl_ee_interp(pt, params, bg, l_values, n_k_fine=3000)

    print(f"\n{'l':>5s}  {'TT err':>10s}  {'EE err':>10s}")
    print("-" * 30)
    for i, l in enumerate(l_values):
        tt_err = (float(cl_tt[i]) - ref['tt'][l]) / ref['tt'][l] * 100
        ee_err = (float(cl_ee[i]) - ref['ee'][l]) / ref['ee'][l] * 100
        sub_tt = " ***" if abs(tt_err) < 1.0 else ""
        sub_ee = " ***" if abs(ee_err) < 1.0 else ""
        print(f"{l:5d}  {tt_err:+10.3f}%{sub_tt:4s}  {ee_err:+10.3f}%{sub_ee}")

print("\nDone.", flush=True)
