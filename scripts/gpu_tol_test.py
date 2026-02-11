"""Test ODE tolerance sensitivity for TT C_l.

If rtol matters at high l, the error is from numerical diffusion in the ODE solver.
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

params = CosmoParams()

# Compute bg/th once
prec0 = PrecisionParams.planck_cl()
bg = background_solve(params, prec0)
th = thermodynamics_solve(params, prec0, bg)

l_test = [100, 200, 500, 700, 1000]

print("ODE tolerance scan:", flush=True)
for rtol in [1e-5, 1e-6, 1e-7, 1e-8]:
    prec = PrecisionParams(
        pt_k_max_cl=1.0,
        pt_k_per_decade=60,
        pt_tau_n_points=5000,
        pt_l_max_g=50,
        pt_l_max_pol_g=50,
        pt_l_max_ur=50,
        pt_ode_rtol=rtol,
        pt_ode_atol=rtol * 1e-5,
        ode_max_steps=262144,
    )
    pt = perturbations_solve(params, prec, bg, th)
    cl_tt = compute_cl_tt_interp(pt, params, bg, l_test, n_k_fine=3000)
    cl_ee = compute_cl_ee_interp(pt, params, bg, l_test, n_k_fine=3000)

    print(f"\nrtol={rtol:.0e}:", flush=True)
    print(f"  {'l':>5s}  {'TT err':>10s}  {'EE err':>10s}")
    for i, l in enumerate(l_test):
        tt_err = (float(cl_tt[i]) - ref['tt'][l]) / ref['tt'][l] * 100
        ee_err = (float(cl_ee[i]) - ref['ee'][l]) / ref['ee'][l] * 100
        sub_tt = " ***" if abs(tt_err) < 1.0 else ""
        sub_ee = " ***" if abs(ee_err) < 1.0 else ""
        print(f"  {l:5d}  {tt_err:+10.3f}%{sub_tt:4s}  {ee_err:+10.3f}%{sub_ee}")

print("\nDone.", flush=True)
