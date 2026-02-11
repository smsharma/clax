"""Dump T_l(k) transfer function at specific l values.

Compare the integrand P_R(k)|T_l(k)|^2 against CLASS to see where
in k-space the power deficit occurs.

Also tests: what happens at l=1000 if we increase k_max from 1.0 to 3.0?
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

# ---- Test 1: k_max sensitivity ----
print("=" * 60, flush=True)
print("Test 1: k_max sensitivity for TT", flush=True)
print("=" * 60, flush=True)

l_values = [20, 100, 200, 500, 700, 1000, 1500]

for k_max in [1.0, 2.0, 3.0]:
    prec = PrecisionParams(
        pt_k_max_cl=k_max,
        pt_k_per_decade=60,
        pt_tau_n_points=5000,
        pt_l_max_g=50,
        pt_l_max_pol_g=50,
        pt_l_max_ur=50,
        pt_ode_rtol=1e-6,
        pt_ode_atol=1e-11,
        ode_max_steps=131072,
    )

    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    print(f"\nk_max={k_max}, n_k={len(pt.k_grid)}, k_range=[{float(pt.k_grid[0]):.4e}, {float(pt.k_grid[-1]):.4e}]", flush=True)

    cl_tt = compute_cl_tt_interp(pt, params, bg, l_values, n_k_fine=5000)

    for i, l in enumerate(l_values):
        if l < len(ref['tt']) and ref['tt'][l] != 0:
            err = (float(cl_tt[i]) - ref['tt'][l]) / ref['tt'][l] * 100
            sub = " ***" if abs(err) < 1.0 else ""
            print(f"  l={l:5d}: err={err:+.3f}%{sub}")

# ---- Test 2: ODE tolerance sensitivity ----
print("\n" + "=" * 60, flush=True)
print("Test 2: ODE tolerance sensitivity for TT at l=1000", flush=True)
print("=" * 60, flush=True)

l_test = [200, 500, 700, 1000]
bg0 = background_solve(params, PrecisionParams.planck_cl())
th0 = thermodynamics_solve(params, PrecisionParams.planck_cl(), bg0)

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
        ode_max_steps=131072,
    )
    pt = perturbations_solve(params, prec, bg0, th0)
    cl_tt = compute_cl_tt_interp(pt, params, bg0, l_test, n_k_fine=3000)
    print(f"\nrtol={rtol:.0e}:", flush=True)
    for i, l in enumerate(l_test):
        err = (float(cl_tt[i]) - ref['tt'][l]) / ref['tt'][l] * 100
        sub = " ***" if abs(err) < 1.0 else ""
        print(f"  l={l:5d}: err={err:+.3f}%{sub}")

# ---- Test 3: n_k_fine sensitivity ----
print("\n" + "=" * 60, flush=True)
print("Test 3: n_k_fine (interpolation density) sensitivity", flush=True)
print("=" * 60, flush=True)

prec0 = PrecisionParams.planck_cl()
pt0 = perturbations_solve(params, prec0, bg0, th0)

for n_k_fine in [1000, 3000, 5000, 10000]:
    cl_tt = compute_cl_tt_interp(pt0, params, bg0, l_test, n_k_fine=n_k_fine)
    print(f"\nn_k_fine={n_k_fine}:", flush=True)
    for i, l in enumerate(l_test):
        err = (float(cl_tt[i]) - ref['tt'][l]) / ref['tt'][l] * 100
        sub = " ***" if abs(err) < 1.0 else ""
        print(f"  l={l:5d}: err={err:+.3f}%{sub}")

print("\nDone.", flush=True)
