"""Scan l_max to see if higher hierarchy truncation fixes TT high-l error.

Tests l_max = 25, 50, 100 with a reduced k-grid for speed.
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
cl_tt_ref = ref['tt']
cl_ee_ref = ref['ee']

params = CosmoParams()
l_values = [20, 50, 100, 200, 300, 500, 700, 1000]

for l_max in [25, 50, 100]:
    print(f"\n{'='*60}", flush=True)
    print(f"Testing l_max = {l_max}", flush=True)
    print(f"{'='*60}", flush=True)

    prec = PrecisionParams(
        pt_k_max_cl=0.5,        # smaller k-range for speed
        pt_k_per_decade=40,
        pt_tau_n_points=3000,
        pt_l_max_g=l_max,
        pt_l_max_pol_g=min(l_max, 50),
        pt_l_max_ur=min(l_max, 50),
        pt_ode_rtol=1e-6,
        pt_ode_atol=1e-11,
        ode_max_steps=131072,
    )

    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

    # Compute C_l^TT
    cl_tt = compute_cl_tt_interp(pt, params, bg, l_values, n_k_fine=2000)

    print(f"\n{'l':>5s}  {'TT err':>10s}  {'TT':>14s}  {'TT_ref':>14s}")
    print("-" * 50)
    for i, l in enumerate(l_values):
        idx = np.argmin(np.abs(ell_ref - l))
        cl_class = cl_tt_ref[idx]
        cl_ours = float(cl_tt[i])
        if abs(cl_class) > 1e-30:
            err = (cl_ours - cl_class) / cl_class * 100
            print(f"{l:5d}  {err:+10.3f}%  {cl_ours:14.6e}  {cl_class:14.6e}")
        else:
            print(f"{l:5d}  (no ref)")

print("\nDone.", flush=True)
