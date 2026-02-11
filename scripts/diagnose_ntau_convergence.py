"""Test convergence of TT C_l with n_tau (number of tau grid points).

If the error changes with n_tau, the issue is tau-grid resolution.
If it doesn't, the issue is in the physics (source functions, hierarchy).
"""
import sys
sys.path.insert(0, '.')
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp

# Load CLASS reference
ref = np.load('reference_data/lcdm_fiducial/cls.npz')
cl_tt_ref = ref['tt']

params = CosmoParams()

l_test = [30, 50, 100, 300, 700]

print("=== n_tau convergence test for TT C_l ===")
print(f"Testing l = {l_test}\n")

for n_tau in [3000, 5000, 8000]:
    prec = PrecisionParams(
        n_tau=n_tau,
        l_max_g=50,
        l_max_pol_g=50,
        l_max_ur=50,
        k_min=1e-4,
        k_max=1.0,
        k_per_decade=60,
    )

    print(f"n_tau={n_tau}: computing...", flush=True)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    cl = np.array(compute_cl_tt_interp(pt, params, bg, l_test, n_k_fine=3000))

    errs = [(cl[i] / cl_tt_ref[l] - 1) * 100 for i, l in enumerate(l_test)]
    err_str = ", ".join(f"l={l}: {e:+.2f}%" for l, e in zip(l_test, errs))
    print(f"  {err_str}")

print("\nDone!")
