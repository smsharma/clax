"""Test C_l accuracy with RSA relaxation damping DISABLED.

Runs perturbations_solve with RSA damping disabled to see if accuracy improves.
Compare against the known results with RSA damping ON.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("Devices:", jax.devices(), flush=True)

# DISABLE RSA damping BEFORE importing perturbations
import jaxclass.perturbations as pert_mod
pert_mod._RSA_DAMPING_ENABLED = False
print(f"RSA damping enabled: {pert_mod._RSA_DAMPING_ENABLED}", flush=True)

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
l_test = [10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000]

# Known results with RSA ON (from mode decomposition):
rsa_on_errors = {10: -7.397, 20: -0.285, 30: 1.535, 50: 1.671, 100: 0.638,
                 150: -0.041, 200: 0.192, 300: -0.677, 500: -0.568, 700: -1.576, 1000: -7.232}

print("\n=== RSA DAMPING OFF — C_l^TT ===")
print(f"{'l':>5} | {'C_l':>11} {'CLASS':>11} {'OFF_err%':>9} {'ON_err%':>9} {'better':>7}")
for l in l_test:
    cl = float(compute_cl_tt_interp(pt, p, bg, [l], n_k_fine=3000)[0])
    cl_ref = ref['tt'][l]
    err = (cl / cl_ref - 1) * 100
    err_on = rsa_on_errors.get(l, float('nan'))
    better = "OFF" if abs(err) < abs(err_on) else "ON"
    print(f"{l:>5} | {cl:>11.4e} {cl_ref:>11.4e} {err:>+9.3f} {err_on:>+9.3f} {better:>7}", flush=True)

print("\nDone!", flush=True)
