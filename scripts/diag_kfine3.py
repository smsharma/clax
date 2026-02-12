"""Test n_k_fine=20000 for high-l convergence."""
import sys
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()
prec = PrecisionParams.planck_cl()

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']
ells = [100, 300, 500, 700, 1000, 1500, 2000]

print("Running pipeline...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

n_fine = 20000
print(f"\nn_k_fine = {n_fine}", flush=True)
cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=n_fine)
cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=n_fine)

print(f"  {'l':>6} {'TT err%':>10} {'EE err%':>10}")
for i, ell in enumerate(ells):
    idx = np.argmin(np.abs(ell_ref - ell))
    tt_class = ref_cls['tt'][idx]
    ee_class = ref_cls['ee'][idx]
    tt_err = (float(cl_tt[i]) - tt_class) / abs(tt_class) * 100
    ee_err = (float(cl_ee[i]) - ee_class) / abs(ee_class) * 100
    tt_mark = " ***" if abs(tt_err) < 0.1 else (" **" if abs(tt_err) < 0.5 else (" *" if abs(tt_err) < 1.0 else ""))
    ee_mark = " ***" if abs(ee_err) < 0.1 else (" **" if abs(ee_err) < 0.5 else (" *" if abs(ee_err) < 1.0 else ""))
    print(f"  {ell:6d} {tt_err:+10.3f}{tt_mark} {ee_err:+10.3f}{ee_mark}")

print("\nDone!", flush=True)
