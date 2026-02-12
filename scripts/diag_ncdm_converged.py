"""Converged C_l diagnostic with ncdm (ρ+p) correction and n_k_fine=10000."""
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
from jaxclass.harmonic import (compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp)

params = CosmoParams()
prec = PrecisionParams.planck_cl()
ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']
ells = [10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000]

bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

n_fine = 10000
cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=n_fine)
cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=n_fine)
cl_te = compute_cl_te_interp(pt, params, bg, ells, n_k_fine=n_fine)

print(f"\nn_k_fine={n_fine}, with ncdm (rho+p) correction:")
print(f"{'l':>6} {'TT err%':>10} {'EE err%':>10} {'TE err%':>10}")
print("-" * 40)
for i, ell in enumerate(ells):
    idx = np.argmin(np.abs(ell_ref - ell))
    tt_class, ee_class, te_class = ref_cls['tt'][idx], ref_cls['ee'][idx], ref_cls['te'][idx]
    tt_err = (float(cl_tt[i]) - tt_class) / abs(tt_class) * 100 if abs(tt_class) > 1e-30 else 0
    ee_err = (float(cl_ee[i]) - ee_class) / abs(ee_class) * 100 if abs(ee_class) > 1e-30 else 0
    te_err = (float(cl_te[i]) - te_class) / abs(te_class) * 100 if abs(te_class) > 1e-30 else 0
    tt_m = " ***" if abs(tt_err) < 0.1 else (" **" if abs(tt_err) < 0.5 else (" *" if abs(tt_err) < 1.0 else ""))
    ee_m = " ***" if abs(ee_err) < 0.1 else (" **" if abs(ee_err) < 0.5 else (" *" if abs(ee_err) < 1.0 else ""))
    print(f"{ell:6d} {tt_err:+10.3f}{tt_m} {ee_err:+10.3f}{ee_m} {te_err:+10.3f}")
print("\nDone!", flush=True)
