"""Quick C_l accuracy diagnostic - compares jaxCLASS vs CLASS at key multipoles."""
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
from jaxclass.harmonic import (compute_cl_tt_interp, compute_cl_ee_interp,
                                compute_cl_te_interp)

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()
prec = PrecisionParams.planck_cl()

# Load CLASS reference
ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']
ells = [10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000, 1500]

print("Running background...", flush=True)
bg = background_solve(params, prec)
print("Running thermodynamics...", flush=True)
th = thermodynamics_solve(params, prec, bg)
print("Running perturbations...", flush=True)
pt = perturbations_solve(params, prec, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

print("\nComputing C_l^TT...", flush=True)
cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000)
print("Computing C_l^EE...", flush=True)
cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=3000)
print("Computing C_l^TE...", flush=True)
cl_te = compute_cl_te_interp(pt, params, bg, ells, n_k_fine=3000)

print(f"\n{'l':>6} {'TT err%':>10} {'EE err%':>10} {'TE err%':>10}")
print("-" * 40)
for i, ell in enumerate(ells):
    idx = np.argmin(np.abs(ell_ref - ell))

    tt_class = ref_cls['tt'][idx]
    ee_class = ref_cls['ee'][idx]
    te_class = ref_cls['te'][idx]

    tt_err = (float(cl_tt[i]) - tt_class) / abs(tt_class) * 100 if abs(tt_class) > 1e-30 else 0
    ee_err = (float(cl_ee[i]) - ee_class) / abs(ee_class) * 100 if abs(ee_class) > 1e-30 else 0
    te_err = (float(cl_te[i]) - te_class) / abs(te_class) * 100 if abs(te_class) > 1e-30 else 0

    tt_mark = " ***" if abs(tt_err) < 0.1 else (" **" if abs(tt_err) < 0.5 else (" *" if abs(tt_err) < 1.0 else ""))
    ee_mark = " ***" if abs(ee_err) < 0.1 else (" **" if abs(ee_err) < 0.5 else (" *" if abs(ee_err) < 1.0 else ""))
    te_mark = " ***" if abs(te_err) < 0.1 else (" **" if abs(te_err) < 0.5 else (" *" if abs(te_err) < 1.0 else ""))

    print(f"{ell:6d} {tt_err:+10.3f}{tt_mark} {ee_err:+10.3f}{ee_mark} {te_err:+10.3f}{te_mark}")

print("\nDone!", flush=True)
