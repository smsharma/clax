"""Separate T1 and T2 contributions to identify l=30-50 error source."""
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
from jaxclass.harmonic import compute_cl_tt_interp

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()
prec = PrecisionParams.planck_cl()

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']
ells = [20, 30, 50, 100, 200, 300, 500]

print("Running pipeline...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
n_fine = 10000

for mode in ["T0", "T0+T1", "T0+T2", "T0+T1+T2"]:
    print(f"\nMode: {mode}", flush=True)
    cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=n_fine, tt_mode=mode)
    for i, ell in enumerate(ells):
        idx = np.argmin(np.abs(ell_ref - ell))
        tt_class = ref_cls['tt'][idx]
        tt_err = (float(cl_tt[i]) - tt_class) / abs(tt_class) * 100
        mark = " ***" if abs(tt_err) < 0.1 else (" **" if abs(tt_err) < 0.5 else (" *" if abs(tt_err) < 1.0 else ""))
        print(f"  l={ell:5d}: {tt_err:+.3f}%{mark}")

print("\nDone!", flush=True)
