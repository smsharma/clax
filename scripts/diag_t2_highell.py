"""Check T2 contribution impact at l=500-700 with ncdm correction."""
import sys
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp

params = CosmoParams()
prec = PrecisionParams.planck_cl()
ref = np.load('reference_data/lcdm_fiducial/cls.npz')
ells = [200, 300, 500, 700, 1000]

bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)

for mode in ["T0", "T0+T1", "T0+T1+T2"]:
    cl = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=10000, tt_mode=mode)
    errs = []
    for i, l in enumerate(ells):
        idx = np.argmin(np.abs(ref['ell'] - l))
        e = (float(cl[i]) - ref['tt'][idx]) / abs(ref['tt'][idx]) * 100
        errs.append(f"l={l}:{e:+.2f}%")
    print(f"{mode:12s}: {', '.join(errs)}", flush=True)
print("Done!")
