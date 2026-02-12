"""Test linear k-grid improvement on C_l accuracy."""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp

print(f"Devices: {jax.devices()}", flush=True)

params = CosmoParams()
prec = PrecisionParams.planck_cl()

# Key ells where we had errors
ELLS = [20, 30, 50, 100, 300, 500, 700, 1000, 1500, 2000]

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']

print("Running pipeline...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
t0 = time.time()
pt = perturbations_solve(params, prec, bg, th)
print(f"Perturbations: {time.time()-t0:.1f}s", flush=True)

for n_k in [10000, 20000]:
    print(f"\n--- n_k_fine = {n_k} (LINEAR k-grid) ---", flush=True)
    t0 = time.time()
    cl_tt = compute_cl_tt_interp(pt, params, bg, ELLS, n_k_fine=n_k)
    cl_ee = compute_cl_ee_interp(pt, params, bg, ELLS, n_k_fine=n_k)
    cl_te = compute_cl_te_interp(pt, params, bg, ELLS, n_k_fine=n_k)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    print(f"  {'l':>6} {'TT%':>8} {'EE%':>8} {'TE%':>8}")
    print(f"  {'-'*35}")
    for i, ell in enumerate(ELLS):
        idx = np.argmin(np.abs(ell_ref - ell))
        tt_c, ee_c, te_c = ref_cls['tt'][idx], ref_cls['ee'][idx], ref_cls['te'][idx]
        tt_e = (float(cl_tt[i]) - tt_c) / abs(tt_c) * 100 if abs(tt_c) > 1e-30 else 0
        ee_e = (float(cl_ee[i]) - ee_c) / abs(ee_c) * 100 if abs(ee_c) > 1e-30 else 0
        te_e = (float(cl_te[i]) - te_c) / abs(te_c) * 100 if abs(te_c) > 1e-30 else 0
        tt_m = " *" if abs(tt_e) > 0.1 else ""
        ee_m = " *" if abs(ee_e) > 0.1 else ""
        print(f"  {ell:6d} {tt_e:+8.3f}{tt_m} {ee_e:+8.3f}{ee_m} {te_e:+8.3f}")

print("\nDone!", flush=True)
