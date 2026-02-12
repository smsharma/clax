"""Phase 1b Diagnostic: n_k_fine sweep to confirm k-integration as the remaining error source.

Since l_max sweep showed NO effect (hierarchy truncation not the issue),
test n_k_fine=5000 vs 10000 to confirm k-integration is the bottleneck.
"""
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
prec = PrecisionParams.planck_cl()  # l_max=50, k_max=1.0
ELLS = [20, 30, 50, 100, 300, 500, 700, 1000, 2000]
N_K_FINE_VALUES = [5000, 10000]

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']

print("Running background...", flush=True)
bg = background_solve(params, prec)
print("Running thermodynamics...", flush=True)
th = thermodynamics_solve(params, prec, bg)
print("Running perturbations...", flush=True)
t0 = time.time()
pt = perturbations_solve(params, prec, bg, th)
t1 = time.time()
print(f"Perturbations done in {t1-t0:.1f}s, n_k={len(pt.k_grid)}", flush=True)

results = {}
for n_k_fine in N_K_FINE_VALUES:
    print(f"\n--- n_k_fine = {n_k_fine} ---", flush=True)
    t0 = time.time()
    cl_tt = compute_cl_tt_interp(pt, params, bg, ELLS, n_k_fine=n_k_fine)
    cl_ee = compute_cl_ee_interp(pt, params, bg, ELLS, n_k_fine=n_k_fine)
    cl_te = compute_cl_te_interp(pt, params, bg, ELLS, n_k_fine=n_k_fine)
    t1 = time.time()
    print(f"  All spectra done in {t1-t0:.1f}s", flush=True)

    errs = {'TT': {}, 'EE': {}, 'TE': {}}
    for i, ell in enumerate(ELLS):
        idx = np.argmin(np.abs(ell_ref - ell))
        for spec, cl_arr, ref_key in [('TT', cl_tt, 'tt'), ('EE', cl_ee, 'ee'), ('TE', cl_te, 'te')]:
            cl_class = ref_cls[ref_key][idx]
            if abs(cl_class) > 1e-30:
                errs[spec][ell] = (float(cl_arr[i]) - cl_class) / abs(cl_class) * 100
            else:
                errs[spec][ell] = 0.0
    results[n_k_fine] = errs

# Summary
for spec in ['TT', 'EE', 'TE']:
    print(f"\n{'='*60}")
    print(f"  {spec} error (%) vs n_k_fine")
    print(f"{'='*60}")
    header = f"{'l':>6}"
    for nk in N_K_FINE_VALUES:
        header += f" {'nk='+str(nk):>12}"
    print(header)
    print("-" * (6 + 13 * len(N_K_FINE_VALUES)))
    for ell in ELLS:
        row = f"{ell:6d}"
        for nk in N_K_FINE_VALUES:
            row += f" {results[nk][spec][ell]:+12.3f}"
        print(row)

print("\nDone!", flush=True)
