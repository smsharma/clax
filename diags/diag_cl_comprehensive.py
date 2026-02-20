"""Comprehensive C_l diagnostic: dense l-sampling + n_k_fine convergence test.

Runs perturbation solve once, then computes C_l at n_k_fine=10000 and 20000
for many l values to identify both accuracy bottlenecks and k-convergence.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp

params = CosmoParams()
prec = PrecisionParams.planck_cl()

# Load CLASS RECFAST reference
ref = np.load('reference_data/lcdm_fiducial/cls_recfast_thermo.npz')
ell_ref = ref['cls_ell']

# Dense l-sampling, especially around 400-500 and high-l
ells = [20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 420, 450, 480, 500,
        550, 600, 650, 700, 800, 900, 1000, 1200, 1500, 2000]

t0 = time.time()
print("Running pipeline...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"Pipeline: {time.time()-t0:.0f}s (n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)})", flush=True)

def get_err(val, ell):
    idx = np.argmin(np.abs(ell_ref - ell))
    rv = ref['cls_tt'][idx] if 'tt' in str(type(val)) else ref['cls_tt'][idx]
    return 0.0

def compute_and_compare(n_k_fine, label):
    """Compute TT/EE C_l and compare vs CLASS RECFAST."""
    t1 = time.time()
    cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=n_k_fine)
    cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=n_k_fine)
    cl_te = compute_cl_te_interp(pt, params, bg, ells, n_k_fine=n_k_fine)
    dt = time.time() - t1

    print(f"\n{'='*70}")
    print(f"{label} (n_k_fine={n_k_fine}, {dt:.0f}s)")
    print(f"{'='*70}")
    print(f"{'l':>6} {'TT err%':>10} {'EE err%':>10} {'TE err%':>10}")
    print("-" * 46)

    tt_pass = ee_pass = te_pass = 0
    for i, ell in enumerate(ells):
        idx = int(ell)  # ell array is [0,1,...,2500]
        tt_cls = ref['cls_tt'][idx]
        ee_cls = ref['cls_ee'][idx]
        te_cls = ref['cls_te'][idx]

        tt_e = (float(cl_tt[i]) - tt_cls) / abs(tt_cls) * 100 if abs(tt_cls) > 1e-30 else 0
        ee_e = (float(cl_ee[i]) - ee_cls) / abs(ee_cls) * 100 if abs(ee_cls) > 1e-30 else 0
        te_e = (float(cl_te[i]) - te_cls) / abs(te_cls) * 100 if abs(te_cls) > 1e-30 else 0

        mk = lambda e: " ***" if abs(e) < 0.1 else (" **" if abs(e) < 0.25 else (" *" if abs(e) < 0.5 else ""))
        if abs(tt_e) < 0.1: tt_pass += 1
        if abs(ee_e) < 0.1: ee_pass += 1
        if abs(te_e) < 0.1: te_pass += 1
        print(f"{ell:6d} {tt_e:+10.4f}{mk(tt_e)} {ee_e:+10.4f}{mk(ee_e)} {te_e:+10.4f}{mk(te_e)}")

    print(f"Sub-0.1%: TT {tt_pass}/{len(ells)}, EE {ee_pass}/{len(ells)}, TE {te_pass}/{len(ells)}")
    return cl_tt, cl_ee, cl_te

# Run at two n_k_fine values
cl_tt_10k, cl_ee_10k, cl_te_10k = compute_and_compare(10000, "n_k_fine=10000")
cl_tt_20k, cl_ee_20k, cl_te_20k = compute_and_compare(20000, "n_k_fine=20000")

# Convergence: difference between 10k and 20k
print(f"\n{'='*70}")
print(f"{'k-convergence: |10k - 20k| / |20k|':^70}")
print(f"{'='*70}")
print(f"{'l':>6} {'TT Δ%':>10} {'EE Δ%':>10}")
print("-" * 30)
for i, ell in enumerate(ells):
    tt_conv = abs(float(cl_tt_10k[i]) - float(cl_tt_20k[i])) / abs(float(cl_tt_20k[i])) * 100
    ee_conv = abs(float(cl_ee_10k[i]) - float(cl_ee_20k[i])) / abs(float(cl_ee_20k[i])) * 100
    print(f"{ell:6d} {tt_conv:10.4f} {ee_conv:10.4f}")

print(f"\nTotal: {time.time()-t0:.0f}s")
print("Done!")
