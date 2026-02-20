"""Fast C_l diagnostic v2: compare against CLASS massless-ncdm RECFAST reference.

This is the apples-to-apples comparison: both use massless ncdm + RECFAST.
Any remaining error is purely from our CODE, not from ncdm physics.
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

N_K_FINE = 10000

params = CosmoParams(m_ncdm=0.0)  # massless ncdm
prec = PrecisionParams.planck_cl()

# Load CLASS massless-ncdm RECFAST reference (apples-to-apples)
ref = np.load('reference_data/cls_massless_recfast.npz')
ell_ref = ref['ell']

ells = [20, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000]

t0 = time.time()
print("Running background...", flush=True)
bg = background_solve(params, prec)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("Running thermodynamics...", flush=True)
th = thermodynamics_solve(params, prec, bg)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("Running perturbations...", flush=True)
pt = perturbations_solve(params, prec, bg, th)
print(f"  n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)} ({time.time()-t0:.0f}s)", flush=True)

print(f"\nComputing C_l^TT (n_k_fine={N_K_FINE})...", flush=True)
cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=N_K_FINE)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("Computing C_l^EE...", flush=True)
cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=N_K_FINE)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("Computing C_l^TE...", flush=True)
cl_te = compute_cl_te_interp(pt, params, bg, ells, n_k_fine=N_K_FINE)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

def get_err(our_val, ref_ell, ref_spec, ell):
    idx = np.argmin(np.abs(ref_ell - ell))
    ref_val = ref_spec[idx]
    if abs(ref_val) > 1e-30:
        return (float(our_val) - ref_val) / abs(ref_val) * 100
    return 0.0

def mark(err):
    if abs(err) < 0.1: return " ***"
    if abs(err) < 0.25: return " **"
    if abs(err) < 0.5: return " *"
    return ""

print(f"\n{'='*70}")
print(f"{'vs CLASS massless-ncdm RECFAST (apples-to-apples CODE test)':^70}")
print(f"{'='*70}")
print(f"{'l':>6} {'TT err%':>10} {'EE err%':>10} {'TE err%':>10}")
print("-" * 46)
tt_pass = ee_pass = te_pass = 0
for i, ell in enumerate(ells):
    tt_e = get_err(cl_tt[i], ell_ref, ref['tt'], ell)
    ee_e = get_err(cl_ee[i], ell_ref, ref['ee'], ell)
    te_e = get_err(cl_te[i], ell_ref, ref['te'], ell)
    if abs(tt_e) < 0.1: tt_pass += 1
    if abs(ee_e) < 0.1: ee_pass += 1
    if abs(te_e) < 0.1: te_pass += 1
    print(f"{ell:6d} {tt_e:+10.4f}{mark(tt_e)} {ee_e:+10.4f}{mark(ee_e)} {te_e:+10.4f}{mark(te_e)}")
print(f"\nSub-0.1%: TT {tt_pass}/{len(ells)}, EE {ee_pass}/{len(ells)}, TE {te_pass}/{len(ells)}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done!")
