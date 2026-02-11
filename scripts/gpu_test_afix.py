"""Test impact of a''/a fix on TT C_l accuracy.

Tests the fix of Bug #22: a_primeprime_over_a in TCA slip formula was
(aH)^2 + a*H'_tau (which is (a'/a)'), but should be 2*(aH)^2 + a*H'_tau
(which is a''/a). cf. CLASS perturbations.c:10032.

Run: python scripts/gpu_test_afix.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("Devices:", jax.devices(), flush=True)

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp

# Setup
p = CosmoParams()
pp = PrecisionParams.planck_cl()
print(f"Preset: planck_cl (l_max_g={pp.pt_l_max_g}, k_max={pp.pt_k_max_cl})", flush=True)

bg = background_solve(p, pp)
th = thermodynamics_solve(p, pp, bg)
print(f"tau_star = {float(th.tau_star):.2f} Mpc", flush=True)

pt = perturbations_solve(p, pp, bg, th)
print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

# Reference
ref = np.load('reference_data/lcdm_fiducial/cls.npz')

# Test l values focused on the problem region
ells = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000]

print("\nComputing TT (source-interp, 3000 fine k)...", flush=True)
cl_tt = compute_cl_tt_interp(pt, p, bg, ells, n_k_fine=3000)

print("Computing EE...", flush=True)
cl_ee = compute_cl_ee_interp(pt, p, bg, ells, n_k_fine=3000)

print("Computing TE...", flush=True)
cl_te = compute_cl_te_interp(pt, p, bg, ells, n_k_fine=3000)

# Results
print("\n" + "="*70)
print("C_l ACCURACY (after a''/a fix in TCA slip)")
print("="*70)

for spec, cl, key in [("TT", cl_tt, 'tt'), ("EE", cl_ee, 'ee'), ("TE", cl_te, 'te')]:
    print(f"\nC_l^{spec}:")
    for i, ell in enumerate(ells):
        cl_ref = ref[key][ell]
        cl_ours = float(cl[i])
        if abs(cl_ref) > 1e-30:
            err = (cl_ours - cl_ref) / abs(cl_ref) * 100
            marker = " ***" if abs(err) < 1.0 else " **" if abs(err) < 2.0 else ""
            print(f"  l={ell:5d}: err={err:+7.3f}%{marker}", flush=True)

print("\nDone!", flush=True)
