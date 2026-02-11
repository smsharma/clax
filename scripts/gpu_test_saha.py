"""Test Saha equilibrium fix: x_e accuracy + C_l TT/EE.

Uses th_z/th_x_e dense reference for x_e, then runs C_l computation.
"""
import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

print(f"JAX device: {jax.devices()}", flush=True)

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve

params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Solving background + thermo...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)

# Check x_e using dense CLASS reference (th_z, th_x_e)
ref = np.load('reference_data/lcdm_fiducial/thermodynamics.npz')
thz = ref['th_z']
txe = ref['th_x_e']

z_test = np.array([2000, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1089, 1000, 900])
xe_class = np.interp(z_test, thz, txe)

print(f"\n{'z':>6s}  {'x_e(ours)':>12s}  {'x_e(CLASS)':>12s}  {'err%':>8s}")
print("=" * 50)
for i, z in enumerate(z_test):
    loga = float(jnp.log(1.0 / (1.0 + z)))
    xe_ours = float(th.xe_of_loga.evaluate(jnp.array(loga)))
    err = (xe_ours - xe_class[i]) / max(abs(xe_class[i]), 1e-30) * 100
    print(f"{z:6.0f}  {xe_ours:12.6f}  {xe_class[i]:12.6f}  {err:+8.3f}%")

# Compute C_l
print("\nSolving perturbations...", flush=True)
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

pt = perturbations_solve(params, prec, bg, th)

l_test = [20, 50, 100, 200, 300, 500, 700, 1000]
print("Computing TT...", flush=True)
cl_tt = compute_cl_tt_interp(pt, params, bg, l_test)
print("Computing EE...", flush=True)
cl_ee = compute_cl_ee_interp(pt, params, bg, l_test)

# Load CLASS reference C_l
ref_cl = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cl['ell']
tt_ref = ref_cl['tt']
ee_ref = ref_cl['ee']

print(f"\n{'l':>6s}  {'TT_err':>10s}  {'EE_err':>10s}")
print("=" * 35)
for i, l in enumerate(l_test):
    idx_ref = np.searchsorted(ell_ref, l)
    tt_r = tt_ref[idx_ref]
    ee_r = ee_ref[idx_ref]
    tt_e = float((cl_tt[i] - tt_r) / abs(tt_r) * 100)
    ee_e = float((cl_ee[i] - ee_r) / abs(ee_r) * 100)
    star_tt = "***" if abs(tt_e) < 1.0 else ""
    star_ee = "***" if abs(ee_e) < 1.0 else ""
    print(f"{l:6d}  {tt_e:+10.3f}% {star_tt:3s}  {ee_e:+10.3f}% {star_ee:3s}")

print("\nDone!", flush=True)
