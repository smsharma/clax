"""Diagnostic: compare IBP vs nonIBP TT transfer modes after theta_b' fix.

Measures C_l^TT accuracy at several l-values using both modes to quantify
the impact of Bug #1 (theta_b' extraction mismatch) fix.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt, compute_cl_ee, compute_cl_te

# Load CLASS reference
ref = np.load('reference_data/lcdm_fiducial/cls.npz')

# Run pipeline at fast_cl resolution
params = CosmoParams()
prec = PrecisionParams.fast_cl()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)

l_values = [10, 30, 50, 100, 200]

print("=" * 70)
print("C_l^TT: IBP (T0) vs nonIBP modes — after theta_b' fix + TCA dual criteria")
print("=" * 70)

for mode_name, mode in [("T0 (IBP)", "T0"), ("nonIBP", "nonIBP")]:
    cl_tt = compute_cl_tt(pt, params, bg, l_values, tt_mode=mode)
    print(f"\n  Mode: {mode_name}")
    for i, l in enumerate(l_values):
        cl_us = float(cl_tt[i])
        cl_class = float(ref['tt'][l])
        if cl_class != 0:
            ratio = cl_us / cl_class
            err = abs(ratio - 1) * 100
            print(f"    l={l:4d}: ratio={ratio:.4f}  err={err:.1f}%  (jax={cl_us:.4e}, CLASS={cl_class:.4e})")

print("\n" + "=" * 70)
print("C_l^EE")
print("=" * 70)
cl_ee = compute_cl_ee(pt, params, bg, l_values)
for i, l in enumerate(l_values):
    cl_us = float(cl_ee[i])
    cl_class = float(ref['ee'][l])
    if cl_class != 0:
        ratio = cl_us / cl_class
        err = abs(ratio - 1) * 100
        print(f"  l={l:4d}: ratio={ratio:.4f}  err={err:.1f}%")

print("\n" + "=" * 70)
print("C_l^TE")
print("=" * 70)
cl_te = compute_cl_te(pt, params, bg, l_values)
for i, l in enumerate(l_values):
    cl_us = float(cl_te[i])
    cl_class = float(ref['te'][l])
    if cl_class != 0:
        ratio = cl_us / cl_class
        err = abs(ratio - 1) * 100
        print(f"  l={l:4d}: ratio={ratio:.4f}  err={err:.1f}%")
