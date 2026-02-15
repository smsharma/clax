#!/usr/bin/env python -u
"""Quick multi-cosmology validation: 3 parameter points.

Tests omega_b_high, omega_cdm_low, h_high against CLASS reference.
Uses medium_cl preset to fit in V100-32GB.

Usage:
    python -u scripts/diag_multicosmo_quick.py
"""
import sys
import time
sys.path.insert(0, '.')
# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cls_all_interp

# medium_cl fits in V100-32GB (89 k-modes vs planck_cl 300)
prec = replace(PrecisionParams.medium_cl(), th_n_points=100000, ode_max_steps=262144)

# 3 representative cosmologies
VARIATIONS = {
    'omega_b_high': {'omega_b': 0.02237 * 1.20},
    'omega_cdm_low': {'omega_cdm': 0.1200 * 0.80},
    'h_high': {'h': 0.6736 * 1.10},
}

TEST_ELLS = [20, 50, 100, 200, 300, 500]

print("=" * 70)
print("Quick multi-cosmology validation (3 points, medium_cl)")
print("=" * 70)

for i, (name, overrides) in enumerate(VARIATIONS.items()):
    print(f"\n[{i+1}/{len(VARIATIONS)}] {name}: {overrides}")
    cls_ref = dict(np.load(f'reference_data/{name}/cls.npz'))

    params = replace(CosmoParams(), **overrides)
    t0 = time.time()

    bg = background_solve(params, prec)
    print(f"  bg: {time.time()-t0:.0f}s")

    th = thermodynamics_solve(params, prec, bg)
    print(f"  th: {time.time()-t0:.0f}s")

    pt = perturbations_solve(params, prec, bg, th)
    print(f"  pt: {time.time()-t0:.0f}s")

    cls = compute_cls_all_interp(pt, params, bg, l_max=2500, n_k_fine=5000)
    print(f"  cls: {time.time()-t0:.0f}s")

    print(f"  {'l':>6}  {'TT%':>8}  {'EE%':>8}  {'TE%':>8}")
    for l in TEST_ELLS:
        tt_ref = cls_ref['tt'][l]
        ee_ref = cls_ref['ee'][l]
        te_ref = cls_ref['te'][l]
        tt_err = (cls['tt'][l] - tt_ref) / abs(tt_ref) * 100 if abs(tt_ref) > 1e-30 else 0
        ee_err = (cls['ee'][l] - ee_ref) / abs(ee_ref) * 100 if abs(ee_ref) > 1e-30 else 0
        te_err = (cls['te'][l] - te_ref) / abs(te_ref) * 100 if abs(te_ref) > 1e-30 else 0
        print(f"  {l:6d}  {float(tt_err):+8.3f}  {float(ee_err):+8.3f}  {float(te_err):+8.3f}")

print("\nDONE")
