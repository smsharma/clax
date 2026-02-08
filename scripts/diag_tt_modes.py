#!/usr/bin/env python3
"""Diagnostic: isolate TT transfer decomposition impact.

Single perturbation solve, then sweeps TT modes in-memory.
Prints hash of harmonic.py for traceability.

Usage:
    python scripts/diag_tt_modes.py
    # Or on Bridges2:
    python diag_tt_modes.py
"""
import hashlib
import os
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
import jaxclass.harmonic as hm

# 1) Preflight: prove which harmonic.py is active
harm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "jaxclass", "harmonic.py")
if not os.path.exists(harm_path):
    harm_path = os.path.join(os.path.dirname(__file__), "..", "jaxclass", "harmonic.py")
with open(harm_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()[:16]
print(f"harmonic.py: {harm_path}")
print(f"sha256[:16]: {sha}")
print(f"_DEFAULT_TT_MODE: {getattr(hm, '_DEFAULT_TT_MODE', 'MISSING')}")

# 2) Config
params = CosmoParams()
prec = PrecisionParams.science_cl()
ells = [10, 30, 50, 100, 200]
modes = ["T0", "T0+T1", "T0+T1+T2", "T0-T1+T2"]

n_k = int(jnp.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade)
print(f"preset: science_cl ({n_k} k-modes, l_max={prec.pt_l_max_g})", flush=True)

# 3) Single expensive solve
t0 = time.time()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"solve_time_s={time.time()-t0:.1f}", flush=True)

# 4) Load CLASS TT reference
ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference_data", "lcdm_fiducial", "cls.npz")
if not os.path.exists(ref_path):
    ref_path = os.path.join(os.path.dirname(__file__), "..", "reference_data", "lcdm_fiducial", "cls.npz")
ref = np.load(ref_path, allow_pickle=True)
tt_ref = ref["tt"]

# 5) Mode sweep without re-solving perturbations
print(f"\n{'l':>3} | {'mode':>11} | {'ratio':>8} | {'err':>7}", flush=True)
print("-" * 38, flush=True)
for l in ells:
    for mode in modes:
        cl = float(hm.compute_cl_tt(
            pt, params, bg, [l],
            k_interp_factor=3, l_switch=100000, delta_l=50,
            tt_mode=mode,
        )[0])
        ratio = cl / float(tt_ref[l])
        err = abs(ratio - 1) * 100
        print(f"{l:3d} | {mode:>11} | {ratio: .4f} | {err:5.1f}%", flush=True)
    print("-" * 38, flush=True)

print("Done", flush=True)
