"""Timing test: baseline vs fewer k-modes vs float32."""
import sys, time, os
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

# ============================================================
# Test 1: Baseline (float64, planck_cl defaults)
# ============================================================
print("=" * 60)
print("TEST 1: Baseline (float64, planck_cl defaults)")
print("=" * 60)
jax.config.update("jax_enable_x64", True)

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp

params = CosmoParams(m_ncdm=0.0)
prec = PrecisionParams.planck_cl()
ells = [20, 100, 500, 1000]

print(f"k_per_decade={prec.pt_k_per_decade}, k_max={prec.pt_k_max_cl}")

# First call (JIT compile + run)
t0 = time.time()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
cl1 = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=10000)
t_first = time.time() - t0
print(f"First call (compile+run): {t_first:.1f}s")
print(f"  C_l = {[f'{float(c):.6e}' for c in cl1]}")

# Second call (cached)
params2 = CosmoParams(m_ncdm=0.0, h=0.70, omega_cdm=0.13)
t0 = time.time()
bg2 = background_solve(params2, prec)
th2 = thermodynamics_solve(params2, prec, bg2)
pt2 = perturbations_solve(params2, prec, bg2, th2)
cl2 = compute_cl_tt_interp(pt2, params2, bg2, ells, n_k_fine=10000)
t_second = time.time() - t0
print(f"Second call (cached): {t_second:.1f}s")

# Save baseline for accuracy comparison
cl1_baseline = np.array([float(c) for c in cl1])

# ============================================================
# Test 2: Fewer k-modes (30 k/decade instead of 60)
# ============================================================
print()
print("=" * 60)
print("TEST 2: Fewer k-modes (30 k/decade, float64)")
print("=" * 60)

prec_fast = PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=30, pt_tau_n_points=5000,
    th_n_points=100000, pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-6, pt_ode_atol=1e-12, ode_max_steps=131072,
)

t0 = time.time()
bg_f = background_solve(params, prec_fast)
th_f = thermodynamics_solve(params, prec_fast, bg_f)
pt_f = perturbations_solve(params, prec_fast, bg_f, th_f)
cl_fast = compute_cl_tt_interp(pt_f, params, bg_f, ells, n_k_fine=10000)
t_first_fast = time.time() - t0
print(f"First call (compile+run): {t_first_fast:.1f}s")

# Second call
t0 = time.time()
bg2_f = background_solve(params2, prec_fast)
th2_f = thermodynamics_solve(params2, prec_fast, bg2_f)
pt2_f = perturbations_solve(params2, prec_fast, bg2_f, th2_f)
cl2_fast = compute_cl_tt_interp(pt2_f, params2, bg2_f, ells, n_k_fine=10000)
t_second_fast = time.time() - t0
print(f"Second call (cached): {t_second_fast:.1f}s")

cl_fast_arr = np.array([float(c) for c in cl_fast])
err = (cl_fast_arr - cl1_baseline) / np.abs(cl1_baseline) * 100
print(f"  vs baseline: {[f'{e:+.3f}%' for e in err]}")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline f64 (60 k/dec):  1st={t_first:.0f}s  2nd={t_second:.0f}s")
print(f"Fewer k (30 k/dec) f64:   1st={t_first_fast:.0f}s  2nd={t_second_fast:.0f}s")
