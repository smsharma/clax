"""Diagnose TT C_l errors: compare IBP vs nonIBP, all TT modes, at science_cl.

Tests multiple TT transfer modes to identify the error source:
- "T0" (IBP): all terms in j_l radial
- "nonIBP": Doppler via j_l', rest via j_l
- "T0+T1": adds ISW dipole
- "T0+T1+T2": CLASS full form (adds quadrupole)
"""
import sys
sys.path.insert(0, '.')
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt, compute_cl_ee

print("Devices:", jax.devices())

params = CosmoParams()
prec = PrecisionParams.science_cl()

print("Solving background + thermo + perturbations (science_cl)...")
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']
cl_tt_ref = ref_cls['tt']
cl_ee_ref = ref_cls['ee']

ells = [10, 20, 30, 50, 100, 150, 200]

# Test all TT modes
modes = ["T0", "nonIBP", "T0+T1", "T0+T1+T2"]
for mode in modes:
    print(f"\n--- TT mode: {mode} ---")
    cl_tt = compute_cl_tt(pt, params, bg, ells, l_switch=100000, tt_mode=mode)
    for i, ell in enumerate(ells):
        idx = np.argmin(np.abs(ell_ref - ell))
        err = (float(cl_tt[i]) - cl_tt_ref[idx]) / cl_tt_ref[idx] * 100
        fac = ell*(ell+1)/(2*np.pi)
        print(f"  l={ell:4d}: err={err:+.2f}%")

# EE for comparison (should be 1-3%)
print(f"\n--- EE (for reference) ---")
cl_ee = compute_cl_ee(pt, params, bg, ells, l_switch=100000)
for i, ell in enumerate(ells):
    idx = np.argmin(np.abs(ell_ref - ell))
    err = (float(cl_ee[i]) - cl_ee_ref[idx]) / max(abs(cl_ee_ref[idx]), 1e-30) * 100
    print(f"  l={ell:4d}: err={err:+.2f}%")

# Also test: what if we use CLASS g(tau) for sources?
# This isolates whether the error is in thermo or perturbations.
print("\n--- Source function diagnostics ---")
print(f"  k_grid: {len(pt.k_grid)} modes, k_min={float(pt.k_grid[0]):.5f}, k_max={float(pt.k_grid[-1]):.4f}")
print(f"  tau_grid: {len(pt.tau_grid)} points, tau_min={float(pt.tau_grid[0]):.1f}, tau_max={float(pt.tau_grid[-1]):.1f}")

# Check source T0 peak amplitude
for ik_test in [0, len(pt.k_grid)//4, len(pt.k_grid)//2, 3*len(pt.k_grid)//4, -1]:
    k = float(pt.k_grid[ik_test])
    src_max = float(jnp.max(jnp.abs(pt.source_T0[ik_test])))
    src_sw_max = float(jnp.max(jnp.abs(pt.source_SW[ik_test])))
    src_dop_max = float(jnp.max(jnp.abs(pt.source_Doppler[ik_test])))
    src_isw_vis_max = float(jnp.max(jnp.abs(pt.source_ISW_vis[ik_test])))
    print(f"  k={k:.4f}: max|S_T0|={src_max:.3e}, max|SW|={src_sw_max:.3e}, "
          f"max|Dop|={src_dop_max:.3e}, max|ISW_vis|={src_isw_vis_max:.3e}")

print("\nDone!")
