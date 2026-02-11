"""Diagnostic: decompose TT into T0, T1, T2 contributions and test l_max sensitivity."""
import sys
sys.path.insert(0, '.')
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']

ells = [10, 20, 30, 50, 100, 200, 500, 700, 1000]

# ---- Test 1: decompose T0 vs T0+T1+T2 at planck_cl ----
print("\n=== Test 1: T0 vs T0+T1+T2 (planck_cl, l_max=50) ===", flush=True)
prec = PrecisionParams.planck_cl()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)

cl_T0 = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000, tt_mode="T0")
cl_T0T1 = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000, tt_mode="T0+T1")
cl_T0T1T2 = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000, tt_mode="T0+T1+T2")

print(f"\n{'l':>6s} {'CLASS':>12s} {'T0 err':>10s} {'T0T1 err':>10s} {'T0T1T2 err':>10s}")
for i, ell in enumerate(ells):
    idx = np.argmin(np.abs(ell_ref - ell))
    cl_class = ref_cls['tt'][idx]
    if abs(cl_class) < 1e-30:
        continue
    e0 = (float(cl_T0[i]) - cl_class) / cl_class * 100
    e01 = (float(cl_T0T1[i]) - cl_class) / cl_class * 100
    e012 = (float(cl_T0T1T2[i]) - cl_class) / cl_class * 100
    print(f"  {ell:5d} {cl_class:12.4e} {e0:+9.2f}% {e01:+9.2f}% {e012:+9.2f}%", flush=True)

# ---- Test 2: l_max sensitivity ----
print("\n=== Test 2: l_max sensitivity (k_max=1.0) ===", flush=True)
for lmax_g in [50, 80]:
    print(f"\n--- l_max_g = {lmax_g} ---", flush=True)
    prec2 = PrecisionParams(
        pt_k_max_cl=1.0,
        pt_k_per_decade=60,
        pt_tau_n_points=5000,
        pt_l_max_g=lmax_g,
        pt_l_max_pol_g=lmax_g,
        pt_l_max_ur=lmax_g,
        pt_ode_rtol=1e-6,
        pt_ode_atol=1e-11,
        ode_max_steps=131072,
    )
    bg2 = background_solve(params, prec2)
    th2 = thermodynamics_solve(params, prec2, bg2)
    pt2 = perturbations_solve(params, prec2, bg2, th2)
    print(f"n_k={len(pt2.k_grid)}, n_tau={len(pt2.tau_grid)}", flush=True)

    cl_tt = compute_cl_tt_interp(pt2, params, bg2, ells, n_k_fine=3000, tt_mode="T0+T1+T2")

    for i, ell in enumerate(ells):
        idx = np.argmin(np.abs(ell_ref - ell))
        cl_class = ref_cls['tt'][idx]
        if abs(cl_class) < 1e-30:
            continue
        err = (float(cl_tt[i]) - cl_class) / cl_class * 100
        sub = " ***" if abs(err) < 1.0 else ""
        print(f"  l={ell:5d}: err={err:+.2f}%{sub}", flush=True)

print("\nDone!", flush=True)
