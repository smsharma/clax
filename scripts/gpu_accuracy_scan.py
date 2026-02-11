"""Systematic accuracy diagnostic: test l_max and tau resolution effects on C_l."""
import sys
sys.path.insert(0, '.')
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']
ells = [10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000]

def run_test(label, prec):
    print(f"\n{'='*60}")
    print(f"CONFIG: {label}")
    print(f"  l_max_g={prec.pt_l_max_g}, n_tau={prec.pt_tau_n_points}, "
          f"k_max={prec.pt_k_max_cl}, k/dec={prec.pt_k_per_decade}")
    print(f"{'='*60}", flush=True)

    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

    cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000)
    cl_ee = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=3000)

    for spec, cl, key in [("TT", cl_tt, 'tt'), ("EE", cl_ee, 'ee')]:
        print(f"\n  C_l^{spec}:")
        for i, ell in enumerate(ells):
            idx = np.argmin(np.abs(ell_ref - ell))
            cl_class = ref_cls[key][idx]
            cl_ours = float(cl[i])
            if abs(cl_class) > 1e-30:
                err = (cl_ours - cl_class) / cl_class * 100
                marker = " ***" if abs(err) < 0.1 else (" **" if abs(err) < 1.0 else "")
                print(f"    l={ell:5d}: err={err:+.3f}%{marker}", flush=True)

# Baseline: planck_cl preset (l_max=50, 5000 tau)
run_test("BASELINE (l_max=50, tau=5000)", PrecisionParams.planck_cl())

# Test 1: Higher hierarchy l_max
run_test("l_max=80", PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=5000,
    pt_l_max_g=80, pt_l_max_pol_g=80, pt_l_max_ur=80,
    pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
))

# Test 2: More tau points
run_test("tau=8000", PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=8000,
    pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
))

# Test 3: Tighter ODE tolerance
run_test("tighter_ode (rtol=1e-8)", PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=5000,
    pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-8, pt_ode_atol=1e-13, ode_max_steps=262144,
))

print("\n\nDone!", flush=True)
