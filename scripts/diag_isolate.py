"""Isolate C_l error sources: T0-only, tighter ODE, more tau points."""
import sys
sys.path.insert(0, '.')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

print("Devices:", jax.devices(), flush=True)
params = CosmoParams()

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']
ells = [20, 30, 50, 100, 200, 300, 500, 700, 1000]

def run_test(label, prec, tt_mode="T0+T1+T2"):
    print(f"\n{'='*60}")
    print(f"{label} (mode={tt_mode})")
    print(f"  l_max_g={prec.pt_l_max_g}, rtol={prec.pt_ode_rtol}, n_tau={prec.pt_tau_n_points}")
    print(f"{'='*60}", flush=True)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    print(f"  n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

    cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000, tt_mode=tt_mode)

    print(f"\n  {'l':>6} {'TT err%':>10}")
    print(f"  {'-'*20}")
    for i, ell in enumerate(ells):
        idx = np.argmin(np.abs(ell_ref - ell))
        tt_class = ref_cls['tt'][idx]
        tt_err = (float(cl_tt[i]) - tt_class) / abs(tt_class) * 100
        mark = " ***" if abs(tt_err) < 0.1 else (" **" if abs(tt_err) < 0.5 else (" *" if abs(tt_err) < 1.0 else ""))
        print(f"  {ell:6d} {tt_err:+10.3f}{mark}", flush=True)

# Test 1: Baseline T0+T1+T2
prec_base = PrecisionParams.planck_cl()
run_test("BASELINE T0+T1+T2", prec_base, "T0+T1+T2")

# Test 2: T0 only (IBP monopole - no T1, T2)
run_test("T0 ONLY", prec_base, "T0")

# Test 3: Tighter ODE tolerance
prec_tight = PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=5000,
    pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-8, pt_ode_atol=1e-13, ode_max_steps=262144,
)
run_test("TIGHT ODE (rtol=1e-8)", prec_tight, "T0+T1+T2")

# Test 4: More tau points
prec_moretau = PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=10000,
    pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
)
run_test("MORE TAU (n=10000)", prec_moretau, "T0+T1+T2")

print("\n\nDone!", flush=True)
