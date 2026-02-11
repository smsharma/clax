"""Planck-quality C_l diagnostic: test ODE tolerance and TT mode effects."""
import sys
sys.path.insert(0, '.')
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
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
ells = [10, 20, 30, 50, 100, 150, 200, 300, 500, 700, 1000]

def run_tt_test(label, prec, tt_mode="T0+T1+T2"):
    print(f"\n{'='*60}")
    print(f"{label} (mode={tt_mode})")
    print(f"  l_max_g={prec.pt_l_max_g}, rtol={prec.pt_ode_rtol}")
    print(f"{'='*60}", flush=True)
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    print(f"n_k={len(pt.k_grid)}, n_tau={len(pt.tau_grid)}", flush=True)

    cl_tt = compute_cl_tt_interp(pt, params, bg, ells, n_k_fine=3000, tt_mode=tt_mode)

    print(f"\n  C_l^TT ({tt_mode}):")
    for i, ell in enumerate(ells):
        idx = np.argmin(np.abs(ell_ref - ell))
        cl_class = ref_cls['tt'][idx]
        cl_ours = float(cl_tt[i])
        if abs(cl_class) > 1e-30:
            err = (cl_ours - cl_class) / cl_class * 100
            marker = " ***" if abs(err) < 0.1 else (" **" if abs(err) < 1.0 else "")
            print(f"    l={ell:5d}: err={err:+.3f}%{marker}", flush=True)

prec_base = PrecisionParams.planck_cl()

# Test 1: Baseline T0+T1+T2
run_tt_test("BASELINE", prec_base, "T0+T1+T2")

# Test 2: T0 only (IBP monopole) — no T1, T2
run_tt_test("T0 ONLY", prec_base, "T0")

# Test 3: T0+T2 (no T1)
run_tt_test("T0+T2", prec_base, "T0+T2")

# Test 4: Tighter ODE with full T0+T1+T2
prec_tight = PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=5000,
    pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-8, pt_ode_atol=1e-13, ode_max_steps=262144,
)
run_tt_test("TIGHTER ODE", prec_tight, "T0+T1+T2")

# Test 5: Higher l_max=65 (should fit in V100 32GB with fewer k)
prec_lmax = PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=40, pt_tau_n_points=5000,
    pt_l_max_g=65, pt_l_max_pol_g=65, pt_l_max_ur=65,
    pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
)
run_tt_test("L_MAX=65 (40 k/dec)", prec_lmax, "T0+T1+T2")

print("\n\nDone!", flush=True)
