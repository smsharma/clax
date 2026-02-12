"""Test if better thermodynamics precision improves EE accuracy."""
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
ells = [100, 200, 300, 500, 700, 1000]

# Test 1: Baseline thermodynamics
prec_base = PrecisionParams.planck_cl()
print("Baseline thermo:", flush=True)
bg = background_solve(params, prec_base)
th = thermodynamics_solve(params, prec_base, bg)
from jaxclass.thermodynamics import xe_of_z
print(f"  x_e(z_star)={float(xe_of_z(th, float(th.z_star))):.6f}", flush=True)
print(f"  z_star={float(th.z_star):.2f}", flush=True)
pt = perturbations_solve(params, prec_base, bg, th)
cl_ee_base = compute_cl_ee_interp(pt, params, bg, ells, n_k_fine=10000)

# Test 2: Higher thermo precision
prec_hires = PrecisionParams(
    pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=5000,
    pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
    th_n_points=20000, th_z_max=1e4,  # 4x more thermo points
)
print("\nHigh-res thermo:", flush=True)
th2 = thermodynamics_solve(params, prec_hires, bg)
print(f"  x_e(z_star)={float(xe_of_z(th2, float(th2.z_star))):.6f}", flush=True)
print(f"  z_star={float(th2.z_star):.2f}", flush=True)
pt2 = perturbations_solve(params, prec_hires, bg, th2)
cl_ee_hires = compute_cl_ee_interp(pt2, params, bg, ells, n_k_fine=10000)

print(f"\n{'l':>6} {'EE base%':>10} {'EE hires%':>10} {'change':>10}")
for i, ell in enumerate(ells):
    idx = np.argmin(np.abs(ell_ref - ell))
    ee_class = ref_cls['ee'][idx]
    err_base = (float(cl_ee_base[i]) - ee_class) / abs(ee_class) * 100
    err_hires = (float(cl_ee_hires[i]) - ee_class) / abs(ee_class) * 100
    print(f"{ell:6d} {err_base:+10.3f} {err_hires:+10.3f} {err_hires-err_base:+10.4f}")

print("\nDone!", flush=True)
