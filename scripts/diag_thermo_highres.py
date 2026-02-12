"""Test if higher thermo resolution improves C_l."""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp

print(f"Devices: {jax.devices()}", flush=True)

params = CosmoParams()
ELLS = [20, 30, 100, 300, 500, 700, 1000]
N_K_FINE = 5000  # fast, converged at l<=500

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref_cls['ell']

bg = background_solve(params, PrecisionParams.planck_cl())

for n_th in [20000, 100000]:
    print(f"\n--- th_n_points = {n_th} ---", flush=True)
    prec = PrecisionParams(
        pt_k_max_cl=1.0, pt_k_per_decade=60, pt_tau_n_points=5000,
        pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
        pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
        th_n_points=n_th, th_z_max=5e4,
    )
    th = thermodynamics_solve(params, prec, bg)
    t0 = time.time()
    pt = perturbations_solve(params, prec, bg, th)
    print(f"  Perturbations: {time.time()-t0:.1f}s", flush=True)

    cl_tt = compute_cl_tt_interp(pt, params, bg, ELLS, n_k_fine=N_K_FINE)
    cl_ee = compute_cl_ee_interp(pt, params, bg, ELLS, n_k_fine=N_K_FINE)

    print(f"  {'l':>6} {'TT%':>8} {'EE%':>8}")
    print(f"  {'-'*22}")
    for i, ell in enumerate(ELLS):
        idx = np.argmin(np.abs(ell_ref - ell))
        tt_c, ee_c = ref_cls['tt'][idx], ref_cls['ee'][idx]
        tt_e = (float(cl_tt[i]) - tt_c) / abs(tt_c) * 100 if abs(tt_c) > 1e-30 else 0
        ee_e = (float(cl_ee[i]) - ee_c) / abs(ee_c) * 100 if abs(ee_c) > 1e-30 else 0
        tt_m = " *" if abs(tt_e) > 0.1 else ""
        ee_m = " *" if abs(ee_e) > 0.1 else ""
        print(f"  {ell:6d} {tt_e:+8.3f}{tt_m} {ee_e:+8.3f}{ee_m}")

print("\nDone!", flush=True)
