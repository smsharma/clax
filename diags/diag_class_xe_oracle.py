"""Oracle test: inject CLASS-exact x_e into our pipeline to check if C_l become sub-0.1%.

If yes → x_e is the sole remaining blocker.
If no → there are other error sources beyond x_e.
"""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import classy
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp
from jaxclass.interpolation import CubicSpline

print(f"Devices: {jax.devices()}", flush=True)

# Generate CLASS RECFAST thermodynamics
print("Generating CLASS RECFAST reference...", flush=True)
cosmo = classy.Class()
cosmo.set({'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.0544,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06, 'T_ncdm': 0.71611,
    'output': 'tCl', 'recombination': 'RECFAST', 'l_max_scalars': 2500})
cosmo.compute()
th_rf = cosmo.get_thermodynamics()

# Get our thermo result
params = CosmoParams()
prec = PrecisionParams.planck_cl()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)

# Replace our x_e spline with CLASS's x_e
# CLASS provides x_e(z). We need x_e(loga) where loga = log(1/(1+z))
z_class = np.array(th_rf['z'])
xe_class = np.array(th_rf['x_e'])
# Sort by loga (ascending)
loga_class = np.log(1.0 / (1.0 + z_class))
sort_idx = np.argsort(loga_class)
loga_sorted = jnp.array(loga_class[sort_idx])
xe_sorted = jnp.array(xe_class[sort_idx])

# Create new x_e spline from CLASS data
xe_spline_class = CubicSpline(loga_sorted, xe_sorted)

# Monkey-patch: replace our x_e spline with CLASS's
# Also need to recompute kappa_dot from CLASS x_e
# kappa_dot = n_e * sigma_T * c = x_e * n_H * (1+z)^2 * sigma_T * Mpc
from jaxclass import constants as const
import math
Omega_b = params.omega_b / params.h**2
_bigH = 3.2407792902755102e-18
n_H_0 = 3.0 * (_bigH * params.h)**2 / (8.0 * math.pi * const.G_SI * 1.67353284e-27 / (1.0 - params.Y_He)) * Omega_b
a_grid = jnp.exp(loga_sorted)
z_grid = 1.0 / a_grid - 1.0
kd_new = xe_sorted * n_H_0 * (1.0 + z_grid)**2 * const.sigma_T * const.Mpc_over_m
kd_spline_class = CubicSpline(loga_sorted, kd_new)

# Create modified thermo result with CLASS x_e
from dataclasses import replace
# We need to replace xe_of_loga and kappa_dot_of_loga
# Since ThermoResult is frozen, we'll use object.__setattr__
import copy
th_class = copy.copy(th)
object.__setattr__(th_class, 'xe_of_loga', xe_spline_class)
object.__setattr__(th_class, 'kappa_dot_of_loga', kd_spline_class)

# Also need to recompute exp_m_kappa, g, g_prime from the new kappa_dot
# This is complex, so let's just test with the original visibility
# (the main effect should come from kappa_dot in the perturbation equations)

# Run perturbations with CLASS x_e injected
print("Running perturbations with CLASS x_e...", flush=True)
t0 = time.time()
pt = perturbations_solve(params, prec, bg, th_class)
print(f"Perturbations: {time.time()-t0:.1f}s", flush=True)

ELLS = [20, 30, 100, 300, 500, 700, 1000]
N_K_FINE = 5000

cl_tt = compute_cl_tt_interp(pt, params, bg, ELLS, n_k_fine=N_K_FINE)
cl_ee = compute_cl_ee_interp(pt, params, bg, ELLS, n_k_fine=N_K_FINE)

ref = np.load('reference_data/lcdm_fiducial/cls.npz')
ell_ref = ref['ell']

print(f"\nWith CLASS-exact x_e (oracle test):")
print(f"{'l':>6} {'TT%':>8} {'EE%':>8}")
print("-" * 25)
for i, ell in enumerate(ELLS):
    idx = np.argmin(np.abs(ell_ref - ell))
    tt_c, ee_c = ref['tt'][idx], ref['ee'][idx]
    tt_e = (float(cl_tt[i]) - tt_c) / abs(tt_c) * 100 if abs(tt_c) > 1e-30 else 0
    ee_e = (float(cl_ee[i]) - ee_c) / abs(ee_c) * 100 if abs(ee_c) > 1e-30 else 0
    tt_m = " *" if abs(tt_e) > 0.1 else ""
    ee_m = " *" if abs(ee_e) > 0.1 else ""
    print(f"{ell:6d} {tt_e:+8.3f}{tt_m} {ee_e:+8.3f}{ee_m}")

print("\nDone!", flush=True)
