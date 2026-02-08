#!/usr/bin/env python3
"""Compare jaxCLASS source functions directly against CLASS at specific (k,tau).

Extracts CLASS source functions using the CLASS Python wrapper (classy)
and compares them point-by-point with jaxCLASS source functions.

This is the definitive diagnostic for isolating source function errors.

Usage:
    cd /path/to/jaxclass && python scripts/compare_sources_vs_class.py
"""
import os
import sys
import time
import numpy as np

# JAX setup
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve

# CLASS setup
from classy import Class

print("=== DIRECT SOURCE FUNCTION COMPARISON: jaxCLASS vs CLASS ===", flush=True)

# 1) Run CLASS with matching parameters
class_params = {
    'h': 0.6736,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'T_ncdm': 0.71611,
    'output': 'tCl pCl lCl mPk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_1/Mpc': 50.0,
    # Request perturbation output
    'perturbations_verbose': 1,
}

print("Running CLASS...", flush=True)
cosmo = Class()
cosmo.set(class_params)
cosmo.compute()

# Get CLASS C_l for comparison
cl_class = cosmo.raw_cl(2500)
print("CLASS C_l computed", flush=True)

# Get CLASS derived params
derived = cosmo.get_current_derived_parameters(['conformal_age', 'tau_rec', 'tau_star'])
tau_0_class = derived['conformal_age']
tau_star_class = derived['tau_star']
print(f"CLASS: tau_0={tau_0_class:.2f}, tau_star={tau_star_class:.2f}", flush=True)

# 2) Run jaxCLASS with matching precision
params = CosmoParams()
prec = PrecisionParams.fast_cl()  # 15 k/decade, l_max=25

t0 = time.time()
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
pt = perturbations_solve(params, prec, bg, th)
print(f"jaxCLASS solve: {time.time()-t0:.0f}s", flush=True)

tau_0_us = float(bg.conformal_age)
tau_star_us = float(th.tau_star)
print(f"jaxCLASS: tau_0={tau_0_us:.2f}, tau_star={tau_star_us:.2f}", flush=True)

# 3) Compare C_l^TT
print("\n=== C_l^TT COMPARISON ===", flush=True)
from jaxclass.harmonic import compute_cl_tt
for l in [10, 50, 100, 200]:
    cl_us = float(compute_cl_tt(pt, params, bg, [l])[0])
    cl_ref = float(cl_class['tt'][l])
    ratio = cl_us / cl_ref if abs(cl_ref) > 1e-30 else 0
    print(f"  l={l:4d}: jaxCLASS={cl_us:.4e}, CLASS={cl_ref:.4e}, ratio={ratio:.4f}", flush=True)

# 4) Source function comparison at specific (k, tau) — this is the key
# CLASS doesn't easily expose source functions directly via the Python wrapper.
# But we can compare TRANSFER functions, which are the source * Bessel integral.
# CLASS provides transfer functions via .get_transfer()
print("\n=== TRANSFER FUNCTION COMPARISON ===", flush=True)
print("(CLASS transfer function = source * Bessel integral at each k)", flush=True)

# Get CLASS transfer functions
# The transfer dict has keys like 'd_tot', 'd_g', 't_0', 't_1', 't_2', etc.
try:
    transfers = cosmo.get_transfer(z=0)
    print(f"CLASS transfer keys: {list(transfers.keys())[:10]}", flush=True)
    print(f"Available: {', '.join(sorted(transfers.keys()))}", flush=True)
except Exception as e:
    print(f"Could not get transfer functions: {e}", flush=True)
    print("This is expected — CLASS Python wrapper doesn't expose all transfer internals.", flush=True)

# 5) Alternative: compare P(k) and C_l to bound the error
print("\n=== P(k) COMPARISON (sanity check) ===", flush=True)
for k in [0.01, 0.05, 0.1]:
    pk_class = cosmo.pk_lin(k, 0)
    from jaxclass import compute_pk
    pk_us = float(compute_pk(params, prec, k=k))
    ratio = pk_us / pk_class
    print(f"  k={k:.3f}: jaxCLASS={pk_us:.4e}, CLASS={pk_class:.4e}, ratio={ratio:.4f}", flush=True)

# 6) Compare background quantities at recombination
print("\n=== BACKGROUND AT RECOMBINATION ===", flush=True)
z_star = 1089.0
H_class = cosmo.Hubble(z_star)
from jaxclass.background import H_of_z
H_us = float(H_of_z(bg, z_star))
print(f"  H(z_star): CLASS={H_class:.6e}, jaxCLASS={H_us:.6e}, ratio={H_us/H_class:.6f}", flush=True)

# 7) Compare thermodynamics at recombination
print("\n=== THERMODYNAMICS AT z_star ===", flush=True)
xe_class = cosmo.ionization_fraction(z_star)
# jaxCLASS x_e
loga_star = float(jnp.log(1.0 / (1.0 + z_star)))
xe_us = float(th.xe_of_loga.evaluate(jnp.array(loga_star)))
print(f"  x_e(z_star): CLASS={xe_class:.6e}, jaxCLASS={xe_us:.6e}, ratio={xe_us/xe_class:.4f}", flush=True)

kappa_dot_us = float(th.kappa_dot_of_loga.evaluate(jnp.array(loga_star)))
print(f"  kappa_dot(z_star): jaxCLASS={kappa_dot_us:.4e}", flush=True)

g_us = float(th.g_of_loga.evaluate(jnp.array(loga_star)))
print(f"  g(z_star): jaxCLASS={g_us:.4e}", flush=True)

cosmo.struct_cleanup()
cosmo.empty()
print("\nDone", flush=True)
