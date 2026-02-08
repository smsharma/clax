#!/usr/bin/env python3
"""Compare perturbation variables at specific (k, tau) between jaxCLASS and CLASS.

CLASS exposes perturbation output via get_perturbations(). This gives us
delta_g, theta_b, shear_g, etc. at each tau for each k-mode. We compare
these directly to isolate where the source function error originates.

Usage:
    cd /path/to/jaxclass && python3 scripts/diag_perturbation_vars.py
"""
import os, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import (perturbations_solve, _build_indices,
                                     _adiabatic_ic, _perturbation_rhs)
import diffrax

from classy import Class

print("=== PERTURBATION VARIABLE COMPARISON ===", flush=True)

# 1) Run CLASS with perturbation output
class_params = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.0544,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06, 'T_ncdm': 0.71611,
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_1/Mpc': 50.0,
    # Request perturbation output at specific k values
    'k_output_values': '0.01, 0.05, 0.1',
}

print("Running CLASS with perturbation output...", flush=True)
cosmo = Class()
cosmo.set(class_params)
cosmo.compute()

# Get CLASS perturbation data
try:
    pert = cosmo.get_perturbations()
    scalar_pert = pert['scalar']
    print(f"CLASS perturbations: {len(scalar_pert)} k-modes", flush=True)
    if len(scalar_pert) > 0:
        print(f"Keys: {list(scalar_pert[0].keys())[:15]}...", flush=True)
except Exception as e:
    print(f"Error getting perturbations: {e}", flush=True)
    scalar_pert = []

derived = cosmo.get_current_derived_parameters(['conformal_age', 'tau_star'])
tau_0_class = derived['conformal_age']
tau_star_class = derived['tau_star']

# 2) Run jaxCLASS
params = CosmoParams()
prec = PrecisionParams.fast_cl()

bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
tau_star_us = float(th.tau_star)
print(f"tau_star: CLASS={tau_star_class:.2f}, jaxCLASS={tau_star_us:.2f}", flush=True)

# 3) For each CLASS k-mode, solve jaxCLASS at the same k and compare variables
l_max = prec.pt_l_max_g
idx = _build_indices(l_max, prec.pt_l_max_pol_g, prec.pt_l_max_ur)
n_eq = idx['n_eq']

k_test = [0.01, 0.05, 0.1]

for ik_class, k in enumerate(k_test):
    if ik_class >= len(scalar_pert):
        print(f"\nk={k}: no CLASS data", flush=True)
        continue

    class_data = scalar_pert[ik_class]
    tau_class = class_data['tau [Mpc]']

    print(f"\n=== k = {k} Mpc^-1 ===", flush=True)

    # Solve jaxCLASS at this k
    tau_ini = 0.5
    tau_end = float(bg.conformal_age) * 0.999
    y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, params, idx, n_eq)

    # Save at CLASS tau points near recombination
    tau_compare = []
    for tc in tau_class:
        if tau_star_class - 50 < tc < tau_star_class + 50:
            tau_compare.append(tc)
    if len(tau_compare) > 10:
        tau_compare = tau_compare[::len(tau_compare)//10]  # subsample
    tau_compare = np.array(tau_compare)

    # Also compare at tau_star
    tau_compare = np.sort(np.unique(np.append(tau_compare, [tau_star_class])))

    ode_args = (k, bg, th, params, idx, l_max, prec.pt_l_max_pol_g, prec.pt_l_max_ur)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini, t1=tau_end, dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.array(tau_compare)),
        stepsize_controller=diffrax.PIDController(
            rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol
        ),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=prec.ode_max_steps,
        args=ode_args,
    )

    print(f"  tau range: [{tau_compare[0]:.1f}, {tau_compare[-1]:.1f}], {len(tau_compare)} points", flush=True)

    # Compare key variables at each tau
    print(f"  {'tau':>8} | {'delta_g us':>12} | {'delta_g CL':>12} | {'ratio':>8} | {'theta_b us':>12} | {'theta_b CL':>12} | {'ratio':>8}", flush=True)
    print("  " + "-" * 90, flush=True)

    for i, tau in enumerate(tau_compare):
        y = sol.ys[i]

        # jaxCLASS variables
        delta_g_us = float(y[idx['F_g_0']])
        theta_b_us = float(y[idx['theta_b']])
        F_g_2_us = float(y[idx['F_g_2']])

        # CLASS variables (interpolate to this tau)
        j_class = np.argmin(np.abs(tau_class - tau))
        delta_g_cl = class_data['delta_g'][j_class] if 'delta_g' in class_data else None
        theta_b_cl = class_data['theta_b'][j_class] if 'theta_b' in class_data else None

        if delta_g_cl is not None and abs(delta_g_cl) > 1e-30:
            dg_ratio = delta_g_us / delta_g_cl
        else:
            dg_ratio = 0

        if theta_b_cl is not None and abs(theta_b_cl) > 1e-30:
            tb_ratio = theta_b_us / theta_b_cl
        else:
            tb_ratio = 0

        dg_cl_str = f"{delta_g_cl:12.4e}" if delta_g_cl is not None else "      N/A   "
        tb_cl_str = f"{theta_b_cl:12.4e}" if theta_b_cl is not None else "      N/A   "

        print(f"  {tau:8.1f} | {delta_g_us:12.4e} | {dg_cl_str} | {dg_ratio:8.4f} | {theta_b_us:12.4e} | {tb_cl_str} | {tb_ratio:8.4f}", flush=True)

cosmo.struct_cleanup()
cosmo.empty()
print("\nDone", flush=True)
