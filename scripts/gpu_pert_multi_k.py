"""Compare perturbation variables at multiple k values vs CLASS.

Test k=0.01 (large scale, TT accurate), k=0.05, k=0.1, k=0.2
to see how errors change with k.

Focus on variables at tau_star (recombination).
"""
import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

print(f"JAX device: {jax.devices()}", flush=True)

from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import _adiabatic_ic, _perturbation_rhs, _build_indices
import diffrax

params = CosmoParams()
prec = PrecisionParams.planck_cl()

print("Solving background + thermo...", flush=True)
bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)

tau_star = float(th.tau_star)
print(f"tau_star = {tau_star:.2f} Mpc", flush=True)

l_max_g = prec.pt_l_max_g
l_max_pol = prec.pt_l_max_pol_g
l_max_ur = prec.pt_l_max_ur

idx = _build_indices(l_max_g, l_max_pol, l_max_ur)
n_eq = idx['n_eq']

# k values to test
k_values = [0.01, 0.05, 0.1]

# Print header
print(f"\n{'k':>8s}  {'var':>10s}  {'ours':>14s}  {'CLASS':>14s}  {'err%':>10s}")
print("=" * 70)

for k_val in k_values:
    # Load CLASS reference
    ref_file = f'reference_data/lcdm_fiducial/perturbations_k{k_val:.4f}.npz'
    try:
        ref = np.load(ref_file)
    except FileNotFoundError:
        print(f"\nk={k_val}: reference file not found ({ref_file})")
        continue

    tau_ref = ref['tau_Mpc']

    k = jnp.array(k_val)
    k2 = k_val**2
    tau_ini = 0.1 / prec.pt_k_max_cl

    y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, params, idx, n_eq)
    ode_args = (k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur)

    # Sample at tau_star only
    tau_compare = jnp.array([tau_star])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini,
        t1=float(bg.conformal_age) * 0.999,
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=tau_compare),
        stepsize_controller=diffrax.PIDController(
            rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol,
        ),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=prec.ode_max_steps,
        args=ode_args,
    )

    y = sol.ys[0]

    # Background at tau_star
    loga = float(bg.loga_of_tau.evaluate(jnp.array(tau_star)))
    a = np.exp(loga)
    H = float(bg.H_of_loga.evaluate(jnp.array(loga)))
    a_prime_over_a = a * H
    a2 = a**2

    # Synchronous gauge variables
    eta = float(y[idx['eta']])
    delta_g_S = float(y[idx['F_g_0']])
    delta_b_S = float(y[idx['delta_b']])
    delta_cdm_S = float(y[idx['delta_cdm']])
    theta_b_S = float(y[idx['theta_b']])
    theta_g_S = 3.0 * k_val * float(y[idx['F_g_1']]) / 4.0
    theta_ur_S = 3.0 * k_val * float(y[idx['F_ur_1']]) / 4.0
    F_ur_0 = float(y[idx['F_ur_0']])
    shear_g = float(y[idx['F_g_2']]) / 2.0

    # Background densities
    rho_g = float(bg.rho_g_of_loga.evaluate(jnp.array(loga)))
    rho_b = float(bg.rho_b_of_loga.evaluate(jnp.array(loga)))
    rho_cdm = float(bg.rho_cdm_of_loga.evaluate(jnp.array(loga)))
    rho_ur = float(bg.rho_ur_of_loga.evaluate(jnp.array(loga)))
    rho_ncdm = float(bg.rho_ncdm_of_loga.evaluate(jnp.array(loga)))

    # Recompute h' from constraint
    delta_rho = (rho_g * delta_g_S + rho_b * delta_b_S
                 + rho_cdm * delta_cdm_S + rho_ur * F_ur_0 + rho_ncdm * F_ur_0)
    h_prime = (k2 * eta + 1.5 * a2 * delta_rho) / (0.5 * a_prime_over_a)

    rpt_sum = (4./3.*rho_g*theta_g_S + rho_b*theta_b_S
               + 4./3.*rho_ur*theta_ur_S + 4./3.*rho_ncdm*theta_ur_S)
    eta_prime = 1.5 * a2 * rpt_sum / k2

    alpha = (h_prime + 6.0 * eta_prime) / (2.0 * k2)

    # Newtonian gauge
    delta_g_N = delta_g_S - 4.0 * a_prime_over_a * alpha
    delta_b_N = delta_b_S - 3.0 * a_prime_over_a * alpha
    theta_g_N = theta_g_S + k2 * alpha
    theta_b_N = theta_b_S + k2 * alpha
    phi_N = eta - a_prime_over_a * alpha

    # CLASS reference at tau_star
    idx_ref = np.argmin(np.abs(tau_ref - tau_star))

    def pct(ours, theirs):
        if abs(theirs) < 1e-30:
            return 0.0
        return (ours - theirs) / abs(theirs) * 100

    class_dg = ref['delta_g'][idx_ref]
    class_db = ref['delta_b'][idx_ref]
    class_tg = ref['theta_g'][idx_ref]
    class_tb = ref['theta_b'][idx_ref]
    class_phi = ref['phi'][idx_ref]
    class_shear = ref['shear_g'][idx_ref]

    print(f"\n{k_val:8.4f}  {'delta_g':>10s}  {delta_g_N:14.6f}  {class_dg:14.6f}  {pct(delta_g_N, class_dg):+10.4f}")
    print(f"{'':>8s}  {'theta_g':>10s}  {theta_g_N:14.6e}  {class_tg:14.6e}  {pct(theta_g_N, class_tg):+10.4f}")
    print(f"{'':>8s}  {'delta_b':>10s}  {delta_b_N:14.6f}  {class_db:14.6f}  {pct(delta_b_N, class_db):+10.4f}")
    print(f"{'':>8s}  {'theta_b':>10s}  {theta_b_N:14.6e}  {class_tb:14.6e}  {pct(theta_b_N, class_tb):+10.4f}")
    print(f"{'':>8s}  {'phi':>10s}  {phi_N:14.6f}  {class_phi:14.6f}  {pct(phi_N, class_phi):+10.4f}")
    print(f"{'':>8s}  {'shear_g':>10s}  {shear_g:14.6e}  {class_shear:14.6e}  {pct(shear_g, class_shear):+10.4f}")

print("\nDone!", flush=True)
