"""Direct state-variable comparison: jaxCLASS vs CLASS at matched (k, tau).

Solves ONE k-mode ODE at exactly CLASS's k value, saves state at CLASS's tau
points near recombination, and compares raw perturbation variables:
  - delta_g, theta_b, delta_b, delta_cdm (state variables)
  - phi, psi (Newtonian potentials, gauge-invariant)
  - theta_b + k^2*alpha (gauge-invariant velocity)

This is the decisive diagnostic: any discrepancy here propagates to all C_l.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import (
    _build_indices, _adiabatic_ic, _perturbation_rhs, _extract_sources,
    _compute_tca_criterion,
)


def solve_single_k_at_tau(k, tau_save, params, prec, bg, th):
    """Solve perturbation ODE for one k-mode, saving state at specific tau values.

    Returns the raw state vector y at each tau in tau_save.
    """
    l_max_g = prec.pt_l_max_g
    l_max_pol = prec.pt_l_max_pol_g
    l_max_ur = prec.pt_l_max_ur
    idx = _build_indices(l_max_g, l_max_pol, l_max_ur)
    n_eq = idx['n_eq']

    tau_ini = 0.1 / k  # kτ_ini = 0.1 << 1
    tau_ini = max(tau_ini, float(bg.tau_table[0]) * 1.1)

    y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, params, idx, n_eq)
    ode_args = (k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini,
        t1=float(tau_save[-1]) * 1.001,
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(ts=tau_save),
        stepsize_controller=diffrax.PIDController(
            rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol,
        ),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=prec.ode_max_steps,
        args=ode_args,
    )
    return sol.ys, idx  # shape (n_tau, n_eq)


def extract_gauge_invariant(y, k, tau, bg, th, idx):
    """Extract gauge-invariant quantities from raw state at a single (k, tau)."""
    loga = bg.loga_of_tau.evaluate(tau)
    a = jnp.exp(loga)
    H = bg.H_of_loga.evaluate(loga)
    aH = a * H
    k2 = k * k
    a2 = a * a

    eta = y[idx['eta']]
    delta_g = y[idx['F_g_0']]
    F_g_1 = y[idx['F_g_1']]
    F_g_2 = y[idx['F_g_2']]
    theta_b = y[idx['theta_b']]
    delta_b = y[idx['delta_b']]
    delta_cdm = y[idx['delta_cdm']]
    F_ur_0 = y[idx['F_ur_0']]
    F_ur_1 = y[idx['F_ur_1']]
    F_ur_2 = y[idx['F_ur_2']]

    theta_g = 3.0 * k * F_g_1 / 4.0
    theta_ur = 3.0 * k * F_ur_1 / 4.0

    # Background densities
    rho_g = bg.rho_g_of_loga.evaluate(loga)
    rho_b = bg.rho_b_of_loga.evaluate(loga)
    rho_cdm = bg.rho_cdm_of_loga.evaluate(loga)
    rho_ur = bg.rho_ur_of_loga.evaluate(loga)
    rho_ncdm = bg.rho_ncdm_of_loga.evaluate(loga)

    # Einstein constraints (same as in perturbation RHS)
    delta_rho = (rho_g * delta_g + rho_b * delta_b + rho_cdm * delta_cdm
                 + rho_ur * F_ur_0 + rho_ncdm * F_ur_0)
    rho_plus_p_theta = (4.0/3.0 * rho_g * theta_g + rho_b * theta_b
                        + 4.0/3.0 * rho_ur * theta_ur
                        + 4.0/3.0 * rho_ncdm * theta_ur)

    h_prime = (k2 * eta + 1.5 * a2 * delta_rho) / (0.5 * aH)
    eta_prime = 1.5 * a2 * rho_plus_p_theta / k2
    alpha = (h_prime + 6.0 * eta_prime) / (2.0 * k2)

    # Newtonian gauge potentials
    phi = eta - aH * alpha  # Φ = η - ℋα

    # Ψ from anisotropic stress (trace-free spatial Einstein)
    rho_plus_p_shear = (2.0/3.0 * rho_g * F_g_2
                        + 2.0/3.0 * rho_ur * F_ur_2
                        + 2.0/3.0 * rho_ncdm * F_ur_2)
    psi = phi - 4.5 * (a2 / k2) * rho_plus_p_shear  # approximate

    # Gauge-invariant velocity
    theta_b_gi = theta_b + k2 * alpha

    # Photon shear (CLASS convention: shear_g = F_g_2 / 2)
    shear_g = F_g_2 / 2.0

    return {
        'delta_g': delta_g, 'theta_g': theta_g, 'shear_g': shear_g,
        'delta_b': delta_b, 'theta_b': theta_b,
        'delta_cdm': delta_cdm,
        'phi': phi, 'psi': psi,
        'eta': eta, 'h_prime': h_prime,
        'theta_b_gi': theta_b_gi,
        'alpha': alpha,
    }


def main():
    k_val = 0.05  # Mpc^-1 — matches CLASS reference exactly

    # Load CLASS reference
    d = np.load(f'reference_data/lcdm_fiducial/perturbations_k{k_val:.4f}.npz')
    tau_class = d['tau_Mpc']

    # Setup — solve once
    params = CosmoParams()
    prec = PrecisionParams.fast_cl()
    print("Solving background + thermodynamics...")
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    tau_star = float(th.tau_star)

    # Pick 3 tau points near recombination + a few at late times
    tau_compare = np.array([
        tau_star - 20, tau_star - 5, tau_star, tau_star + 5, tau_star + 20,
        tau_star + 100, tau_star + 500, 1000.0, 5000.0,
    ])
    # Filter to CLASS range
    tau_compare = tau_compare[(tau_compare > tau_class[0]) & (tau_compare < tau_class[-1])]

    print(f"Solving perturbation ODE at k={k_val} Mpc^-1...")
    print(f"Saving at {len(tau_compare)} tau points: {tau_compare}")

    ys, idx = solve_single_k_at_tau(
        jnp.array(k_val), jnp.array(tau_compare), params, prec, bg, th
    )

    # Extract jaxCLASS quantities at each tau
    print(f"\n{'='*80}")
    print(f"k = {k_val} Mpc^-1, tau_star = {tau_star:.2f} Mpc")
    print(f"{'='*80}")

    # Interpolate CLASS reference to same tau points
    from scipy.interpolate import interp1d
    class_vars = {}
    for name in ['delta_g', 'theta_g', 'shear_g', 'delta_b', 'theta_b',
                  'delta_cdm', 'phi', 'psi', 'theta_ur', 'delta_ur']:
        key = name
        if key in d:
            class_vars[name] = interp1d(tau_class, d[key], kind='linear',
                                         bounds_error=False, fill_value=0.0)

    # Compare variable by variable
    compare_names = ['delta_g', 'theta_b', 'delta_b', 'delta_cdm', 'phi', 'psi', 'shear_g']

    for var_name in compare_names:
        if var_name not in class_vars:
            continue
        print(f"\n  {var_name}:")
        print(f"  {'tau':>8s}  {'CLASS':>14s}  {'jaxCLASS':>14s}  {'ratio':>8s}  {'err':>8s}")
        print(f"  {'-'*60}")

        for i, tau_s in enumerate(tau_compare):
            y_i = ys[i]
            jax_vals = extract_gauge_invariant(y_i, jnp.array(k_val), jnp.array(tau_s), bg, th, idx)
            jax_val = float(jax_vals[var_name])
            class_val = float(class_vars[var_name](tau_s))

            if abs(class_val) > 1e-30:
                ratio = jax_val / class_val
                err = abs(ratio - 1) * 100
                marker = " <<<" if err > 5 else ""
                print(f"  {tau_s:8.1f}  {class_val:14.6e}  {jax_val:14.6e}  {ratio:8.4f}  {err:7.2f}%{marker}")
            else:
                print(f"  {tau_s:8.1f}  {class_val:14.6e}  {jax_val:14.6e}  {'N/A':>8s}")

    # Also check gauge-invariant combo: theta_b + k^2*alpha
    print(f"\n  theta_b + k^2*alpha (gauge-invariant velocity):")
    print(f"  {'tau':>8s}  {'jaxCLASS':>14s}")
    print(f"  {'-'*30}")
    for i, tau_s in enumerate(tau_compare):
        y_i = ys[i]
        jax_vals = extract_gauge_invariant(y_i, jnp.array(k_val), jnp.array(tau_s), bg, th, idx)
        print(f"  {tau_s:8.1f}  {float(jax_vals['theta_b_gi']):14.6e}")


if __name__ == '__main__':
    main()
