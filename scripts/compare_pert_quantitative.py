"""Quantitative comparison of jaxCLASS vs CLASS perturbation variables.

Computes ratios of phi, delta_g, theta_b at matched (k, tau) points
to identify exactly which variable diverges and when.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy import interpolate

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import (
    perturbations_solve, _build_indices, _adiabatic_ic, _perturbation_rhs,
    _extract_sources, _compute_tca_criterion,
)
from jaxclass.interpolation import CubicSpline


def get_jaxclass_phi_at_tau(pt, bg, th, ik, k, tau_eval):
    """Reconstruct Newtonian potential Phi from jaxCLASS perturbation output.

    Phi = eta - H*alpha where alpha = (h' + 6*eta')/(2k^2).
    We can reconstruct this from the stored source functions.
    """
    # We need to recompute from the ODE state. Since we only store source functions,
    # not the raw state, we use the source_lens = exp(-kappa) * 2 * Phi
    # So Phi = source_lens / (2 * exp(-kappa))
    tau_grid = np.array(pt.tau_grid)

    source_lens_k = np.array(pt.source_lens[ik, :])
    lens_spline = CubicSpline(jnp.array(tau_grid), jnp.array(source_lens_k))

    # exp(-kappa) at eval point
    loga = bg.loga_of_tau.evaluate(jnp.array(tau_eval))
    exp_m_kappa = float(th.exp_m_kappa_of_loga.evaluate(loga))

    source_lens_val = float(lens_spline.evaluate(jnp.array(tau_eval)))
    if abs(exp_m_kappa) > 1e-30:
        phi = source_lens_val / (2.0 * exp_m_kappa)
    else:
        phi = 0.0
    return phi


def compare_k(k_val):
    """Compare perturbation variables at a single k-mode."""
    # Load CLASS
    fname = f'reference_data/lcdm_fiducial/perturbations_k{k_val:.4f}.npz'
    d = np.load(fname)
    tau_class = d['tau_Mpc']

    # Run jaxCLASS
    params = CosmoParams()
    prec = PrecisionParams.fast_cl()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)

    k_grid = np.array(pt.k_grid)
    ik = np.argmin(np.abs(k_grid - k_val))
    k_jax = float(k_grid[ik])
    tau_star = float(th.tau_star)

    print(f"\n{'='*70}")
    print(f"k = {k_val} Mpc^-1 (jaxCLASS: {k_jax:.6f}, mismatch: {abs(k_jax-k_val)/k_val*100:.1f}%)")
    print(f"tau_star = {tau_star:.2f} Mpc")
    print(f"{'='*70}")

    # Phi comparison (Newtonian potential)
    if 'phi' in d:
        phi_class_interp = interpolate.interp1d(tau_class, d['phi'], kind='linear',
                                                 bounds_error=False, fill_value=0)

        print(f"\n  Newtonian potential Phi:")
        print(f"  {'tau':>8s}  {'CLASS':>12s}  {'jaxCLASS':>12s}  {'ratio':>8s}  {'err':>6s}")
        for tau_s in [200, 250, tau_star-10, tau_star, tau_star+10, 300, 400, 500, 1000]:
            if tau_s > tau_class[0] and tau_s < tau_class[-1]:
                phi_c = phi_class_interp(tau_s)
                phi_j = get_jaxclass_phi_at_tau(pt, bg, th, ik, k_jax, tau_s)
                if abs(phi_c) > 1e-30:
                    ratio = phi_j / phi_c
                    err = abs(ratio - 1) * 100
                    print(f"  {tau_s:8.1f}  {phi_c:12.6e}  {phi_j:12.6e}  {ratio:8.4f}  {err:5.1f}%")

    # Source_T0 at recombination
    source_T0_jax = np.array(pt.source_T0[ik, :])
    tau_jax = np.array(pt.tau_grid)
    s_spline = CubicSpline(jnp.array(tau_jax), jnp.array(source_T0_jax))

    # Doppler source
    source_dop_jax = np.array(pt.source_Doppler[ik, :])
    dop_spline = CubicSpline(jnp.array(tau_jax), jnp.array(source_dop_jax))

    # SW source
    source_sw_jax = np.array(pt.source_SW[ik, :])
    sw_spline = CubicSpline(jnp.array(tau_jax), jnp.array(source_sw_jax))

    print(f"\n  Source decomposition at tau_star = {tau_star:.1f}:")
    for name, spline in [("SW", sw_spline), ("Doppler(IBP)", dop_spline), ("T0(total)", s_spline)]:
        val = float(spline.evaluate(jnp.array(tau_star)))
        print(f"    {name:20s}: {val:12.6e}")

    # Peak source value
    max_idx = np.argmax(np.abs(source_T0_jax))
    peak_tau = tau_jax[max_idx]
    peak_val = source_T0_jax[max_idx]
    print(f"    Peak |source_T0| at tau={peak_tau:.1f}: {peak_val:.6e}")


if __name__ == '__main__':
    for k in [0.01, 0.05, 0.1]:
        compare_k(k)
