"""Diagnose source function accuracy: visibility, source assembly, and transfer.

Since perturbation variables match CLASS to 0.2% at recombination,
the remaining C_l error must come from:
  1. Visibility function g(tau), g'(tau), kappa_dot accuracy
  2. Source function assembly (how pert vars are combined into S_T0)
  3. Transfer integration (Bessel, k-quadrature)

This script isolates each layer.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.interpolate import interp1d

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve


def compare_visibility():
    """Compare g(tau), g'(tau), kappa_dot against CLASS thermodynamics."""
    params = CosmoParams()
    prec = PrecisionParams.fast_cl()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)

    # Load CLASS thermodynamics reference
    d = np.load('reference_data/lcdm_fiducial/thermodynamics.npz')
    tau_star = float(th.tau_star)

    # CLASS thermodynamics table
    th_z_class = d['th_z']
    th_xe_class = d['th_x_e']
    th_kappa_dot_class = d.get('th_kappa_dot', None)
    th_g_class = d.get('th_g', None)

    # Convert CLASS z to tau via our background
    # CLASS outputs on z-grid, we need tau-grid comparison
    # Use a few key tau points around recombination
    tau_test = np.array([
        tau_star - 30, tau_star - 15, tau_star - 5, tau_star,
        tau_star + 5, tau_star + 15, tau_star + 30, tau_star + 60,
        tau_star + 200, tau_star + 1000,
    ])

    print("=" * 70)
    print("VISIBILITY FUNCTION COMPARISON")
    print(f"tau_star = {tau_star:.2f} Mpc")
    print("=" * 70)

    # Evaluate jaxCLASS visibility at test points
    print(f"\n  g(tau) [visibility function]:")
    print(f"  {'tau':>8s}  {'jaxCLASS':>14s}  {'peak-norm':>10s}")
    print(f"  {'-'*40}")

    g_values = []
    g_prime_values = []
    kappa_dot_values = []
    for tau_s in tau_test:
        loga = float(bg.loga_of_tau.evaluate(jnp.array(tau_s)))
        g_val = float(th.g_of_loga.evaluate(jnp.array(loga)))
        g_values.append(g_val)

        # g' = dg/dloga * aH
        a = np.exp(loga)
        H = float(bg.H_of_loga.evaluate(jnp.array(loga)))
        aH = a * H
        dg_dloga = float(th.g_of_loga.derivative(jnp.array(loga)))
        g_prime = dg_dloga * aH
        g_prime_values.append(g_prime)

        kd = float(th.kappa_dot_of_loga.evaluate(jnp.array(loga)))
        kappa_dot_values.append(kd)

    g_peak = max(abs(g) for g in g_values)
    for i, tau_s in enumerate(tau_test):
        norm = g_values[i] / g_peak if g_peak > 0 else 0
        print(f"  {tau_s:8.1f}  {g_values[i]:14.6e}  {norm:10.4f}")

    # Now compare against CLASS
    # CLASS gives kappa_dot and g on z-grid. We need to match tau.
    if th_kappa_dot_class is not None and len(th_kappa_dot_class) > 0:
        # CLASS z is reversed (high z first), tau increases with z decreasing
        # Convert CLASS z to tau using our background
        z_class = th_z_class[::-1]  # ascending z
        kd_class = th_kappa_dot_class[::-1]

        # CLASS also gives conf time in background table
        bg_ref = np.load('reference_data/lcdm_fiducial/background.npz')
        if 'bg_conf_time' in bg_ref and 'bg_z' in bg_ref:
            z_bg = bg_ref['bg_z'][::-1]  # ascending z
            tau_bg = bg_ref['bg_conf_time'][::-1]  # ascending tau (decreasing z)

            # Interpolate CLASS kappa_dot to tau
            # First get tau(z) from CLASS background
            tau_of_z = interp1d(z_bg, tau_bg, kind='linear', bounds_error=False, fill_value='extrapolate')
            tau_class_kd = tau_of_z(z_class)

            # Now interpolate kappa_dot to our test tau points
            kd_of_tau = interp1d(tau_class_kd, kd_class, kind='linear',
                                  bounds_error=False, fill_value=0)

            print(f"\n  kappa_dot comparison:")
            print(f"  {'tau':>8s}  {'CLASS':>14s}  {'jaxCLASS':>14s}  {'ratio':>8s}  {'err':>8s}")
            print(f"  {'-'*60}")
            for i, tau_s in enumerate(tau_test):
                kd_c = kd_of_tau(tau_s)
                kd_j = kappa_dot_values[i]
                if abs(kd_c) > 1e-30:
                    ratio = kd_j / kd_c
                    err = abs(ratio - 1) * 100
                    marker = " <<<" if err > 2 else ""
                    print(f"  {tau_s:8.1f}  {kd_c:14.6e}  {kd_j:14.6e}  {ratio:8.4f}  {err:7.2f}%{marker}")

        # Also compare g(tau) if available
        if th_g_class is not None and len(th_g_class) > 0:
            g_class = th_g_class[::-1]
            g_of_tau = interp1d(tau_class_kd, g_class, kind='linear',
                                 bounds_error=False, fill_value=0)

            print(f"\n  g(tau) comparison:")
            print(f"  {'tau':>8s}  {'CLASS':>14s}  {'jaxCLASS':>14s}  {'ratio':>8s}  {'err':>8s}")
            print(f"  {'-'*60}")
            for i, tau_s in enumerate(tau_test):
                g_c = g_of_tau(tau_s)
                g_j = g_values[i]
                if abs(g_c) > 1e-30:
                    ratio = g_j / g_c
                    err = abs(ratio - 1) * 100
                    marker = " <<<" if err > 2 else ""
                    print(f"  {tau_s:8.1f}  {g_c:14.6e}  {g_j:14.6e}  {ratio:8.4f}  {err:7.2f}%{marker}")

    # Compare x_e
    z_test = np.array([500, 800, 1000, 1050, 1089, 1100, 1150, 1200, 1500, 2000])
    xe_class_interp = interp1d(th_z_class, th_xe_class, kind='linear',
                                bounds_error=False, fill_value=1.0)
    print(f"\n  x_e(z) comparison:")
    print(f"  {'z':>6s}  {'CLASS':>14s}  {'jaxCLASS':>14s}  {'ratio':>8s}  {'err':>8s}")
    print(f"  {'-'*56}")
    for z in z_test:
        loga = np.log(1.0 / (1.0 + z))
        xe_j = float(th.xe_of_loga.evaluate(jnp.array(loga)))
        xe_c = float(xe_class_interp(z))
        if abs(xe_c) > 1e-10:
            ratio = xe_j / xe_c
            err = abs(ratio - 1) * 100
            marker = " <<<" if err > 5 else ""
            print(f"  {z:6.0f}  {xe_c:14.6e}  {xe_j:14.6e}  {ratio:8.4f}  {err:7.2f}%{marker}")

    print()
    return bg, th


if __name__ == '__main__':
    compare_visibility()
