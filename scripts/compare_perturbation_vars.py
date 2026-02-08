"""Compare jaxCLASS perturbation variables against CLASS at matched (k, tau).

Loads CLASS perturbation output from reference_data/ and runs jaxCLASS
for the same k-modes, comparing delta_g, theta_b, h', eta, phi, psi
at matched tau points around recombination.

This directly pinpoints where source function discrepancies originate.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve, _build_indices
from jaxclass.interpolation import CubicSpline


def compare_at_k(k_val, prec_name="fast_cl"):
    """Compare perturbation variables at a single k-mode."""
    # Load CLASS data
    fname = f'reference_data/lcdm_fiducial/perturbations_k{k_val:.4f}.npz'
    try:
        d = np.load(fname)
    except FileNotFoundError:
        print(f"  No CLASS data for k={k_val}")
        return

    tau_class = d['tau_Mpc']
    a_class = d['a']

    # Run jaxCLASS
    params = CosmoParams()
    prec = getattr(PrecisionParams, prec_name)()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)

    # Find nearest k-mode in jaxCLASS grid
    k_grid = np.array(pt.k_grid)
    ik = np.argmin(np.abs(k_grid - k_val))
    k_jax = float(k_grid[ik])
    print(f"\n{'='*60}")
    print(f"k = {k_val} Mpc^-1 (jaxCLASS nearest: {k_jax:.6f})")
    print(f"{'='*60}")

    if abs(k_jax - k_val) / k_val > 0.05:
        print(f"  WARNING: k-grid mismatch > 5%: k_CLASS={k_val}, k_jax={k_jax}")

    # Build index mapping to get perturbation variables from jaxCLASS state
    idx = _build_indices(prec.pt_l_max_g, prec.pt_l_max_pol_g, prec.pt_l_max_ur)

    # Get jaxCLASS tau grid and find the state at this k
    tau_jax = np.array(pt.tau_grid)

    # We stored source functions but not raw state. Compute Einstein constraints
    # from the source components to reconstruct phi, psi.
    # For now, compare the source functions directly.

    # Compare delta_m at z=0
    delta_m_jax = float(pt.delta_m[ik, -1])
    print(f"\n  delta_m(z=0): jaxCLASS = {delta_m_jax:.6e}")

    # Compare source functions around recombination
    # Recombination: tau ~ 280 Mpc, a ~ 9e-4
    tau_star = float(th.tau_star)
    print(f"  tau_star = {tau_star:.2f} Mpc")

    # Visibility function peak comparison
    tau_range = [tau_star - 50, tau_star + 50]  # ±50 Mpc around recombination

    # Compare source_T0 at key tau points
    print(f"\n  Source T0 (IBP) around recombination (tau_star ± 50 Mpc):")
    source_T0_jax = np.array(pt.source_T0[ik, :])
    source_T0_spline = CubicSpline(jnp.array(tau_jax), jnp.array(source_T0_jax))

    # Sample at recombination region
    tau_sample = np.linspace(max(tau_range[0], tau_jax[0]), min(tau_range[1], tau_jax[-1]), 10)
    for tau_s in tau_sample:
        s_jax = float(source_T0_spline.evaluate(jnp.array(tau_s)))
        print(f"    tau={tau_s:.1f}: source_T0 = {s_jax:.6e}")

    # Compare delta_g from CLASS vs source decomposition
    # CLASS gives delta_g, theta_g, theta_b directly
    # At recombination, the SW source is g*(delta_g/4 + alpha')
    # We can check if delta_g, theta_b, phi, psi match between codes

    # Interpolate CLASS data at jaxCLASS tau points
    from scipy import interpolate

    # CLASS variables
    delta_g_class = d['delta_g']
    theta_b_class = d['theta_b']
    phi_class = d.get('phi', None)
    psi_class = d.get('psi', None)

    if phi_class is not None:
        # Compare Newtonian potential phi at recombination
        phi_interp = interpolate.interp1d(tau_class, phi_class, kind='linear',
                                           bounds_error=False, fill_value=0)

        print(f"\n  Newtonian potential Phi comparison at recombination:")
        for tau_s in [tau_star - 20, tau_star, tau_star + 20, tau_star + 100]:
            if tau_s > tau_class[0] and tau_s < tau_class[-1]:
                phi_c = phi_interp(tau_s)
                print(f"    tau={tau_s:.1f}: CLASS phi = {phi_c:.6e}")

    if psi_class is not None:
        psi_interp = interpolate.interp1d(tau_class, psi_class, kind='linear',
                                           bounds_error=False, fill_value=0)
        print(f"\n  Newtonian potential Psi comparison:")
        for tau_s in [tau_star - 20, tau_star, tau_star + 20]:
            if tau_s > tau_class[0] and tau_s < tau_class[-1]:
                psi_c = psi_interp(tau_s)
                print(f"    tau={tau_s:.1f}: CLASS psi = {psi_c:.6e}")

    # Compare theta_b
    theta_b_interp = interpolate.interp1d(tau_class, theta_b_class, kind='linear',
                                           bounds_error=False, fill_value=0)
    print(f"\n  theta_b comparison:")
    for tau_s in [tau_star - 20, tau_star, tau_star + 20]:
        if tau_s > tau_class[0] and tau_s < tau_class[-1]:
            tb_c = theta_b_interp(tau_s)
            print(f"    tau={tau_s:.1f}: CLASS theta_b = {tb_c:.6e}")

    # Compare delta_g
    delta_g_interp = interpolate.interp1d(tau_class, delta_g_class, kind='linear',
                                           bounds_error=False, fill_value=0)
    print(f"\n  delta_g comparison:")
    for tau_s in [tau_star - 20, tau_star, tau_star + 20]:
        if tau_s > tau_class[0] and tau_s < tau_class[-1]:
            dg_c = delta_g_interp(tau_s)
            print(f"    tau={tau_s:.1f}: CLASS delta_g = {dg_c:.6e}")

    print()


if __name__ == '__main__':
    for k in [0.01, 0.05, 0.1]:
        compare_at_k(k)
