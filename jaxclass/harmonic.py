"""Harmonic (C_l) module for jaxCLASS.

Computes angular power spectra C_l^TT, C_l^EE, C_l^TE from source functions
and primordial P(k).

The source function S_T0 uses the CLASS IBP form (all terms x j_l only):
    T_l(k) = int dtau S_T0(k,tau) j_l(k*chi)
    C_l^TT = 4pi int dlnk P_R(k) |T_l(k)|^2

E-polarization uses the type-2 radial function:
    E_l(k) = sqrt((l+2)!/(l-2)!) int dtau S_E(k,tau) j_l(k*chi) / (k*chi)^2
    C_l^EE = 4pi int dlnk P_R(k) |E_l(k)|^2
    C_l^TE = 4pi int dlnk P_R(k) T_l(k) E_l(k)

For l > l_limber, uses Limber approximation:
    T_l^limber(k) ~ sqrt(pi/(2l+1)) * S(k, tau_0 - (l+0.5)/k) / k
which replaces the expensive Bessel integral with a single source evaluation.
Smooth sigmoid blending in the transition region.

References:
    Dodelson "Modern Cosmology" (2003) eq. 9.35
    CLASS harmonic.c: cl_integrand = 4pi/k x P_R x T^2 x dk
    CLASS perturbations.c:7660-7678: source function assembly
    CLASS transfer.c:3606-3674: Limber approximation
    Zaldarriaga & Seljak (1997) eq. 18
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxclass.background import BackgroundResult
from jaxclass.bessel import spherical_jl, spherical_jl_backward
from jaxclass.interpolation import CubicSpline
from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.primordial import primordial_scalar_pk, primordial_tensor_pk
from jaxclass.perturbations import PerturbationResult, TensorPerturbationResult


# ---------------------------------------------------------------------------
# k-integration helpers
# ---------------------------------------------------------------------------

def _cl_k_integral(T_l, k_grid, params, k_interp_factor=1):
    """Integrate C_l = 4pi int dlnk P_R(k) |T_l(k)|^2.

    Uses trapezoidal rule on the log-k grid by default (k_interp_factor=1).
    CubicSpline refinement (k_interp_factor>1) is available but can introduce
    ringing for oscillatory T_l(k) — use with caution.
    """
    log_k = jnp.log(k_grid)
    P_R = primordial_scalar_pk(k_grid, params)
    integrand = P_R * T_l**2
    dlnk = jnp.diff(log_k)

    if k_interp_factor <= 1:
        return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    n_fine = len(k_grid) * k_interp_factor
    log_k_fine = jnp.linspace(log_k[0], log_k[-1], n_fine)
    k_fine = jnp.exp(log_k_fine)

    T_l_fine = CubicSpline(log_k, T_l).evaluate(log_k_fine)
    P_R_fine = primordial_scalar_pk(k_fine, params)
    integrand_fine = P_R_fine * T_l_fine**2
    dlnk_fine = jnp.diff(log_k_fine)
    return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_fine[:-1] + integrand_fine[1:]) * dlnk_fine)


def _cl_k_integral_cross(T_l, E_l, k_grid, params, k_interp_factor=1):
    """Like _cl_k_integral but for cross-spectrum C_l = 4pi int dlnk P_R T_l E_l."""
    log_k = jnp.log(k_grid)
    P_R = primordial_scalar_pk(k_grid, params)
    integrand = P_R * T_l * E_l
    dlnk = jnp.diff(log_k)

    if k_interp_factor <= 1:
        return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    n_fine = len(k_grid) * k_interp_factor
    log_k_fine = jnp.linspace(log_k[0], log_k[-1], n_fine)
    k_fine = jnp.exp(log_k_fine)

    T_l_fine = CubicSpline(log_k, T_l).evaluate(log_k_fine)
    E_l_fine = CubicSpline(log_k, E_l).evaluate(log_k_fine)
    P_R_fine = primordial_scalar_pk(k_fine, params)
    integrand_fine = P_R_fine * T_l_fine * E_l_fine
    dlnk_fine = jnp.diff(log_k_fine)
    return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_fine[:-1] + integrand_fine[1:]) * dlnk_fine)


# ---------------------------------------------------------------------------
# Limber approximation (CLASS transfer.c:3606-3674)
# ---------------------------------------------------------------------------

def _limber_transfer_tt(source_T0, tau_grid, k_grid, tau_0, l):
    """Limber approximation for temperature transfer function T_l(k).

    For large l, j_l(kX) ~ sqrt(pi/(2l+1)) * delta(kX - (l+0.5)), so:
        T_l(k) ~ IPhiFlat * S_T0(k, tau_0 - (l+0.5)/k) / k
    """
    l_fl = float(l)
    nu = l_fl + 0.5
    tau_limber = tau_0 - nu / k_grid

    IPhiFlat = jnp.sqrt(jnp.pi / (2.0 * l_fl + 1.0)) * (
        1.0 - 1.0 / (8.0 * (2.0 * l_fl + 1.0))
    )

    def eval_source(ik):
        return jnp.interp(tau_limber[ik], tau_grid, source_T0[ik, :], left=0.0, right=0.0)

    S_at_limber = jax.vmap(eval_source)(jnp.arange(len(k_grid)))
    return IPhiFlat * S_at_limber / k_grid


def _limber_transfer_ee(source_E, tau_grid, k_grid, tau_0, l):
    """Limber approximation for E-mode transfer function E_l(k).

    For type-2 radial j_l(x)/x^2, Limber gives extra 1/nu^2 factor.
    """
    l_fl = float(l)
    nu = l_fl + 0.5
    tau_limber = tau_0 - nu / k_grid

    prefactor = jnp.sqrt(l_fl * (l_fl + 1.0) * (l_fl - 1.0) * (l_fl + 2.0))
    IPhiFlat = jnp.sqrt(jnp.pi / (2.0 * l_fl + 1.0)) * (
        1.0 - 1.0 / (8.0 * (2.0 * l_fl + 1.0))
    )

    def eval_source(ik):
        return jnp.interp(tau_limber[ik], tau_grid, source_E[ik, :], left=0.0, right=0.0)

    S_at_limber = jax.vmap(eval_source)(jnp.arange(len(k_grid)))
    return prefactor * IPhiFlat * S_at_limber / (k_grid * nu * nu)


# ---------------------------------------------------------------------------
# Exact Bessel transfer functions
# ---------------------------------------------------------------------------

def _exact_transfer_tt(source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l,
                       source_T1=None, source_T2=None, mode=None, **kwargs):
    """Exact Bessel transfer function T_l(k) for temperature.

    CLASS sums three transfer types for scalar TT (harmonic.c:962):
        T_total = T0 + T1 + T2
    with different radial functions (transfer.c:4168-4190):
        T0: j_l(x)                                    [x = k*chi]
        T1: j_l'(x)                                   [derivative of spherical Bessel]
        T2: (1/2)(3*j_l''(x) + j_l(x))               [second derivative combination]

    For flat space (K=0), sqrt_absK_over_k = 1.0 (transfer.c:4056-4058).

    IMPORTANT: source_T2 in our code is g*Pi (perturbations.py:667), but CLASS
    uses g*P where P = Pi/8 (perturbations.c:7565,7676). The 1/8 is applied here.

    Args:
        mode: "T0", "T0+T1", "T0+T1+T2", or "T0-T1+T2" (sign test).
              Default: _DEFAULT_TT_MODE ("T0").

    cf. CLASS harmonic.c:962, transfer.c:4168-4190
    """
    if mode is None:
        mode = _DEFAULT_TT_MODE
    l_int = int(l)

    # Non-IBP mode: source_T0_noDopp*j_l + source_Doppler_nonIBP*j_l'
    if mode == "nonIBP":
        source_noDopp = kwargs.get('source_T0_noDopp', source_T0)  # fallback to IBP
        source_dop_nonIBP = kwargs.get('source_Doppler_nonIBP', None)

        def transfer_single_k_nonibp(ik):
            k = k_grid[ik]
            x = k * chi_grid
            x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
            jl = spherical_jl_backward(l_int, x)

            # Non-Doppler sources use j_l
            integrand = source_noDopp[ik, :] * jl

            # Doppler uses j_l'(x) = l/x*j_l - j_{l+1}
            if source_dop_nonIBP is not None:
                jl_p1 = spherical_jl_backward(l_int + 1, x)
                jl_prime = (l_int / x_safe) * jl - jl_p1
                integrand = integrand + source_dop_nonIBP[ik, :] * jl_prime

            return jnp.sum(integrand * dtau_mid)

        return jax.vmap(transfer_single_k_nonibp)(jnp.arange(len(k_grid)))

    # Standard IBP modes
    include_T1 = "T1" in mode and source_T1 is not None
    include_T2 = "T2" in mode and source_T2 is not None
    T1_sign = -1.0 if mode == "T0-T1+T2" else 1.0

    def transfer_single_k(ik):
        k = k_grid[ik]
        x = k * chi_grid
        x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)

        jl = spherical_jl_backward(l_int, x)

        # T0 contribution: source_T0 * j_l
        integrand = source_T0[ik, :] * jl

        if include_T1 or include_T2:
            jl_p1 = spherical_jl_backward(l_int + 1, x)

        # T1 contribution: source_T1 * j_l'(x)
        if include_T1:
            jl_prime = (l_int / x_safe) * jl - jl_p1
            integrand = integrand + T1_sign * source_T1[ik, :] * jl_prime

        # T2 contribution: (source_T2/8) * (1/2)(3*j_l''(x) + j_l(x))
        if include_T2:
            jl_pp = (l_int * (l_int - 1) / (x_safe * x_safe) - 1.0) * jl + (2.0 / x_safe) * jl_p1
            radial_T2 = 0.5 * (3.0 * jl_pp + jl)
            integrand = integrand + (source_T2[ik, :] / 8.0) * radial_T2

        return jnp.sum(integrand * dtau_mid)

    return jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))


def _exact_transfer_ee(source_E, tau_grid, k_grid, chi_grid, dtau_mid, l):
    """Exact Bessel transfer function E_l(k) for E-mode polarization."""
    l_int = int(l)
    l_fl = float(l)
    prefactor = jnp.sqrt(l_fl * (l_fl + 1.0) * (l_fl - 1.0) * (l_fl + 2.0))

    def transfer_single_k(ik):
        k = k_grid[ik]
        x = k * chi_grid
        jl = spherical_jl_backward(l_int, x)
        x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
        radial_E = jl / (x_safe * x_safe)
        return prefactor * jnp.sum(source_E[ik, :] * radial_E * dtau_mid)

    return jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))


# ---------------------------------------------------------------------------
# Blended transfer (exact + Limber)
# ---------------------------------------------------------------------------

_DEFAULT_L_SWITCH = 100000  # Effectively disabled — Limber fails for CMB primaries
_DEFAULT_DELTA_L = 50       # Blending half-width (unused when l_switch >> l_max)

# Default TT transfer mode.
# "T0": IBP form (source_T0 * j_l) — correct physics, all terms in j_l radial
# "T0+T1": adds ISW dipole (j_l' radial)
# "T0+T1+T2": adds polarization quadrupole (CLASS full form)
# "nonIBP": Non-IBP Doppler (source_T0_noDopp*j_l + source_Doppler_nonIBP*j_l')
_DEFAULT_TT_MODE = "T0+T1+T2"  # CLASS full form (harmonic.c:962)


def _get_transfer_tt(pt, bg, l, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
                     tt_mode=None):
    """Compute T_l(k) choosing exact vs Limber based on l.

    NOTE: Limber always uses source_T0 (IBP form with j_l radial), since the
    Limber approximation j_l(kX) ~ delta(kX-(l+0.5)) only applies to the j_l
    radial function. For nonIBP mode (which uses j_l' for Doppler), Limber
    falls back to the IBP form automatically.
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - tau_grid
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    l_fl = float(l)
    # Extra kwargs for non-IBP mode
    extra = {}
    if hasattr(pt, 'source_T0_noDopp'):
        extra['source_T0_noDopp'] = pt.source_T0_noDopp
    if hasattr(pt, 'source_Doppler_nonIBP'):
        extra['source_Doppler_nonIBP'] = pt.source_Doppler_nonIBP

    if l_fl > l_switch + 2 * delta_l:
        # Limber always uses IBP source_T0 (j_l radial)
        return _limber_transfer_tt(pt.source_T0, tau_grid, k_grid, tau_0, l)
    elif l_fl < l_switch - 2 * delta_l:
        return _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l,
                                  source_T1=pt.source_T1, source_T2=pt.source_T2,
                                  mode=tt_mode, **extra)
    else:
        T_exact = _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l,
                                     source_T1=pt.source_T1, source_T2=pt.source_T2,
                                     mode=tt_mode, **extra)
        # Limber uses IBP source_T0 for consistency
        T_limber = _limber_transfer_tt(pt.source_T0, tau_grid, k_grid, tau_0, l)
        w = 1.0 / (1.0 + jnp.exp(-(l_fl - l_switch) / delta_l))
        return (1.0 - w) * T_exact + w * T_limber


def _get_transfer_ee(pt, bg, l, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L):
    """Compute E_l(k) choosing exact vs Limber based on l."""
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - tau_grid
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    l_fl = float(l)
    if l_fl > l_switch + 2 * delta_l:
        return _limber_transfer_ee(pt.source_E, tau_grid, k_grid, tau_0, l)
    elif l_fl < l_switch - 2 * delta_l:
        return _exact_transfer_ee(pt.source_E, tau_grid, k_grid, chi_grid, dtau_mid, l)
    else:
        E_exact = _exact_transfer_ee(pt.source_E, tau_grid, k_grid, chi_grid, dtau_mid, l)
        E_limber = _limber_transfer_ee(pt.source_E, tau_grid, k_grid, tau_0, l)
        w = 1.0 / (1.0 + jnp.exp(-(l_fl - l_switch) / delta_l))
        return (1.0 - w) * E_exact + w * E_limber


# ---------------------------------------------------------------------------
# Public C_l functions
# ---------------------------------------------------------------------------

def compute_cl_tt(
    pt, params, bg, l_values,
    k_interp_factor=1, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    tt_mode=None,
):
    """Compute unlensed C_l^TT. Uses Limber for l > l_switch.

    tt_mode: "T0", "T0+T1", "T0+T1+T2", "T0-T1+T2", or None (uses global default).
    """
    cls = []
    for l in l_values:
        T_l = _get_transfer_tt(pt, bg, l, l_switch, delta_l, tt_mode=tt_mode)
        cl = _cl_k_integral(T_l, pt.k_grid, params, k_interp_factor)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_ee(
    pt, params, bg, l_values,
    k_interp_factor=1, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
):
    """Compute unlensed C_l^EE. Uses Limber for l > l_switch."""
    cls = []
    for l in l_values:
        E_l = _get_transfer_ee(pt, bg, l, l_switch, delta_l)
        cl = _cl_k_integral(E_l, pt.k_grid, params, k_interp_factor)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_te(
    pt, params, bg, l_values,
    k_interp_factor=1, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    tt_mode=None,
):
    """Compute unlensed C_l^TE. Uses Limber for l > l_switch."""
    cls = []
    for l in l_values:
        T_l = _get_transfer_tt(pt, bg, l, l_switch, delta_l, tt_mode=tt_mode)
        E_l = _get_transfer_ee(pt, bg, l, l_switch, delta_l)
        cl = _cl_k_integral_cross(T_l, E_l, pt.k_grid, params, k_interp_factor)
        cls.append(cl)
    return jnp.array(cls)


# ---------------------------------------------------------------------------
# Source-interpolated C_l (robust against oscillatory T_l(k))
# ---------------------------------------------------------------------------

def _interp_sources_to_fine_k(sources_list, log_k_coarse, log_k_fine):
    """Interpolate source functions S(k,τ) from coarse to fine k-grid.

    Source functions vary slowly in k (BAO scale ~0.02 Mpc⁻¹), so
    CubicSpline interpolation in log(k) is well-conditioned.

    Args:
        sources_list: list of [n_k, n_tau] source arrays
        log_k_coarse: log(k) values of the perturbation k-grid
        log_k_fine: log(k) values of the fine output grid

    Returns:
        list of [n_k_fine, n_tau] interpolated source arrays
    """
    result = []
    for src in sources_list:
        # Interpolate each tau column independently
        # Build splines along k-axis for each tau point
        def interp_one_tau(itau):
            return CubicSpline(log_k_coarse, src[:, itau]).evaluate(log_k_fine)
        src_fine = jax.vmap(interp_one_tau)(jnp.arange(src.shape[1]))  # [n_tau, n_k_fine]
        result.append(src_fine.T)  # [n_k_fine, n_tau]
    return result


def compute_cl_tt_interp(
    pt, params, bg, l_values,
    n_k_fine=3000,
    l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    tt_mode=None,
):
    """Compute C_l^TT with source interpolation to a fine k-grid.

    This is the robust version that handles the oscillatory T_l(k) by:
    1. Interpolating smooth source functions S(k,τ) to a fine k-grid
    2. Computing T_l(k_fine) = ∫ S(k_fine,τ) × j_l(k_fine×χ) dτ exactly
    3. Integrating C_l = 4π ∫ P_R |T_l|² dlnk on the fine grid

    The fine k-grid resolves the Bessel oscillation period (π/χ_star),
    ensuring the C_l integral converges independent of the perturbation
    k-grid density.

    Args:
        n_k_fine: number of fine k-points (default 3000, ~660 k/decade)
    """
    if tt_mode is None:
        tt_mode = _DEFAULT_TT_MODE

    # Fine k-grid
    log_k_coarse = jnp.log(pt.k_grid)
    log_k_fine = jnp.linspace(log_k_coarse[0], log_k_coarse[-1], n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    # Interpolate sources to fine grid
    sources_to_interp = [pt.source_T0]
    include_T1 = "T1" in tt_mode and pt.source_T1 is not None
    include_T2 = "T2" in tt_mode and pt.source_T2 is not None
    if include_T1:
        sources_to_interp.append(pt.source_T1)
    if include_T2:
        sources_to_interp.append(pt.source_T2)

    fine_sources = _interp_sources_to_fine_k(sources_to_interp, log_k_coarse, log_k_fine)
    source_T0_fine = fine_sources[0]
    source_T1_fine = fine_sources[1] if include_T1 else None
    source_T2_fine = fine_sources[1 + int(include_T1)] if include_T2 else None

    # Setup tau-grid quantities
    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - pt.tau_grid
    dtau = jnp.diff(pt.tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    # Compute T_l(k_fine) and C_l for each l
    cls = []
    for l in l_values:
        T_l_fine = _exact_transfer_tt(
            source_T0_fine, pt.tau_grid, k_fine, chi_grid, dtau_mid, l,
            source_T1=source_T1_fine, source_T2=source_T2_fine,
            mode=tt_mode)
        cl = _cl_k_integral(T_l_fine, k_fine, params, k_interp_factor=1)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_ee_interp(
    pt, params, bg, l_values,
    n_k_fine=3000,
    l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
):
    """Compute C_l^EE with source interpolation to a fine k-grid."""
    log_k_coarse = jnp.log(pt.k_grid)
    log_k_fine = jnp.linspace(log_k_coarse[0], log_k_coarse[-1], n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    fine_sources = _interp_sources_to_fine_k([pt.source_E], log_k_coarse, log_k_fine)
    source_E_fine = fine_sources[0]

    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - pt.tau_grid
    dtau = jnp.diff(pt.tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    cls = []
    for l in l_values:
        E_l_fine = _exact_transfer_ee(source_E_fine, pt.tau_grid, k_fine, chi_grid, dtau_mid, l)
        cl = _cl_k_integral(E_l_fine, k_fine, params, k_interp_factor=1)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_te_interp(
    pt, params, bg, l_values,
    n_k_fine=3000,
    l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    tt_mode=None,
):
    """Compute C_l^TE with source interpolation to a fine k-grid."""
    if tt_mode is None:
        tt_mode = _DEFAULT_TT_MODE

    log_k_coarse = jnp.log(pt.k_grid)
    log_k_fine = jnp.linspace(log_k_coarse[0], log_k_coarse[-1], n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    # Interpolate all needed sources
    sources_to_interp = [pt.source_T0, pt.source_E]
    include_T1 = "T1" in tt_mode and pt.source_T1 is not None
    include_T2 = "T2" in tt_mode and pt.source_T2 is not None
    if include_T1:
        sources_to_interp.append(pt.source_T1)
    if include_T2:
        sources_to_interp.append(pt.source_T2)

    fine_sources = _interp_sources_to_fine_k(sources_to_interp, log_k_coarse, log_k_fine)
    source_T0_fine = fine_sources[0]
    source_E_fine = fine_sources[1]
    source_T1_fine = fine_sources[2] if include_T1 else None
    source_T2_fine = fine_sources[2 + int(include_T1)] if include_T2 else None

    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - pt.tau_grid
    dtau = jnp.diff(pt.tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    cls = []
    for l in l_values:
        T_l_fine = _exact_transfer_tt(
            source_T0_fine, pt.tau_grid, k_fine, chi_grid, dtau_mid, l,
            source_T1=source_T1_fine, source_T2=source_T2_fine,
            mode=tt_mode)
        E_l_fine = _exact_transfer_ee(source_E_fine, pt.tau_grid, k_fine, chi_grid, dtau_mid, l)
        cl = _cl_k_integral_cross(T_l_fine, E_l_fine, k_fine, params, k_interp_factor=1)
        cls.append(cl)
    return jnp.array(cls)


# ---------------------------------------------------------------------------
# Sparse l-sampling + full spectrum API
# ---------------------------------------------------------------------------

def sparse_l_grid(l_max=2500):
    """Generate sparse l-sampling for efficient C_l computation.

    Mirrors CLASS strategy: dense at low l, sparser at high l.
    Returns ~100 l-values.
    """
    l_list = list(range(2, min(31, l_max + 1)))
    l_list += list(range(35, min(101, l_max + 1), 5))
    l_list += list(range(120, min(501, l_max + 1), 20))
    l_list += list(range(550, l_max + 1, 50))
    return np.array(l_list, dtype=int)


def compute_cls_all(
    pt, params, bg, l_max=2500,
    k_interp_factor=1, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    tt_mode=None,
):
    """Compute all unlensed C_l spectra at l=2..l_max.

    Uses sparse l-sampling + spline interpolation for efficiency.
    At l > l_switch, uses Limber approximation.

    Returns:
        dict with 'ell', 'tt', 'ee', 'te' (arrays of length l_max+1)
    """
    l_sparse = sparse_l_grid(l_max)

    cl_tt_sparse = compute_cl_tt(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l,
                                 tt_mode=tt_mode)
    cl_ee_sparse = compute_cl_ee(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l)
    cl_te_sparse = compute_cl_te(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l,
                                 tt_mode=tt_mode)

    l_dense = jnp.arange(2, l_max + 1, dtype=jnp.float64)
    l_sp = jnp.array(l_sparse.astype(float))

    cl_tt_dense = CubicSpline(l_sp, cl_tt_sparse).evaluate(l_dense)
    cl_ee_dense = CubicSpline(l_sp, cl_ee_sparse).evaluate(l_dense)
    cl_te_dense = CubicSpline(l_sp, cl_te_sparse).evaluate(l_dense)

    ell = jnp.arange(l_max + 1, dtype=jnp.float64)
    tt = jnp.concatenate([jnp.zeros(2), cl_tt_dense])
    ee = jnp.concatenate([jnp.zeros(2), cl_ee_dense])
    te = jnp.concatenate([jnp.zeros(2), cl_te_dense])

    return {'ell': ell, 'tt': tt, 'ee': ee, 'te': te}


def compute_cls_all_interp(
    pt, params, bg, l_max=2500,
    n_k_fine=3000, tt_mode=None,
):
    """Compute all unlensed C_l spectra at l=2..l_max with source interpolation.

    The robust version: interpolates source functions to a fine k-grid
    before computing transfer functions. Convergent regardless of the
    perturbation k-density. Use this for science-quality results.

    Returns:
        dict with 'ell', 'tt', 'ee', 'te' (arrays of length l_max+1)
    """
    l_sparse = sparse_l_grid(l_max)

    cl_tt_sparse = compute_cl_tt_interp(pt, params, bg, l_sparse.tolist(),
                                         n_k_fine=n_k_fine, tt_mode=tt_mode)
    cl_ee_sparse = compute_cl_ee_interp(pt, params, bg, l_sparse.tolist(),
                                         n_k_fine=n_k_fine)
    cl_te_sparse = compute_cl_te_interp(pt, params, bg, l_sparse.tolist(),
                                         n_k_fine=n_k_fine, tt_mode=tt_mode)

    l_dense = jnp.arange(2, l_max + 1, dtype=jnp.float64)
    l_sp = jnp.array(l_sparse.astype(float))

    cl_tt_dense = CubicSpline(l_sp, cl_tt_sparse).evaluate(l_dense)
    cl_ee_dense = CubicSpline(l_sp, cl_ee_sparse).evaluate(l_dense)
    cl_te_dense = CubicSpline(l_sp, cl_te_sparse).evaluate(l_dense)

    ell = jnp.arange(l_max + 1, dtype=jnp.float64)
    tt = jnp.concatenate([jnp.zeros(2), cl_tt_dense])
    ee = jnp.concatenate([jnp.zeros(2), cl_ee_dense])
    te = jnp.concatenate([jnp.zeros(2), cl_te_dense])

    return {'ell': ell, 'tt': tt, 'ee': ee, 'te': te}


# ---------------------------------------------------------------------------
# Tensor B-mode (no Limber needed -- tensor spectra peak at low l)
# ---------------------------------------------------------------------------

def compute_cl_bb(tpt, params, bg, l_values):
    """Compute unlensed C_l^BB from tensor perturbation source functions."""
    tau_grid = tpt.tau_grid
    k_grid = tpt.k_grid
    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - tau_grid

    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])
    log_k = jnp.log(k_grid)

    def compute_cl_single_l(l):
        l_int = int(l)
        prefactor = jnp.sqrt(l * (l + 1.0) * (l - 1.0) * (l + 2.0))

        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl_backward(l_int, x)
            x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
            radial_B = jl / (x_safe * x_safe)
            return prefactor * jnp.sum(tpt.source_p[ik, :] * radial_B * dtau_mid)

        B_l = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))
        P_T = primordial_tensor_pk(k_grid, params)
        integrand = P_T * B_l**2
        dlnk = jnp.diff(log_k)
        return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    cls = []
    for l in l_values:
        cls.append(compute_cl_single_l(l))
    return jnp.array(cls)
