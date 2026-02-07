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
from jaxclass.bessel import spherical_jl
from jaxclass.interpolation import CubicSpline
from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.primordial import primordial_scalar_pk, primordial_tensor_pk
from jaxclass.perturbations import PerturbationResult, TensorPerturbationResult


# ---------------------------------------------------------------------------
# k-integration helpers
# ---------------------------------------------------------------------------

def _cl_k_integral(T_l, k_grid, params, k_interp_factor=3):
    """Integrate C_l = 4pi int dlnk P_R(k) |T_l(k)|^2 with k-refinement."""
    log_k = jnp.log(k_grid)

    if k_interp_factor <= 1:
        P_R = primordial_scalar_pk(k_grid, params)
        integrand = P_R * T_l**2
        dlnk = jnp.diff(log_k)
        return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    n_fine = len(k_grid) * k_interp_factor
    log_k_fine = jnp.linspace(log_k[0], log_k[-1], n_fine)
    k_fine = jnp.exp(log_k_fine)

    T_l_fine = CubicSpline(log_k, T_l).evaluate(log_k_fine)
    P_R_fine = primordial_scalar_pk(k_fine, params)
    integrand_fine = P_R_fine * T_l_fine**2
    dlnk_fine = jnp.diff(log_k_fine)
    return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_fine[:-1] + integrand_fine[1:]) * dlnk_fine)


def _cl_k_integral_cross(T_l, E_l, k_grid, params, k_interp_factor=3):
    """Like _cl_k_integral but for cross-spectrum C_l = 4pi int dlnk P_R T_l E_l."""
    log_k = jnp.log(k_grid)

    if k_interp_factor <= 1:
        P_R = primordial_scalar_pk(k_grid, params)
        integrand = P_R * T_l * E_l
        dlnk = jnp.diff(log_k)
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

def _exact_transfer_tt(source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l):
    """Exact Bessel transfer function T_l(k) for temperature."""
    l_int = int(l)

    def transfer_single_k(ik):
        k = k_grid[ik]
        x = k * chi_grid
        jl = spherical_jl(l_int, x)
        return jnp.sum(source_T0[ik, :] * jl * dtau_mid)

    return jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))


def _exact_transfer_ee(source_E, tau_grid, k_grid, chi_grid, dtau_mid, l):
    """Exact Bessel transfer function E_l(k) for E-mode polarization."""
    l_int = int(l)
    l_fl = float(l)
    prefactor = jnp.sqrt(l_fl * (l_fl + 1.0) * (l_fl - 1.0) * (l_fl + 2.0))

    def transfer_single_k(ik):
        k = k_grid[ik]
        x = k * chi_grid
        jl = spherical_jl(l_int, x)
        x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
        radial_E = jl / (x_safe * x_safe)
        return prefactor * jnp.sum(source_E[ik, :] * radial_E * dtau_mid)

    return jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))


# ---------------------------------------------------------------------------
# Blended transfer (exact + Limber)
# ---------------------------------------------------------------------------

_DEFAULT_L_SWITCH = 400  # Limber transition center
_DEFAULT_DELTA_L = 50    # Blending half-width


def _get_transfer_tt(pt, bg, l, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L):
    """Compute T_l(k) choosing exact vs Limber based on l."""
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - tau_grid
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

    l_fl = float(l)
    if l_fl > l_switch + 2 * delta_l:
        return _limber_transfer_tt(pt.source_T0, tau_grid, k_grid, tau_0, l)
    elif l_fl < l_switch - 2 * delta_l:
        return _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l)
    else:
        T_exact = _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l)
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
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

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
    k_interp_factor=3, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
):
    """Compute unlensed C_l^TT. Uses Limber for l > l_switch."""
    cls = []
    for l in l_values:
        T_l = _get_transfer_tt(pt, bg, l, l_switch, delta_l)
        cl = _cl_k_integral(T_l, pt.k_grid, params, k_interp_factor)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_ee(
    pt, params, bg, l_values,
    k_interp_factor=3, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
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
    k_interp_factor=3, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
):
    """Compute unlensed C_l^TE. Uses Limber for l > l_switch."""
    cls = []
    for l in l_values:
        T_l = _get_transfer_tt(pt, bg, l, l_switch, delta_l)
        E_l = _get_transfer_ee(pt, bg, l, l_switch, delta_l)
        cl = _cl_k_integral_cross(T_l, E_l, pt.k_grid, params, k_interp_factor)
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
    k_interp_factor=3, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
):
    """Compute all unlensed C_l spectra at l=2..l_max.

    Uses sparse l-sampling + spline interpolation for efficiency.
    At l > l_switch, uses Limber approximation.

    Returns:
        dict with 'ell', 'tt', 'ee', 'te' (arrays of length l_max+1)
    """
    l_sparse = sparse_l_grid(l_max)

    cl_tt_sparse = compute_cl_tt(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l)
    cl_ee_sparse = compute_cl_ee(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l)
    cl_te_sparse = compute_cl_te(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l)

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
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])
    log_k = jnp.log(k_grid)

    def compute_cl_single_l(l):
        l_int = int(l)
        prefactor = jnp.sqrt(l * (l + 1.0) * (l - 1.0) * (l + 2.0))

        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)
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
