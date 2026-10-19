"""Harmonic (C_l) module for clax.

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

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from clax.background import BackgroundResult
from clax.bessel import spherical_jl, spherical_jl_backward, sparse_l_grid, build_jl_table
from clax.interpolation import CubicSpline
from clax.params import CosmoParams, PrecisionParams
from clax.primordial import primordial_scalar_pk, primordial_tensor_pk
from clax.perturbations import PerturbationResult, TensorPerturbationResult


# ---------------------------------------------------------------------------
# k-integration helpers
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(3,))
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


@functools.partial(jax.jit, static_argnums=(4,))
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
# (Batched Bessel removed — normalization issues with backward recurrence
# rescaling. Standard per-l approach used instead.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exact Bessel transfer functions
# ---------------------------------------------------------------------------

def _chunked_vmap(fn, n, chunk_size=2000):
    """Apply fn via vmap in chunks to avoid GPU OOM.

    Args:
        fn: function mapping index -> scalar
        n: total number of elements
        chunk_size: max elements per vmap batch
    Returns:
        array of shape (n,)
    """
    if n <= chunk_size:
        return jax.vmap(fn)(jnp.arange(n))
    results = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = jax.vmap(fn)(jnp.arange(start, end))
        results.append(chunk)
    return jnp.concatenate(results)


@functools.partial(jax.jit, static_argnums=(5, 8))
def _exact_transfer_tt(source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l,
                       source_T1=None, source_T2=None, mode=None,
                       source_T0_noDopp=None, source_Doppler_nonIBP=None):
    """Exact Bessel transfer function T_l(k) for temperature.

    CLASS sums three transfer types for scalar TT (harmonic.c:962):
        T_total = T0 + T1 + T2
    with radial functions (transfer.c:4168-4190):
        T0: j_l(x)                                         [x = k*chi]
        T1: sqrt(|K|)/k * j_l'(x)                          [vanishes for flat K=0]
        T2: (1/(2*s2)) * (3*(K/k^2)*j_l''(x) + j_l(x))    [= 0.5*j_l for K=0]

    For FLAT space (K=0): T1 contributes nothing (sqrt(|K|)/k = 0), and
    T2 simplifies to 0.5*j_l since the K/k^2 * j_l'' term vanishes.

    IMPORTANT: source_T2 in our code is g*Pi (perturbations.py), but CLASS
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
        source_noDopp = source_T0_noDopp if source_T0_noDopp is not None else source_T0
        source_dop_nonIBP = source_Doppler_nonIBP

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

        return _chunked_vmap(transfer_single_k_nonibp, len(k_grid))

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

    return _chunked_vmap(transfer_single_k, len(k_grid))


@functools.partial(jax.jit, static_argnums=(5,))
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

    return _chunked_vmap(transfer_single_k, len(k_grid))


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
    tau_0 = bg.conformal_age
    chi_grid = tau_0 - tau_grid
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    l_fl = float(l)
    # Extra kwargs for non-IBP mode
    src_noDopp = getattr(pt, 'source_T0_noDopp', None)
    src_dop_nonIBP = getattr(pt, 'source_Doppler_nonIBP', None)

    if l_fl > l_switch + 2 * delta_l:
        # Limber always uses IBP source_T0 (j_l radial)
        return _limber_transfer_tt(pt.source_T0, tau_grid, k_grid, tau_0, l)
    elif l_fl < l_switch - 2 * delta_l:
        return _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l,
                                  source_T1=pt.source_T1, source_T2=pt.source_T2,
                                  mode=tt_mode,
                                  source_T0_noDopp=src_noDopp,
                                  source_Doppler_nonIBP=src_dop_nonIBP)
    else:
        T_exact = _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l,
                                     source_T1=pt.source_T1, source_T2=pt.source_T2,
                                     mode=tt_mode,
                                     source_T0_noDopp=src_noDopp,
                                     source_Doppler_nonIBP=src_dop_nonIBP)
        # Limber uses IBP source_T0 for consistency
        T_limber = _limber_transfer_tt(pt.source_T0, tau_grid, k_grid, tau_0, l)
        w = 1.0 / (1.0 + jnp.exp(-(l_fl - l_switch) / delta_l))
        return (1.0 - w) * T_exact + w * T_limber


def _get_transfer_ee(pt, bg, l, l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L):
    """Compute E_l(k) choosing exact vs Limber based on l."""
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = bg.conformal_age
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


def _fine_log_k_grid(log_k_coarse, n_k_fine):
    """The dense log-uniform evaluation grid for the C_l k-integral.
    Single definition for the four former inline copies. The integration
    grid deliberately stays log-uniform + trapezoid: T_l(k) is
    oscillatory, and density -- not spacing -- controls its integral
    (CHANGELOG.md:2142-2163)."""
    return jnp.linspace(log_k_coarse[0], log_k_coarse[-1], n_k_fine)


@jax.jit
def _interp_single_source(src, log_k_coarse, log_k_fine):
    """Interpolate a single source array from coarse to fine k-grid (JIT-cached).

    Args:
        src: shape [n_k_coarse, n_tau]
        log_k_coarse: shape [n_k_coarse]
        log_k_fine: shape [n_k_fine]
    Returns:
        shape [n_k_fine, n_tau]
    """
    def interp_one_tau(itau):
        return CubicSpline(log_k_coarse, src[:, itau]).evaluate(log_k_fine)
    return jax.vmap(interp_one_tau)(jnp.arange(src.shape[1])).T


def _barycentric_matrix(log_k_coarse, log_k_fine):
    """Dense barycentric evaluation matrix B with (B @ y) equal to
    Chebyshev interpolation of y from Lobatto nodes log_k_coarse to
    log_k_fine. Built once per C_l call; applied to each source as a
    single matmul (replaces one Thomas solve per (source, tau))."""
    n = log_k_coarse.shape[0]
    w = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    w = w.at[0].mul(0.5).at[-1].mul(0.5)
    d = log_k_fine[:, None] - log_k_coarse[None, :]
    hit = jnp.abs(d) < 1e-14 * jnp.abs(log_k_coarse[-1] - log_k_coarse[0])
    d_safe = jnp.where(hit, 1.0, d)
    c = w[None, :] / d_safe
    bary = c / jnp.sum(c, axis=1, keepdims=True)
    row_hit = jnp.any(hit, axis=1, keepdims=True)
    return jnp.where(row_hit, hit.astype(bary.dtype), bary)


def _interp_sources_to_fine_k(sources_list, log_k_coarse, log_k_fine, method="spline"):
    """Interpolate source functions S(k,τ) from coarse to fine k-grid.

    Source functions vary slowly in k (BAO scale ~0.02 Mpc⁻¹), so
    CubicSpline interpolation in log(k) is well-conditioned.

    Args:
        sources_list: list of [n_k, n_tau] source arrays
        log_k_coarse: log(k) values of the perturbation k-grid
        log_k_fine: log(k) values of the fine output grid
        method: "spline" (default, unchanged CubicSpline path) or
            "chebyshev" -- builds one barycentric evaluation matrix
            B = _barycentric_matrix(log_k_coarse, log_k_fine) and applies
            it to every source as a matmul. Requires the perturbation
            solve to have used pt_k_grid_type="chebyshev" (Lobatto nodes
            in log k) -- the barycentric weights assume Lobatto spacing.

    Returns:
        list of [n_k_fine, n_tau] interpolated source arrays
    """
    if method == "chebyshev":
        B = _barycentric_matrix(log_k_coarse, log_k_fine)
        return [B @ src for src in sources_list]
    return [_interp_single_source(src, log_k_coarse, log_k_fine) for src in sources_list]


def compute_cl_tt_interp(
    pt, params, bg, l_values,
    n_k_fine=5000,
    l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    tt_mode=None,
    l_limber=0,
    k_interp_method="spline",
):
    """Compute C_l^TT with source interpolation to a fine k-grid.

    This is the robust version that handles the oscillatory T_l(k) by:
    1. Interpolating smooth source functions S(k,τ) to a fine k-grid
    2. Computing T_l(k_fine) = ∫ S(k_fine,τ) × j_l(k_fine×χ) dτ exactly
    3. Integrating C_l = 4π ∫ P_R |T_l|² dlnk on the fine grid

    For l >= l_limber (if > 0), uses Limber approximation instead of exact
    Bessel integrals, providing massive speedup at high l.

    Args:
        l_values: tuple of multipole values (must be tuple, not list, for JIT caching)
        n_k_fine: number of fine k-points (default 10000, converged for l≤1200)
        l_limber: use Limber for l >= this value (0=disabled)
        k_interp_method: "spline" (default) or "chebyshev" -- passed through
            to _interp_sources_to_fine_k. "chebyshev" requires the
            perturbation solve to have used pt_k_grid_type="chebyshev"
            (Lobatto nodes in log k) -- the barycentric weights assume
            Lobatto spacing.
    """
    if tt_mode is None:
        tt_mode = _DEFAULT_TT_MODE

    # Fine k-grid
    log_k_coarse = jnp.log(pt.k_grid)
    log_k_fine = _fine_log_k_grid(log_k_coarse, n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    # Interpolate sources to fine grid
    sources_to_interp = [pt.source_T0]
    include_T1 = "T1" in tt_mode and pt.source_T1 is not None
    include_T2 = "T2" in tt_mode and pt.source_T2 is not None
    if include_T1:
        sources_to_interp.append(pt.source_T1)
    if include_T2:
        sources_to_interp.append(pt.source_T2)

    fine_sources = _interp_sources_to_fine_k(sources_to_interp, log_k_coarse, log_k_fine,
                                              method=k_interp_method)
    source_T0_fine = fine_sources[0]
    source_T1_fine = fine_sources[1] if include_T1 else None
    source_T2_fine = fine_sources[1 + int(include_T1)] if include_T2 else None

    # Setup tau-grid quantities
    tau_0 = bg.conformal_age
    chi_grid = tau_0 - pt.tau_grid
    dtau = jnp.diff(pt.tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    # Compute T_l(k_fine) and C_l for each l
    cls = []
    for l in l_values:
        if l_limber > 0 and l >= l_limber:
            # Limber: O(n_k) instead of O(n_k × n_tau × l)
            T_l_fine = _limber_transfer_tt(source_T0_fine, pt.tau_grid, k_fine, tau_0, l)
        else:
            T_l_fine = _exact_transfer_tt(
                source_T0_fine, pt.tau_grid, k_fine, chi_grid, dtau_mid, l,
                source_T1=source_T1_fine, source_T2=source_T2_fine,
                mode=tt_mode)
        cl = _cl_k_integral(T_l_fine, k_fine, params, k_interp_factor=1)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_ee_interp(
    pt, params, bg, l_values,
    n_k_fine=5000,
    l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    l_limber=0,
    k_interp_method="spline",
):
    """Compute C_l^EE with source interpolation to a fine k-grid.

    Args:
        k_interp_method: "spline" (default) or "chebyshev" -- passed
            through to _interp_sources_to_fine_k. "chebyshev" requires
            the perturbation solve to have used pt_k_grid_type="chebyshev"
            (Lobatto nodes in log k) -- the barycentric weights assume
            Lobatto spacing.
    """
    log_k_coarse = jnp.log(pt.k_grid)
    log_k_fine = _fine_log_k_grid(log_k_coarse, n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    fine_sources = _interp_sources_to_fine_k([pt.source_E], log_k_coarse, log_k_fine,
                                              method=k_interp_method)
    source_E_fine = fine_sources[0]

    tau_0 = bg.conformal_age
    chi_grid = tau_0 - pt.tau_grid
    dtau = jnp.diff(pt.tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    cls = []
    for l in l_values:
        if l_limber > 0 and l >= l_limber:
            E_l_fine = _limber_transfer_ee(source_E_fine, pt.tau_grid, k_fine, tau_0, l)
        else:
            E_l_fine = _exact_transfer_ee(source_E_fine, pt.tau_grid, k_fine, chi_grid, dtau_mid, l)
        cl = _cl_k_integral(E_l_fine, k_fine, params, k_interp_factor=1)
        cls.append(cl)
    return jnp.array(cls)


def compute_cl_te_interp(
    pt, params, bg, l_values,
    n_k_fine=5000,
    l_switch=_DEFAULT_L_SWITCH, delta_l=_DEFAULT_DELTA_L,
    tt_mode=None,
    l_limber=0,
    k_interp_method="spline",
):
    """Compute C_l^TE with source interpolation to a fine k-grid.

    Args:
        k_interp_method: "spline" (default) or "chebyshev" -- passed
            through to _interp_sources_to_fine_k. "chebyshev" requires
            the perturbation solve to have used pt_k_grid_type="chebyshev"
            (Lobatto nodes in log k) -- the barycentric weights assume
            Lobatto spacing.
    """
    if tt_mode is None:
        tt_mode = _DEFAULT_TT_MODE

    log_k_coarse = jnp.log(pt.k_grid)
    log_k_fine = _fine_log_k_grid(log_k_coarse, n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    # Interpolate all needed sources
    sources_to_interp = [pt.source_T0, pt.source_E]
    include_T1 = "T1" in tt_mode and pt.source_T1 is not None
    include_T2 = "T2" in tt_mode and pt.source_T2 is not None
    if include_T1:
        sources_to_interp.append(pt.source_T1)
    if include_T2:
        sources_to_interp.append(pt.source_T2)

    fine_sources = _interp_sources_to_fine_k(sources_to_interp, log_k_coarse, log_k_fine,
                                              method=k_interp_method)
    source_T0_fine = fine_sources[0]
    source_E_fine = fine_sources[1]
    source_T1_fine = fine_sources[2] if include_T1 else None
    source_T2_fine = fine_sources[2 + int(include_T1)] if include_T2 else None

    tau_0 = bg.conformal_age
    chi_grid = tau_0 - pt.tau_grid
    dtau = jnp.diff(pt.tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    cls = []
    for l in l_values:
        if l_limber > 0 and l >= l_limber:
            T_l_fine = _limber_transfer_tt(source_T0_fine, pt.tau_grid, k_fine, tau_0, l)
            E_l_fine = _limber_transfer_ee(source_E_fine, pt.tau_grid, k_fine, tau_0, l)
        else:
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

# sparse_l_grid is now imported from clax.bessel


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
    l_sparse = _sparse_l_padded(l_max)

    cl_tt_sparse = compute_cl_tt(pt, params, bg, tuple(l_sparse.tolist()),
                                 k_interp_factor, l_switch, delta_l,
                                 tt_mode=tt_mode)
    cl_ee_sparse = compute_cl_ee(pt, params, bg, tuple(l_sparse.tolist()),
                                 k_interp_factor, l_switch, delta_l)
    cl_te_sparse = compute_cl_te(pt, params, bg, tuple(l_sparse.tolist()),
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
    n_k_fine=5000, tt_mode=None,
    l_limber=0,
    k_interp_method="spline",
):
    """Compute all unlensed C_l spectra at l=2..l_max with source interpolation.

    The robust version: interpolates source functions to a fine k-grid
    before computing transfer functions. Convergent regardless of the
    perturbation k-density. Use this for science-quality results.

    Args:
        l_limber: use Limber approximation for l >= this value (0=disabled)
        k_interp_method: "spline" (default) or "chebyshev" -- forwarded to
            compute_cl_tt_interp/compute_cl_ee_interp/compute_cl_te_interp.
            "chebyshev" requires the perturbation solve to have used
            pt_k_grid_type="chebyshev" (Lobatto nodes in log k) -- the
            barycentric weights assume Lobatto spacing.

    Returns:
        dict with 'ell', 'tt', 'ee', 'te' (arrays of length l_max+1)
    """
    l_sparse = _sparse_l_padded(l_max)

    cl_tt_sparse = compute_cl_tt_interp(pt, params, bg, tuple(l_sparse.tolist()),
                                         n_k_fine=n_k_fine, tt_mode=tt_mode,
                                         l_limber=l_limber,
                                         k_interp_method=k_interp_method)
    cl_ee_sparse = compute_cl_ee_interp(pt, params, bg, tuple(l_sparse.tolist()),
                                         n_k_fine=n_k_fine,
                                         l_limber=l_limber,
                                         k_interp_method=k_interp_method)
    cl_te_sparse = compute_cl_te_interp(pt, params, bg, tuple(l_sparse.tolist()),
                                         n_k_fine=n_k_fine, tt_mode=tt_mode,
                                         l_limber=l_limber,
                                         k_interp_method=k_interp_method)

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
# Fast all-l-at-once C_l computation (table-based Bessel)
# ---------------------------------------------------------------------------

# _build_jl_table is now build_jl_table, imported from clax.bessel
_build_jl_table = build_jl_table

# The sparse-l grid is extended this far beyond the requested l_max so that
# the cubic spline's end condition sits outside the returned range. Without
# the pad the last ~50 multipoles are off by up to 2% (1.87% at l=983 for
# l_max=1000; precision sweep follow-up, Oct 12, 2026). Measured by splining
# CLASS's own spectra on the sparse grid: the end-condition residual over the
# last 50 multipoles is 1.94% TT / 1.80% EE at l_max=1000 and 0.26% / 2.7%
# at 1500 unpadded, 0.03% TT at both l_max and 0.09% EE at 1500 with the
# pad; larger pads gain nothing. The whole-range spline error is a separate
# matter, set by the grid's 500 -> 550 step: EE 1.13% at l=526, TT 0.16% at
# l=527, EE 0.43% at l=832, TT ~0.10% over l=680-1580; <= 0.03% TT and
# <= 0.11% EE below l=500 (CHANGELOG, Oct 12, 2026, follow-ups). Costs two
# extra l rows.
L_SPLINE_PAD = 100


def _sparse_l_padded(l_max):
    return sparse_l_grid(l_max + L_SPLINE_PAD)

# Bessel lookup-table resolution (points per unit x). Linear interpolation of
# j_l at spacing h biases the amplitude by ~ -h^2/12 on average; the historical
# 3 points per unit x (h ~ 0.33 at k_max = 1/Mpc) gave a -1% TT offset at
# l < 50 and +20% in EE at l = 2 (precision sweep, Oct 2026,
# docs/precision_sweep_2026-10-10.md finding 6). At 30 the fast path matches
# the exact-j_l path to 0.01% for l <= 500; the table costs ~0.7 GB at
# l_max 2500 and the harmonic stage ~12 s instead of ~4 s on a V100. Those
# were measured on the then cosmology-dependent range x_max = 1.05 k_max tau_0
# (~14800 at k_max 1/Mpc). On the fixed x-range of #54, x_max = 1.05
# pt_k_max_cl hr_tau0_max = 21000, the table is 2 x 105 x 630000 doubles =
# 1.06 GB at l_max 2500 (0.86 GB at 1500), built once per process and kept
# (see _jl_tables).
JL_TABLE_X_PER_UNIT = 30.0


def _table_bounds(pt, prec):
    """Static (k_bound, tau0_max) for the fixed-range Bessel table (#54).

    With prec: k_bound = prec.pt_k_max_cl (the fine k-grid ends at pt.k_grid[-1]
    = pt_k_max_cl) and tau0_max = prec.hr_tau0_max -- both static, so jit, jvp
    and grad work. Without prec (eager callers only), k_bound is read from
    pt.k_grid; under tracing that read fails and we ask for prec instead.
    """
    if prec is not None:
        return float(prec.pt_k_max_cl), float(prec.hr_tau0_max)
    try:
        k_bound = float(pt.k_grid[-1])
    except jax.errors.ConcretizationTypeError as err:
        raise TypeError(
            "compute_cls_all_fast: pass prec=<the PrecisionParams of the perturbation "
            "solve> to jit or differentiate; without prec the Bessel table's k-range "
            "is read from pt.k_grid, which is traced here.") from err
    return k_bound, float(PrecisionParams().hr_tau0_max)


def _table_grid(k_bound, tau0_max, x_per_unit):
    """x_max = 1.05 k_bound tau0_max and n_x = ceil(x_per_unit x_max), at
    least 30000 points: the fixed x-range at the density the caller passes
    (compute_cls_all_fast's x_per_unit, default JL_TABLE_X_PER_UNIT)."""
    x_max = 1.05 * k_bound * tau0_max
    return x_max, max(30000, int(math.ceil(x_per_unit * x_max)))


@functools.lru_cache(maxsize=8)
def _jl_tables(l_max_padded, n_x, x_max):
    """j_l, j_l' on the fixed x-range for the rows of sparse_l_grid(l_max_padded);
    compute_cls_all_fast passes l_max + L_SPLINE_PAD so the rows match
    _sparse_l_padded(l_max). Built once per (l_max_padded, n_x, x_max) and
    reused across cosmologies and calls -- concrete arrays, never traced.
    Each entry holds 2 x n_l x n_x doubles for the life of the process
    (fit_cl at 30 points per unit x: 2 x 85 x 630000 x 8 B = 0.86 GB)."""
    with jax.ensure_compile_time_eval():
        return build_jl_table(l_max_padded, n_x=n_x, x_max=x_max)


def _table_position(x_flat, dx_tab, n_x):
    """Left table index and fractional position t in [0, 1] on the uniform
    x-table. The index carries no derivative; t carries d/dx = 1/dx_tab."""
    u = x_flat / dx_tab
    i_left = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_x - 2)
    return i_left, u - i_left.astype(u.dtype)


def _lookup_jl(jl_row, jlp_row, l_val, ll1_val, i_left, t, dx_tab, mode):
    """j_l(x) and j_l'(x) from one table row at positions (i_left, t)."""
    i_right = i_left + 1
    jl0, jl1 = jl_row[i_left], jl_row[i_right]
    jp0, jp1 = jlp_row[i_left], jlp_row[i_right]
    if mode == "linear":
        return jl0 + t * (jl1 - jl0), jp0 + t * (jp1 - jp0)
    raise ValueError(f"unknown Bessel lookup mode {mode!r}")


def compute_cls_all_fast(
    pt, params, bg, l_max=1500,
    n_k_fine=5000, tt_mode=None,
    k_interp_method="spline",
    x_per_unit=JL_TABLE_X_PER_UNIT,
    *, prec=None, _jl_lookup="linear",
):
    """Fast C_l computation using precomputed j_l/j_l' tables + source interpolation.

    Architecture:
    1. Interpolate smooth source functions S(k,τ) to fine k-grid
    2. Build j_l(x) and j_l'(x) lookup tables (backward+upward, one JIT)
    3. For each l: interpolate tables at x = k*χ, compute radial_T2 on-the-fly,
       integrate over τ → T_l(k) with all 3 transfer types (T0+T1+T2)
    4. Integrate C_l = 4π ∫ dlnk P_R |T_l|² for all l at once

    Full TT transfer function (cf. CLASS harmonic.c:962):
        T_l(k) = ∫ dτ [S_T0*j_l + S_T1*j_l' + (S_T2/8)*radial_T2]
    where radial_T2 = 0.5*([3l(l+1)/x²-2]*j_l - 6/x*j_l').

    T1 (ISW dipole * j_l') is the dominant correction at low l (~20% at l=20).
    T2 (polarization quadrupole) contributes ~0.2-0.8% at all l.

    Performance: ~3-5s harmonic (vs ~800s per-l approach) on V100.

    Args:
        pt: PerturbationResult with source functions
        params: CosmoParams for primordial spectrum
        bg: BackgroundResult for conformal age
        l_max: maximum multipole (default 1500)
        n_k_fine: fine k-grid points for source interpolation (default 5000)
        tt_mode: unused (always uses T0+T1+T2)
        k_interp_method: "spline" (default) or "chebyshev" -- passed
            through to _interp_sources_to_fine_k. "chebyshev" requires
            the perturbation solve to have used pt_k_grid_type="chebyshev"
            (Lobatto nodes in log k) -- the barycentric weights assume
            Lobatto spacing.
        x_per_unit: Bessel-table points per unit x (default
            JL_TABLE_X_PER_UNIT = 30) on the table's fixed x-range
            [0, 1.05 k_bound hr_tau0_max] (#54). Values below ~10 bias C_l
            low by ~ -h^2/12 per unit amplitude; see the module constant.
        prec: PrecisionParams of the perturbation solve; required to jit or
            differentiate (fixes the Bessel table's x-range from pt_k_max_cl
            and hr_tau0_max). _jl_lookup: internal, measurement only (#54).

    Accuracy (Oct 2026 sweep, massless RECFAST reference, planck_cl-class
    perturbations): with n_k_fine 20000 and the default table this path
    matches the exact-j_l path (compute_cls_all_interp) to 0.011% for
    l <= 500, 0.024% for l <= 1000, 0.17% for l <= 2000; n_k_fine 5000 is
    percent-level above l ~ 500.

    Returns:
        dict with 'ell', 'tt', 'ee', 'te' (arrays of length l_max+1)
    """
    # 1. Source interpolation to fine k-grid
    log_k_coarse = jnp.log(pt.k_grid)
    log_k_fine = _fine_log_k_grid(log_k_coarse, n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    fine_sources = _interp_sources_to_fine_k(
        [pt.source_T0, pt.source_E, pt.source_T1, pt.source_T2],
        log_k_coarse, log_k_fine, method=k_interp_method
    )
    source_T0_fine = fine_sources[0]
    source_E_fine = fine_sources[1]
    source_T1_fine = fine_sources[2]
    source_T2_fine = fine_sources[3]

    # 2. Setup grids
    k_bound, tau0_max = _table_bounds(pt, prec)
    x_max, n_x = _table_grid(k_bound, tau0_max, x_per_unit)
    tau_0 = eqx.error_if(
        bg.conformal_age, bg.conformal_age > tau0_max,
        f"compute_cls_all_fast: conformal age tau_0 exceeds PrecisionParams.hr_tau0_max "
        f"= {tau0_max:g} Mpc, the Bessel table's design bound; raise hr_tau0_max.")
    chi_grid = tau_0 - pt.tau_grid
    n_tau = pt.tau_grid.shape[0]
    dtau = jnp.diff(pt.tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    # Precompute source × dtau weights
    src_T0_dtau = source_T0_fine * dtau_mid[None, :]           # T0: * j_l
    src_T1_dtau = source_T1_fine * dtau_mid[None, :]           # T1: * j_l'
    src_T2_dtau = (source_T2_fine / 8.0) * dtau_mid[None, :]  # T2/8: * radial_T2
    x_query = k_fine[:, None] * chi_grid[None, :]              # (n_k, n_tau)
    x_query = eqx.error_if(
        x_query, jnp.max(x_query) > x_max,
        f"compute_cls_all_fast: k*chi exceeds the Bessel table (x_max = {x_max:g}); "
        "prec.pt_k_max_cl is smaller than the perturbation grid's k_max -- pass the "
        "prec used for the perturbation solve.")
    x_safe = jnp.where(jnp.abs(x_query) < 1e-30, 1e-30, x_query)
    src_E_dtau = source_E_fine * dtau_mid[None, :] / (x_safe * x_safe)

    # Precompute 1/x and 1/x² for radT2 on-the-fly computation
    x_flat_query = x_query.flatten()
    x_safe_flat = jnp.where(jnp.abs(x_flat_query) < 1e-30, 1e-30, x_flat_query)
    inv_x_flat = 1.0 / x_safe_flat
    inv_x2_flat = inv_x_flat * inv_x_flat

    # 3. j_l and j_l' tables on the fixed x-range (#54): independent of cosmology;
    #    rows match _sparse_l_padded(l_max)
    x_table, jl_table, jlp_table = _jl_tables(l_max + L_SPLINE_PAD, n_x, x_max)
    dx_tab = x_max / (n_x - 1)

    # 4. Table positions (shared across all l and both tables)
    i_left, t_pos = _table_position(x_flat_query, dx_tab, n_x)

    # 5. Compute transfer functions via table lookup (scan over l)
    l_sparse = _sparse_l_padded(l_max)
    n_l = len(l_sparse)
    l_sparse_jnp = jnp.array(l_sparse, dtype=jnp.float64)

    # Precompute l-dependent constants for radT2 and EE
    ll1 = l_sparse_jnp * (l_sparse_jnp + 1.0)  # l(l+1) for radT2
    ee_prefactors = jnp.sqrt(
        l_sparse_jnp * (l_sparse_jnp + 1.0) *
        (l_sparse_jnp - 1.0) * (l_sparse_jnp + 2.0)
    )

    # jax.checkpoint: the backward pass recomputes each l's lookups instead of
    # storing ~(n_l x n_k x n_tau) residuals (~46 GB at fit_cl; #54).
    @jax.checkpoint
    def scan_body(carry, l_idx):
        jl_flat, jlp_flat = _lookup_jl(
            jl_table[l_idx], jlp_table[l_idx], l_sparse_jnp[l_idx], ll1[l_idx],
            i_left, t_pos, dx_tab, _jl_lookup)

        # Compute radial_T2 on-the-fly from j_l and j_l':
        # j_l''(x) = [l(l+1)/x² - 1]*j_l - 2/x*j_l'
        # radial_T2 = 0.5*(3*j_l'' + j_l) = 0.5*([3*l(l+1)/x² - 2]*j_l - 6/x*j_l')
        # cf. CLASS transfer.c:4168-4190
        rT2_flat = 0.5 * ((3.0 * ll1[l_idx] * inv_x2_flat - 2.0) * jl_flat
                          - 6.0 * inv_x_flat * jlp_flat)

        jl_interp = jl_flat.reshape(n_k_fine, n_tau)        # (n_k, n_tau)
        jlp_interp = jlp_flat.reshape(n_k_fine, n_tau)
        rT2_interp = rT2_flat.reshape(n_k_fine, n_tau)

        # TT transfer: T_l(k) = Σ_τ [S_T0*j_l + S_T1*j_l' + (S_T2/8)*radial_T2] * dτ
        # cf. CLASS harmonic.c:962 (T_total = T0 + T1 + T2)
        T_l = jnp.sum(src_T0_dtau * jl_interp
                      + src_T1_dtau * jlp_interp
                      + src_T2_dtau * rT2_interp, axis=-1)  # (n_k,)
        # EE transfer: E_l(k) = prefactor * Σ_τ source_E * j_l/x² * dτ
        E_l = ee_prefactors[l_idx] * jnp.sum(src_E_dtau * jl_interp, axis=-1)

        return carry, (T_l, E_l)

    _, (T_all, E_all) = jax.lax.scan(scan_body, None, jnp.arange(n_l))
    # T_all: (n_l, n_k_fine), E_all: (n_l, n_k_fine)

    # 6. Vectorized k-integration for ALL l at once
    P_R = primordial_scalar_pk(k_fine, params)
    dlnk = jnp.diff(jnp.log(k_fine))

    # C_l^TT = 4π ∫ dlnk P_R |T_l|²
    integrand_tt = P_R[None, :] * T_all**2
    cl_tt_sparse = 4.0 * jnp.pi * jnp.sum(
        0.5 * (integrand_tt[:, :-1] + integrand_tt[:, 1:]) * dlnk[None, :], axis=1
    )

    # C_l^EE = 4π ∫ dlnk P_R |E_l|²
    integrand_ee = P_R[None, :] * E_all**2
    cl_ee_sparse = 4.0 * jnp.pi * jnp.sum(
        0.5 * (integrand_ee[:, :-1] + integrand_ee[:, 1:]) * dlnk[None, :], axis=1
    )

    # C_l^TE = 4π ∫ dlnk P_R T_l E_l
    integrand_te = P_R[None, :] * T_all * E_all
    cl_te_sparse = 4.0 * jnp.pi * jnp.sum(
        0.5 * (integrand_te[:, :-1] + integrand_te[:, 1:]) * dlnk[None, :], axis=1
    )

    # 7. Spline interpolation to dense l-grid
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

def compute_cl_bb(tpt, params, bg, l_values, *, n_k_fine=2000):
    """Compute unlensed C_l^BB from tensor perturbation source functions.

    Uses the CLASS tensor B-polarization radial kernel
    K_l^B(x) = 0.5 * (j_l'(x) + 2 * j_l(x) / x)
    (cf. CLASS transfer.c:4263-4272 case TENSOR_POLARISATION_B, flat-space limit
    K=0 where cscKgen=cotKgen=1/x and rescale_function=1).

    The source_p(k, tau) is interpolated from the perturbation k-grid to a fine
    log-uniform k-grid (n_k_fine modes) before the transfer integral, since the
    Bessel-driven oscillation period in k is comparable to the spacing of the
    perturbation k-grid at the BB recombination peak. Mirrors the n_k_fine
    interpolation already used by ``compute_cls_all_fast`` for scalar T,E.
    """
    tau_grid = tpt.tau_grid
    k_grid_coarse = tpt.k_grid
    log_k_coarse = jnp.log(k_grid_coarse)
    log_k_fine = jnp.linspace(log_k_coarse[0], log_k_coarse[-1], n_k_fine)
    k_fine = jnp.exp(log_k_fine)

    def interp_one_tau(itau):
        return CubicSpline(log_k_coarse, tpt.source_p[:, itau]).evaluate(log_k_fine)
    source_p_fine = jax.vmap(interp_one_tau)(jnp.arange(len(tau_grid))).T

    chi_grid = bg.conformal_age - tau_grid
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1] / 2, (dtau[:-1] + dtau[1:]) / 2, dtau[-1:] / 2])

    def compute_cl_single_l(l):
        l_int = int(l)

        def transfer_single_k(ik):
            k = k_fine[ik]
            x = k * chi_grid
            x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
            jl = spherical_jl_backward(l_int, x)
            jlm1 = spherical_jl_backward(l_int - 1, x)
            # j_l'(x) = j_{l-1}(x) - (l+1)/x * j_l(x)
            jl_prime = jlm1 - (l_int + 1) / x_safe * jl
            radial_B = 0.5 * (jl_prime + 2.0 / x_safe * jl)
            return jnp.sum(source_p_fine[ik, :] * radial_B * dtau_mid)

        B_l = jax.vmap(transfer_single_k)(jnp.arange(n_k_fine))
        P_T = primordial_tensor_pk(k_fine, params)
        integrand = P_T * B_l**2
        dlnk = jnp.diff(log_k_fine)
        return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    cls = []
    for l in l_values:
        cls.append(compute_cl_single_l(l))
    return jnp.array(cls)
