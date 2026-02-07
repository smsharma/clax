"""Harmonic (C_l) module for jaxCLASS.

Computes angular power spectra C_l^TT, C_l^EE, C_l^TE from source functions
and primordial P(k).

The source function S_T0 uses the CLASS IBP form (all terms × j_l only):
    T_l(k) = ∫ dτ S_T0(k,τ) j_l(kχ)
    C_l^TT = 4π ∫ dlnk P_R(k) |T_l(k)|²

E-polarization uses the type-2 radial function:
    E_l(k) = sqrt((l+2)!/(l-2)!) ∫ dτ S_E(k,τ) j_l(kχ) / (kχ)²
    C_l^EE = 4π ∫ dlnk P_R(k) |E_l(k)|²
    C_l^TE = 4π ∫ dlnk P_R(k) T_l(k) E_l(k)

For l > l_limber_switch, uses Limber approximation:
    T_l^limber(k) ≈ sqrt(π/(2l+1)) * S(k, τ_0 - (l+0.5)/k) / k
which replaces the expensive Bessel integral with a single source evaluation.
Smooth sigmoid blending in the transition region.

References:
    Dodelson "Modern Cosmology" (2003) eq. 9.35
    CLASS harmonic.c: cl_integrand = 4π/k × P_R × T² × Δk
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

def _cl_k_integral(T_l: Float[Array, "Nk"], k_grid: Float[Array, "Nk"],
                   params: CosmoParams, k_interp_factor: int = 3) -> float:
    """Integrate C_l = 4π ∫ dlnk P_R(k) |T_l(k)|² with optional k-refinement.

    When k_interp_factor > 1, spline-interpolates T_l onto a finer k-grid
    before integration.

    Args:
        T_l: transfer function values on coarse k-grid
        k_grid: coarse k-grid from perturbation solve
        params: cosmological parameters (for primordial spectrum)
        k_interp_factor: refinement factor (1 = no refinement)

    Returns:
        C_l value (scalar)
    """
    log_k = jnp.log(k_grid)

    if k_interp_factor <= 1:
        P_R = primordial_scalar_pk(k_grid, params)
        integrand = P_R * T_l**2
        dlnk = jnp.diff(log_k)
        return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dlnk)

    n_fine = len(k_grid) * k_interp_factor
    log_k_fine = jnp.linspace(log_k[0], log_k[-1], n_fine)
    k_fine = jnp.exp(log_k_fine)

    T_l_spline = CubicSpline(log_k, T_l)
    T_l_fine = T_l_spline.evaluate(log_k_fine)

    P_R_fine = primordial_scalar_pk(k_fine, params)
    integrand_fine = P_R_fine * T_l_fine**2
    dlnk_fine = jnp.diff(log_k_fine)
    return 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_fine[:-1] + integrand_fine[1:]) * dlnk_fine)


def _cl_k_integral_cross(T_l: Float[Array, "Nk"], E_l: Float[Array, "Nk"],
                         k_grid: Float[Array, "Nk"], params: CosmoParams,
                         k_interp_factor: int = 3) -> float:
    """Like _cl_k_integral but for cross-spectrum C_l = 4π ∫ dlnk P_R T_l E_l."""
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
# Limber approximation
# ---------------------------------------------------------------------------

def _limber_transfer_tt(source_T0, tau_grid, k_grid, tau_0, l):
    """Limber approximation for temperature transfer function T_l(k).

    For large l, j_l(kχ) ≈ sqrt(π/(2l+1)) * δ(kχ - (l+0.5)), so:
        T_l(k) ≈ IPhiFlat * S_T0(k, τ_0 - (l+0.5)/k) / k

    cf. CLASS transfer.c:3606-3674

    Args:
        source_T0: shape (Nk, Ntau) - temperature source function
        tau_grid: shape (Ntau,) - conformal time grid
        k_grid: shape (Nk,) - wavenumber grid
        tau_0: conformal age (scalar)
        l: multipole (Python int)

    Returns:
        T_l: shape (Nk,) - transfer function values
    """
    l_fl = float(l)

    # Limber point: τ_limber = τ_0 - (l+0.5)/k
    nu = l_fl + 0.5
    tau_limber = tau_0 - nu / k_grid

    # Higher-order correction to Limber (CLASS transfer.c:3664)
    IPhiFlat = jnp.sqrt(jnp.pi / (2.0 * l_fl + 1.0)) * (
        1.0 - 1.0 / (8.0 * (2.0 * l_fl + 1.0))
    )

    # Evaluate source at each (k, tau_limber) via linear interp in tau
    def eval_source_at_limber(ik):
        source_k = source_T0[ik, :]
        tau_l = tau_limber[ik]
        return jnp.interp(tau_l, tau_grid, source_k, left=0.0, right=0.0)

    S_at_limber = jax.vmap(eval_source_at_limber)(jnp.arange(len(k_grid)))

    return IPhiFlat * S_at_limber / k_grid


def _limber_transfer_ee(source_E, tau_grid, k_grid, tau_0, l):
    """Limber approximation for E-mode transfer function E_l(k).

    For type-2 radial function j_l(x)/x², the Limber limit gives:
        E_l(k) ≈ prefactor * IPhiFlat * S_E(k, τ_limber) / (k * ν²)

    where ν = l + 0.5 and prefactor = sqrt(l(l+1)(l-1)(l+2)).

    Args:
        source_E: shape (Nk, Ntau)
        tau_grid, k_grid, tau_0, l: as in _limber_transfer_tt

    Returns:
        E_l: shape (Nk,)
    """
    l_fl = float(l)
    nu = l_fl + 0.5
    tau_limber = tau_0 - nu / k_grid

    prefactor = jnp.sqrt(l_fl * (l_fl + 1.0) * (l_fl - 1.0) * (l_fl + 2.0))
    IPhiFlat = jnp.sqrt(jnp.pi / (2.0 * l_fl + 1.0)) * (
        1.0 - 1.0 / (8.0 * (2.0 * l_fl + 1.0))
    )

    def eval_source_at_limber(ik):
        source_k = source_E[ik, :]
        tau_l = tau_limber[ik]
        return jnp.interp(tau_l, tau_grid, source_k, left=0.0, right=0.0)

    S_at_limber = jax.vmap(eval_source_at_limber)(jnp.arange(len(k_grid)))

    # Type-2 radial: extra 1/x² evaluated at x=ν gives 1/ν²
    return prefactor * IPhiFlat * S_at_limber / (k_grid * nu * nu)


# ---------------------------------------------------------------------------
# Transfer function computation (exact + Limber blend)
# ---------------------------------------------------------------------------

def _exact_transfer_tt(source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l):
    """Exact Bessel transfer function T_l(k) for temperature."""
    l_int = int(l)

    def transfer_single_k(ik):
        k = k_grid[ik]
        x = k * chi_grid
        jl = spherical_jl(l_int, x)
        S0 = source_T0[ik, :]
        return jnp.sum(S0 * jl * dtau_mid)

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
        S_E = source_E[ik, :]
        return prefactor * jnp.sum(S_E * radial_E * dtau_mid)

    return jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))


def _compute_transfer_tt(pt, bg, l, l_switch=400, delta_l=50):
    """Compute T_l(k) with exact/Limber blend based on l value.

    For l < l_switch - 2*delta_l: pure exact Bessel
    For l > l_switch + 2*delta_l: pure Limber (fast)
    In between: smooth sigmoid blend
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - tau_grid
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

    l_fl = float(l)

    if l_fl > l_switch + 2 * delta_l:
        # Pure Limber
        return _limber_transfer_tt(pt.source_T0, tau_grid, k_grid, tau_0, l)
    elif l_fl < l_switch - 2 * delta_l:
        # Pure exact
        return _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l)
    else:
        # Blend
        T_exact = _exact_transfer_tt(pt.source_T0, tau_grid, k_grid, chi_grid, dtau_mid, l)
        T_limber = _limber_transfer_tt(pt.source_T0, tau_grid, k_grid, tau_0, l)
        w = 1.0 / (1.0 + jnp.exp(-(l_fl - l_switch) / delta_l))
        return (1.0 - w) * T_exact + w * T_limber


def _compute_transfer_ee(pt, bg, l, l_switch=400, delta_l=50):
    """Compute E_l(k) with exact/Limber blend."""
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
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
    k_interp_factor: int = 3,
    l_switch: int = 200,
    delta_l: int = 50,
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^TT from perturbation source functions.

    Uses exact Bessel integration for l < l_switch, Limber approximation
    for l > l_switch, with smooth sigmoid blending in between.

    C_l = 4π ∫ dlnk P_R(k) |T_l(k)|²

    Args:
        pt: perturbation result with source function tables
        params: cosmological parameters
        bg: background result
        l_values: multipole values (as list of ints)
        k_interp_factor: k-grid refinement for C_l integration (default 3)
        l_switch: multipole where Limber starts blending in (default 200)
        delta_l: blending width (default 50)

    Returns:
        C_l values (dimensionless raw C_l)
    """
    k_grid = pt.k_grid

    cls = []
    for l in l_values:
        T_l = _compute_transfer_tt(pt, bg, l, l_switch, delta_l)
        cl = _cl_k_integral(T_l, k_grid, params, k_interp_factor)
        cls.append(cl)

    return jnp.array(cls)


def compute_cl_ee(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
    k_interp_factor: int = 3,
    l_switch: int = 200,
    delta_l: int = 50,
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^EE from E-polarization source functions.

    C_l^EE = 4π ∫ dlnk P_R(k) |E_l(k)|²

    Args:
        pt: perturbation result with source function tables
        params: cosmological parameters
        bg: background result
        l_values: multipole values (as list of ints, must be >= 2)
        k_interp_factor: k-grid refinement for C_l integration (default 3)
        l_switch: multipole where Limber starts blending in (default 200)
        delta_l: blending width (default 50)

    Returns:
        C_l^EE values (dimensionless raw C_l)
    """
    k_grid = pt.k_grid

    cls = []
    for l in l_values:
        E_l = _compute_transfer_ee(pt, bg, l, l_switch, delta_l)
        cl = _cl_k_integral(E_l, k_grid, params, k_interp_factor)
        cls.append(cl)

    return jnp.array(cls)


def compute_cl_te(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
    k_interp_factor: int = 3,
    l_switch: int = 200,
    delta_l: int = 50,
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^TE (temperature-polarization cross-correlation).

    C_l^TE = 4π ∫ dlnk P_R(k) T_l(k) E_l(k)

    Args:
        pt: perturbation result with source function tables
        params: cosmological parameters
        bg: background result
        l_values: multipole values (as list of ints, must be >= 2)
        k_interp_factor: k-grid refinement for C_l integration (default 3)
        l_switch: multipole where Limber starts blending in (default 200)
        delta_l: blending width (default 50)

    Returns:
        C_l^TE values (dimensionless raw C_l, can be negative)
    """
    k_grid = pt.k_grid

    cls = []
    for l in l_values:
        T_l = _compute_transfer_tt(pt, bg, l, l_switch, delta_l)
        E_l = _compute_transfer_ee(pt, bg, l, l_switch, delta_l)
        cl = _cl_k_integral_cross(T_l, E_l, k_grid, params, k_interp_factor)
        cls.append(cl)

    return jnp.array(cls)


# ---------------------------------------------------------------------------
# Sparse l-sampling + full spectrum API
# ---------------------------------------------------------------------------

def sparse_l_grid(l_max: int = 2500) -> np.ndarray:
    """Generate sparse l-sampling for efficient C_l computation.

    Mirrors CLASS strategy: dense at low l, sparser at high l.

    Returns:
        Array of l-values (Python ints)
    """
    l_list = list(range(2, min(31, l_max + 1)))       # every l up to 30
    l_list += list(range(35, min(101, l_max + 1), 5))  # every 5 up to 100
    l_list += list(range(120, min(501, l_max + 1), 20)) # every 20 up to 500
    l_list += list(range(550, l_max + 1, 50))           # every 50 up to l_max
    return np.array(l_list, dtype=int)


def compute_cls_all(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_max: int = 2500,
    k_interp_factor: int = 3,
    l_switch: int = 200,
    delta_l: int = 50,
) -> dict:
    """Compute all unlensed C_l spectra at l=2..l_max.

    Uses sparse l-sampling + spline interpolation for efficiency.
    At l > l_switch, uses Limber approximation instead of Bessel.

    Args:
        pt: perturbation result
        params: cosmological parameters
        bg: background result
        l_max: maximum multipole
        k_interp_factor: k-grid refinement
        l_switch: Limber transition multipole
        delta_l: Limber blending width

    Returns:
        dict with keys 'ell', 'tt', 'ee', 'te' (arrays of length l_max+1,
        indexed by l, with l=0,1 set to 0)
    """
    l_sparse = sparse_l_grid(l_max)
    l_sparse_float = l_sparse.astype(float)

    # Compute at sparse l-values
    cl_tt_sparse = compute_cl_tt(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l)
    cl_ee_sparse = compute_cl_ee(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l)
    cl_te_sparse = compute_cl_te(pt, params, bg, l_sparse.tolist(),
                                 k_interp_factor, l_switch, delta_l)

    # Spline interpolate to all integer l
    l_dense = jnp.arange(2, l_max + 1, dtype=jnp.float64)

    cl_tt_dense = CubicSpline(jnp.array(l_sparse_float), cl_tt_sparse).evaluate(l_dense)
    cl_ee_dense = CubicSpline(jnp.array(l_sparse_float), cl_ee_sparse).evaluate(l_dense)
    cl_te_dense = CubicSpline(jnp.array(l_sparse_float), cl_te_sparse).evaluate(l_dense)

    # Assemble full arrays (l=0,1 are zero)
    ell = jnp.arange(l_max + 1, dtype=jnp.float64)
    tt = jnp.concatenate([jnp.zeros(2), cl_tt_dense])
    ee = jnp.concatenate([jnp.zeros(2), cl_ee_dense])
    te = jnp.concatenate([jnp.zeros(2), cl_te_dense])

    return {'ell': ell, 'tt': tt, 'ee': ee, 'te': te}


# ---------------------------------------------------------------------------
# Tensor B-mode (no Limber needed — tensor spectra peak at low l)
# ---------------------------------------------------------------------------

def compute_cl_bb(
    tpt: TensorPerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^BB from tensor perturbation source functions.

    B-mode polarization comes exclusively from tensor modes (at linear order).
    The tensor polarization source S_p uses the type-2 radial function
    (same as scalar E-mode):
        B_l(k) = sqrt((l+2)!/(l-2)!) * integral dτ S_p(k,τ) j_l(kχ) / (kχ)²

    C_l^BB = 4π integral dlnk P_T(k) |B_l(k)|²

    where P_T(k) = A_s * r * (k/k_pivot)^{n_t} is the tensor primordial spectrum.

    cf. CLASS harmonic.c:1096-1101
    cf. Zaldarriaga & Seljak (1997) for tensor B-mode decomposition

    Args:
        tpt: tensor perturbation result with source function tables
        params: cosmological parameters (r_t for tensor amplitude)
        bg: background result
        l_values: multipole values (as list of ints, must be >= 2)

    Returns:
        C_l^BB values (dimensionless raw C_l)
    """
    tau_grid = tpt.tau_grid
    k_grid = tpt.k_grid
    tau_0 = float(bg.conformal_age)

    chi_grid = tau_0 - tau_grid

    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

    log_k = jnp.log(k_grid)

    def compute_cl_single_l(l):
        l_int = int(l)

        # Prefactor: sqrt(l(l+1)(l-1)(l+2))
        prefactor = jnp.sqrt(l * (l + 1.0) * (l - 1.0) * (l + 2.0))

        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)

            # Radial function for B-mode: j_l(x) / x^2 (type-2)
            x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
            radial_B = jl / (x_safe * x_safe)

            S_p = tpt.source_p[ik, :]
            return prefactor * jnp.sum(S_p * radial_B * dtau_mid)

        B_l_coarse = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))

        # C_l^BB = 4π integral dlnk P_T(k) |B_l(k)|^2
        P_T_coarse = primordial_tensor_pk(k_grid, params)
        integrand_k = P_T_coarse * B_l_coarse**2
        dlnk = jnp.diff(log_k)
        cl = 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_k[:-1] + integrand_k[1:]) * dlnk)
        return cl

    cls_bb = []
    for l in l_values:
        cl = compute_cl_single_l(l)
        cls_bb.append(cl)

    return jnp.array(cls_bb)


# ---------------------------------------------------------------------------
# Unified high-l C_l computation with Limber + sparse l-sampling
# ---------------------------------------------------------------------------

def compute_cls_all(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_max: int = 2500,
    k_interp_factor: int = 3,
) -> dict:
    """Compute unlensed C_l^TT, C_l^EE, C_l^TE for l=2..l_max.

    Uses exact Bessel line-of-sight integration at all l. Computes at
    sparse l-values (~100 points) and spline-interpolates to all integer l.

    NOTE: Limber approximation is NOT used for TT/EE/TE because the CMB
    source functions have sharp features (visibility peak) that the
    single-point Limber evaluation misses. Limber is only appropriate
    for smooth sources (lensing potential, galaxy clustering).
    The sparse l-sampling provides the speedup instead.

    This is the main C_l computation function for science-quality results.

    Args:
        pt: perturbation result with source function tables
        params: cosmological parameters
        bg: background result
        l_max: maximum multipole (default 2500)
        k_interp_factor: k-grid refinement for C_l integration (default 3)

    Returns:
        dict with keys 'ell', 'tt', 'ee', 'te' — arrays indexed by l (0..l_max)
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)
    chi_grid = tau_0 - tau_grid

    # Trapezoidal weights for τ integration
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

    # Sparse l-grid (~100 values for l_max=2500)
    l_sparse = sparse_l_grid(l_max)

    # --- Compute C_l at each sparse l via exact Bessel ---
    cl_tt_sparse = []
    cl_ee_sparse = []
    cl_te_sparse = []

    for l in l_sparse:
        l_int = int(l)
        l_fl = float(l)

        # Temperature transfer T_l(k)
        def transfer_tt_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)
            return jnp.sum(pt.source_T0[ik, :] * jl * dtau_mid)

        # E-mode transfer E_l(k)
        prefactor = jnp.sqrt(l_fl * (l_fl + 1.0) * (l_fl - 1.0) * (l_fl + 2.0))
        def transfer_ee_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)
            x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
            radial_E = jl / (x_safe * x_safe)
            return prefactor * jnp.sum(pt.source_E[ik, :] * radial_E * dtau_mid)

        T_l = jax.vmap(transfer_tt_k)(jnp.arange(len(k_grid)))
        E_l = jax.vmap(transfer_ee_k)(jnp.arange(len(k_grid)))

        cl_tt_sparse.append(_cl_k_integral(T_l, k_grid, params, k_interp_factor))
        cl_ee_sparse.append(_cl_k_integral(E_l, k_grid, params, k_interp_factor))
        cl_te_sparse.append(_cl_k_integral_cross(T_l, E_l, k_grid, params, k_interp_factor))

    cl_tt_sparse = jnp.array(cl_tt_sparse)
    cl_ee_sparse = jnp.array(cl_ee_sparse)
    cl_te_sparse = jnp.array(cl_te_sparse)

    # --- Spline interpolate to all integer l ---
    l_sparse_fl = jnp.array(l_sparse, dtype=jnp.float64)

    cl_tt_all = jnp.zeros(l_max + 1)
    cl_ee_all = jnp.zeros(l_max + 1)
    cl_te_all = jnp.zeros(l_max + 1)

    # Interpolate for l >= 2
    l_interp = jnp.arange(2, l_max + 1, dtype=jnp.float64)
    cl_tt_all = cl_tt_all.at[2:].set(CubicSpline(l_sparse_fl, cl_tt_sparse).evaluate(l_interp))
    cl_ee_all = cl_ee_all.at[2:].set(CubicSpline(l_sparse_fl, cl_ee_sparse).evaluate(l_interp))
    cl_te_all = cl_te_all.at[2:].set(CubicSpline(l_sparse_fl, cl_te_sparse).evaluate(l_interp))

    return {
        'ell': np.arange(l_max + 1),
        'tt': cl_tt_all,
        'ee': cl_ee_all,
        'te': cl_te_all,
    }
