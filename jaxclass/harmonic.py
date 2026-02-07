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

References:
    Dodelson "Modern Cosmology" (2003) eq. 9.35
    CLASS harmonic.c: cl_integrand = 4π/k × P_R × T² × Δk
    CLASS perturbations.c:7660-7678: source function assembly
    CLASS transfer.c: radial_function type 2 for E-mode
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


def compute_cl_tt(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^TT from perturbation source functions.

    Uses the CLASS IBP source (S_T0 includes SW + ISW + Doppler after
    integration by parts), so only j_l is needed.

    C_l = 4π ∫ dlnk P_R(k) |T_l(k)|²

    Args:
        pt: perturbation result with source function tables
        params: cosmological parameters
        bg: background result
        l_values: multipole values (as numpy array of ints)

    Returns:
        C_l values (dimensionless raw C_l)
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)

    chi_grid = tau_0 - tau_grid  # comoving distance

    # Weights for trapezoidal integration over τ
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

    log_k = jnp.log(k_grid)

    def compute_cl_single_l(l):
        """Compute C_l at a single multipole l."""
        l_int = int(l)

        # Compute T_l(k) on coarse k-grid via line-of-sight integration
        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)

            # S_T0 contains SW + ISW + Doppler (IBP form)
            S0 = pt.source_T0[ik, :]
            return jnp.sum(S0 * jl * dtau_mid)

        T_l_coarse = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))

        # C_l = 4π ∫ dlnk P_R(k) |T_l(k)|²
        # cf. Dodelson (2003) eq. 9.35: C_l = (2/π) ∫ k² P(k) |T_l|² dk
        # where P(k) = (2π²/k³) P_R → C_l = 4π ∫ P_R |T_l|² dlnk
        P_R_coarse = primordial_scalar_pk(k_grid, params)
        integrand_k = P_R_coarse * T_l_coarse**2
        dlnk = jnp.diff(log_k)
        cl = 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_k[:-1] + integrand_k[1:]) * dlnk)
        return cl

    cls = []
    for l in l_values:
        cl = compute_cl_single_l(l)
        cls.append(cl)

    return jnp.array(cls)


def compute_cl_ee(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^EE from E-polarization source functions.

    The E-mode transfer uses the type-2 radial function:
        E_l(k) = sqrt((l+2)!/(l-2)!) * ∫ dτ S_E(k,τ) j_l(kχ) / (kχ)²
    where sqrt((l+2)!/(l-2)!) = sqrt(l(l+1)(l-1)(l+2)) for l >= 2.

    C_l^EE = 4π ∫ dlnk P_R(k) |E_l(k)|²

    cf. CLASS transfer.c: radial function type 2
    cf. Zaldarriaga & Seljak (1997) eq. 18

    Args:
        pt: perturbation result with source function tables
        params: cosmological parameters
        bg: background result
        l_values: multipole values (as numpy array of ints, must be >= 2)

    Returns:
        C_l^EE values (dimensionless raw C_l)
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)

    chi_grid = tau_0 - tau_grid  # comoving distance

    # Weights for trapezoidal integration over τ
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

    log_k = jnp.log(k_grid)

    def compute_cl_single_l(l):
        """Compute C_l^EE at a single multipole l."""
        l_int = int(l)

        # Prefactor: sqrt(l(l+1)(l-1)(l+2)) = sqrt((l+2)!/(l-2)!)
        prefactor = jnp.sqrt(l * (l + 1.0) * (l - 1.0) * (l + 2.0))

        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)

            # Radial function for E-mode: j_l(x) / x²
            # Avoid division by zero for small x (chi → 0 at τ → τ_0)
            x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
            radial_E = jl / (x_safe * x_safe)

            S_E = pt.source_E[ik, :]
            return prefactor * jnp.sum(S_E * radial_E * dtau_mid)

        E_l_coarse = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))

        P_R_coarse = primordial_scalar_pk(k_grid, params)
        integrand_k = P_R_coarse * E_l_coarse**2
        dlnk = jnp.diff(log_k)
        cl = 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_k[:-1] + integrand_k[1:]) * dlnk)
        return cl

    cls = []
    for l in l_values:
        cl = compute_cl_single_l(l)
        cls.append(cl)

    return jnp.array(cls)


def compute_cl_te(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^TE (temperature-polarization cross-correlation).

    C_l^TE = 4π ∫ dlnk P_R(k) T_l(k) E_l(k)

    This can be negative (anti-correlated at some multipoles).

    cf. CLASS harmonic.c
    cf. Zaldarriaga & Seljak (1997)

    Args:
        pt: perturbation result with source function tables
        params: cosmological parameters
        bg: background result
        l_values: multipole values (as numpy array of ints, must be >= 2)

    Returns:
        C_l^TE values (dimensionless raw C_l, can be negative)
    """
    tau_grid = pt.tau_grid
    k_grid = pt.k_grid
    tau_0 = float(bg.conformal_age)

    chi_grid = tau_0 - tau_grid  # comoving distance

    # Weights for trapezoidal integration over τ
    dtau = jnp.diff(tau_grid)
    dtau_mid = jnp.concatenate([dtau[:1], (dtau[:-1] + dtau[1:]) / 2, dtau[-1:]])

    log_k = jnp.log(k_grid)

    def compute_cl_single_l(l):
        """Compute C_l^TE at a single multipole l."""
        l_int = int(l)

        # Prefactor for E-mode: sqrt(l(l+1)(l-1)(l+2))
        prefactor = jnp.sqrt(l * (l + 1.0) * (l - 1.0) * (l + 2.0))

        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)

            # T transfer (same as compute_cl_tt)
            S0 = pt.source_T0[ik, :]
            T_l = jnp.sum(S0 * jl * dtau_mid)

            # E transfer
            x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
            radial_E = jl / (x_safe * x_safe)
            S_E = pt.source_E[ik, :]
            E_l = prefactor * jnp.sum(S_E * radial_E * dtau_mid)

            return T_l, E_l

        T_l_coarse, E_l_coarse = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))

        P_R_coarse = primordial_scalar_pk(k_grid, params)
        integrand_k = P_R_coarse * T_l_coarse * E_l_coarse
        dlnk = jnp.diff(log_k)
        cl = 4.0 * jnp.pi * jnp.sum(0.5 * (integrand_k[:-1] + integrand_k[1:]) * dlnk)
        return cl

    cls = []
    for l in l_values:
        cl = compute_cl_single_l(l)
        cls.append(cl)

    return jnp.array(cls)


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
        l_values: multipole values (as numpy array of ints, must be >= 2)

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

    cls = []
    for l in l_values:
        cl = compute_cl_single_l(l)
        cls.append(cl)

    return jnp.array(cls)
