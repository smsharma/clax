"""Harmonic (C_l) module for jaxCLASS.

Computes angular power spectra C_l^TT from source functions and primordial P(k).

The transfer function Δ_l(k) is:
    Δ_l(k) = ∫ dτ [S_T0(k,τ) * j_l(kχ) + S_T1(k,τ) * j_l'(kχ)]

where χ = τ₀ - τ, S_T0 is the SW+ISW source, S_T1 is the Doppler source.

    C_l = ∫ dlnk P_R(k) |Δ_l(k)|²

References:
    CLASS harmonic.c line 1073: C_l normalization
    CLASS transfer.c: line-of-sight integration
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
from jaxclass.primordial import primordial_scalar_pk
from jaxclass.perturbations import PerturbationResult


def _spherical_jl_derivative(l: int, x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Compute j_l'(x) = dj_l/dx.

    Uses the recurrence: j_l'(x) = j_{l-1}(x) - (l+1)/x * j_l(x)
    """
    if l == 0:
        return -spherical_jl(1, x)

    jl = spherical_jl(l, x)
    jlm1 = spherical_jl(l - 1, x)
    x_safe = jnp.where(jnp.abs(x) < 1e-30, 1e-30, x)
    return jlm1 - (l + 1.0) / x_safe * jl


def compute_cl_tt(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^TT from perturbation source functions.

    Includes both monopole (SW+ISW) and Doppler terms:
        Δ_l(k) = ∫ dτ [S_T0 * j_l(kχ) + S_T1 * j_l'(kχ)]

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

    # Build fine k-grid by interpolating the transfer function
    n_k_fine = 500
    log_k_fine = jnp.linspace(jnp.log(k_grid[0]), jnp.log(k_grid[-1]), n_k_fine)
    k_fine = jnp.exp(log_k_fine)
    log_k = jnp.log(k_grid)

    def compute_cl_single_l(l):
        """Compute C_l at a single multipole l."""
        l_int = int(l)

        # Compute Δ_l(k) on coarse k-grid
        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)

            # Monopole (SW + ISW)
            S0 = pt.source_T0[ik, :]
            delta_l = jnp.sum(S0 * jl * dtau_mid)

            # Doppler (× j_l')
            S1 = pt.source_T1[ik, :]
            jl_prime = _spherical_jl_derivative(l_int, x)
            delta_l = delta_l + jnp.sum(S1 * jl_prime * dtau_mid)

            return delta_l

        Delta_l_coarse = jax.vmap(transfer_single_k)(jnp.arange(len(k_grid)))

        # Interpolate Δ_l(k) to fine k-grid
        Delta_l_spline = CubicSpline(log_k, Delta_l_coarse)
        Delta_l_fine = Delta_l_spline.evaluate(log_k_fine)

        P_R_fine = primordial_scalar_pk(k_fine, params)

        # C_l = ∫ dlnk P_R(k) |Δ_l(k)|²
        dlnk_f = jnp.diff(log_k_fine)
        integrand_k = P_R_fine * Delta_l_fine**2
        cl = jnp.sum(0.5 * (integrand_k[:-1] + integrand_k[1:]) * dlnk_f)
        return cl

    cls = []
    for l in l_values:
        cl = compute_cl_single_l(l)
        cls.append(cl)

    return jnp.array(cls)
