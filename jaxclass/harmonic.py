"""Harmonic (C_l) module for jaxCLASS.

Computes angular power spectra C_l^TT from source functions and primordial P(k).

The temperature source function S_T0 is computed in the CLASS synchronous gauge
IBP form (perturbations.py), which includes SW, ISW, and Doppler (after
integration by parts). Only j_l is needed for S_T0.

    Δ_l(k) = ∫ dτ S_T0(k,τ) * j_l(k(τ₀-τ))
    C_l = ∫ dlnk P_R(k) |Δ_l(k)|²

References:
    CLASS harmonic.c line 1073: C_l normalization
    CLASS transfer.c: line-of-sight integration
    CLASS perturbations.c:7660-7678: source function assembly
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


def compute_cl_tt(
    pt: PerturbationResult,
    params: CosmoParams,
    bg: BackgroundResult,
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Compute unlensed C_l^TT from perturbation source functions.

    The source function S_T0 already includes SW + ISW + Doppler (IBP form),
    so we only need to multiply by j_l(kχ) and integrate.

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

        # Compute Δ_l(k) on coarse k-grid via line-of-sight integration
        def transfer_single_k(ik):
            k = k_grid[ik]
            x = k * chi_grid
            jl = spherical_jl(l_int, x)

            # S_T0 already contains SW + ISW + Doppler (IBP form)
            S0 = pt.source_T0[ik, :]
            delta_l = jnp.sum(S0 * jl * dtau_mid)

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
