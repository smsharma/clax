"""Harmonic (C_l) module for jaxCLASS.

Computes angular power spectra C_l^TT from source functions and primordial P(k).

The source function S_T0 uses the CLASS IBP form (all terms × j_l only):
    T_l(k) = ∫ dτ S_T0(k,τ) j_l(kχ)
    C_l = 4π ∫ dlnk P_R(k) |T_l(k)|²

References:
    Dodelson "Modern Cosmology" (2003) eq. 9.35
    CLASS harmonic.c: cl_integrand = 4π/k × P_R × T² × Δk
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
