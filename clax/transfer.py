"""Transfer function module for clax.

Computes transfer functions Δ_l(k) from source functions via line-of-sight
integration:
    Δ_l(k) = ∫ dτ S(k,τ) j_l(k(τ₀-τ))

For large l, uses the Limber approximation:
    Δ_l(k) ≈ √(π/(2l+1)) * S(k, τ₀ - (l+1/2)/k) / k

References:
    CLASS source: transfer.c
    DESIGN.md Section 4.8
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from clax.background import BackgroundResult
from clax.bessel import spherical_jl
from clax.interpolation import CubicSpline
from clax.params import PrecisionParams
from clax.perturbations import PerturbationResult


# Placeholder: transfer functions will be computed from perturbation source functions
# and Bessel functions. Full implementation requires:
# 1. Source function interpolation on (k, τ) grid
# 2. Bessel function evaluation j_l(k(τ₀-τ))
# 3. Integration over τ for each (l, k) pair
# 4. Limber approximation for large l

# For v1, we compute P(k) directly from perturbation output (δ_m at z=0)
# and defer the full transfer function / C_l computation to later.

def compute_pk_from_perturbations(
    pt: PerturbationResult,
    bg: BackgroundResult,
    k_eval: Float[Array, "Nk_eval"],
    z: float = 0.0,
) -> Float[Array, "Nk_eval"]:
    """Compute linear matter density contrast delta_m(k) at redshift z.

    Extracts delta_m from the perturbation output grid, interpolating in
    both conformal time (tau) and wavenumber (k).

    Args:
        pt: Perturbation results containing delta_m(k, tau)
        bg: Background results (for tau(z) conversion)
        k_eval: Wavenumber array to evaluate at (Mpc^-1)
        z: Redshift (default 0)

    Returns:
        delta_m(k) at the requested redshift, shape (Nk_eval,)
    """
    from clax.background import tau_of_z

    tau_grid = pt.tau_grid  # shape (Ntau,)
    log_k_pt = jnp.log(pt.k_grid)
    log_k_eval = jnp.log(k_eval)

    if z == 0.0:
        # Fast path: last time step
        delta_m_at_z = pt.delta_m[:, -1]  # shape (Nk,)
    else:
        # Find tau at requested z
        tau_z = tau_of_z(bg, z)
        # Interpolate delta_m at each k to the target tau
        # delta_m has shape (Nk, Ntau) — interpolate along tau axis
        def _interp_single_k(delta_m_k):
            spline = CubicSpline(tau_grid, delta_m_k)
            return spline.evaluate(tau_z)
        delta_m_at_z = jax.vmap(_interp_single_k)(pt.delta_m)

    # Interpolate to evaluation k-grid
    delta_m_spline = CubicSpline(log_k_pt, delta_m_at_z)
    delta_m_eval = delta_m_spline.evaluate(log_k_eval)

    return delta_m_eval
