"""Transfer function module for jaxCLASS.

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

from jaxclass.background import BackgroundResult
from jaxclass.bessel import spherical_jl
from jaxclass.interpolation import CubicSpline
from jaxclass.params import PrecisionParams
from jaxclass.perturbations import PerturbationResult


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
) -> Float[Array, "Nk_eval"]:
    """Compute linear matter power spectrum P(k) at z=0.

    This uses the matter density contrast δ_m from the perturbation output
    at the last time step (z≈0).

    P(k) = (2π²/k³) * A_s * (k/k_pivot)^(n_s-1) * |T(k)|²

    where T(k) = δ_m(k, z=0) is the matter transfer function
    (normalized so that T → 1 as k → 0 in the matter-dominated era).

    For now, this interpolates δ_m from the perturbation k-grid.
    """
    # δ_m at z=0 (last time step)
    delta_m_z0 = pt.delta_m[:, -1]  # shape (Nk,)

    # Interpolate to evaluation k-grid
    log_k_pt = jnp.log(pt.k_grid)
    log_k_eval = jnp.log(k_eval)

    # Use log-space spline for δ_m(k)
    # Note: δ_m can be negative (at high k due to oscillations), so interpolate directly
    delta_m_spline = CubicSpline(log_k_pt, delta_m_z0)
    delta_m_eval = delta_m_spline.evaluate(log_k_eval)

    return delta_m_eval
