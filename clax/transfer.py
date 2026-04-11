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
import numpy as np
from jaxtyping import Array, Float

from clax.background import BackgroundResult
from clax.bessel import spherical_jl
from clax.interpolation import CubicSpline
from clax.params import CosmoParams, PrecisionParams
from clax.perturbations import MatterPerturbationResult, PerturbationResult
from clax.primordial import primordial_scalar_pk


# Placeholder: transfer functions will be computed from perturbation source functions
# and Bessel functions. Full implementation requires:
# 1. Source function interpolation on (k, τ) grid
# 2. Bessel function evaluation j_l(k(τ₀-τ))
# 3. Integration over τ for each (l, k) pair
# 4. Limber approximation for large l

# For v1, we compute P(k) directly from perturbation output (δ_m at z=0)
# and defer the full transfer function / C_l computation to later.


def _validate_k_eval_support(
    k_grid: Float[Array, "Nk"],
    k_eval: Float[Array, "Nk_eval"],
) -> None:
    """Reject table queries outside the solved perturbation support.

    The public table-backed ``P(k)`` APIs solve the perturbations only on
    ``pt.k_grid`` and then interpolate in ``log k``. Silent extrapolation far
    beyond that range produces misleading spectra, so require callers to stay
    within the solved support explicitly.
    """
    k_grid_np = np.asarray(k_grid, dtype=float)
    k_eval_np = np.asarray(k_eval, dtype=float)
    k_min = float(k_grid_np[0])
    k_max = float(k_grid_np[-1])
    tol = 1.0e-12
    below = k_eval_np < (1.0 - tol) * k_min
    above = k_eval_np > (1.0 + tol) * k_max
    if np.any(below) or np.any(above):
        req_min = float(np.min(k_eval_np))
        req_max = float(np.max(k_eval_np))
        raise ValueError(
            "k_eval must lie within the solved perturbation grid "
            f"[{k_min:.6g}, {k_max:.6g}] Mpc^-1; got "
            f"[{req_min:.6g}, {req_max:.6g}] Mpc^-1."
        )


def compute_pk_from_perturbations(
    pt: PerturbationResult | MatterPerturbationResult,
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

    _validate_k_eval_support(pt.k_grid, k_eval)

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


def compute_linear_matter_pk_from_perturbations(
    pt: PerturbationResult | MatterPerturbationResult,
    bg: BackgroundResult,
    params: CosmoParams,
    k_eval: Float[Array, "Nk_eval"],
    z: float = 0.0,
) -> Float[Array, "Nk_eval"]:
    """Compute linear matter ``P(k, z)`` from a perturbation-table solve.

    Reuses the perturbation-table interpolation for ``delta_m(k, z)`` and
    applies the primordial normalization at the requested ``k`` values.

    Args:
        pt: perturbation results containing ``delta_m(k, tau)``
        bg: background results for ``tau(z)`` conversion
        params: cosmological parameters for the primordial spectrum
        k_eval: wavenumbers in ``Mpc^-1``
        z: redshift at which to evaluate the spectrum

    Returns:
        Linear matter power spectrum ``P(k, z)`` in ``Mpc^3``
    """
    k_eval = jnp.asarray(k_eval)
    delta_m = compute_pk_from_perturbations(pt, bg, k_eval, z=z)
    primordial = primordial_scalar_pk(k_eval, params)
    return 2.0 * jnp.pi**2 / k_eval**3 * primordial * delta_m**2
