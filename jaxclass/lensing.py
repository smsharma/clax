"""Lensing module for jaxCLASS.

Computes lensed CMB power spectra from unlensed C_l's and the lensing
potential power spectrum C_l^φφ.

The lensed spectra are computed using the first-order approximation:
    C_l^{TT,lensed} ≈ C_l^{TT,unlensed} * exp(-l(l+1) σ²/2)
where σ² is the RMS lensing deflection angle.

For v1: simple exponential damping approximation.
Future: full correlation function method (CLASS approach).

References:
    CLASS lensing.c
    Lewis & Challinor (2006) for the full method
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def lens_cl_tt(
    cl_tt_unlensed: Float[Array, "Nl"],
    cl_pp: Float[Array, "Nl"],
    l_values: Float[Array, "Nl"],
) -> Float[Array, "Nl"]:
    """Apply lensing to C_l^TT using the simple exponential damping approximation.

    C_l^{lensed} ≈ C_l^{unlensed} * exp(-l(l+1) * R_lens)

    where R_lens = ∫ dl l(l+1)(2l+1) C_l^φφ / (4π) is the lensing RMS.

    This is a rough approximation -- the full method uses correlation functions.

    Args:
        cl_tt_unlensed: unlensed TT power spectrum
        cl_pp: lensing potential power spectrum C_l^φφ
        l_values: multipole values

    Returns:
        Lensed C_l^TT (approximate)
    """
    # Compute RMS lensing deflection
    # σ² = Σ_l l(l+1)(2l+1)/(4π) C_l^φφ
    sigma_sq = jnp.sum(
        l_values * (l_values + 1) * (2 * l_values + 1) / (4 * jnp.pi) * cl_pp
    )

    # Simple lensing: smoothing of acoustic peaks
    # This is the zeroth-order approximation
    lensing_factor = jnp.exp(-l_values * (l_values + 1) * sigma_sq / 2)

    return cl_tt_unlensed * lensing_factor
