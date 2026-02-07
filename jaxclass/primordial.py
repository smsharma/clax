"""Primordial power spectrum for jaxCLASS.

Implements the standard power-law parameterization:
    P_R(k) = A_s * (k / k_pivot)^{n_s - 1 + (1/2)*alpha_s*ln(k/k_pivot)}

For tensor modes:
    P_T(k) = A_s * r * (k / k_pivot)^{n_t}

References:
    CLASS source: primordial.c
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxclass.params import CosmoParams


def primordial_scalar_pk(k: Float[Array, "..."], params: CosmoParams) -> Float[Array, "..."]:
    """Compute primordial scalar power spectrum P_R(k).

    Args:
        k: wavenumber(s) in Mpc^-1
        params: cosmological parameters

    Returns:
        P_R(k) (dimensionless)
    """
    A_s = jnp.exp(params.ln10A_s) / 1e10
    ln_k_over_pivot = jnp.log(k / params.k_pivot)

    # Spectral index with optional running
    ns_eff = params.n_s - 1.0 + 0.5 * params.alpha_s * ln_k_over_pivot

    return A_s * (k / params.k_pivot) ** ns_eff


def primordial_tensor_pk(k: Float[Array, "..."], params: CosmoParams) -> Float[Array, "..."]:
    """Compute primordial tensor power spectrum P_T(k).

    Args:
        k: wavenumber(s) in Mpc^-1
        params: cosmological parameters

    Returns:
        P_T(k) (dimensionless)
    """
    A_s = jnp.exp(params.ln10A_s) / 1e10
    return A_s * params.r_t * (k / params.k_pivot) ** params.n_t
