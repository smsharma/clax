"""Non-linear corrections to the matter power spectrum (HaloFit).

Implements the Takahashi et al. (2012) revision of the HaloFit fitting formula
for converting linear P(k) to non-linear P(k). Includes the Bird et al. (2012)
massive neutrino corrections.

All functions are pure JAX (differentiable, JIT-compatible).

References:
    Smith et al. (2003), MNRAS 341, 1311 (original HaloFit)
    Takahashi et al. (2012), ApJ 761, 152 (revised fitting formulas)
    Bird, Viel & Haehnelt (2012), MNRAS 420, 2551 (massive neutrino correction)
    CLASS: external/Halofit/halofit.c

Mirrors CLASS source: external/Halofit/halofit.c, source/fourier.c
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# sigma(R): variance of the linear density field smoothed with Gaussian window
# ---------------------------------------------------------------------------

def _sigma_integrals(
    R: float,
    lnk: Float[Array, "N"],
    pk: Float[Array, "N"],
) -> tuple[float, float, float]:
    """Compute the three sigma integrals needed by HaloFit at a given R.

    Uses Gaussian window W(kR) = exp(-(kR)^2/2). The integrals are:
        sigma^2(R) = 1/(2 pi^2) int dk k^2 P(k) exp(-(kR)^2)
        d1(R)      = 1/(2 pi^2) int dk k^2 P(k) 2*(kR)^2 exp(-(kR)^2)
        d2(R)      = 1/(2 pi^2) int dk k^2 P(k) 4*(kR)^2*(1-(kR)^2) exp(-(kR)^2)

    Note: CLASS uses a full Gaussian exp(-(kR)^2), not exp(-(kR)^2/2).
    cf. halofit.c:518-520

    The integration is done over ln(k) using trapezoidal rule:
        int dk f(k) = int d(ln k) k f(k)

    Args:
        R: smoothing radius [Mpc]
        lnk: log wavenumbers, shape (N,)
        pk: linear power spectrum P(k) [Mpc^3], shape (N,)

    Returns:
        (sum1, sum2, sum3): the three integral values
    """
    anorm = 1.0 / (2.0 * jnp.pi**2)
    k = jnp.exp(lnk)
    x2 = (k * R) ** 2

    # Base integrand: P(k) * k^2 * (1/(2pi^2)) * exp(-x^2) * k (for dlnk)
    # cf. halofit.c:518
    base = pk * k**3 * anorm * jnp.exp(-x2)

    # integral_one: just the base
    integrand1 = base
    # integral_two: base * 2 * x^2
    # cf. halofit.c:519
    integrand2 = base * 2.0 * x2
    # integral_three: base * 4 * x^2 * (1 - x^2)
    # cf. halofit.c:520
    integrand3 = base * 4.0 * x2 * (1.0 - x2)

    # Trapezoidal integration over ln(k)
    dlnk = lnk[1:] - lnk[:-1]
    sum1 = jnp.sum(0.5 * (integrand1[1:] + integrand1[:-1]) * dlnk)
    sum2 = jnp.sum(0.5 * (integrand2[1:] + integrand2[:-1]) * dlnk)
    sum3 = jnp.sum(0.5 * (integrand3[1:] + integrand3[:-1]) * dlnk)

    return sum1, sum2, sum3


def sigma_R(
    R: float,
    lnk: Float[Array, "N"],
    pk: Float[Array, "N"],
) -> float:
    """Compute sigma(R), the RMS linear density fluctuation in a Gaussian window.

    sigma(R) = sqrt(1/(2 pi^2) int dk k^2 P(k) exp(-(kR)^2))

    Args:
        R: smoothing radius [Mpc]
        lnk: log wavenumbers, shape (N,)
        pk: linear power spectrum P(k) [Mpc^3], shape (N,)

    Returns:
        sigma(R)
    """
    sum1, _, _ = _sigma_integrals(R, lnk, pk)
    return jnp.sqrt(sum1)


# ---------------------------------------------------------------------------
# Find k_sigma, n_eff, C by bisection on sigma(R) = 1
# ---------------------------------------------------------------------------

def _bisect_sigma_body(carry, _):
    """One bisection step to find R where sigma(R) = 1.

    cf. halofit.c:315-358
    """
    xlogr1, xlogr2, lnk, pk, tol = carry
    xlogr_mid = 0.5 * (xlogr1 + xlogr2)
    rmid = jnp.power(10.0, xlogr_mid)
    sum1, _, _ = _sigma_integrals(rmid, lnk, pk)
    sigma = jnp.sqrt(sum1)
    diff = sigma - 1.0

    # If diff > 0 (sigma too large => R too small), move lower bound up
    # If diff < 0 (sigma too small => R too large), move upper bound down
    xlogr1 = jnp.where(diff > tol, xlogr_mid, xlogr1)
    xlogr2 = jnp.where(diff < -tol, xlogr_mid, xlogr2)

    return (xlogr1, xlogr2, lnk, pk, tol), None


def halofit_parameters(
    lnk: Float[Array, "N"],
    pk: Float[Array, "N"],
) -> tuple[float, float, float]:
    """Find the non-linear scale k_sigma, effective slope n_eff, and curvature C.

    Uses bisection to find R_nl where sigma(R_nl) = 1, then computes:
        k_sigma = 1 / R_nl
        n_eff = -3 - d1  (effective spectral index at k_sigma)
        C = -d2           (curvature of the spectrum at k_sigma)

    where d1 = -sum2/sum1 and d2 = -sum2^2/sum1^2 - sum3/sum1.
    cf. halofit.c:394-401

    Args:
        lnk: log wavenumbers, shape (N,)
        pk: linear power spectrum P(k) [Mpc^3], shape (N,)

    Returns:
        (k_sigma, n_eff, C)
    """
    k = jnp.exp(lnk)
    tol = 1e-6  # cf. CLASS: halofit_tol_sigma

    # Lower bound for log10(R): small R => large sigma
    # Minimum R such that integral is converged: exp(-(k_max*R)^2) < epsilon
    # cf. halofit.c:229
    R_min = jnp.sqrt(-jnp.log(1e-7)) / k[-1]
    xlogr1 = jnp.log10(R_min)

    # Upper bound: large R => small sigma
    # cf. halofit.c:286: R = 1/halofit_min_k_nonlinear
    # Use a generous upper bound
    R_max = 1.0 / 0.001  # 1000 Mpc
    xlogr2 = jnp.log10(R_max)

    # Run bisection for 60 iterations (converges to ~10^-18 in log10(R))
    carry = (xlogr1, xlogr2, lnk, pk, tol)
    carry, _ = jax.lax.scan(_bisect_sigma_body, carry, None, length=60)
    xlogr1, xlogr2, _, _, _ = carry

    # Final R
    rmid = jnp.power(10.0, 0.5 * (xlogr1 + xlogr2))

    # Compute all three integrals at this R
    sum1, sum2, sum3 = _sigma_integrals(rmid, lnk, pk)

    # cf. halofit.c:394-401
    d1 = -sum2 / sum1
    d2 = -sum2**2 / sum1**2 - sum3 / sum1

    k_sigma = 1.0 / rmid       # rknl in CLASS
    n_eff = -3.0 - d1           # rneff in CLASS
    C = -d2                     # rncur in CLASS

    return k_sigma, n_eff, C


# ---------------------------------------------------------------------------
# HaloFit non-linear P(k) (Takahashi 2012 + Bird 2012)
# ---------------------------------------------------------------------------

def halofit_nl_pk(
    k: Float[Array, "N"],
    pk_lin: Float[Array, "N"],
    Omega_m: float,
    Omega_v: float,
    w: float,
    fnu: float,
    h: float,
) -> Float[Array, "N"]:
    """Compute the non-linear power spectrum using HaloFit (Takahashi 2012).

    Applies the Takahashi et al. (2012) revised fitting formulas with
    Bird et al. (2012) massive neutrino corrections.

    P_NL(k) = P_quasi(k) + P_halo(k)

    cf. halofit.c:404-468

    Args:
        k: wavenumbers [Mpc^-1], shape (N,)
        pk_lin: linear P(k) [Mpc^3], shape (N,)
        Omega_m: matter density fraction at the redshift of interest
        Omega_v: vacuum/DE density fraction at the redshift of interest
        w: dark energy equation of state at the redshift of interest
        fnu: neutrino fraction Omega_ncdm / Omega_m
        h: dimensionless Hubble parameter

    Returns:
        P_NL(k) [Mpc^3], shape (N,)
    """
    lnk = jnp.log(k)

    # Find non-linear scale and spectral parameters
    k_sigma, n_eff, C = halofit_parameters(lnk, pk_lin)

    # Abbreviations
    # cf. halofit.c:410
    anorm = 1.0 / (2.0 * jnp.pi**2)

    # Dimensionless power: Delta^2(k) = k^3 P(k) / (2 pi^2)
    pk_lin_dimless = pk_lin * k**3 * anorm

    y = k / k_sigma

    # --- Takahashi 2012 fitting formulas ---
    # cf. halofit.c:421-427
    gam = 0.1971 - 0.0843 * n_eff + 0.8460 * C

    a_coeff = 10.0 ** (
        1.5222 + 2.8553 * n_eff + 2.3706 * n_eff**2
        + 0.9903 * n_eff**3 + 0.2250 * n_eff**4
        - 0.6038 * C + 0.1749 * Omega_v * (1.0 + w)
    )

    b_coeff = 10.0 ** (
        -0.5642 + 0.5864 * n_eff + 0.5716 * n_eff**2
        - 1.5474 * C + 0.2279 * Omega_v * (1.0 + w)
    )

    c_coeff = 10.0 ** (
        0.3698 + 2.0404 * n_eff + 0.8161 * n_eff**2 + 0.5869 * C
    )

    xmu = 0.0
    xnu = 10.0 ** (5.2105 + 3.6902 * n_eff)

    alpha = jnp.abs(
        6.0835 + 1.3373 * n_eff - 0.1959 * n_eff**2 - 5.5274 * C
    )

    beta = (
        2.0379 - 0.7354 * n_eff + 0.3157 * n_eff**2
        + 1.2490 * n_eff**3 + 0.3980 * n_eff**4
        - 0.1682 * C + fnu * (1.081 + 0.395 * n_eff**2)
    )

    # --- f1, f2, f3 factors for Omega evolution ---
    # cf. halofit.c:431-447
    f1a = Omega_m ** (-0.0732)
    f2a = Omega_m ** (-0.1423)
    f3a = Omega_m ** (0.0725)
    f1b = Omega_m ** (-0.0307)
    f2b = Omega_m ** (-0.0585)
    f3b = Omega_m ** (0.0743)

    frac = Omega_v / (1.0 - Omega_m)
    # Guard against Omega_m ~ 1 where frac is undefined
    frac = jnp.where(jnp.abs(1.0 - Omega_m) > 0.01, frac, 0.0)

    f1 = jnp.where(jnp.abs(1.0 - Omega_m) > 0.01,
                    frac * f1b + (1.0 - frac) * f1a, 1.0)
    f2 = jnp.where(jnp.abs(1.0 - Omega_m) > 0.01,
                    frac * f2b + (1.0 - frac) * f2a, 1.0)
    f3 = jnp.where(jnp.abs(1.0 - Omega_m) > 0.01,
                    frac * f3b + (1.0 - frac) * f3a, 1.0)

    # --- Halo term ---
    # cf. halofit.c:450-451
    pk_halo = (
        a_coeff * y ** (f1 * 3.0)
        / (1.0 + b_coeff * y**f2 + (f3 * c_coeff * y) ** (3.0 - gam))
    )
    pk_halo = pk_halo / (1.0 + xmu * y**(-1) + xnu * y**(-2)) * (1.0 + fnu * 0.977)

    # --- Quasi-linear term ---
    # cf. halofit.c:459-460
    # Bird 2012 neutrino correction to pk_lin
    pk_linaa = pk_lin_dimless * (
        1.0 + fnu * 47.48 * (k / h) ** 2 / (1.0 + 1.5 * (k / h) ** 2)
    )
    pk_quasi = (
        pk_lin_dimless
        * (1.0 + pk_linaa) ** beta
        / (1.0 + pk_linaa * alpha)
        * jnp.exp(-y / 4.0 - y**2 / 8.0)
    )

    # --- Total dimensionless power ---
    pk_nl_dimless = pk_halo + pk_quasi

    # Convert back to P(k) [Mpc^3]: P(k) = Delta^2(k) / (k^3 / (2 pi^2))
    # cf. halofit.c:462
    pk_nl = pk_nl_dimless / (k**3 * anorm)

    # Below the non-linear scale, just use linear P(k)
    # cf. halofit.c:467: if rk <= halofit_min_k_nonlinear, pk_nl = pk_lin
    k_min_nl = 0.001  # cf. CLASS precision: halofit_min_k_nonlinear = 0.001
    pk_nl = jnp.where(k > k_min_nl, pk_nl, pk_lin)

    return pk_nl


# ---------------------------------------------------------------------------
# Convenience wrapper: compute P_NL from background quantities
# ---------------------------------------------------------------------------

def compute_pk_nonlinear(
    k: Float[Array, "N"],
    pk_lin: Float[Array, "N"],
    Omega_m_0: float,
    Omega_lambda_0: float,
    Omega_r_0: float,
    w0: float = -1.0,
    wa: float = 0.0,
    fnu: float = 0.0,
    h: float = 0.6736,
    z: float = 0.0,
) -> Float[Array, "N"]:
    """Compute non-linear P(k) at a given redshift from linear P(k) at that redshift.

    This is a convenience wrapper that computes the redshift-dependent Omega_m and
    Omega_v needed by the HaloFit fitting formulas.

    Args:
        k: wavenumbers [Mpc^-1], shape (N,)
        pk_lin: linear P(k) at redshift z [Mpc^3], shape (N,)
        Omega_m_0: total matter density fraction today
        Omega_lambda_0: cosmological constant + DE density fraction today
        Omega_r_0: radiation density fraction today
        w0: dark energy EOS parameter
        wa: dark energy EOS derivative
        fnu: neutrino fraction Omega_ncdm / Omega_m
        h: dimensionless Hubble parameter
        z: redshift

    Returns:
        P_NL(k) [Mpc^3], shape (N,)
    """
    a = 1.0 / (1.0 + z)

    # Compute Omega_m(z) and Omega_v(z)
    # cf. halofit.c:107-108
    # Omega_m(a) = Omega_m_0 / a^3 / E^2(a)
    # Omega_v(a) = 1 - Omega_m(a) - Omega_r(a)
    # where E^2(a) = Omega_m_0/a^3 + Omega_r_0/a^4 + Omega_lambda_0*f_de(a)
    # For Lambda: f_de = 1; for w0wa: f_de = a^{-3(1+w0+wa)} * exp(-3*wa*(1-a))

    # DE density scaling
    rho_de_ratio = a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))
    # For pure Lambda, this gives 1.0

    E2 = (
        Omega_m_0 / a**3
        + Omega_r_0 / a**4
        + Omega_lambda_0 * rho_de_ratio
    )

    Omega_m = Omega_m_0 / a**3 / E2
    Omega_v = 1.0 - Omega_m - Omega_r_0 / a**4 / E2

    # Dark energy EOS at this redshift (CPL parameterization)
    w = w0 + wa * (1.0 - a)

    return halofit_nl_pk(k, pk_lin, Omega_m, Omega_v, w, fnu, h)
