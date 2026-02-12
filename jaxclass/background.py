"""Background cosmology module for jaxCLASS.

Solves the Friedmann equation to obtain H(tau), a(tau), and all derived
background quantities (distances, growth factor, sound horizon, etc.)

The background ODE is integrated in log(a) from a_ini ~ 1e-14 to a = 1,
following CLASS's approach (background.c:background_solve).

Key functions:
    background_solve(params, prec) -> BackgroundResult

Mirrors CLASS source: source/background.c
    - background_functions() at line 371
    - background_derivs() at line 2589
    - background_ncdm_momenta() at line 1600
    - background_initial_conditions() at line 2169

Design choices:
    - Pre-tabulate neutrino density/pressure as splines on log(a) grid
      (DISCO-EB pattern), avoiding repeated Gauss-Laguerre quadrature.
    - Use Diffrax Tsit5 (non-stiff explicit RK) for the background ODE.
    - All results stored in BackgroundResult as CubicSpline objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxclass import constants as const
from jaxclass.interpolation import CubicSpline
from jaxclass.ode import solve_nonstiff
from jaxclass.params import CosmoParams, PrecisionParams


# ---------------------------------------------------------------------------
# BackgroundResult
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BackgroundResult:
    """Output of the background module.

    Contains spline interpolation tables for all background quantities
    as functions of log(a), plus derived scalar quantities.
    """

    # Grid
    loga_table: Float[Array, "N"]     # log(a) grid
    tau_table: Float[Array, "N"]      # conformal time at each grid point

    # Splines (functions of log(a))
    tau_of_loga: CubicSpline          # conformal time tau(log a)
    loga_of_tau: CubicSpline          # inverse: log(a)(tau)
    H_of_loga: CubicSpline            # Hubble rate H(log a) [Mpc^-1]
    rho_g_of_loga: CubicSpline        # photon density
    rho_b_of_loga: CubicSpline        # baryon density
    rho_cdm_of_loga: CubicSpline      # CDM density
    rho_ur_of_loga: CubicSpline       # ultra-relativistic density
    rho_ncdm_of_loga: CubicSpline     # massive neutrino density
    p_ncdm_of_loga: CubicSpline       # massive neutrino pressure
    w_ncdm_of_loga: CubicSpline       # ncdm equation of state w = p/rho
    ca2_ncdm_of_loga: CubicSpline     # ncdm adiabatic sound speed squared
    rho_de_of_loga: CubicSpline       # dark energy density
    rho_lambda_of_loga: CubicSpline   # cosmological constant density
    rs_of_loga: CubicSpline           # comoving sound horizon
    D_of_loga: CubicSpline            # growth factor D(a)
    f_of_loga: CubicSpline            # growth rate f = d ln D / d ln a

    # Derived scalars
    conformal_age: float              # tau_0 = tau(a=1) [Mpc]
    age_Gyr: float                    # proper age [Gyr]
    z_eq: float                       # matter-radiation equality redshift
    tau_eq: float                     # conformal time at equality [Mpc]
    H0: float                         # H0 in Mpc^-1 (= h * 100 km/s/Mpc / c)
    Omega_g: float                    # photon density fraction today
    Omega_b: float                    # baryon density fraction today
    Omega_cdm: float                  # CDM density fraction today
    Omega_ur: float                   # ultra-relativistic density fraction today
    Omega_ncdm: float                 # massive neutrino density fraction today
    Omega_de: float                   # dark energy density fraction today
    Omega_lambda: float               # cosmological constant density fraction today

    def tree_flatten(self):
        fields = [
            self.loga_table, self.tau_table,
            self.tau_of_loga, self.loga_of_tau,
            self.H_of_loga, self.rho_g_of_loga, self.rho_b_of_loga,
            self.rho_cdm_of_loga, self.rho_ur_of_loga,
            self.rho_ncdm_of_loga, self.p_ncdm_of_loga,
            self.w_ncdm_of_loga, self.ca2_ncdm_of_loga,
            self.rho_de_of_loga, self.rho_lambda_of_loga,
            self.rs_of_loga, self.D_of_loga, self.f_of_loga,
            self.conformal_age, self.age_Gyr, self.z_eq, self.tau_eq,
            self.H0, self.Omega_g, self.Omega_b, self.Omega_cdm,
            self.Omega_ur, self.Omega_ncdm, self.Omega_de, self.Omega_lambda,
        ]
        return fields, None

    @classmethod
    def tree_unflatten(cls, aux, fields):
        return cls(*fields)


# ---------------------------------------------------------------------------
# Internal helper: density computations
# ---------------------------------------------------------------------------

def _H0_from_h(h: float) -> float:
    """Convert dimensionless h to H0 in CLASS units [Mpc^-1].

    H0 [Mpc^-1] = h * 100 km/s/Mpc * (1e3 m/km) / (c [m/s] * Mpc_over_m [m/Mpc])
                 = h * 1e5 / (c_SI * Mpc_over_m)

    cf. CLASS: H0 is stored as H0/c in Mpc^-1.
    Actually CLASS stores H0 directly in Mpc^-1 with the conversion:
    H0 = pba->h * 1.e5 / _c_ / _Mpc_over_m_ ... wait, let me check.

    Actually in CLASS, H0 is in 1/Mpc with:
    H0 = h * 1e5 / _c_  (since _c_ is in m/s and 1e5 converts 100 km/s to m/s)
    But wait, we also need 1/Mpc_over_m since H0 should be in 1/Mpc not 1/m.
    H0 [1/s] = h * 100 * 1e3 / Mpc_over_m  (because 100 km/s/Mpc = 1e5 m/s/Mpc)
    H0 [1/Mpc] = H0 [1/s] * Mpc_over_m / c  ... no.

    Let me just follow CLASS exactly:
    In CLASS, H0 is pba->H0 with units such that the Friedmann eq is H^2 = sum(rho).
    From input.c: pba->H0 = pba->h * 1.e5 / _c_
    This gives H0 in s^-1 / (m/s) ... no.

    Actually _c_ = 2.998e8 m/s. So:
    pba->H0 = h * 1e5 / 2.998e8 = h * 3.336e-4 [1/Mpc? or 1/s?]

    Looking more carefully at CLASS units: CLASS uses Mpc as the unit of length
    and 1/Mpc as the unit of H. The conversion is:
    H0 [km/s/Mpc] = h * 100
    H0 [1/Mpc] = h * 100 [km/s/Mpc] * 1e3 [m/km] / c [m/s] = h * 1e5 / c
    No wait, 1/Mpc means inverse megaparsecs. Let me think again.

    If length unit is Mpc and time unit is Mpc/c (so that c=1 in these units),
    then H has units of 1/(Mpc/c) = c/Mpc. And:
    H0 = h * 100 km/s/Mpc

    To convert: H0 [c/Mpc] = H0 [km/s/Mpc] * (1e3 m/km) / (c_SI m/s)
    = h * 100 * 1e3 / c_SI = h * 1e5 / c_SI

    And indeed CLASS does: pba->H0 = pba->h * 1.e5 / _c_

    So H0 is in units of 1/Mpc (where time is measured in Mpc, i.e. c=1).
    """
    return h * 1e5 / const.c_SI


def _compute_omega_g(T_cmb: float, H0: float) -> float:
    """Compute photon density parameter Omega_g from T_cmb.

    cf. CLASS background.c, background_functions() line 425:
        rho_g = Omega0_g * H0^2 / a^4

    Omega_g = (4 * sigma_B * T_cmb^4) / (c^3) * (8 * pi * G) / (3 * H0^2)
    But in CLASS units where H^2 = rho, we have:
    rho_g_0 = Omega_g * H0^2

    The physical photon energy density is:
    rho_g_phys = (pi^2/15) * (k_B T)^4 / (hbar^3 c^3) = (4 sigma_B / c) * T^4

    In CLASS units (H^2 = sum rho_i, with H in 1/Mpc):
    Omega_g = (8 pi G) / (3 H0^2 c^2) * (4 sigma_B / c) * T_cmb^4 * (Mpc_over_m / c)^2 ... messy.

    Simpler: just use CLASS's formula. In CLASS:
    Omega0_g = (4.0 * sigma_B / pow(_c_,3)) * pow(T_cmb,4) / (3.0 * pow(_c_,2) * pow(H0,2) / (8.0 * _PI_ * _G_))

    Which simplifies to:
    Omega0_g = 32 * pi * G * sigma_B * T_cmb^4 / (3 * c^5 * H0^2)

    But H0 here is in 1/s. Wait, CLASS stores H0 = h * 1e5 / c in Mpc^-1... no.

    Let me just compute it directly. The key relation is:
    Omega_g * H0^2 [Mpc^-2] = (8piG/3) * rho_g_phys [J/m^3] * (Mpc_over_m)^2 / c^4

    rho_g_phys = (4 sigma_B / c) * T_cmb^4  [J/m^3]

    So:
    Omega_g = (8piG/3) * (4 sigma_B / c) * T_cmb^4 * Mpc_over_m^2 / (c^4 * H0^2)

    With H0 in Mpc^-1 (CLASS units).
    """
    rho_g_phys = 4.0 * const.sigma_B / const.c_SI * T_cmb**4  # J/m^3
    # Convert to CLASS units: rho_class = (8piG/3c^2) * rho_phys * (Mpc/c)^2
    # = (8piG / 3) * rho_phys * Mpc^2 / c^4
    rho_g_class = (
        8.0 * math.pi * const.G_SI / 3.0
        * rho_g_phys
        * const.Mpc_over_m**2
        / const.c_SI**4
    )
    return rho_g_class / H0**2


def _compute_omega_ur(N_ur: float, Omega_g: float) -> float:
    """Compute ultra-relativistic density parameter from N_ur and Omega_g.

    Omega_ur = N_ur * (7/8) * (4/11)^(4/3) * Omega_g

    This uses the standard relation between neutrino and photon energy densities
    for massless species.
    """
    return N_ur * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * Omega_g


# ---------------------------------------------------------------------------
# Neutrino momentum integrals (pre-tabulation)
# ---------------------------------------------------------------------------

def _gauss_laguerre_nodes_weights(n: int):
    """Compute Gauss-Laguerre quadrature nodes and weights.

    For integrating ∫_0^∞ f(q) * q^2 * f_FD(q) dq where f_FD = 1/(e^q + 1).
    We use the substitution and Gauss-Laguerre with appropriate weights.

    Returns fixed numpy arrays (not traced).
    """
    import numpy as np
    from numpy.polynomial.laguerre import laggauss
    q, w = laggauss(n)
    # Modify weights to include the Fermi-Dirac factor
    # ∫_0^∞ f(q) dq ≈ Σ_i w_i * f(q_i) * e^{q_i}  (standard Gauss-Laguerre)
    # We want ∫_0^∞ q^2 / (e^q + 1) * g(q) dq
    # = ∫_0^∞ e^{-q} * [q^2 * e^q / (e^q + 1)] * g(q) dq
    # ≈ Σ_i w_i * [q_i^2 * e^{q_i} / (e^{q_i} + 1)] * g(q_i)
    w_modified = w * q**2 * np.exp(q) / (np.exp(q) + 1.0)
    return jnp.array(q), jnp.array(w_modified)


def _ncdm_momenta(
    q: Float[Array, "Nq"],
    w: Float[Array, "Nq"],
    M: float,
    a: float,
) -> tuple[float, float]:
    """Compute massive neutrino energy density and pressure at scale factor a.

    cf. CLASS background.c:1600, background_ncdm_momenta()

    rho_ncdm = factor * (1+z)^4 * Σ_i w_i * epsilon_i
    P_ncdm   = factor * (1+z)^4 * Σ_i w_i * q_i^2 / (3 * epsilon_i)

    where epsilon_i = sqrt(q_i^2 + (M * a)^2) and M = m_ncdm / T_ncdm.
    The (1+z)^4 factor and the overall normalization 'factor' are applied
    outside this function.

    Here we compute the UN-normalized integrals (without factor or (1+z)^4),
    i.e., just Σ w_i * epsilon and Σ w_i * q^2/(3*epsilon).

    Args:
        q: quadrature nodes (comoving momenta)
        w: quadrature weights (including q^2 * f_FD factor)
        M: dimensionless mass m_ncdm / T_ncdm
        a: scale factor

    Returns:
        (rho_unnorm, P_unnorm): unnormalized density and pressure integrals
    """
    # epsilon = sqrt(q^2 + (M/T * T * a)^2) but in CLASS convention
    # M = m / T_ncdm in units where q is dimensionless
    # At redshift z: epsilon = sqrt(q^2 + M^2 * a^2)
    # Wait: CLASS has epsilon = sqrt(q^2 + M^2/(1+z)^2) = sqrt(q^2 + M^2 * a^2)
    # cf. background.c:1638: epsilon = sqrt(q2+M*M/(1.+z)/(1.+z))
    epsilon = jnp.sqrt(q**2 + (M * a) ** 2)

    rho_unnorm = jnp.dot(w, epsilon)
    P_unnorm = jnp.dot(w, q**2 / (3.0 * epsilon))
    # Pseudo-pressure for ncdm sound speed: Σ w_i * q^4 / (3 * epsilon^3)
    # cf. CLASS background.c: background_ncdm_momenta() pseudo_p term
    pseudo_p_unnorm = jnp.dot(w, q**4 / (3.0 * epsilon**3))

    return rho_unnorm, P_unnorm, pseudo_p_unnorm


def _pretabulate_ncdm(
    params: CosmoParams,
    prec: PrecisionParams,
    loga_grid: Float[Array, "N"],
    H0: float,
) -> tuple[CubicSpline, CubicSpline]:
    """Pre-tabulate massive neutrino density and pressure as splines.

    Following DISCO-EB pattern: compute on a fine log(a) grid, fit splines.
    This avoids re-doing quadrature at every ODE step.

    Returns splines of log(rho_ncdm) and log(P_ncdm) vs log(a),
    in CLASS units (so that rho_ncdm [Mpc^-2] can be used directly in Friedmann).
    """
    q, w = _gauss_laguerre_nodes_weights(prec.ncdm_q_size)

    # Dimensionless mass M = m_ncdm [eV] / (k_B * T_ncdm [K])
    # T_ncdm = T_ncdm_over_T_cmb * T_cmb
    T_ncdm_K = params.T_ncdm_over_T_cmb * params.T_cmb
    # Convert T_ncdm to eV: k_B * T [K] / eV
    T_ncdm_eV = const.k_B_SI * T_ncdm_K / const.eV_SI
    M = params.m_ncdm / params.N_ncdm / T_ncdm_eV  # mass per species in units of T_ncdm

    # Normalization factor to convert dimensionless integral to CLASS density units.
    #
    # CLASS background_ncdm_init() computes:
    #   factor_ncdm = deg * 4*pi / (2*pi)^3 * (kB * T_ncdm / (hbar * c))^3
    # This is the number density prefactor [1/m^3] when multiplied by T_ncdm^3.
    # The energy density adds another power of (kB * T_ncdm) from epsilon.
    #
    # Then background_ncdm_momenta() computes:
    #   rho = factor * (1+z)^4 * sum_i w_i * q_i^2 * epsilon_i
    # where the w_i already include f_FD(q_i).
    #
    # But our w_i already include q^2 * f_FD, so:
    #   rho_unnorm = sum_i w_i * epsilon_i = integral of q^2 * f_FD * epsilon dq
    #
    # The physical energy density [J/m^3] is:
    #   rho_phys = deg * (kB * T)^4 / (2*pi^2 * (hbar*c)^3) * (1+z)^4 * rho_unnorm
    #
    # Note: deg_ncdm = 1 means ONE FAMILY (particle + antiparticle) = g* = 2.
    # The integral ∫ q^2 f_FD dq counts ONE helicity state. For one family,
    # g* = 2 (two helicity states for Dirac, or particle+antiparticle).
    # CLASS uses: deg_ncdm * 4*pi/(2*pi)^3 = deg/(2*pi^2), and the default
    # deg_ncdm = 1 gives the correct result for one Dirac neutrino family.
    # The factor 4*pi/(2*pi)^3 = 1/(2*pi^2) already accounts for the angular
    # integration of d^3p = 4*pi*p^2*dp over the momentum space volume (2*pi*hbar)^3.
    #
    # To convert to CLASS internal units [Mpc^-2]:
    #   rho_class = (8*pi*G / 3*c^2) * rho_phys * (Mpc_over_m / c)^2
    #
    # Combined: prefactor_per_species [Mpc^-2] =
    #   deg * (kB*T)^4 / (2*pi^2 * hbar^3 * c^3) * (8*pi*G/3) * (Mpc/c)^2
    # multiplied by N_ncdm for total neutrino contribution.

    # The factor of 2 accounts for particle + antiparticle (one family).
    # CLASS's deg_ncdm = 1 means one family (g* = 2). Our integral
    # ∫ q^2 f_FD epsilon dq counts one helicity state. We need 2x for one family.
    # cf. CLASS: factor = deg * 4pi/(2pi)^3 * (kB T/hbar/c)^3
    #          = deg / (2pi^2) * (kB T/hbar/c)^3
    # The factor 4pi/(2pi)^3 = 1/(2pi^2) gives the phase space for one spin.
    # For g* = 2*deg, the total is 2*deg/(2pi^2).
    hbar = const.h_P_SI / (2.0 * math.pi)
    prefactor = (
        2.0  # particle + antiparticle (one Dirac family = g* = 2)
        * params.deg_ncdm
        * params.N_ncdm
        * (const.k_B_SI * T_ncdm_K) ** 4
        / (2.0 * math.pi**2 * hbar**3 * const.c_SI**3)
        * (8.0 * math.pi * const.G_SI / 3.0)
        * const.Mpc_over_m**2
        / const.c_SI**4
    )

    a_grid = jnp.exp(loga_grid)

    def compute_log_at_a(a):
        # Compute the unnormalized integrals (these are O(1) for any a)
        rho_unnorm, P_unnorm, pseudo_p_unnorm = _ncdm_momenta(q, w, M, a)
        # The full density is: prefactor * (1+z)^4 * rho_unnorm = prefactor / a^4 * rho_unnorm
        # Work in log space to avoid overflow at small a:
        log_rho = jnp.log(prefactor) - 4.0 * jnp.log(a) + jnp.log(rho_unnorm)
        log_P = jnp.log(prefactor) - 4.0 * jnp.log(a) + jnp.log(P_unnorm)
        log_pseudo_p = jnp.log(prefactor) - 4.0 * jnp.log(a) + jnp.log(pseudo_p_unnorm)
        return log_rho, log_P, log_pseudo_p

    log_rho_grid, log_P_grid, log_pseudo_p_grid = jax.vmap(compute_log_at_a)(a_grid)

    log_rho_spline = CubicSpline(loga_grid, log_rho_grid)
    log_P_spline = CubicSpline(loga_grid, log_P_grid)
    log_pseudo_p_spline = CubicSpline(loga_grid, log_pseudo_p_grid)

    return log_rho_spline, log_P_spline, log_pseudo_p_spline


# ---------------------------------------------------------------------------
# Dark energy equation of state
# ---------------------------------------------------------------------------

def _w_fld(a: float, w0: float, wa: float) -> float:
    """CPL dark energy equation of state: w(a) = w0 + wa * (1 - a).

    cf. CLASS background.c: background_w_fld()
    """
    return w0 + wa * (1.0 - a)


# ---------------------------------------------------------------------------
# Background functions: compute all densities and H from scale factor
# ---------------------------------------------------------------------------

def _background_functions(
    a: float,
    params: CosmoParams,
    H0: float,
    Omega_g: float,
    Omega_b: float,
    Omega_cdm: float,
    Omega_ur: float,
    Omega_lambda: float,
    log_rho_ncdm_spline: CubicSpline,
    rho_fld: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Compute background quantities at scale factor a.

    cf. CLASS background.c:371, background_functions()

    Returns: (H, rho_g, rho_b, rho_cdm, rho_ur, rho_ncdm, P_ncdm, rho_de, rho_lambda)
    All in CLASS units [Mpc^-2].
    """
    H0_sq = H0**2

    # cf. background.c:425-429
    rho_g = Omega_g * H0_sq / a**4
    # cf. background.c:432-435
    rho_b = Omega_b * H0_sq / a**3
    # cf. background.c:438-443
    rho_cdm = Omega_cdm * H0_sq / a**3
    # cf. background.c:558-564
    rho_ur = Omega_ur * H0_sq / a**4
    # cf. background.c:533-536
    rho_lambda = Omega_lambda * H0_sq

    # Massive neutrinos from pre-tabulated spline
    # cf. background.c:493-530
    loga = jnp.log(a)
    rho_ncdm = jnp.exp(log_rho_ncdm_spline.evaluate(loga))

    # Dark energy (fluid): passed from ODE integration state
    rho_de = rho_fld

    # Total density
    # cf. background.c:575-579
    rho_tot = rho_g + rho_b + rho_cdm + rho_ur + rho_ncdm + rho_lambda + rho_de

    # Hubble rate: H = sqrt(rho_tot - K/a^2)
    # For flat universe (K=0): H = sqrt(rho_tot)
    H = jnp.sqrt(rho_tot)

    return H, rho_g, rho_b, rho_cdm, rho_ur, rho_ncdm, 0.0, rho_de, rho_lambda


# ---------------------------------------------------------------------------
# Background ODE right-hand side
# ---------------------------------------------------------------------------

def _background_rhs(loga, y, args):
    """Background ODE right-hand side, integrated in log(a).

    cf. CLASS background.c:2589, background_derivs()

    State vector y = [tau, t, rs, D, D', rho_fld]
    (rho_fld only if has_fld = True, but we always include it for fixed shapes)

    args is a tuple: (params, H0, Omega_g, Omega_b, Omega_cdm, Omega_ur,
                      Omega_lambda, log_rho_ncdm_spline, w0, wa)
    All elements must be valid JAX pytree leaves.

    Derivatives:
        d(tau)/d(loga) = 1/(a*H)         [conformal time]
        d(t)/d(loga)   = 1/H             [proper time]
        d(rs)/d(loga)  = c_s/(a*H)       [sound horizon]
        d(D)/d(loga)   = D'/(a*H)        [growth factor]
        d(D')/d(loga)  = -D' + 1.5*a*rho_M*D/H  [growth equation]
        d(rho_fld)/d(loga) = -3*(1+w)*rho_fld   [dark energy conservation]
    """
    (H0, Omega_g, Omega_b, Omega_cdm, Omega_ur,
     Omega_lambda, log_rho_ncdm_spline, w0, wa) = args

    a = jnp.exp(loga)

    # Unpack state
    rho_fld = y[5]

    # Compute H and all densities
    H, rho_g, rho_b, rho_cdm, rho_ur, rho_ncdm, _, rho_de, rho_lambda_val = _background_functions(
        a, None, H0, Omega_g, Omega_b, Omega_cdm, Omega_ur,
        Omega_lambda, log_rho_ncdm_spline, rho_fld,
    )

    # cf. background.c:2623
    dtau_dloga = 1.0 / (a * H)
    # cf. background.c:2626
    dt_dloga = 1.0 / H

    # Sound horizon: cf. background.c:2633
    # c_s = 1/sqrt(3*(1 + 3*rho_b/(4*rho_g)))
    R = 3.0 * rho_b / (4.0 * rho_g)
    cs = 1.0 / jnp.sqrt(3.0 * (1.0 + R))
    drs_dloga = cs / (a * H)

    # Growth factor: cf. background.c:2635-2646
    rho_M = rho_b + rho_cdm
    D = y[3]
    D_prime = y[4]
    dD_dloga = D_prime / (a * H)
    dDprime_dloga = -D_prime + 1.5 * a * rho_M * D / H

    # Dark energy: cf. background.c:2658-2661
    w = _w_fld(a, w0, wa)
    drho_fld_dloga = -3.0 * (1.0 + w) * rho_fld

    dy = jnp.array([dtau_dloga, dt_dloga, drs_dloga, dD_dloga, dDprime_dloga, drho_fld_dloga])
    return dy


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def background_solve(
    params: CosmoParams,
    prec: PrecisionParams = PrecisionParams(),
) -> BackgroundResult:
    """Solve the background cosmology.

    Integrates the Friedmann equation from a_ini to a=1, building interpolation
    tables for all background quantities.

    Args:
        params: cosmological parameters (JAX-traced)
        prec: precision parameters (static)

    Returns:
        BackgroundResult with all background spline tables
    """
    # --- Compute derived quantities ---
    H0 = _H0_from_h(params.h)
    Omega_g = _compute_omega_g(params.T_cmb, H0)
    Omega_b = params.omega_b / params.h**2
    Omega_cdm = params.omega_cdm / params.h**2
    Omega_ur = _compute_omega_ur(params.N_ur, Omega_g)

    # --- Pre-tabulate neutrino integrals ---
    loga_min = jnp.log(prec.bg_a_ini_default)
    loga_max = 0.0  # log(1) = 0
    ncdm_loga_grid = jnp.linspace(loga_min, loga_max, prec.ncdm_bg_n_points)
    log_rho_ncdm_spline, log_P_ncdm_spline, log_pseudo_p_ncdm_spline = _pretabulate_ncdm(
        params, prec, ncdm_loga_grid, H0
    )

    # Neutrino density today (a=1)
    Omega_ncdm = jnp.exp(log_rho_ncdm_spline.evaluate(0.0)) / H0**2

    # Dark energy: determine if we need fluid (w != -1) or just Lambda
    has_fld = (params.w0 != -1.0) | (params.wa != 0.0)

    # Closure: Omega_Lambda or Omega_de fills the rest
    Omega_lambda = jnp.where(
        has_fld,
        0.0,
        1.0 - Omega_g - Omega_b - Omega_cdm - Omega_ur - Omega_ncdm - params.Omega_k,
    )
    Omega_de = jnp.where(
        has_fld,
        1.0 - Omega_g - Omega_b - Omega_cdm - Omega_ur - Omega_ncdm - params.Omega_k,
        0.0,
    )

    # --- Set up ODE ---
    loga_grid = jnp.linspace(loga_min, loga_max, prec.bg_n_points)
    a_ini = jnp.exp(loga_min)

    # Initial conditions at a_ini (deep radiation domination)
    # cf. CLASS background.c:2169, background_initial_conditions()
    # At early times: H ≈ sqrt(rho_r) / a^2, tau ≈ a / (a_ini * H_ini)
    rho_r_ini = (Omega_g + Omega_ur) * H0**2 / a_ini**4
    rho_ncdm_ini = jnp.exp(log_rho_ncdm_spline.evaluate(loga_min))
    rho_tot_ini = rho_r_ini + rho_ncdm_ini  # matter negligible at a ~ 1e-14
    H_ini = jnp.sqrt(rho_tot_ini)

    tau_ini = 1.0 / (a_ini * H_ini)
    t_ini = 1.0 / (2.0 * H_ini)  # t ≈ 1/(2H) in radiation domination
    rs_ini = tau_ini / jnp.sqrt(3.0)  # c_s ≈ 1/sqrt(3) at early times

    # Growth factor: D ∝ a^2 in radiation domination, D' = 2*a*H*D
    D_ini = a_ini**2
    D_prime_ini = 2.0 * a_ini * H_ini * D_ini

    # DE initial density
    rho_fld_ini = jnp.where(has_fld, Omega_de * H0**2, 0.0)
    # For w0wa, at very early times: rho_de ∝ a^{-3(1+w0+wa)} * exp(-3*wa*(1-a))
    # At a << 1: rho_de ≈ rho_de_0 * a^{-3(1+w0+wa)} * exp(3*wa) ≈ tiny
    # We just start from rho_de_0 and integrate forward (like CLASS)
    # Actually CLASS integrates from a_ini forward, starting with the analytical solution
    # rho_fld(a) = Omega_fld * H0^2 * a^{-3(1+w0+wa)} * exp(-3*wa*(1-a))
    w_ini = _w_fld(a_ini, params.w0, params.wa)
    rho_fld_ini = jnp.where(
        has_fld,
        Omega_de * H0**2 * a_ini ** (-3.0 * (1.0 + params.w0 + params.wa))
        * jnp.exp(-3.0 * params.wa * (1.0 - a_ini)),
        0.0,
    )

    y0 = jnp.array([tau_ini, t_ini, rs_ini, D_ini, D_prime_ini, rho_fld_ini])

    # --- Integrate ---
    # ODE args as a plain tuple (valid JAX pytree)
    ode_args = (H0, Omega_g, Omega_b, Omega_cdm, Omega_ur, Omega_lambda,
                log_rho_ncdm_spline, params.w0, params.wa)

    # Initial step size: small fraction of the log(a) range
    dt0 = (loga_max - loga_min) / prec.bg_n_points

    sol = solve_nonstiff(
        rhs_fn=_background_rhs,
        t0=loga_min,
        t1=loga_max,
        y0=y0,
        saveat=diffrax.SaveAt(ts=loga_grid),
        args=ode_args,
        rtol=prec.bg_tol,
        atol=prec.bg_tol * 1e-3,
        max_steps=262144,  # background spans 32 decades in a, needs many steps
    )

    # --- Extract solution ---
    tau_grid = sol.ys[:, 0]
    t_grid = sol.ys[:, 1]
    rs_grid = sol.ys[:, 2]
    D_grid = sol.ys[:, 3]
    D_prime_grid = sol.ys[:, 4]
    rho_fld_grid = sol.ys[:, 5]

    # Recompute all densities and H at each grid point
    a_grid = jnp.exp(loga_grid)

    def compute_bg_at_a(a, rho_fld_val):
        H, rho_g, rho_b, rho_cdm, rho_ur, rho_ncdm, _, rho_de, rho_lam = _background_functions(
            a, None, H0, Omega_g, Omega_b, Omega_cdm, Omega_ur, Omega_lambda,
            log_rho_ncdm_spline, rho_fld_val,
        )
        return H, rho_g, rho_b, rho_cdm, rho_ur, rho_ncdm, rho_de, rho_lam

    # Vectorize over grid
    bg_quantities = jax.vmap(compute_bg_at_a)(a_grid, rho_fld_grid)
    H_grid, rho_g_grid, rho_b_grid, rho_cdm_grid, rho_ur_grid, rho_ncdm_grid, rho_de_grid, rho_lambda_grid = bg_quantities

    # Growth rate f = d ln D / d ln a = D' / (a * H * D)
    f_grid = D_prime_grid / (a_grid * H_grid * D_grid)

    # --- Build splines ---
    tau_of_loga = CubicSpline(loga_grid, tau_grid)
    loga_of_tau = CubicSpline(tau_grid, loga_grid)
    H_of_loga = CubicSpline(loga_grid, H_grid)
    rho_g_of_loga = CubicSpline(loga_grid, rho_g_grid)
    rho_b_of_loga = CubicSpline(loga_grid, rho_b_grid)
    rho_cdm_of_loga = CubicSpline(loga_grid, rho_cdm_grid)
    rho_ur_of_loga = CubicSpline(loga_grid, rho_ur_grid)
    rho_ncdm_of_loga = CubicSpline(loga_grid, rho_ncdm_grid)
    P_ncdm_at_grid = jnp.exp(log_P_ncdm_spline.evaluate(loga_grid))
    p_ncdm_of_loga = CubicSpline(loga_grid, P_ncdm_at_grid)
    # ncdm equation of state and adiabatic sound speed
    w_ncdm_grid = P_ncdm_at_grid / jnp.maximum(rho_ncdm_grid, 1e-100)
    pseudo_p_ncdm_grid = jnp.exp(log_pseudo_p_ncdm_spline.evaluate(loga_grid))
    # c_a² = w/(3*(1+w)) * (5 - pseudo_p/p)
    # cf. CLASS perturbations.c:9513
    ca2_ncdm_grid = w_ncdm_grid / (3.0 * (1.0 + w_ncdm_grid)) * (
        5.0 - pseudo_p_ncdm_grid / jnp.maximum(P_ncdm_at_grid, 1e-100))
    w_ncdm_of_loga = CubicSpline(loga_grid, w_ncdm_grid)
    ca2_ncdm_of_loga = CubicSpline(loga_grid, ca2_ncdm_grid)
    rho_de_of_loga = CubicSpline(loga_grid, rho_de_grid)
    rho_lambda_of_loga = CubicSpline(loga_grid, rho_lambda_grid)
    rs_of_loga = CubicSpline(loga_grid, rs_grid)
    D_of_loga = CubicSpline(loga_grid, D_grid)
    f_of_loga = CubicSpline(loga_grid, f_grid)

    # --- Derived quantities ---
    conformal_age = tau_grid[-1]
    # t_grid is in Mpc (c=1 units). CLASS's Gyr_over_Mpc = 306.6 means 1 Gyr = 306.6 Mpc.
    # So age_Gyr = t_Mpc / Gyr_over_Mpc
    # cf. CLASS output.c where age = pba->age * _Gyr_over_Mpc_ but pba->age is already in Gyr
    # Actually in CLASS, proper time is stored in Mpc and converted: age = t_Mpc / _Gyr_over_Mpc_
    # Wait, let me check CLASS: background_output_budget has pba->age in Gyr
    # and conformal_age is tau in Mpc. The conversion: 1 Mpc = 1/Gyr_over_Mpc Gyr.
    # No: Gyr_over_Mpc = 3.066e2 means 1 Mpc of TIME = Gyr_over_Mpc Gyr? No...
    # Let me just compute directly: 1 Mpc (time) = Mpc_over_m / c_SI seconds
    age_Gyr = t_grid[-1] * const.Mpc_over_m / const.c_SI / (365.25 * 24 * 3600 * 1e9)

    # Matter-radiation equality: rho_m = rho_r
    # cf. CLASS background.c:523-528: split ncdm into relativistic and non-relativistic
    # rho_r += 3 * P_ncdm (relativistic contribution)
    # rho_m += rho_ncdm - 3 * P_ncdm (non-relativistic contribution)
    rho_m_grid = rho_b_grid + rho_cdm_grid + (rho_ncdm_grid - 3.0 * P_ncdm_at_grid)
    rho_r_grid = rho_g_grid + rho_ur_grid + 3.0 * P_ncdm_at_grid
    # ncdm relativistic contribution: 3*P_ncdm
    # Find where rho_m crosses rho_r
    diff = rho_m_grid - rho_r_grid
    # Find the crossing index (from negative to positive)
    cross_idx = jnp.argmax(diff > 0)
    # Linear interpolation for z_eq
    loga_eq = loga_grid[cross_idx - 1] + (
        -diff[cross_idx - 1]
        / (diff[cross_idx] - diff[cross_idx - 1])
        * (loga_grid[cross_idx] - loga_grid[cross_idx - 1])
    )
    z_eq = 1.0 / jnp.exp(loga_eq) - 1.0
    tau_eq = tau_of_loga.evaluate(loga_eq)

    return BackgroundResult(
        loga_table=loga_grid,
        tau_table=tau_grid,
        tau_of_loga=tau_of_loga,
        loga_of_tau=loga_of_tau,
        H_of_loga=H_of_loga,
        rho_g_of_loga=rho_g_of_loga,
        rho_b_of_loga=rho_b_of_loga,
        rho_cdm_of_loga=rho_cdm_of_loga,
        rho_ur_of_loga=rho_ur_of_loga,
        rho_ncdm_of_loga=rho_ncdm_of_loga,
        p_ncdm_of_loga=p_ncdm_of_loga,
        w_ncdm_of_loga=w_ncdm_of_loga,
        ca2_ncdm_of_loga=ca2_ncdm_of_loga,
        rho_de_of_loga=rho_de_of_loga,
        rho_lambda_of_loga=rho_lambda_of_loga,
        rs_of_loga=rs_of_loga,
        D_of_loga=D_of_loga,
        f_of_loga=f_of_loga,
        conformal_age=conformal_age,
        age_Gyr=age_Gyr,
        z_eq=z_eq,
        tau_eq=tau_eq,
        H0=H0,
        Omega_g=Omega_g,
        Omega_b=Omega_b,
        Omega_cdm=Omega_cdm,
        Omega_ur=Omega_ur,
        Omega_ncdm=Omega_ncdm,
        Omega_de=Omega_de,
        Omega_lambda=Omega_lambda,
    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def H_of_z(bg: BackgroundResult, z: float) -> float:
    """Hubble rate at redshift z in Mpc^-1."""
    loga = jnp.log(1.0 / (1.0 + z))
    return bg.H_of_loga.evaluate(loga)


def tau_of_z(bg: BackgroundResult, z: float) -> float:
    """Conformal time at redshift z in Mpc."""
    loga = jnp.log(1.0 / (1.0 + z))
    return bg.tau_of_loga.evaluate(loga)


def comoving_distance(bg: BackgroundResult, z: float) -> float:
    """Comoving distance to redshift z in Mpc.

    chi(z) = tau_0 - tau(z)  (for flat universe)
    """
    return bg.conformal_age - tau_of_z(bg, z)


def angular_diameter_distance(bg: BackgroundResult, z: float) -> float:
    """Angular diameter distance to redshift z in Mpc.

    D_A(z) = chi(z) / (1 + z)  (for flat universe)
    """
    return comoving_distance(bg, z) / (1.0 + z)


def luminosity_distance(bg: BackgroundResult, z: float) -> float:
    """Luminosity distance to redshift z in Mpc.

    D_L(z) = chi(z) * (1 + z)  (for flat universe)
    """
    return comoving_distance(bg, z) * (1.0 + z)
