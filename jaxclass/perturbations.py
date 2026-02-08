"""Perturbations module for jaxCLASS.

Integrates the linearized Einstein-Boltzmann equations for each Fourier mode k,
producing source functions S(k,τ) for temperature, E-polarization, and lensing.

Uses approximation-free integration: the full Boltzmann hierarchy is solved at
all times with a stiff implicit solver. No TCA/RSA/UFA switching.

Key function:
    perturbations_solve(params, prec, bg, th) -> PerturbationResult

The state vector for each k-mode (synchronous gauge) contains:
    - Metric: η, h' (or equivalently h_prime_over_6)
    - CDM: δ_cdm
    - Baryons: δ_b, θ_b
    - Photons: F_γ,0 through F_γ,l_max (monopole through l_max)
    - Polarization: G_γ,0 through G_γ,l_max
    - Massless neutrinos: F_ur,0 through F_ur,l_max
    - (Optional) Massive neutrinos: Ψ_l(q_i) for each q bin and multipole
    - (Optional) Dark energy: δ_de, θ_de

References:
    CLASS source: perturbations.c (10528 lines)
    Ma & Bertschinger (1995) ApJ 455, 7
    DISCO-EB: perturbations.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxclass import constants as const
from jaxclass.background import BackgroundResult
from jaxclass.interpolation import CubicSpline
from jaxclass.params import CosmoParams, PrecisionParams
from jaxclass.thermodynamics import ThermoResult


# ---------------------------------------------------------------------------
# State vector index layout
# ---------------------------------------------------------------------------

def _build_indices(l_max_g: int, l_max_pol: int, l_max_ur: int):
    """Build index mapping for the perturbation state vector.

    Returns a dict mapping variable names to indices.
    """
    idx = {}
    i = 0

    # Metric perturbations (synchronous gauge)
    idx['eta'] = i; i += 1
    idx['h_prime'] = i; i += 1  # h' = dh/dτ (trace part)

    # CDM
    idx['delta_cdm'] = i; i += 1
    # theta_cdm = 0 by synchronous gauge convention

    # Baryons
    idx['delta_b'] = i; i += 1
    idx['theta_b'] = i; i += 1

    # Photons (temperature hierarchy)
    idx['F_g_start'] = i
    for l in range(l_max_g + 1):
        idx[f'F_g_{l}'] = i; i += 1
    idx['F_g_end'] = i

    # Photon polarization hierarchy
    idx['G_g_start'] = i
    for l in range(l_max_pol + 1):
        idx[f'G_g_{l}'] = i; i += 1
    idx['G_g_end'] = i

    # Massless neutrinos (ultra-relativistic)
    idx['F_ur_start'] = i
    for l in range(l_max_ur + 1):
        idx[f'F_ur_{l}'] = i; i += 1
    idx['F_ur_end'] = i

    idx['n_eq'] = i
    return idx


# ---------------------------------------------------------------------------
# PerturbationResult
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PerturbationResult:
    """Output of the perturbation module.

    Contains source function tables on a (k, τ) grid.
    """
    k_grid: Float[Array, "Nk"]
    tau_grid: Float[Array, "Ntau"]

    # Source functions: shape (Nk, Ntau)
    source_T0: Float[Array, "Nk Ntau"]   # Temperature monopole source (SW + ISW + Doppler)
    source_T1: Float[Array, "Nk Ntau"]   # Temperature dipole source (ISW dipole)
    source_T2: Float[Array, "Nk Ntau"]   # Temperature quadrupole source (g*Pi)
    source_E: Float[Array, "Nk Ntau"]    # E-polarization source
    source_lens: Float[Array, "Nk Ntau"]  # Lensing potential source
    delta_m: Float[Array, "Nk Ntau"]     # Total matter density contrast

    # Decomposed T0 subterms for diagnostics
    source_SW: Float[Array, "Nk Ntau"]       # Sachs-Wolfe: g*(delta_g/4 + alpha')
    source_ISW_vis: Float[Array, "Nk Ntau"]  # ISW visibility: g*(eta - alpha' - 2*H*alpha)
    source_ISW_fs: Float[Array, "Nk Ntau"]   # ISW free-streaming: exp(-kappa)*2*Phi'
    source_Doppler: Float[Array, "Nk Ntau"]  # Doppler IBP: (g*theta_b' + g'*theta_b)/k^2

    # Non-IBP Doppler sources (for alternative TT computation)
    source_Doppler_nonIBP: Float[Array, "Nk Ntau"]  # g*theta_b_shifted/k (uses j_l' radial)
    source_T0_noDopp: Float[Array, "Nk Ntau"]       # SW + ISW (no Doppler)

    def tree_flatten(self):
        return [
            self.k_grid, self.tau_grid,
            self.source_T0, self.source_T1, self.source_T2,
            self.source_E, self.source_lens, self.delta_m,
            self.source_SW, self.source_ISW_vis, self.source_ISW_fs, self.source_Doppler,
            self.source_Doppler_nonIBP, self.source_T0_noDopp,
        ], None

    @classmethod
    def tree_unflatten(cls, aux, fields):
        return cls(*fields)


# ---------------------------------------------------------------------------
# Adiabatic initial conditions (Ma & Bertschinger 1995)
# ---------------------------------------------------------------------------

def _adiabatic_ic(k, tau_ini, bg, params, idx, n_eq):
    """Compute adiabatic initial conditions at conformal time tau_ini.

    Deep in the radiation era (kτ << 1), all modes are super-horizon.
    The adiabatic mode satisfies: all species share the same curvature perturbation.

    cf. Ma & Bertschinger (1995), DISCO-EB perturbations.py
    cf. CLASS perturbations.c: perturbations_initial_conditions()

    Normalization: we set the initial curvature perturbation to 1.
    """
    y0 = jnp.zeros(n_eq)

    # Get background quantities at tau_ini
    loga_ini = bg.loga_of_tau.evaluate(tau_ini)
    a_ini = jnp.exp(loga_ini)
    H_ini = bg.H_of_loga.evaluate(loga_ini)
    a_prime_over_a = a_ini * H_ini

    # Fractional densities at tau_ini
    rho_g = bg.rho_g_of_loga.evaluate(loga_ini)
    rho_b = bg.rho_b_of_loga.evaluate(loga_ini)
    rho_cdm = bg.rho_cdm_of_loga.evaluate(loga_ini)
    rho_ur = bg.rho_ur_of_loga.evaluate(loga_ini)
    rho_ncdm = bg.rho_ncdm_of_loga.evaluate(loga_ini)

    # Radiation and matter densities
    rho_r = rho_g + rho_ur + rho_ncdm  # all radiation (ncdm relativistic at early times)
    rho_m = rho_b + rho_cdm

    # Neutrino fraction of radiation
    rho_nu = rho_ur + rho_ncdm
    fracnu = rho_nu / rho_r
    fracg = rho_g / rho_r

    # omega parameter (matter/radiation ratio parameter)
    # cf. CLASS perturbations.c:5398
    om = a_ini * rho_m / jnp.sqrt(rho_r)

    # Initial curvature perturbation normalization
    # cf. CLASS: ppr->curvature_ini = 1 (default)
    curvature_ini = 1.0
    s2_squared = 1.0  # flat space

    ktau_two = k**2 * tau_ini**2
    ktau_three = k * tau_ini * ktau_two

    # --- CLASS adiabatic initial conditions (perturbations.c:5432-5511) ---
    # These are valid at leading order in (kτ) and (om*τ)

    # Photon density: cf. CLASS line 5432
    delta_g = -ktau_two / 3.0 * (1.0 - om * tau_ini / 5.0) * curvature_ini * s2_squared

    # Photon velocity: cf. CLASS line 5436
    theta_g = -k * ktau_three / 36.0 * (
        1.0 - 3.0 * (1.0 + 5.0 * rho_b / rho_m - fracnu) / 20.0 / (1.0 - fracnu) * om * tau_ini
    ) * curvature_ini * s2_squared

    # Baryons: cf. CLASS lines 5440-5441
    delta_b = 3.0 / 4.0 * delta_g
    theta_b = theta_g

    # CDM: cf. CLASS line 5444
    delta_cdm = 3.0 / 4.0 * delta_g

    # Neutrinos: cf. CLASS lines 5496-5503
    delta_ur = delta_g

    theta_ur = -k * ktau_three / 36.0 / (4.0 * fracnu + 15.0) * (
        4.0 * fracnu + 11.0 + 12.0 * s2_squared
        - 3.0 * (8.0 * fracnu**2 + 50.0 * fracnu + 275.0) / 20.0 / (2.0 * fracnu + 15.0) * tau_ini * om
    ) * curvature_ini * s2_squared

    shear_ur = ktau_two / (45.0 + 12.0 * fracnu) * (3.0 * s2_squared - 1.0) * (
        1.0 + (4.0 * fracnu - 5.0) / 4.0 / (2.0 * fracnu + 15.0) * tau_ini * om
    ) * curvature_ini

    # Metric η: cf. CLASS line 5511
    eta = curvature_ini * (
        1.0 - ktau_two / 12.0 / (15.0 + 4.0 * fracnu) * (
            5.0 + 4.0 * s2_squared * fracnu
            - (16.0 * fracnu**2 + 280.0 * fracnu + 325.0) / 10.0 / (2.0 * fracnu + 15.0) * tau_ini * om
        )
    )

    # h' from h'' + aH h' = source → at early times h' ≈ 2k²τ*η (leading order)
    # cf. CLASS: h is obtained from integration, h' from the trace Einstein eq
    h_prime = 2.0 * k**2 * tau_ini * eta

    # Set state vector
    y0 = y0.at[idx['eta']].set(eta)
    y0 = y0.at[idx['h_prime']].set(h_prime)
    y0 = y0.at[idx['delta_cdm']].set(delta_cdm)
    y0 = y0.at[idx['delta_b']].set(delta_b)
    y0 = y0.at[idx['theta_b']].set(theta_b)

    # Photon hierarchy
    y0 = y0.at[idx['F_g_0']].set(delta_g)
    y0 = y0.at[idx['F_g_1']].set(4.0 * theta_g / (3.0 * k))

    # Neutrino hierarchy
    y0 = y0.at[idx['F_ur_0']].set(delta_ur)
    y0 = y0.at[idx['F_ur_1']].set(4.0 * theta_ur / (3.0 * k))
    y0 = y0.at[idx['F_ur_2']].set(2.0 * shear_ur)  # F_2 = 2σ

    # Polarization starts at 0 (correct for adiabatic IC)

    return y0


# ---------------------------------------------------------------------------
# Shared TCA helpers (used by both RHS and source extraction)
# ---------------------------------------------------------------------------

def _compute_tca_criterion(kappa_dot, a_prime_over_a, k):
    """Compute smooth TCA switching criterion matching CLASS dual criteria.

    CLASS (perturbations.c:6178-6179) uses:
        tca_on when tau_c/tau_h < 0.005 AND tau_c/tau_k < 0.01
    Returns is_tca: ~1 when tightly coupled, ~0 when free-streaming.
    """
    tau_c = 1.0 / jnp.maximum(kappa_dot, 1e-30)
    tau_h = 1.0 / a_prime_over_a
    tau_k = 1.0 / k
    crit1 = tau_c / tau_h  # must be < 0.005
    crit2 = tau_c / tau_k  # must be < 0.01
    tca_ratio = jnp.maximum(crit1 / 0.005, crit2 / 0.01)
    _TCA_WIDTH = 5.0
    return jax.nn.sigmoid(-_TCA_WIDTH * jnp.log(jnp.maximum(tca_ratio, 1e-30))), tau_c


def _compute_theta_b_prime_blended(
    theta_b, delta_b, theta_g, delta_g, F_g_2, G_g_0, G_g_2,
    a_prime_over_a, cs2, k, k2, kappa_dot,
    rho_g, rho_b, bg, loga, a, h_prime, eta_prime, is_tca, tau_c,
):
    """Compute theta_b' with proper TCA/full blending, matching the ODE RHS.

    This must be called from both _perturbation_rhs and _extract_sources
    to ensure the Doppler IBP source uses the same theta_b' as the evolution.

    cf. CLASS perturbations.c:9100-9103 (TCA), Ma & Bertschinger eq. 31 (full)
    """
    R = 4.0 * rho_g / (3.0 * rho_b)
    metric_continuity = h_prime / 2.0
    metric_shear = (h_prime + 6.0 * eta_prime) / 2.0

    # --- TCA theta_b' ---
    # Shear: cf. CLASS perturbations.c:10193
    tca_shear_g = 16.0 / 45.0 * tau_c * (theta_g + metric_shear)

    # Slip: cf. CLASS perturbations.c:10168 (first_order_CLASS)
    F_tca = tau_c / (1.0 + R)
    dtau_c_over_tau_c = 2.0 * a_prime_over_a
    dH_dloga = bg.H_of_loga.derivative(loga)
    a_primeprime_over_a = a_prime_over_a * (a_prime_over_a + a * dH_dloga)
    tca_slip = (dtau_c_over_tau_c - 2.0 * a_prime_over_a / (1.0 + R)) * (theta_b - theta_g) \
        + F_tca * (
            -a_primeprime_over_a * theta_b
            + k2 * (
                -a_prime_over_a * delta_g / 2.0
                + cs2 * (-theta_b - metric_continuity)
                - (1.0/3.0) * (-theta_g - metric_continuity)
            )
        )

    theta_b_tca = (-a_prime_over_a * theta_b
                   + k2 * (cs2 * delta_b + R * (delta_g / 4.0 - tca_shear_g))
                   + R * tca_slip) / (1.0 + R)

    # --- Full (non-TCA) theta_b' ---
    theta_b_full = -a_prime_over_a * theta_b + cs2 * k2 * delta_b + R * kappa_dot * (theta_g - theta_b)

    # Hard switch: TCA or full equations (not blended — blending changes the physics)
    return jnp.where(is_tca > 0.5, theta_b_tca, theta_b_full)


# ---------------------------------------------------------------------------
# Boltzmann hierarchy RHS (synchronous gauge)
# ---------------------------------------------------------------------------

def _perturbation_rhs(tau, y, args):
    """Right-hand side of the perturbation ODE system in synchronous gauge.

    KEY INSIGHT (from CLASS perturbations.c:6529-6675):
    h' is NOT an evolved variable. It is computed at each step from the
    00 Einstein CONSTRAINT equation. Only η is evolved as a metric variable.
    This is critical for getting the correct growth rate.

    Implements Tight Coupling Approximation (TCA) for early times when
    Thomson scattering is fast (κ' >> k). During TCA:
    - Photon-baryon system evolves as a single fluid
    - Photon shear σ_g = (16/45) * τ_c * (θ_g + metric_shear)
    - Higher photon multipoles (l>=3) are damped to zero
    - Photon-baryon slip Δ = θ_b - θ_g expanded to first order in τ_c
    Uses jnp.where for smooth switching (JAX-traceable, no branching).

    State: y = [η, h'(dummy), δ_cdm, δ_b, θ_b, F_g_0..l_max, G_g_0..l_max, F_ur_0..l_max]
    (h' slot exists for compatibility but its evolution equation is not used)

    cf. CLASS perturbations.c: perturbations_derivs() + perturbations_einstein()
    cf. CLASS perturbations.c:9960-10200: perturbations_tca_slip_and_shear()
    cf. Ma & Bertschinger (1995) Eqs. (25)-(56)
    """
    (k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur) = args

    # Background quantities at this τ
    loga = bg.loga_of_tau.evaluate(tau)
    a = jnp.exp(loga)
    H = bg.H_of_loga.evaluate(loga)
    a_prime_over_a = a * H  # conformal Hubble aH
    a2 = a * a
    k2 = k * k

    rho_g = bg.rho_g_of_loga.evaluate(loga)
    rho_b = bg.rho_b_of_loga.evaluate(loga)
    rho_cdm = bg.rho_cdm_of_loga.evaluate(loga)
    rho_ur = bg.rho_ur_of_loga.evaluate(loga)
    rho_ncdm = bg.rho_ncdm_of_loga.evaluate(loga)

    # Thermodynamic quantities
    kappa_dot = th.kappa_dot_of_loga.evaluate(loga)
    cs2 = th.cs2_of_loga.evaluate(loga)

    # Unpack state
    eta = y[idx['eta']]
    delta_cdm = y[idx['delta_cdm']]
    delta_b = y[idx['delta_b']]
    theta_b = y[idx['theta_b']]

    F_g = y[idx['F_g_start']:idx['F_g_end']]
    delta_g = F_g[0]
    theta_g = 3.0 * k * F_g[1] / 4.0

    G_g = y[idx['G_g_start']:idx['G_g_end']]

    F_ur = y[idx['F_ur_start']:idx['F_ur_end']]
    delta_ur = F_ur[0]
    theta_ur = 3.0 * k * F_ur[1] / 4.0

    Pi = F_g[2] + G_g[0] + G_g[2]

    # === TCA CRITERION (shared helper, dual criteria matching CLASS) ===
    is_tca, tau_c = _compute_tca_criterion(kappa_dot, a_prime_over_a, k)

    # === EINSTEIN EQUATIONS (constraint approach, matching CLASS) ===
    # cf. CLASS perturbations.c:6611-6644

    # Total density perturbation δρ = Σ ρ_i δ_i
    # ncdm perturbations are approximated using ur variables (δ_ncdm ≈ δ_ur, θ_ncdm ≈ θ_ur)
    # but with correct background equation of state from p_ncdm(a).
    delta_rho = rho_g * delta_g + rho_b * delta_b + rho_cdm * delta_cdm + rho_ur * delta_ur + rho_ncdm * delta_ur

    # Total (ρ+p)θ
    # ncdm approximated as massless: ρ_ncdm + p_ncdm ≈ 4/3 ρ_ncdm, θ_ncdm ≈ θ_ur
    # TODO: use actual p_ncdm(a) once ncdm perturbation variables are implemented
    rho_plus_p_theta = (4.0/3.0 * rho_g * theta_g + rho_b * theta_b
                        + 4.0/3.0 * rho_ur * theta_ur + 4.0/3.0 * rho_ncdm * theta_ur)

    # h' from 00 Einstein CONSTRAINT (NOT evolved!)
    # cf. CLASS line 6612: h' = (k2*eta + 1.5*a2*delta_rho) / (0.5*a'/a)
    h_prime = (k2 * eta + 1.5 * a2 * delta_rho) / (0.5 * a_prime_over_a)

    # η' from 0i Einstein equation
    # cf. CLASS line 6635: η' = 1.5 * a² * (ρ+p)θ / k²  (flat space)
    eta_prime = 1.5 * a2 * rho_plus_p_theta / k2

    # Total pressure perturbation δp = Σ ρ_i δ_i * w_i (for h'')
    delta_p = rho_g * delta_g / 3.0 + rho_ur * delta_ur / 3.0

    # α = (h' + 6η') / (2k²) -- gauge variable
    alpha = (h_prime + 6.0 * eta_prime) / (2.0 * k2)

    # metric_shear = k²α = (h' + 6η')/2, source for l=2 equations
    # cf. CLASS perturbations.c:8984
    metric_shear = k2 * alpha
    # metric_continuity = h'/2 in synchronous gauge
    metric_continuity = h_prime / 2.0

    # === CDM ===
    # cf. CLASS: δ'_cdm = -h'/2 (θ_cdm = 0 in synchronous gauge)
    delta_cdm_prime = -h_prime / 2.0

    # === TCA PHOTON SHEAR (needed for photon hierarchy l=2 TCA) ===
    # cf. CLASS perturbations.c:10193
    tca_shear_g = 16.0 / 45.0 * tau_c * (theta_g + metric_shear)
    tca_F_g_2 = 2.0 * tca_shear_g

    # === BARYON VELOCITY (TCA/full blended via shared helper) ===
    R = 4.0 * rho_g / (3.0 * rho_b)
    theta_b_prime = _compute_theta_b_prime_blended(
        theta_b, delta_b, theta_g, delta_g, F_g[2], G_g[0], G_g[2],
        a_prime_over_a, cs2, k, k2, kappa_dot,
        rho_g, rho_b, bg, loga, a, h_prime, eta_prime, is_tca, tau_c,
    )

    # === TCA PHOTON VELOCITY ===
    # Need TCA theta_b' for photon velocity computation
    # Recompute TCA shear and theta_b_tca for photon velocity
    theta_b_tca = _compute_theta_b_prime_blended(
        theta_b, delta_b, theta_g, delta_g, F_g[2], G_g[0], G_g[2],
        a_prime_over_a, cs2, k, k2, kappa_dot,
        rho_g, rho_b, bg, loga, a, h_prime, eta_prime,
        jnp.ones_like(is_tca), tau_c,  # force TCA mode
    )
    # cf. CLASS perturbations.c:9204-9206
    theta_g_tca = -(theta_b_tca + a_prime_over_a * theta_b - k2 * cs2 * delta_b) / R \
        + k2 * (0.25 * delta_g - tca_shear_g)
    F1_prime_tca = 4.0 * theta_g_tca / (3.0 * k)

    # === FULL (non-TCA) BARYONS ===
    delta_b_prime = -theta_b - h_prime / 2.0

    # === PHOTON HIERARCHY ===
    dy = jnp.zeros_like(y)

    # l=0 (monopole): δ'_γ = -4/3 θ_γ - 2/3 h'
    # In F_l convention: F'_0 = -k F_1 - 2/3 h'
    # Same in both TCA and full (monopole equation is identical)
    dy = dy.at[idx['F_g_0']].set(-k * F_g[1] - 2.0/3.0 * h_prime)

    # l=1 (dipole): different in TCA vs full
    # Full: F'_1 = k/3 (F_0 - 2F_2) - κ'(F_1 - 4θ_b/(3k))
    # cf. CLASS perturbations.c:9127-9130
    F1_source = -kappa_dot * (F_g[1] - 4.0 * theta_b / (3.0 * k))
    F1_prime_full = k/3.0 * (F_g[0] - 2.0*F_g[2]) + F1_source
    F1_prime = jnp.where(is_tca > 0.5, F1_prime_tca, F1_prime_full)
    dy = dy.at[idx['F_g_1']].set(F1_prime)

    # l=2 to l_max-1
    def photon_hierarchy_step(l, dy_acc):
        Fl_prime = k/(2.0*l+1.0) * (l*F_g[l-1] - (l+1.0)*F_g[jnp.minimum(l+1, l_max_g)]) - kappa_dot*F_g[l]
        # For l=2: add metric shear source 8/15*metric_shear (divided by the F_l normalization)
        # cf. CLASS perturbations.c:9137-9140
        Fl_prime = Fl_prime + jnp.where(l == 2, 8.0/15.0 * metric_shear + kappa_dot * Pi / 10.0, 0.0)

        # TCA for l=2: drive F_g_2 toward tca_F_g_2 = 2*sigma_g^{tca}
        # F'_2^{tca} = (tca_F_g_2 - F_g_2) / tau_c  (relax toward TCA value)
        # For l>=3: drive F_g_l toward zero: F'_l^{tca} = -F_g_l / tau_c
        F2_prime_tca = (tca_F_g_2 - F_g[2]) / tau_c
        Fl_prime_tca = jnp.where(l == 2, F2_prime_tca, -F_g[l] / tau_c)

        Fl_prime = jnp.where(is_tca > 0.5, Fl_prime_tca, Fl_prime)
        return dy_acc.at[idx['F_g_start'] + l].set(Fl_prime)
    dy = jax.lax.fori_loop(2, l_max_g, photon_hierarchy_step, dy)

    # l=l_max (truncation)
    tau0_minus_tau = jnp.maximum(bg.conformal_age - tau, 1e-10)
    F_lmax_prime_full = k*F_g[l_max_g-1] - (l_max_g+1.0)/tau0_minus_tau*F_g[l_max_g] - kappa_dot*F_g[l_max_g]
    F_lmax_prime_tca = -F_g[l_max_g] / tau_c
    F_lmax_prime = jnp.where(is_tca > 0.5, F_lmax_prime_tca, F_lmax_prime_full)
    dy = dy.at[idx['F_g_start'] + l_max_g].set(F_lmax_prime)

    # === POLARIZATION HIERARCHY ===
    # During TCA, all polarization is zero (scattering damps it instantly).
    # Drive polarization to zero: G'_l = -G_l / tau_c
    G0_full = -k*G_g[1] - kappa_dot*(G_g[0] - Pi/2.0)
    G0_tca = -G_g[0] / tau_c
    dy = dy.at[idx['G_g_0']].set(jnp.where(is_tca > 0.5, G0_tca, G0_full))

    G1_full = k/3.0*(G_g[0] - 2.0*G_g[2]) - kappa_dot*G_g[1]
    G1_tca = -G_g[1] / tau_c
    dy = dy.at[idx['G_g_1']].set(jnp.where(is_tca > 0.5, G1_tca, G1_full))

    def pol_hierarchy_step(l, dy_acc):
        Gl_prime = k/(2.0*l+1.0)*(l*G_g[l-1] - (l+1.0)*G_g[jnp.minimum(l+1, l_max_pol)]) - kappa_dot*G_g[l]
        Gl_prime = Gl_prime + jnp.where(l == 2, kappa_dot * Pi / 10.0, 0.0)
        Gl_prime_tca = -G_g[l] / tau_c
        Gl_prime = jnp.where(is_tca > 0.5, Gl_prime_tca, Gl_prime)
        return dy_acc.at[idx['G_g_start'] + l].set(Gl_prime)
    dy = jax.lax.fori_loop(2, l_max_pol, pol_hierarchy_step, dy)

    G_lmax_prime_full = k*G_g[l_max_pol-1] - (l_max_pol+1.0)/tau0_minus_tau*G_g[l_max_pol] - kappa_dot*G_g[l_max_pol]
    G_lmax_prime_tca = -G_g[l_max_pol] / tau_c
    dy = dy.at[idx['G_g_start'] + l_max_pol].set(jnp.where(is_tca > 0.5, G_lmax_prime_tca, G_lmax_prime_full))

    # === MASSLESS NEUTRINO HIERARCHY ===
    # Neutrinos have no scattering — no TCA. Full hierarchy at all times.
    dy = dy.at[idx['F_ur_0']].set(-k*F_ur[1] - 2.0/3.0*h_prime)
    dy = dy.at[idx['F_ur_1']].set(k/3.0*(F_ur[0] - 2.0*F_ur[2]))

    def ur_hierarchy_step(l, dy_acc):
        Fl_prime = k/(2.0*l+1.0)*(l*F_ur[l-1] - (l+1.0)*F_ur[jnp.minimum(l+1, l_max_ur)])
        # For l=2: add metric shear source (same as photons but without scattering)
        # cf. CLASS perturbations.c:9434-9439
        Fl_prime = Fl_prime + jnp.where(l == 2, 8.0/15.0 * metric_shear, 0.0)
        return dy_acc.at[idx['F_ur_start'] + l].set(Fl_prime)
    dy = jax.lax.fori_loop(2, l_max_ur, ur_hierarchy_step, dy)

    F_ur_lmax_prime = k*F_ur[l_max_ur-1] - (l_max_ur+1.0)/tau0_minus_tau*F_ur[l_max_ur]
    dy = dy.at[idx['F_ur_start'] + l_max_ur].set(F_ur_lmax_prime)

    # === SET METRIC DERIVATIVES ===
    dy = dy.at[idx['eta']].set(eta_prime)
    dy = dy.at[idx['h_prime']].set(0.0)  # h' is not evolved (constraint), keep dummy at 0

    # === SET MATTER DERIVATIVES ===
    dy = dy.at[idx['delta_cdm']].set(delta_cdm_prime)
    dy = dy.at[idx['delta_b']].set(delta_b_prime)
    dy = dy.at[idx['theta_b']].set(theta_b_prime)

    return dy


# ---------------------------------------------------------------------------
# Source function extraction
# ---------------------------------------------------------------------------

def _extract_sources(y, k, tau, bg, th, idx):
    """Extract CMB source functions from the perturbation state.

    Uses the non-IBP form: S_T0 × j_l + S_T1 × j_l' in the transfer integral.
    S_T0 = SW + ISW, S_T1 = Doppler (original form, not integration-by-parts).

    The Newtonian gauge potential Φ = η - ℋα is used for the ISW free-streaming
    term (more accurate than using η directly), while the SW uses the synchronous
    gauge form g*(δ_g/4 + η) which is numerically stable.

    cf. CLASS perturbations.c: perturbations_source_functions()
    cf. CLASS perturbations.c:6611-6674 (Einstein constraints)
    """
    loga = bg.loga_of_tau.evaluate(tau)
    a = jnp.exp(loga)
    H = bg.H_of_loga.evaluate(loga)
    a_prime_over_a = a * H  # ℋ = conformal Hubble
    k2 = k * k
    a2 = a * a

    kappa_dot = th.kappa_dot_of_loga.evaluate(loga)
    exp_m_kappa = th.exp_m_kappa_of_loga.evaluate(loga)
    g = th.g_of_loga.evaluate(loga)  # visibility

    # Unpack state vector
    eta = y[idx['eta']]
    delta_g = y[idx['F_g_0']]
    F_g_1 = y[idx['F_g_1']]
    F_g_2 = y[idx['F_g_2']]
    theta_b = y[idx['theta_b']]
    delta_b = y[idx['delta_b']]
    delta_cdm = y[idx['delta_cdm']]

    G_g_0 = y[idx['G_g_0']]
    G_g_2 = y[idx['G_g_2']]
    Pi = F_g_2 + G_g_0 + G_g_2

    F_ur_0 = y[idx['F_ur_0']]
    F_ur_1 = y[idx['F_ur_1']]
    F_ur_2 = y[idx['F_ur_2']]

    # Background densities
    rho_g = bg.rho_g_of_loga.evaluate(loga)
    rho_b = bg.rho_b_of_loga.evaluate(loga)
    rho_cdm = bg.rho_cdm_of_loga.evaluate(loga)
    rho_ur = bg.rho_ur_of_loga.evaluate(loga)
    rho_ncdm = bg.rho_ncdm_of_loga.evaluate(loga)

    # --- Einstein constraints ---
    delta_rho = (rho_g * delta_g + rho_b * delta_b + rho_cdm * delta_cdm
                 + rho_ur * F_ur_0 + rho_ncdm * F_ur_0)

    theta_g = 3.0 * k * F_g_1 / 4.0
    theta_ur = 3.0 * k * F_ur_1 / 4.0
    rho_plus_p_theta = (4.0/3.0 * rho_g * theta_g + rho_b * theta_b
                        + 4.0/3.0 * rho_ur * theta_ur
                        + 4.0/3.0 * rho_ncdm * theta_ur)

    # h' from 00 Einstein CONSTRAINT
    # cf. CLASS perturbations.c:6612
    h_prime = (k2 * eta + 1.5 * a2 * delta_rho) / (0.5 * a_prime_over_a)

    # η' from 0i Einstein equation
    # cf. CLASS perturbations.c:6635
    eta_prime = 1.5 * a2 * rho_plus_p_theta / k2

    # α = (h' + 6η')/(2k²) -- gauge shear potential
    # cf. CLASS perturbations.c:6644
    alpha = (h_prime + 6.0 * eta_prime) / (2.0 * k2)

    # α' from trace-free ij Einstein constraint
    # cf. CLASS perturbations.c:6671-6674
    rho_plus_p_shear = (2.0/3.0 * rho_g * F_g_2
                        + 2.0/3.0 * rho_ur * F_ur_2
                        + 2.0/3.0 * rho_ncdm * F_ur_2)
    alpha_prime = (-2.0 * a_prime_over_a * alpha
                   + eta
                   - 4.5 * (a2 / k2) * rho_plus_p_shear)

    # Newtonian gauge potential Φ = η - ℋα
    phi_newt = eta - a_prime_over_a * alpha

    # Φ' = η' - ℋ'α - ℋα' (for ISW)
    dH_dloga = bg.H_of_loga.derivative(loga)
    H_prime_conformal = a_prime_over_a * (a_prime_over_a + a * dH_dloga)
    phi_prime = eta_prime - H_prime_conformal * alpha - a_prime_over_a * alpha_prime

    # === Source functions (CLASS synchronous gauge IBP form) ===
    # cf. CLASS perturbations.c:7660-7678
    # C_l = 4π ∫ dlnk P_R(k) |T_l(k)|²  (Dodelson 2003, eq. 9.35)
    # T_l(k) = ∫ dτ S_T0(k,τ) j_l(kχ)  (IBP form: only j_l needed)
    #
    # The IBP form is the gauge-invariant transfer function formula.
    # Doppler is integrated by parts: g*v_b*j_l' → (g'*v_b + g*v_b')/k² * j_l

    # ℋ' = d(aH)/dτ
    dH_dloga = bg.H_of_loga.derivative(loga)
    H_prime_conformal = a_prime_over_a * (a_prime_over_a + a * dH_dloga)

    # g' = dg/dτ
    dg_dloga = th.g_of_loga.derivative(loga)
    g_prime = dg_dloga * a_prime_over_a

    # θ_b' from the ODE RHS, consistent with TCA/full switching.
    # CRITICAL: CLASS (perturbations.c:7535) uses dy[theta_b] directly from the
    # RHS evaluation, which includes TCA/full switching. We must do the same.
    # Use the shared helper to reproduce the exact same TCA blending as the RHS.
    cs2 = th.cs2_of_loga.evaluate(loga)
    theta_g = 3.0 * k * F_g_1 / 4.0

    is_tca, tau_c = _compute_tca_criterion(kappa_dot, a_prime_over_a, k)
    theta_b_prime = _compute_theta_b_prime_blended(
        theta_b, delta_b, theta_g, delta_g, F_g_2, G_g_0, G_g_2,
        a_prime_over_a, cs2, k, k2, kappa_dot,
        rho_g, rho_b, bg, loga, a, h_prime, eta_prime, is_tca, tau_c,
    )

    # === GAUGE SHIFT for Doppler source (CLASS perturbations.c:7632-7633) ===
    # In sync gauge, the gauge-invariant baryon velocity is θ_b + k²α.
    # CLASS absorbs this shift into θ_b before computing the IBP Doppler source,
    # making the total source function gauge-invariant.
    theta_b_shifted = theta_b + k2 * alpha
    theta_b_prime_shifted = theta_b_prime + k2 * alpha_prime

    # SW: g*(δ_g/4 + α')  [cf. perturbations.c:7660]
    source_SW = g * (delta_g / 4.0 + alpha_prime)

    # ISW (visibility): g*(η - α' - 2ℋα)  [cf. perturbations.c:7662-7664]
    source_ISW_vis = g * (eta - alpha_prime - 2.0 * a_prime_over_a * alpha)

    # ISW (free-streaming): exp(-κ)*2Φ'  [cf. perturbations.c:7665-7667]
    source_ISW_fs = exp_m_kappa * 2.0 * (eta_prime
                                          - H_prime_conformal * alpha
                                          - a_prime_over_a * alpha_prime)

    # Doppler (IBP): (1/k²)*(g*θ_b' + g'*θ_b)  [cf. perturbations.c:7668]
    # Uses SHIFTED velocities (gauge-invariant): θ_b + k²α, θ_b' + k²α'
    source_Doppler = (1.0 / k2) * (g * theta_b_prime_shifted + g_prime * theta_b_shifted)

    # Non-IBP Doppler: g * theta_b_shifted / k (uses j_l' radial function)
    # This bypasses the IBP transformation and directly couples the baryon velocity.
    # The transfer integral becomes: int dtau (g*v_b) * j_l'(kchi) dtau
    # where v_b = theta_b_shifted / k
    source_Doppler_nonIBP = g * theta_b_shifted / k

    # Non-IBP source_T0: SW + ISW_vis + ISW_fs (no Doppler, added via j_l' in harmonic)
    source_T0_noDopp = source_SW + source_ISW_vis + source_ISW_fs

    # IBP source_T0 (original, for comparison)
    source_T0 = source_SW + source_ISW_vis + source_ISW_fs + source_Doppler

    # S_T1: ISW dipole (small in flat space)
    # cf. perturbations.c:7672-7674
    source_T1 = exp_m_kappa * k * (alpha_prime + 2.0 * a_prime_over_a * alpha - eta)

    # S_T2: Quadrupole source
    source_T2 = g * Pi

    # E-polarization source
    # CLASS perturbations.c:7690: source_p = sqrt(6)*g*P where P = Pi/8
    # CLASS transfer.c:4197: radial factor = sqrt(3/8*(l+2)(l+1)l(l-1))
    # Combined factor: sqrt(6)*sqrt(3/8)/8 = 3/16 (no k² dependence!)
    # Our harmonic.py has E_l = sqrt((l+2)(l+1)l(l-1)) * int source_E * j_l/(kchi)^2 dtau
    source_E = 3.0 * g * Pi / 16.0

    # Lensing potential source
    source_lens = exp_m_kappa * 2.0 * phi_newt

    # Matter density contrast (for P(k))
    delta_m = (rho_b * delta_b + rho_cdm * delta_cdm) / (rho_b + rho_cdm)

    return (source_T0, source_T1, source_T2, source_E, source_lens, delta_m,
            source_SW, source_ISW_vis, source_ISW_fs, source_Doppler,
            source_Doppler_nonIBP, source_T0_noDopp)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _make_tau_grid(tau_min, tau_max, tau_star, n_points):
    """Create non-uniform τ grid concentrated around recombination.

    The visibility function g(τ) peaks sharply around τ_* with width ~30 Mpc.
    A uniform grid wastes resolution on late times where the source is smooth.
    We use a piecewise grid: sparse before and after recombination, dense around it.

    Args:
        tau_min: earliest conformal time
        tau_max: latest conformal time (~ conformal age)
        tau_star: recombination conformal time (from thermodynamics)
        n_points: total number of grid points (static for JIT)

    Returns:
        Non-uniform τ grid, shape (n_points,)
    """
    # Allocate points: 10% early, 60% recombination, 30% late
    n_early = n_points // 10
    n_recomb = 6 * n_points // 10
    n_late = n_points - n_early - n_recomb

    # Recombination region: τ_* ± 200 Mpc (covers ~7σ of visibility peak)
    tau_recomb_start = jnp.maximum(tau_star - 200.0, tau_min * 1.01)
    tau_recomb_end = jnp.minimum(tau_star + 300.0, tau_max * 0.99)

    grid_early = jnp.linspace(tau_min, tau_recomb_start, n_early + 1)[:-1]
    grid_recomb = jnp.linspace(tau_recomb_start, tau_recomb_end, n_recomb)
    grid_late = jnp.linspace(tau_recomb_end, tau_max, n_late + 1)[1:]

    return jnp.concatenate([grid_early, grid_recomb, grid_late])


def _k_grid(prec: PrecisionParams) -> Float[Array, "Nk"]:
    """Generate logarithmic k-grid for perturbation integration."""
    n_k = int(jnp.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade)
    return jnp.logspace(jnp.log10(prec.pt_k_min), jnp.log10(prec.pt_k_max_cl), n_k)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def perturbations_solve(
    params: CosmoParams,
    prec: PrecisionParams,
    bg: BackgroundResult,
    th: ThermoResult,
) -> PerturbationResult:
    """Solve the Einstein-Boltzmann system for all k-modes.

    Args:
        params: cosmological parameters
        prec: precision parameters
        bg: background result
        th: thermodynamics result

    Returns:
        PerturbationResult with source function tables
    """
    l_max_g = prec.pt_l_max_g
    l_max_pol = prec.pt_l_max_pol_g
    l_max_ur = prec.pt_l_max_ur

    idx = _build_indices(l_max_g, l_max_pol, l_max_ur)
    n_eq = idx['n_eq']

    # k-grid
    k_grid = _k_grid(prec)
    n_k = len(k_grid)

    # τ-grid for saving source functions
    # Non-uniform: dense around recombination where visibility peaks, sparse elsewhere.
    # This is critical for resolving the narrow visibility function (~30 Mpc width)
    # while covering the full conformal time range for ISW.
    # Initial time: early enough that all k-modes are super-horizon
    # kτ_ini << 1 for all k. Choose τ_ini = 0.1 / k_max
    tau_ini = 0.1 / prec.pt_k_max_cl

    # τ-grid for saving source functions (must start >= tau_ini)
    tau_min = max(float(bg.tau_table[0]) * 1.1, tau_ini * 1.01)
    tau_max = float(bg.conformal_age) * 0.999
    tau_star = th.tau_star  # recombination conformal time (~282 Mpc)
    tau_grid = _make_tau_grid(tau_min, tau_max, tau_star, prec.pt_tau_n_points)

    def solve_single_k(k):
        """Solve for a single k-mode."""
        # Initial conditions
        y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, params, idx, n_eq)

        # ODE args
        ode_args = (k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur)

        # Solve
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_perturbation_rhs),
            solver=diffrax.Kvaerno5(),
            t0=tau_ini,
            t1=float(bg.conformal_age) * 0.999,
            dt0=tau_ini * 0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=tau_grid),
            stepsize_controller=diffrax.PIDController(
                rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol,
            ),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=prec.ode_max_steps,
            args=ode_args,
        )

        # Extract source functions at each τ
        def extract_at_tau(i):
            y_i = sol.ys[i]
            tau_i = tau_grid[i]
            return _extract_sources(y_i, k, tau_i, bg, th, idx)

        sources = jax.vmap(extract_at_tau)(jnp.arange(prec.pt_tau_n_points))
        return sources  # tuple of 12 arrays, each shape (n_tau,)

    # Vectorize over k-modes
    all_sources = jax.vmap(solve_single_k)(k_grid)
    # all_sources is a tuple of 12 arrays, each shape (n_k, n_tau)

    return PerturbationResult(
        k_grid=k_grid,
        tau_grid=tau_grid,
        source_T0=all_sources[0],
        source_T1=all_sources[1],
        source_T2=all_sources[2],
        source_E=all_sources[3],
        source_lens=all_sources[4],
        delta_m=all_sources[5],
        source_SW=all_sources[6],
        source_ISW_vis=all_sources[7],
        source_ISW_fs=all_sources[8],
        source_Doppler=all_sources[9],
        source_Doppler_nonIBP=all_sources[10],
        source_T0_noDopp=all_sources[11],
    )


# ===========================================================================
# TENSOR PERTURBATIONS (gravitational waves → B-mode polarization)
# ===========================================================================

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TensorPerturbationResult:
    """Output of the tensor perturbation module.

    Contains tensor source function tables on a (k, tau) grid for
    computing C_l^BB (and tensor contributions to TT, EE).
    """
    k_grid: Float[Array, "Nk"]
    tau_grid: Float[Array, "Ntau"]

    # Tensor source functions: shape (Nk, Ntau)
    source_t: Float[Array, "Nk Ntau"]   # Tensor temperature source
    source_p: Float[Array, "Nk Ntau"]   # Tensor polarization source (for BB)

    def tree_flatten(self):
        return [self.k_grid, self.tau_grid, self.source_t, self.source_p], None

    @classmethod
    def tree_unflatten(cls, aux, fields):
        return cls(*fields)


def _build_tensor_indices(l_max_g: int, l_max_pol: int, l_max_ur: int):
    """Build index mapping for the tensor perturbation state vector.

    The tensor state vector contains:
        - GW metric perturbation: h, h'
        - Tensor photon hierarchy: F_g,0 through F_g,l_max (intensity)
        - Tensor photon polarization: G_g,0 through G_g,l_max
        - Tensor massless neutrino hierarchy: F_ur,0 through F_ur,l_max

    cf. CLASS perturbations.c: index_pt_gw, index_pt_gwdot, etc.
    """
    idx = {}
    i = 0

    # GW metric perturbation
    idx['gw'] = i; i += 1         # h tensor
    idx['gw_dot'] = i; i += 1     # h' tensor

    # Tensor photon hierarchy (temperature/intensity)
    idx['F_g_start'] = i
    for l in range(l_max_g + 1):
        idx[f'F_g_{l}'] = i; i += 1
    idx['F_g_end'] = i

    # Tensor photon polarization hierarchy
    idx['G_g_start'] = i
    for l in range(l_max_pol + 1):
        idx[f'G_g_{l}'] = i; i += 1
    idx['G_g_end'] = i

    # Tensor massless neutrino hierarchy
    idx['F_ur_start'] = i
    for l in range(l_max_ur + 1):
        idx[f'F_ur_{l}'] = i; i += 1
    idx['F_ur_end'] = i

    idx['n_eq'] = i
    return idx


_SQRT6 = math.sqrt(6.0)
_SQRT2 = math.sqrt(2.0)


def _tensor_ic(k, tau_ini, bg, idx, n_eq):
    """Tensor initial conditions.

    cf. CLASS perturbations.c:5957:
        y[index_pt_gw] = gw_ini / sqrt(6)
    where gw_ini = 1 (default).

    For flat space (K=0), the GW initial condition is h = 1/sqrt(6),
    and the correction at order tau^2 gives:
        h_corr = -h * k^2 / (6 + 8/5 * rho_fs/rho_r) * tau^2
    where rho_fs = rho_ur (free-streaming radiation).

    cf. CLASS perturbations.c:6009-6014
    """
    y0 = jnp.zeros(n_eq)

    loga_ini = bg.loga_of_tau.evaluate(tau_ini)
    a_ini = jnp.exp(loga_ini)

    rho_g = bg.rho_g_of_loga.evaluate(loga_ini)
    rho_ur = bg.rho_ur_of_loga.evaluate(loga_ini)
    rho_ncdm = bg.rho_ncdm_of_loga.evaluate(loga_ini)
    rho_r = rho_g + rho_ur + rho_ncdm
    rho_fs = rho_ur + rho_ncdm  # free-streaming species

    k2 = k * k

    # Leading order: h = 1/sqrt(6) (normalized to unit primordial tensor)
    h0 = 1.0 / _SQRT6

    # Second-order correction (CLASS perturbations.c:6012-6014)
    h_corr = -h0 * k2 / (6.0 + 8.0 / 5.0 * rho_fs / rho_r) * tau_ini**2
    h_val = h0 + h_corr
    hdot_val = 2.0 * h_corr / tau_ini

    y0 = y0.at[idx['gw']].set(h_val)
    y0 = y0.at[idx['gw_dot']].set(hdot_val)

    # Photon/neutrino quadrupoles start at order tau^2 (set by CLASS at 6016-6032)
    # F_g_0 (tensor delta_g) at leading order: proportional to h' * tau
    # For simplicity, start all hierarchy moments at zero (they build up quickly)

    return y0


def _tensor_rhs(tau, y, args):
    """Right-hand side of the tensor perturbation ODE system.

    Implements the full tensor Boltzmann hierarchy in synchronous gauge.

    The GW equation (CLASS perturbations.c:6744, flat space K=0):
        h'' + 2*aH*h' + k^2*h = gw_source

    where gw_source = tensor anisotropic stress from photons + neutrinos.

    The tensor photon Boltzmann hierarchy (CLASS perturbations.c:9815-9868):
        F'_0 = -4/3*theta_g - kappa'*(F_0 + sqrt(6)*P^(2)) + sqrt(6)*h'
        F'_1 (theta_g) = k^2*(F_0/4 - s2*sigma_g) - kappa'*theta_g
        F'_2 (sigma_g) = 4/15*s2*theta_g - 3/10*k*s3*F_3 - kappa'*sigma_g
        F'_l = k/(2l+1)*(l*s_l*F_{l-1} - (l+1)*s_{l+1}*F_{l+1}) - kappa'*F_l

    Tensor polarization hierarchy (CLASS perturbations.c:9851-9868):
        G'_0 = -k*G_1 - kappa'*(G_0 - sqrt(6)*P^(2))
        G'_l = k/(2l+1)*(l*s_l*G_{l-1} - (l+1)*s_{l+1}*G_{l+1}) - kappa'*G_l

    where P^(2) is the tensor polarization term (CLASS perturbations.c:9795-9802):
        P^(2) = -1/sqrt(6) * (1/10*F_0 + 2/7*sigma_g + 3/70*F_4
                               - 3/5*G_0 + 6/7*G_2 - 3/70*G_4)

    Tensor neutrino hierarchy (CLASS perturbations.c:9873-9893):
        F'_ur_0 = -4/3*theta_ur + sqrt(6)*h'
        F'_ur_1 = k^2*(F_ur_0/4 - s2*sigma_ur)
        F'_ur_2 = 4/15*theta_ur - 3/10*k*s3/s2*F_ur_3
        F'_ur_l = k/(2l+1)*(l*s_l*F_ur_{l-1} - (l+1)*s_{l+1}*F_ur_{l+1})

    For flat space, all s_l factors = 1.
    """
    (k, bg, th, idx, l_max_g, l_max_pol, l_max_ur) = args

    loga = bg.loga_of_tau.evaluate(tau)
    a = jnp.exp(loga)
    H = bg.H_of_loga.evaluate(loga)
    a_prime_over_a = a * H
    a2 = a * a
    k2 = k * k

    rho_g = bg.rho_g_of_loga.evaluate(loga)
    rho_ur = bg.rho_ur_of_loga.evaluate(loga)
    rho_ncdm = bg.rho_ncdm_of_loga.evaluate(loga)

    kappa_dot = th.kappa_dot_of_loga.evaluate(loga)

    # Unpack GW state
    gw = y[idx['gw']]
    gw_dot = y[idx['gw_dot']]

    # Photon tensor hierarchy
    F_g = y[idx['F_g_start']:idx['F_g_end']]
    delta_g = F_g[0]
    theta_g = F_g[1]  # NOTE: In tensor mode, theta_g = (3k/4)*F_1, but CLASS stores theta_g directly
    # Actually CLASS stores F_0, theta_g (= 3k/4 * F_1), shear_g (= F_2/2), F_3, ...
    # For simplicity, we store all F_l in our hierarchy and use them directly.
    # theta_g in CLASS = 3k/4 * F_1_ours; sigma_g in CLASS = F_2_ours / 2.

    # Polarization tensor hierarchy
    G_g = y[idx['G_g_start']:idx['G_g_end']]

    # Neutrino tensor hierarchy
    F_ur = y[idx['F_ur_start']:idx['F_ur_end']]

    # === GW ANISOTROPIC STRESS SOURCE ===
    # cf. CLASS perturbations.c:7322-7358
    # gw_source from photons: -sqrt(6)*4*a^2*rho_g * (1/15*F_0 + 4/21*sigma_g + 1/35*F_4)
    # where sigma_g = F_2/2 in our convention
    F_g_4 = jnp.where(l_max_g >= 4, F_g[jnp.minimum(4, l_max_g)], 0.0)
    sigma_g = F_g[jnp.minimum(2, l_max_g)] / 2.0

    gw_source = -_SQRT6 * 4.0 * a2 * rho_g * (
        1.0/15.0 * delta_g + 4.0/21.0 * sigma_g + 1.0/35.0 * F_g_4
    )

    # Neutrino contribution
    F_ur_4 = jnp.where(l_max_ur >= 4, F_ur[jnp.minimum(4, l_max_ur)], 0.0)
    sigma_ur = F_ur[jnp.minimum(2, l_max_ur)] / 2.0

    rho_relativistic = rho_ur + rho_ncdm
    gw_source += -_SQRT6 * 4.0 * a2 * rho_relativistic * (
        1.0/15.0 * F_ur[0] + 4.0/21.0 * sigma_ur + 1.0/35.0 * F_ur_4
    )

    # === GW EQUATION ===
    # h'' = -2*aH*h' - k^2*h + gw_source
    gw_prime_prime = -2.0 * a_prime_over_a * gw_dot - k2 * gw + gw_source

    dy = jnp.zeros_like(y)
    dy = dy.at[idx['gw']].set(gw_dot)
    dy = dy.at[idx['gw_dot']].set(gw_prime_prime)

    # === P^(2) TENSOR POLARIZATION TERM ===
    # cf. CLASS perturbations.c:9795-9802
    G_g_2 = G_g[jnp.minimum(2, l_max_pol)]
    G_g_4 = jnp.where(l_max_pol >= 4, G_g[jnp.minimum(4, l_max_pol)], 0.0)
    P2 = -1.0 / _SQRT6 * (
        1.0/10.0 * delta_g
        + 2.0/7.0 * sigma_g
        + 3.0/70.0 * F_g_4
        - 3.0/5.0 * G_g[0]
        + 6.0/7.0 * G_g_2
        - 3.0/70.0 * G_g_4
    )

    # === PHOTON TENSOR HIERARCHY ===
    # l=0: F'_0 = -4/3*theta_g - kappa'*(F_0 + sqrt(6)*P2) + sqrt(6)*h'
    # cf. CLASS perturbations.c:9816-9820 (synchronous gauge)
    dy = dy.at[idx['F_g_0']].set(
        -4.0/3.0 * theta_g - kappa_dot * (delta_g + _SQRT6 * P2) + _SQRT6 * gw_dot
    )

    # l=1: theta'_g = k^2*(F_0/4 - sigma_g) - kappa'*theta_g
    # cf. CLASS perturbations.c:9822-9825 (synchronous gauge)
    dy = dy.at[idx['F_g_1']].set(
        k2 * (delta_g / 4.0 - sigma_g) - kappa_dot * theta_g
    )

    # l=2: sigma'_g = 4/15*theta_g - 3/10*k*F_3 - kappa'*sigma_g
    # cf. CLASS perturbations.c:9827-9830
    # sigma_g = F_2/2, so F'_2 = 2*sigma'_g
    F_g_3 = F_g[jnp.minimum(3, l_max_g)]
    sigma_g_prime = 4.0/15.0 * theta_g - 3.0/10.0 * k * F_g_3 - kappa_dot * sigma_g
    dy = dy.at[idx['F_g_2']].set(2.0 * sigma_g_prime)

    # l=3: F'_3 = k/7*(6*sigma_g - 4*F_4) - kappa'*F_3
    # cf. CLASS perturbations.c:9832-9835 (with F_2 = 2*sigma_g)
    dy = dy.at[idx['F_g_3']].set(
        k/7.0 * (6.0 * sigma_g - 4.0 * F_g_4) - kappa_dot * F_g_3
    )

    # l=4 to l_max-1
    def photon_tensor_step(l, dy_acc):
        Fl_prime = k/(2.0*l+1.0) * (l*F_g[l-1] - (l+1.0)*F_g[jnp.minimum(l+1, l_max_g)]) - kappa_dot*F_g[l]
        return dy_acc.at[idx['F_g_start'] + l].set(Fl_prime)
    dy = jax.lax.fori_loop(4, l_max_g, photon_tensor_step, dy)

    # l=l_max truncation
    tau0_minus_tau = jnp.maximum(bg.conformal_age - tau, 1e-10)
    dy = dy.at[idx['F_g_start'] + l_max_g].set(
        k * F_g[l_max_g - 1] - (l_max_g + 1.0) / tau0_minus_tau * F_g[l_max_g] - kappa_dot * F_g[l_max_g]
    )

    # === POLARIZATION TENSOR HIERARCHY ===
    # l=0: G'_0 = -k*G_1 - kappa'*(G_0 - sqrt(6)*P2)
    # cf. CLASS perturbations.c:9851-9854
    dy = dy.at[idx['G_g_0']].set(
        -k * G_g[jnp.minimum(1, l_max_pol)] - kappa_dot * (G_g[0] - _SQRT6 * P2)
    )

    # l >= 1 to l_max-1
    def pol_tensor_step(l, dy_acc):
        Gl_prime = k/(2.0*l+1.0) * (l*G_g[l-1] - (l+1.0)*G_g[jnp.minimum(l+1, l_max_pol)]) - kappa_dot*G_g[l]
        return dy_acc.at[idx['G_g_start'] + l].set(Gl_prime)
    dy = jax.lax.fori_loop(1, l_max_pol, pol_tensor_step, dy)

    # l=l_max truncation
    dy = dy.at[idx['G_g_start'] + l_max_pol].set(
        k * G_g[l_max_pol - 1] - (l_max_pol + 1.0) / tau0_minus_tau * G_g[l_max_pol] - kappa_dot * G_g[l_max_pol]
    )

    # === NEUTRINO TENSOR HIERARCHY ===
    # l=0: F'_ur_0 = -4/3*theta_ur + sqrt(6)*h'
    # cf. CLASS perturbations.c:9875
    theta_ur = F_ur[jnp.minimum(1, l_max_ur)]
    dy = dy.at[idx['F_ur_0']].set(-4.0/3.0 * theta_ur + _SQRT6 * gw_dot)

    # l=1: theta'_ur = k^2*(F_ur_0/4 - sigma_ur)
    # cf. CLASS perturbations.c:9877
    dy = dy.at[idx['F_ur_1']].set(k2 * (F_ur[0] / 4.0 - sigma_ur))

    # l=2: sigma'_ur = 4/15*theta_ur - 3/10*k*F_ur_3
    # cf. CLASS perturbations.c:9879-9880
    F_ur_3 = F_ur[jnp.minimum(3, l_max_ur)]
    sigma_ur_prime = 4.0/15.0 * theta_ur - 3.0/10.0 * k * F_ur_3
    dy = dy.at[idx['F_ur_2']].set(2.0 * sigma_ur_prime)

    # l=3
    dy = dy.at[idx['F_ur_3']].set(
        k/7.0 * (6.0 * sigma_ur - 4.0 * F_ur_4)
    )

    # l=4 to l_max-1
    def ur_tensor_step(l, dy_acc):
        Fl_prime = k/(2.0*l+1.0) * (l*F_ur[l-1] - (l+1.0)*F_ur[jnp.minimum(l+1, l_max_ur)])
        return dy_acc.at[idx['F_ur_start'] + l].set(Fl_prime)
    dy = jax.lax.fori_loop(4, l_max_ur, ur_tensor_step, dy)

    # l=l_max truncation
    dy = dy.at[idx['F_ur_start'] + l_max_ur].set(
        k * F_ur[l_max_ur - 1] - (l_max_ur + 1.0) / tau0_minus_tau * F_ur[l_max_ur]
    )

    return dy


def _extract_tensor_sources(y, k, tau, bg, th, idx, l_max_g, l_max_pol):
    """Extract tensor CMB source functions from the tensor perturbation state.

    Tensor temperature source (CLASS perturbations.c:8055-8056):
        S_t = -h' * e^{-kappa} + g * P

    Tensor polarization source (CLASS perturbations.c:8067):
        S_p = sqrt(6) * g * P

    where P = P^(2) is the tensor polarization term.

    cf. CLASS perturbations.c:8036-8067
    """
    loga = bg.loga_of_tau.evaluate(tau)
    exp_m_kappa = th.exp_m_kappa_of_loga.evaluate(loga)
    g = th.g_of_loga.evaluate(loga)
    kappa_dot = th.kappa_dot_of_loga.evaluate(loga)

    gw_dot = y[idx['gw_dot']]

    F_g = y[idx['F_g_start']:idx['F_g_end']]
    G_g = y[idx['G_g_start']:idx['G_g_end']]
    delta_g = F_g[0]
    sigma_g = F_g[jnp.minimum(2, l_max_g)] / 2.0

    # P^(2) (CLASS perturbations.c:8036-8042)
    F_g_4 = jnp.where(l_max_g >= 4, F_g[jnp.minimum(4, l_max_g)], 0.0)
    G_g_2 = G_g[jnp.minimum(2, l_max_pol)]
    G_g_4 = jnp.where(l_max_pol >= 4, G_g[jnp.minimum(4, l_max_pol)], 0.0)

    P = -(1.0/10.0 * delta_g
          + 2.0/7.0 * sigma_g
          + 3.0/70.0 * F_g_4
          - 3.0/5.0 * G_g[0]
          + 6.0/7.0 * G_g_2
          - 3.0/70.0 * G_g_4) / _SQRT6

    # During tight coupling, use TCA expression:
    # P_tca = -1/3 * h' / kappa'  (CLASS perturbations.c:8046-8047)
    P_tca = -1.0/3.0 * gw_dot / jnp.maximum(kappa_dot, 1e-30)
    tau_c = 1.0 / jnp.maximum(kappa_dot, 1e-30)
    is_tca = jax.nn.sigmoid(-5.0 * (jnp.log(tau_c * k + 1e-30) - jnp.log(0.01)))
    P = jnp.where(is_tca > 0.5, P_tca, P)

    # Temperature source: -h' * e^{-kappa} + g * P
    source_t = -gw_dot * exp_m_kappa + g * P

    # Polarization source: sqrt(6) * g * P (CMBFAST/CAMB sign convention)
    # cf. CLASS perturbations.c:8067
    source_p = _SQRT6 * g * P

    return source_t, source_p


def tensor_perturbations_solve(
    params: CosmoParams,
    prec: PrecisionParams,
    bg: BackgroundResult,
    th: ThermoResult,
) -> TensorPerturbationResult:
    """Solve tensor perturbations (gravitational waves) for all k-modes.

    Integrates the tensor GW equation coupled with the tensor photon
    and neutrino Boltzmann hierarchies for each k-mode.

    The tensor power spectrum uses:
        P_T(k) = A_s * r * (k/k_pivot)^{n_t}
    which is applied in the C_l computation (harmonic module).

    Args:
        params: cosmological parameters (r_t > 0 needed for nonzero BB)
        prec: precision parameters
        bg: background result
        th: thermodynamics result

    Returns:
        TensorPerturbationResult with tensor source function tables
    """
    l_max_g = prec.pt_l_max_g
    l_max_pol = prec.pt_l_max_pol_g
    l_max_ur = prec.pt_l_max_ur

    idx = _build_tensor_indices(l_max_g, l_max_pol, l_max_ur)
    n_eq = idx['n_eq']

    # k-grid for tensor modes (use same grid as scalars up to k_max_cl)
    k_grid = _k_grid(prec)

    # tau grid (same approach as scalar)
    tau_ini = 0.1 / prec.pt_k_max_cl
    tau_min = max(float(bg.tau_table[0]) * 1.1, tau_ini * 1.01)
    tau_max = float(bg.conformal_age) * 0.999
    tau_star = th.tau_star
    tau_grid = _make_tau_grid(tau_min, tau_max, tau_star, prec.pt_tau_n_points)

    def solve_single_k(k):
        y0 = _tensor_ic(k, jnp.array(tau_ini), bg, idx, n_eq)
        ode_args = (k, bg, th, idx, l_max_g, l_max_pol, l_max_ur)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_tensor_rhs),
            solver=diffrax.Kvaerno5(),
            t0=tau_ini,
            t1=float(bg.conformal_age) * 0.999,
            dt0=tau_ini * 0.1,
            y0=y0,
            saveat=diffrax.SaveAt(ts=tau_grid),
            stepsize_controller=diffrax.PIDController(
                rtol=prec.pt_ode_rtol, atol=prec.pt_ode_atol,
            ),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            max_steps=prec.ode_max_steps,
            args=ode_args,
        )

        def extract_at_tau(i):
            y_i = sol.ys[i]
            tau_i = tau_grid[i]
            return _extract_tensor_sources(y_i, k, tau_i, bg, th, idx, l_max_g, l_max_pol)

        sources = jax.vmap(extract_at_tau)(jnp.arange(prec.pt_tau_n_points))
        return sources  # tuple of 2 arrays, each shape (n_tau,)

    all_sources = jax.vmap(solve_single_k)(k_grid)

    return TensorPerturbationResult(
        k_grid=k_grid,
        tau_grid=tau_grid,
        source_t=all_sources[0],
        source_p=all_sources[1],
    )
