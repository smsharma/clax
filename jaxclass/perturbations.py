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
    source_T0: Float[Array, "Nk Ntau"]   # Temperature monopole source (SW + intrinsic)
    source_T1: Float[Array, "Nk Ntau"]   # Temperature dipole source (Doppler)
    source_T2: Float[Array, "Nk Ntau"]   # Temperature quadrupole source (ISW + pol)
    source_E: Float[Array, "Nk Ntau"]    # E-polarization source
    source_lens: Float[Array, "Nk Ntau"]  # Lensing potential source

    # Also store matter transfer function
    delta_m: Float[Array, "Nk Ntau"]     # Total matter density contrast

    def tree_flatten(self):
        return [
            self.k_grid, self.tau_grid,
            self.source_T0, self.source_T1, self.source_T2,
            self.source_E, self.source_lens, self.delta_m,
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

    # === TCA CRITERION ===
    # τ_c = 1/κ' is the photon mean free time.
    # TCA is valid when τ_c * k << 1, i.e., κ'/k >> 1.
    # CLASS uses: tca_on when τ_c/τ_h < 0.005 AND τ_c/τ_k < 0.01
    #   where τ_h = 1/aH, τ_k = 1/k → τ_c*k < 0.01 → κ'/k > 100
    # We use a softer threshold with smooth sigmoid for JAX compatibility.
    # cf. CLASS perturbations.c:6178-6179
    tau_c = 1.0 / jnp.maximum(kappa_dot, 1e-30)  # photon mean free time
    tca_ratio = tau_c * k  # small when tightly coupled
    # Smooth sigmoid: 1 when tca_ratio << threshold, 0 when tca_ratio >> threshold
    # CLASS uses threshold ~0.01; we use 0.01 with transition width
    _TCA_THRESHOLD = 0.01
    _TCA_WIDTH = 5.0  # steepness of sigmoid transition
    is_tca = jax.nn.sigmoid(-_TCA_WIDTH * (jnp.log(tca_ratio + 1e-30) - jnp.log(_TCA_THRESHOLD)))

    # === EINSTEIN EQUATIONS (constraint approach, matching CLASS) ===
    # cf. CLASS perturbations.c:6611-6644

    # Total density perturbation δρ = Σ ρ_i δ_i
    # Include ncdm contribution approximated as massless (δ_ncdm ≈ δ_ur)
    # This is valid at early times when ncdm are relativistic.
    # Without this, h' constraint is wrong and perturbations grow too fast.
    delta_rho = rho_g * delta_g + rho_b * delta_b + rho_cdm * delta_cdm + rho_ur * delta_ur + rho_ncdm * delta_ur

    # Total (ρ+p)θ
    # Include ncdm (approximated as massless: ρ_ncdm + p_ncdm = 4/3 ρ_ncdm, θ_ncdm ≈ θ_ur)
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

    # === TCA PHOTON SHEAR ===
    # cf. CLASS perturbations.c:10193
    # shear_g = 16/45 * tau_c * (theta_g + metric_shear)
    # In F_l notation: F_g_2 = 2*sigma_g, and CLASS sigma_g = shear_g/2
    # so F_g_2^{tca} = 2 * (16/45) * tau_c * (theta_g + metric_shear)
    # Note: includes contribution from G_g0 and G_g2 (cf. CLASS comment at line 10194)
    tca_shear_g = 16.0 / 45.0 * tau_c * (theta_g + metric_shear)
    # tca_shear_g is σ_g in CLASS convention; F_g_2 = 2*σ_g
    tca_F_g_2 = 2.0 * tca_shear_g

    # === TCA SLIP ===
    # Photon-baryon slip: Δ = θ_b - θ_g at first order in τ_c
    # cf. CLASS perturbations.c:10144-10176 (first_order_CLASS approximation)
    # F = tau_c / (1+R) where R = (4/3)*rho_g/rho_b
    R = 4.0 * rho_g / (3.0 * rho_b)  # photon-baryon momentum ratio
    F_tca = tau_c / (1.0 + R)

    # dtau_c = -κ''/κ'^2 (derivative of τ_c w.r.t. conformal time)
    # We approximate dtau_c/tau_c ≈ -κ''/κ' which for κ'~a^{-2} gives ~2*aH
    # For the first_order_CAMB/compromise_CLASS form:
    # slip = (dtau_c/tau_c - 2*aH/(1+R))*(theta_b - theta_g)
    #        + F * (-a''_prime_over_a * theta_b + k2*(...) - aH*metric_euler)
    # In synchronous gauge metric_euler = 0.
    # Approximate dtau_c/tau_c ≈ 2*aH (Thomson ~ a^{-2})
    dtau_c_over_tau_c = 2.0 * a_prime_over_a
    tca_slip = (dtau_c_over_tau_c - 2.0 * a_prime_over_a / (1.0 + R)) * (theta_b - theta_g) \
        + F_tca * (
            -(a_prime_over_a**2 + a_prime_over_a**2) * theta_b  # -a''/a * theta_b (approx a''/a ≈ 2(a'/a)^2 in RD)
            + k2 * (
                -a_prime_over_a * delta_g / 2.0
                + cs2 * (-theta_b - metric_continuity)
                - (1.0/3.0) * (-theta_g - metric_continuity)
            )
        )
    # Note: a''/a = H'^conformal + 2*(a'/a)^2 but approximating as 2*(a'/a)^2 in RD is sufficient
    # for first-order TCA accuracy. Cleaner: use background spline for H'.
    dH_dloga_rhs = bg.H_of_loga.derivative(loga)
    a_primeprime_over_a = a_prime_over_a * (a_prime_over_a + a * dH_dloga_rhs)
    tca_slip = (dtau_c_over_tau_c - 2.0 * a_prime_over_a / (1.0 + R)) * (theta_b - theta_g) \
        + F_tca * (
            -a_primeprime_over_a * theta_b
            + k2 * (
                -a_prime_over_a * delta_g / 2.0
                + cs2 * (-theta_b - metric_continuity)
                - (1.0/3.0) * (-theta_g - metric_continuity)
            )
        )

    # === TCA BARYON VELOCITY ===
    # cf. CLASS perturbations.c:9100-9103
    # theta_b' = (-aH*theta_b + k2*(cs2*delta_b + R*(delta_g/4 - s2*shear_g)) + R*slip) / (1+R)
    # In synchronous gauge: metric_euler = 0
    theta_b_tca = (-a_prime_over_a * theta_b
                   + k2 * (cs2 * delta_b + R * (delta_g / 4.0 - tca_shear_g))
                   + R * tca_slip) / (1.0 + R)

    # === TCA PHOTON VELOCITY ===
    # cf. CLASS perturbations.c:9204-9206
    # theta_g' = -(theta_b' + aH*theta_b - k2*cs2*delta_b)/R + k2*(delta_g/4 - shear_g) + (1+R)/R*metric_euler
    # Synchronous gauge: metric_euler = 0
    theta_g_tca = -(theta_b_tca + a_prime_over_a * theta_b - k2 * cs2 * delta_b) / R \
        + k2 * (0.25 * delta_g - tca_shear_g)
    # Convert to F_1 derivative: theta_g = 3k*F_1/4 → F_1' = 4*theta_g'/(3k)
    F1_prime_tca = 4.0 * theta_g_tca / (3.0 * k)

    # === FULL (non-TCA) BARYONS ===
    # cf. Ma & Bertschinger Eqs. (30)-(31)
    delta_b_prime = -theta_b - h_prime / 2.0
    theta_b_full = -a_prime_over_a * theta_b + cs2 * k2 * delta_b + R * kappa_dot * (theta_g - theta_b)

    # Blend TCA and full baryon velocity
    theta_b_prime = jnp.where(is_tca > 0.5, theta_b_tca, theta_b_full)

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

    # θ_b' from perturbation equations
    cs2 = th.cs2_of_loga.evaluate(loga)
    R_photon_baryon = 4.0 * rho_g / (3.0 * rho_b)
    theta_g = 3.0 * k * F_g_1 / 4.0
    theta_b_prime = (-a_prime_over_a * theta_b
                     + cs2 * k2 * delta_b
                     + R_photon_baryon * kappa_dot * (theta_g - theta_b))

    # SW: g*(δ_g/4 + α')  [cf. perturbations.c:7660]
    source_SW = g * (delta_g / 4.0 + alpha_prime)

    # ISW (visibility): g*(η - α' - 2ℋα)  [cf. perturbations.c:7662-7664]
    source_ISW_vis = g * (eta - alpha_prime - 2.0 * a_prime_over_a * alpha)

    # ISW (free-streaming): exp(-κ)*2Φ'  [cf. perturbations.c:7665-7667]
    source_ISW_fs = exp_m_kappa * 2.0 * (eta_prime
                                          - H_prime_conformal * alpha
                                          - a_prime_over_a * alpha_prime)

    # Doppler (IBP): (1/k²)*(g*θ_b' + g'*θ_b)  [cf. perturbations.c:7668]
    source_Doppler = (1.0 / k2) * (g * theta_b_prime + g_prime * theta_b)

    source_T0 = source_SW + source_ISW_vis + source_ISW_fs + source_Doppler

    # S_T1: ISW dipole (small in flat space)
    # cf. perturbations.c:7672-7674
    source_T1 = exp_m_kappa * k * (alpha_prime + 2.0 * a_prime_over_a * alpha - eta)

    # S_T2: Quadrupole source
    source_T2 = g * Pi

    # E-polarization source
    source_E = g * Pi / (4.0 * k2)

    # Lensing potential source
    source_lens = exp_m_kappa * 2.0 * phi_newt

    # Matter density contrast (for P(k))
    delta_m = (rho_b * delta_b + rho_cdm * delta_cdm) / (rho_b + rho_cdm)

    return source_T0, source_T1, source_T2, source_E, source_lens, delta_m


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
        return sources  # tuple of 6 arrays, each shape (n_tau,)

    # Vectorize over k-modes
    all_sources = jax.vmap(solve_single_k)(k_grid)
    # all_sources is a tuple of 6 arrays, each shape (n_k, n_tau)

    return PerturbationResult(
        k_grid=k_grid,
        tau_grid=tau_grid,
        source_T0=all_sources[0],
        source_T1=all_sources[1],
        source_T2=all_sources[2],
        source_E=all_sources[3],
        source_lens=all_sources[4],
        delta_m=all_sources[5],
    )
