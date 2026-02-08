"""Thermodynamics module for jaxCLASS.

Computes the ionization history x_e(z), matter temperature T_m(z),
visibility function g(τ), optical depth κ(τ), and baryon sound speed c_s²(τ).

Uses the Ma & Bertschinger (1995) semi-implicit approach (as in DISCO-EB's
thermodynamics_mb95.py) rather than a full RECFAST ODE solve. This is
numerically stable and JAX-compatible via jax.lax.scan.

Key function:
    thermodynamics_solve(params, prec, bg) -> ThermoResult

References:
    - Ma & Bertschinger (1995) ApJ 455, 7 for the simplified recombination
    - DISCO-EB: src/discoeb/thermodynamics_mb95.py
    - CLASS: source/thermodynamics.c
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxclass import constants as const
from jaxclass.background import BackgroundResult
from jaxclass.interpolation import CubicSpline
from jaxclass.params import CosmoParams, PrecisionParams

# ---------------------------------------------------------------------------
# Constants for recombination
# ---------------------------------------------------------------------------

# MB95 constants (kept for helium Saha)
_thomc0_coeff = 5.0577e-8  # Thomson cooling coefficient (times Tcmb^4)
_barssc_raw = 9.1820e-14   # baryon sound speed prefactor
_tion1 = 2.855e5       # HeII ionization temperature [K]
_tion2 = 6.313e5       # HeIII ionization temperature [K]

# --- RECFAST/CLASS hydrogen recombination constants ---
# cf. CLASS external/HyRec2020/hydrogen.h and hydrogen.c
_EI_eV = 13.598286071938324     # H ionization energy [eV]
_E21_eV = 10.198714553953742    # E_2 - E_1 [eV]
_kBoltz_eV = 8.617343e-5        # Boltzmann constant [eV/K]
_L2s1s = 8.2206                 # 2s→1s two-photon decay rate [s^-1]
_lambda_Lya_cm = 1.215670e-5    # Lyman-alpha wavelength [cm]
_RECFAST_FUDGE = 1.14           # RECFAST fudge factor (Seager et al. 1999)

# Pequignot et al. (1991) case-B recombination coefficient
# alpha_B(T) = F * 4.309e-13 * t4^(-0.6166) / (1 + 0.6703 * t4^0.5300) [cm^3/s]
_ALPHA_B_PREFACTOR = 4.309e-13  # cm^3 s^-1
_ALPHA_B_POWER = -0.6166
_ALPHA_B_DENOM_COEFF = 0.6703
_ALPHA_B_DENOM_POWER = 0.5300

# Saha factor: n_1s,eq / (n_e * n_p) = (2*pi*m_e*kT/h^2)^{-3/2} * exp(E_I/kT) / 4
# SAHA_FACT = 3.016103031869581e21 [eV^{-3/2} cm^{-3}]
_SAHA_FACT = 3.016103031869581e21

# Lyman-alpha escape factor: 8*pi*H / (3 * n_H * (1-x_HII) * lambda_Lya^3)
# LYA_FACT = 4.662899067555897e15 [cm^-3]
_LYA_FACT = 4.662899067555897e15


# ---------------------------------------------------------------------------
# ThermoResult
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ThermoResult:
    """Output of the thermodynamics module."""
    xe_of_loga: CubicSpline
    Tb_of_loga: CubicSpline
    kappa_dot_of_loga: CubicSpline
    exp_m_kappa_of_loga: CubicSpline
    g_of_loga: CubicSpline
    cs2_of_loga: CubicSpline
    z_star: float
    z_rec: float
    tau_star: float
    rs_star: float
    z_reio: float

    def tree_flatten(self):
        return [
            self.xe_of_loga, self.Tb_of_loga,
            self.kappa_dot_of_loga, self.exp_m_kappa_of_loga,
            self.g_of_loga, self.cs2_of_loga,
            self.z_star, self.z_rec, self.tau_star, self.rs_star, self.z_reio,
        ], None

    @classmethod
    def tree_unflatten(cls, aux, fields):
        return cls(*fields)


# ---------------------------------------------------------------------------
# RECFAST Peebles 3-level atom RHS (CLASS/HyRec conventions)
# ---------------------------------------------------------------------------

def _recfast_dxHII_dlna(xe, xHII, nH, H, TM, TR):
    """Peebles 3-level atom: dxHII/d(lna).

    All inputs in CGS: nH [cm^-3], H [s^-1], TM/TR [K].
    Returns dxHII/dlna (dimensionless per e-fold).

    Matches CLASS external/HyRec2020/hydrogen.c:88-109 (rec_TLA_dxHIIdlna)
    with Fudge=1.14 (RECFAST mode).
    """
    # Case-B recombination coefficient: Pequignot et al. (1991) [cm^3/s]
    t4_M = TM / 1e4
    t4_R = TR / 1e4
    t4_M_safe = jnp.maximum(t4_M, 1e-30)
    t4_R_safe = jnp.maximum(t4_R, 1e-30)
    alphaB_TM = _RECFAST_FUDGE * _ALPHA_B_PREFACTOR * t4_M_safe**_ALPHA_B_POWER / (
        1.0 + _ALPHA_B_DENOM_COEFF * t4_M_safe**_ALPHA_B_DENOM_POWER)
    alphaB_TR = _RECFAST_FUDGE * _ALPHA_B_PREFACTOR * t4_R_safe**_ALPHA_B_POWER / (
        1.0 + _ALPHA_B_DENOM_COEFF * t4_R_safe**_ALPHA_B_DENOM_POWER)

    # Temperatures in eV
    TM_eV = _kBoltz_eV * TM
    TR_eV = _kBoltz_eV * TR
    TR_eV_safe = jnp.maximum(TR_eV, 1e-30)

    # Saha factor: s = SAHA_FACT * TR^{3/2} * exp(-EI/TR) / nH
    s = _SAHA_FACT * TR_eV_safe * jnp.sqrt(TR_eV_safe) * jnp.exp(
        -_EI_eV / TR_eV_safe) / jnp.maximum(nH, 1e-30)

    # Photoionization rate: 4*beta_B [s^-1]
    four_betaB = _SAHA_FACT * TR_eV_safe * jnp.sqrt(TR_eV_safe) * jnp.exp(
        -0.25 * _EI_eV / TR_eV_safe) * alphaB_TR

    # Lyman-alpha escape: RLya = LYA_FACT * H / (nH * (1-xHII)) [s^-1]
    x1s = jnp.maximum(1.0 - xHII, 1e-30)
    RLya = _LYA_FACT * H / (jnp.maximum(nH, 1e-30) * x1s)

    # Peebles C factor
    C = (3.0 * RLya + _L2s1s) / (3.0 * RLya + _L2s1s + four_betaB)

    # Saha-departure form (CLASS hydrogen.c:105-107):
    # dxHII/dlna = -(C*nH/H) * [s*(1-x)*(αB_TM - αB_TR) + Δ*αB_TM]
    # where Δ = xe*xHII - s*(1-xHII) is the departure from Saha equilibrium.
    # This form is numerically stable because Δ is O(1) even when the bare
    # rates (alpha*nH, four_betaB) are O(10^5) and would cause stiffness.
    Delta = xe * xHII - s * (1.0 - xHII)
    x1s_safe = jnp.maximum(1.0 - xHII, 1e-30)

    dxHII_dlna = -(C * nH / jnp.maximum(H, 1e-30)) * (
        s * x1s_safe * (alphaB_TM - alphaB_TR) + Delta * alphaB_TM
    )

    return dxHII_dlna


# ---------------------------------------------------------------------------
# Semi-implicit ionization solver (MB95)
# cf. DISCO-EB thermodynamics_mb95.py:ionize()
# ---------------------------------------------------------------------------

def _alpha_B(T_K):
    """Case-B recombination coefficient α_B(T) [cm^3/s].

    Pequignot, Petitjean & Boisson (1991) fitting formula,
    with RECFAST fudge factor F=1.14.

    cf. CLASS external/HyRec2020/hydrogen.c:64-73 (alphaB_PPB)
    """
    t4 = T_K / 1e4  # temperature in units of 10^4 K
    t4_safe = jnp.maximum(t4, 1e-30)
    return _RECFAST_FUDGE * _ALPHA_B_PREFACTOR * t4_safe**_ALPHA_B_POWER / (
        1.0 + _ALPHA_B_DENOM_COEFF * t4_safe**_ALPHA_B_DENOM_POWER
    )


def _ionize(tempb, a, adot, dtau, xe, Y_He, H0, Omega_b):
    """Semi-implicit hydrogen ionization step with Pequignot alpha_B.

    Same MB95 framework (Mpc^-1 code units, semi-implicit stepping) but
    with the Pequignot et al. (1991) case-B recombination coefficient
    and proper Peebles C factor matching CLASS/RECFAST.

    The key change from MB95: replace phi2*alpha0/sqrt(T) with the
    Pequignot formula, and compute C factor with CLASS constants.

    cf. CLASS external/HyRec2020/hydrogen.c:64-73 (alphaB_PPB)
    cf. DISCO-EB thermodynamics_mb95.py:ionize() for stepping framework
    """
    iswitch = 0.5  # semi-implicit

    # --- Recombination coefficient (MB95 original, code units Mpc^-1) ---
    # Recombination coefficient (in sqrt(K)/Mpc)
    # cf. DISCO-EB line 15: alpha0 = 2.3866e-6 * (1-YHe) * Omegab * H0^2
    alpha0 = 2.3866e-6 * (1.0 - Y_He) * Omega_b * H0**2

    # Correction for radiative decay (dimensionless)
    crec = 8.0138e-26 * (1.0 - Y_He) * Omega_b * H0**2

    # Recombination and ionization rates
    _tion = 1.5789e5
    _beta0 = 43.082
    _dec2g = 8.468e14
    phi2 = jnp.maximum(0.448 * jnp.log(_tion / tempb), 0.0)
    alpha = alpha0 / jnp.sqrt(tempb) * phi2 / a**3
    beta = tempb * phi2 * jnp.exp(_beta0 - _tion / tempb)

    # Peebles correction factor
    cp1 = crec * _dec2g * (1.0 - xe) / (a * adot)
    cp2 = crec * tempb * phi2 * jnp.exp(_beta0 - 0.25 * _tion / tempb) * (1.0 - xe) / (a * adot)
    cpeebles = jnp.where(
        tempb <= 200.0,
        1.0,
        (1.0 + cp1) / (1.0 + cp1 + cp2),
    )

    # Semi-implicit step: solve dxe = bb*(1-xe) - aa*xe^2
    aa = a * dtau * alpha * cpeebles
    bb = a * dtau * beta * cpeebles
    b1 = 1.0 + iswitch * bb
    bbxe = bb + xe - (1.0 - iswitch) * (bb * xe + aa * xe * xe)
    rat = iswitch * aa * bbxe / (b1 * b1)

    xe_new = jnp.where(
        rat < 5e-5,
        bbxe / b1 * (1.0 - rat),
        b1 / (2.0 * iswitch * jnp.maximum(aa, 1e-30)) * (jnp.sqrt(jnp.maximum(4.0 * rat + 1.0, 0.0)) - 1.0),
    )
    xe_new = jnp.clip(xe_new, 0.0, 1.0)
    return xe_new


def _ionHe(tempb, a, x0, x1, x2, Y_He, H0, Omega_b):
    """Helium ionization via Saha equation (iterative).

    cf. DISCO-EB thermodynamics_mb95.py:ionHe() lines 47-88
    """
    b0 = 2.150e24 / ((1.0 - Y_He) * Omega_b * H0**2)
    b = b0 * a**3 * tempb * jnp.sqrt(tempb)

    r1 = 4.0 * b * jnp.exp(-_tion1 / tempb)
    r2 = b * jnp.exp(-_tion2 / tempb)

    c = 0.25 * Y_He / (1.0 - Y_He)

    def body_fun(i, vals):
        _, xe, x1, x2 = vals
        xe = x0 + c * (x1 + 2.0 * x2)
        x2new = r1 * r2 / (r1 * r2 + xe * r1 + xe * xe)
        x1 = xe * r1 / (r1 * r2 + xe * r1 + xe * xe)
        err = jnp.abs(x2new - x2)
        return err, xe, x1, x2new

    xe = x0 + c * (x1 + 2.0 * x2)
    out = jax.lax.fori_loop(0, 6, body_fun, (jnp.inf, xe, x1, x2))
    return out[2], out[3]  # x1 (HeII), x2 (HeIII)


# ---------------------------------------------------------------------------
# Reionization (tanh model)
# ---------------------------------------------------------------------------

def _reionization_xe(z, z_reio, Y_He):
    """Tanh reionization, cf. CLASS reio_camb."""
    fHe = Y_He / (4.0 * (1.0 - Y_He))  # simplified He fraction
    xe_after = 1.0 + fHe

    reio_exponent = 1.5
    reio_width = 0.5

    argument = (
        ((1.0 + z_reio) ** reio_exponent - (1.0 + z) ** reio_exponent)
        / (reio_exponent * (1.0 + z_reio) ** (reio_exponent - 1.0))
        / reio_width
    )
    xe_reio = xe_after * (jnp.tanh(argument) + 1.0) / 2.0

    # Helium double reionization at z ~ 3.5
    arg_He = (3.5 - z) / 0.5
    xe_reio += fHe * (jnp.tanh(arg_He) + 1.0) / 2.0

    return xe_reio


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def thermodynamics_solve(
    params: CosmoParams,
    prec: PrecisionParams,
    bg: BackgroundResult,
) -> ThermoResult:
    """Solve the thermodynamics using the MB95 semi-implicit method.

    Args:
        params: cosmological parameters
        prec: precision parameters
        bg: background result from background_solve()

    Returns:
        ThermoResult with all thermodynamic spline tables
    """
    n_thermo = prec.th_n_points
    T_cmb = params.T_cmb
    Y_He = params.Y_He
    H0_kmsMpc = params.h * 100.0
    Omega_b = params.omega_b / params.h**2

    # Conformal time grid (logarithmic spacing)
    # Start from z = th_z_max (not from earliest background point, which is too early
    # and causes numerical instability in the Euler stepping)
    a_start = 1.0 / (1.0 + prec.th_z_max)
    loga_start = jnp.log(a_start)
    # Keep as JAX arrays to allow tracing through for AD
    tau_min_jnp = bg.tau_of_loga.evaluate(loga_start)
    tau_max_jnp = bg.conformal_age
    dlntau = jnp.log(tau_max_jnp / tau_min_jnp) / (n_thermo - 1)

    # Initial conditions (early radiation domination, fully ionized)
    tau0 = tau_min_jnp
    loga0 = loga_start
    a0 = jnp.exp(loga0)
    H0_class = bg.H_of_loga.evaluate(loga0)
    # a' = da/dτ = a^2 * H (since dt = a*dτ and da/dt = aH)
    # cf. DISCO-EB: adot = get_aprimeoa(a) * a where get_aprimeoa returns aH
    adot0 = a0 * a0 * H0_class

    tb0 = T_cmb / a0
    xHII0 = 1.0
    xHeII0 = 0.0
    xHeIII0 = 1.0
    xe0 = xHII0 + 0.25 * Y_He / (1.0 - Y_He) * (xHeII0 + 2.0 * xHeIII0)
    barssc = _barssc_raw * (1.0 - 0.75 * Y_He + (1.0 - Y_He) * xe0)
    cs20 = 4.0 / 3.0 * barssc * tb0

    thomc0 = _thomc0_coeff * T_cmb**4

    # CGS constants for RECFAST (used inside scan_step)
    _H100_cgs = 3.2407792902755e-18  # H0=100 km/s/Mpc in s^-1
    _mH_g = 1.67353284e-24  # proton mass [g]
    _G_cgs = 6.67428e-8
    _c_over_Mpc = const.c_SI / const.Mpc_over_m  # ~9.716e-15 s^-1
    H0_cgs = (H0_kmsMpc / 100.0) * _H100_cgs
    rho_crit_cgs = 3.0 * H0_cgs**2 / (8.0 * math.pi * _G_cgs)
    n_H_0_cgs = (1.0 - Y_He) * Omega_b * rho_crit_cgs / _mH_g

    init = {
        'a': a0, 'adot': adot0, 'tau': tau0, 'tb': tb0,
        'xHII': xHII0, 'xe': xe0, 'xHeII': xHeII0, 'xHeIII': xHeIII0,
        'cs2': cs20,
    }
    keys = ('a', 'adot', 'tau', 'tb', 'xHII', 'xe', 'xHeII', 'xHeIII', 'cs2')

    def scan_step(carry, i):
        a = carry['a']
        adot = carry['adot']
        tau = carry['tau']
        tb = carry['tb']
        xHII = carry['xHII']
        xe = carry['xe']
        xHeII = carry['xHeII']
        xHeIII = carry['xHeIII']

        # New conformal time
        new_tau = tau_min_jnp * jnp.exp(i * dlntau)
        dtau = new_tau - tau

        # Friedmann: advance scale factor (trapezoidal rule)
        new_a = a + adot * dtau
        new_loga = jnp.log(jnp.maximum(new_a, 1e-30))
        new_H = bg.H_of_loga.evaluate(new_loga)
        new_adot = new_a * new_a * new_H  # a' = a^2 * H

        # Trapezoidal refinement
        new_a = a + 2.0 * dtau / (1.0 / adot + 1.0 / new_adot)
        new_loga = jnp.log(jnp.maximum(new_a, 1e-30))
        new_H = bg.H_of_loga.evaluate(new_loga)
        new_adot = new_a * new_a * new_H  # a' = a^2 * H

        # Baryon temperature evolution (Thomson cooling)
        # cf. DISCO-EB thermodynamics_mb95.py:158-177
        tg0 = T_cmb / a  # radiation temperature at current step
        ahalf = 0.5 * (a + new_a)
        adothalf = 0.5 * (adot + new_adot)

        fe = (1.0 - Y_He) * xe / (1.0 - 0.75 * Y_He + (1.0 - Y_He) * xe)
        thomc = thomc0 * fe / adothalf / jnp.maximum(ahalf**3, 1e-30)
        etc = jnp.exp(-thomc * (new_a - a))
        a2t = a**2 * (tb - tg0) * etc - T_cmb / jnp.maximum(thomc, 1e-30) * (1.0 - etc)

        # Taylor expansion for small fe (avoid numerical issues)
        a2t_expansion = (
            (a - new_a) * T_cmb
            + a**2 * (tb - tg0)
            + (0.5 * (a - new_a)**2 * T_cmb + a**2 * (a - new_a) * (tb - tg0)) * thomc
        )
        a2t = jnp.where(fe < 1e-3, a2t_expansion, a2t)

        new_tb = T_cmb / new_a + a2t / new_a**2

        # Ionization step
        tbhalf = 0.5 * (tb + new_tb)

        # RECFAST Peebles ODE in dlna (CLASS-matching coefficients)
        # Convert from dτ stepping to dlna: dlna = (a'/a)*dτ = aH*dτ
        H_half = bg.H_of_loga.evaluate(jnp.log(jnp.maximum(ahalf, 1e-30)))
        dlna_step = ahalf * H_half * dtau
        # n_H in CGS [cm^-3]
        n_H_cgs = n_H_0_cgs / ahalf**3
        # H in CGS [s^-1]
        H_cgs = H_half * _c_over_Mpc
        # Temperatures
        TR_half = T_cmb / ahalf

        # Hydrogen ionization: RECFAST for z < 1600, MB95 for z > 1800,
        # smooth sigmoid blend in between to avoid discontinuities.
        # RECFAST's Saha-departure form is unstable at z > 1600 where
        # the Saha factor s >> 1 causes Delta to be large.
        dxHII_dlna = _recfast_dxHII_dlna(xe, xHII, n_H_cgs, H_cgs, tbhalf, TR_half)
        new_xHII_recfast = jnp.clip(xHII + dlna_step * dxHII_dlna, 0.0, 1.0)
        new_xHII_mb95 = _ionize(tbhalf, ahalf, adothalf, dtau, xHII, Y_He, H0_kmsMpc, Omega_b)

        # Smooth blend: w=1 (RECFAST) at z < 1600, w=0 (MB95) at z > 1800
        z_half = 1.0 / ahalf - 1.0
        w_recfast = jax.nn.sigmoid(-0.02 * (z_half - 1700.0))  # smooth over Δz~100
        new_xHII = w_recfast * new_xHII_recfast + (1.0 - w_recfast) * new_xHII_mb95

        # Helium (Saha iteration)
        new_xHeII, new_xHeIII = _ionHe(
            new_tb, new_a, new_xHII, xHeII, xHeIII, Y_He, H0_kmsMpc, Omega_b
        )

        # Total ionization fraction
        new_xe = new_xHII + 0.25 * Y_He / (1.0 - Y_He) * (new_xHeII + 2.0 * new_xHeIII)

        # Sound speed squared (over c^2)
        dtbdla = -2.0 * new_tb - thomc * a2t / new_a
        barssc = _barssc_raw * (1.0 - 0.75 * Y_He + (1.0 - Y_He) * new_xe)
        new_cs2 = barssc * new_tb * (1.0 - dtbdla / new_tb / 3.0)

        new_carry = {
            'a': new_a, 'adot': new_adot, 'tau': new_tau, 'tb': new_tb,
            'xHII': new_xHII, 'xe': new_xe, 'xHeII': new_xHeII,
            'xHeIII': new_xHeIII, 'cs2': new_cs2,
        }
        return new_carry, new_carry

    _, out = jax.lax.scan(scan_step, init, jnp.arange(1, n_thermo))

    # Prepend initial values
    def prepend(k):
        return jnp.concatenate([jnp.atleast_1d(init[k]), out[k]])

    a_grid = prepend('a')
    tau_grid = prepend('tau')
    tb_grid = prepend('tb')
    xe_raw_grid = prepend('xe')
    cs2_grid = prepend('cs2')
    loga_grid = jnp.log(jnp.maximum(a_grid, 1e-30))

    # --- Reionization ---
    z_grid = 1.0 / a_grid - 1.0
    z_reio = _estimate_z_reio(params.tau_reio)

    xe_reio_grid = jax.vmap(lambda z: _reionization_xe(z, z_reio, Y_He))(z_grid)
    xe_grid = jnp.maximum(xe_raw_grid, xe_reio_grid)

    # --- Derived quantities ---

    # Thomson scattering rate κ' [Mpc^-1]
    # n_e = x_e * n_H(z) = x_e * n_H_0 * (1+z)^3
    # κ'(τ) = n_e * σ_T * a * c converted to Mpc^-1
    mu_H = 1.0 / (1.0 - Y_He)
    _bigH = 3.2407792902755102e-18  # H0=100 km/s/Mpc in s^-1
    n_H_0 = 3.0 * (_bigH * params.h)**2 / (8.0 * math.pi * const.G_SI * 1.67353284e-27 * mu_H) * Omega_b
    kappa_dot_grid = xe_grid * n_H_0 * (1.0 + z_grid)**2 * const.sigma_T * const.Mpc_over_m

    # Optical depth: κ = ∫_τ^τ_0 κ'(τ') dτ' (integrate backwards from today)
    # Use trapezoidal integration on the tau grid
    dtau_grid = jnp.diff(tau_grid)
    kappa_integrand = 0.5 * (kappa_dot_grid[:-1] + kappa_dot_grid[1:]) * dtau_grid
    kappa_cumulative = jnp.cumsum(kappa_integrand[::-1])[::-1]
    kappa_grid = jnp.concatenate([kappa_cumulative, jnp.array([0.0])])

    exp_m_kappa_grid = jnp.exp(-kappa_grid)
    g_grid = kappa_dot_grid * exp_m_kappa_grid

    # --- Find z_star and z_rec ---
    idx_star = jnp.argmax(g_grid)
    z_star = z_grid[idx_star]
    tau_star = bg.tau_of_loga.evaluate(loga_grid[idx_star])
    rs_star = bg.rs_of_loga.evaluate(loga_grid[idx_star])

    idx_rec = jnp.argmin(jnp.abs(kappa_grid - 1.0))
    z_rec = z_grid[idx_rec]

    # --- Build splines on loga grid ---
    # Need to sort by loga (which is increasing as a increases)
    xe_of_loga = CubicSpline(loga_grid, xe_grid)
    Tb_of_loga = CubicSpline(loga_grid, tb_grid)
    kappa_dot_of_loga = CubicSpline(loga_grid, kappa_dot_grid)
    exp_m_kappa_of_loga = CubicSpline(loga_grid, exp_m_kappa_grid)
    g_of_loga = CubicSpline(loga_grid, g_grid)
    cs2_of_loga = CubicSpline(loga_grid, cs2_grid)

    return ThermoResult(
        xe_of_loga=xe_of_loga,
        Tb_of_loga=Tb_of_loga,
        kappa_dot_of_loga=kappa_dot_of_loga,
        exp_m_kappa_of_loga=exp_m_kappa_of_loga,
        g_of_loga=g_of_loga,
        cs2_of_loga=cs2_of_loga,
        z_star=z_star,
        z_rec=z_rec,
        tau_star=tau_star,
        rs_star=rs_star,
        z_reio=z_reio,
    )


def _estimate_z_reio(tau_reio_target):
    """Rough estimate of z_reio from tau_reio."""
    return jnp.clip(2.0 + 150.0 * tau_reio_target, 4.0, 30.0)


# Convenience
def xe_of_z(th: ThermoResult, z: float) -> float:
    loga = jnp.log(1.0 / (1.0 + z))
    return th.xe_of_loga.evaluate(loga)
