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
# All constants match CLASS wrap_recfast.c exactly (CODATA 2006 values).
# cf. CLASS external/RecfastCLASS/wrap_recfast.{c,h}

# Fundamental constants (CLASS values from common.h, thermodynamics.h)
_hP_SI = 6.62606896e-34      # Planck constant [J·s]
_c_SI = 2.99792458e8         # Speed of light [m/s]
_kB_SI = 1.3806504e-23       # Boltzmann constant [J/K]
_me_SI = 9.10938215e-31      # Electron mass [kg]
# CGS versions
_hP_CGS = _hP_SI * 1e7       # [erg·s]
_kB_CGS = _kB_SI * 1e7       # [erg/K]
_me_CGS = _me_SI * 1e3       # [g]

# Inverse wavenumbers (CLASS wrap_recfast.h)
_L_H_ion = 1.096787737e7     # H ionization [m^{-1}]
_L_H_alpha = 8.225916453e6   # H Lyman-alpha [m^{-1}]

# Derived RECFAST constants (CLASS wrap_recfast.c:68-89)
_Lalpha_m = 1.0 / _L_H_alpha                # Lyman-alpha wavelength [m]
_Lalpha_cm = _Lalpha_m * 100.0               # Lyman-alpha wavelength [cm]
_CDB = _hP_SI * _c_SI * (_L_H_ion - _L_H_alpha) / _kB_SI  # 39,462 K (n=2 ionization temp)
_CB1 = _hP_SI * _c_SI * _L_H_ion / _kB_SI                 # 157,807 K (ground-state ionization temp, = CDB + CL)
_CL = _hP_SI * _c_SI * _L_H_alpha / _kB_SI                # 118,348 K (Lyman-alpha temp)
_CK_CGS = _Lalpha_cm**3 / (8.0 * math.pi)  # Peebles K prefactor [cm^3]
_CR_CGS = 2.0 * math.pi * (_me_CGS / _hP_CGS) * (_kB_CGS / _hP_CGS)  # NR number density [K^{-1} cm^{-2}]
_A2s1sH = 8.2245809                         # Einstein 2s→1s coefficient [s^{-1}]

# Pequignot et al. (1991) case-B recombination coefficient (NO fudge factor)
# alpha_B(T) = 4.309e-13 * t4^(-0.6166) / (1 + 0.6703 * t4^0.5300) [cm^3/s]
_ALPHA_B_PREFACTOR = 4.309e-13  # cm^3 s^-1
_ALPHA_B_POWER = -0.6166
_ALPHA_B_DENOM_COEFF = 0.6703
_ALPHA_B_DENOM_POWER = 0.5300

# RECFAST fudge factors (CLASS precisions.h:183-192)
# When Hswitch=True (default): fudge_H = 1.14 + delta = 1.14 - 0.015 = 1.125
_RECFAST_FUDGE_H = 1.14 + (-0.015)   # = 1.125 (RECFAST 1.5.2 with Hswitch)
_RECFAST_FUDGE = _RECFAST_FUDGE_H    # For backward compat with _ionize
_RECFAST_X_H0_TRIGGER2 = 0.995       # Peebles C activation threshold

# Gaussian K correction parameters (RECFAST 1.5/1.5.2, Hswitch=True)
# cf. CLASS precisions.h:187-192
_AGauss1 = -0.14
_AGauss2 = 0.079
_zGauss1 = 7.28    # in ln(1+z)
_zGauss2 = 6.73    # in ln(1+z)
_wGauss1 = 0.18
_wGauss2 = 0.33

# Legacy constants kept for MB95 helium Saha / _ionize
_EI_eV = 13.598286071938324     # H ionization energy [eV]
_kBoltz_eV = 8.617343e-5        # Boltzmann constant [eV/K]
_L2s1s = _A2s1sH                # Alias
_lambda_Lya_cm = _Lalpha_cm     # Alias
_SAHA_FACT = 3.016103031869581e21  # HyRec Saha factor [eV^{-3/2} cm^{-3}]
_LYA_FACT = 4.662899067555897e15   # Lyman-alpha escape factor [cm^{-3}]


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
    g_prime_of_loga: CubicSpline  # dg/dτ, computed analytically
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
            self.g_of_loga, self.g_prime_of_loga, self.cs2_of_loga,
            self.z_star, self.z_rec, self.tau_star, self.rs_star, self.z_reio,
        ], None

    @classmethod
    def tree_unflatten(cls, aux, fields):
        return cls(*fields)


# ---------------------------------------------------------------------------
# RECFAST Peebles 3-level atom RHS (CLASS/HyRec conventions)
# ---------------------------------------------------------------------------

def _recfast_dxHII_dlna(xe, xHII, nH, Hz, z, TM, TR):
    """Peebles 3-level atom: dxHII/d(lna).

    Matches CLASS external/RecfastCLASS/wrap_recfast.c:110-174 exactly
    (recfast_dx_H_dz), converted from dz to dlna.

    Key differences from previous HyRec-style version:
    1. Fudge factor F=1.125 is in the Peebles C coefficient, NOT in alpha_B
    2. Gaussian K correction (RECFAST 1.5, Hswitch=True)
    3. Photoionization uses Tmat (CLASS default: recfast_photoion_Tmat)

    All inputs in CGS: nH [cm^-3], Hz [s^-1], TM/TR [K], z dimensionless.
    Returns dxHII/dlna (dimensionless per e-fold).
    """
    # --- Case-B recombination coefficient alpha_B(Tmat) — NO fudge ---
    # cf. wrap_recfast.c:131
    t4_M = TM / 1e4
    t4_M_safe = jnp.maximum(t4_M, 1e-30)
    Rdown = _ALPHA_B_PREFACTOR * t4_M_safe**_ALPHA_B_POWER / (
        1.0 + _ALPHA_B_DENOM_COEFF * t4_M_safe**_ALPHA_B_DENOM_POWER)

    # --- Photoionization rate Rup (Tmat mode, CLASS default) ---
    # cf. wrap_recfast.c:133-134
    # Rup = Rdown * (CR*Tmat)^{3/2} * exp(-CDB/Tmat) [s^{-1}]
    TM_safe = jnp.maximum(TM, 1e-30)
    Rup = Rdown * (_CR_CGS * TM_safe)**1.5 * jnp.exp(-_CDB / TM_safe)

    # --- K factor with Gaussian correction (RECFAST 1.5, Hswitch=True) ---
    # cf. wrap_recfast.c:141-149
    Hz_safe = jnp.maximum(Hz, 1e-30)
    K = _CK_CGS / Hz_safe
    # Gaussian correction from Rubino-Martin et al. (2010)
    lnz1 = jnp.log(1.0 + z)
    K = K * (1.0
             + _AGauss1 * jnp.exp(-((lnz1 - _zGauss1) / _wGauss1)**2)
             + _AGauss2 * jnp.exp(-((lnz1 - _zGauss2) / _wGauss2)**2))

    # --- Peebles C factor with fudge ---
    # cf. wrap_recfast.c:161-171
    # C = F * (1 + K*A*n_1s) / (1 + K*A*n_1s + F*K*Rup*n_1s)
    # where F = fudge_H = 1.125, A = A2s1sH = 8.2245809
    n_1s = jnp.maximum(nH * (1.0 - xHII), 1e-30)
    KAn = K * _A2s1sH * n_1s
    KRn = K * Rup * n_1s
    C_full = _RECFAST_FUDGE_H * (1.0 + KAn) / jnp.maximum(
        1.0 + KAn + _RECFAST_FUDGE_H * KRn, 1e-30)

    # C = 1 when still fully ionized (x_H >= trigger2 AND z >= z_switch_late)
    # cf. wrap_recfast.c:164
    C = jnp.where((xHII < _RECFAST_X_H0_TRIGGER2) | (z < 800.0), C_full, 1.0)

    # --- ODE: dxH/dz = (x*xH*nH*Rdown - Rup*(1-xH)*exp(-CL/Tmat)) * C / (Hz*(1+z)) ---
    # Convert to dlna: dxH/dlna = dxH/dz * (-(1+z))
    # = -(x*xH*nH*Rdown - Rup*(1-xH)*exp(-CL/Tmat)) * C / Hz
    # cf. wrap_recfast.c:174
    dxHII_dlna = -(
        xe * xHII * nH * Rdown
        - Rup * (1.0 - xHII) * jnp.exp(-_CL / TM_safe)
    ) * C / Hz_safe

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

        z_half = 1.0 / ahalf - 1.0

        # --- Heun's method (predictor-corrector, 2nd order) for RECFAST ---
        # Step 1: Evaluate rate at midpoint (predictor)
        dxHII_1 = _recfast_dxHII_dlna(
            xe, xHII, n_H_cgs, H_cgs, z_half, tbhalf, TR_half)
        xHII_pred = jnp.clip(xHII + dlna_step * dxHII_1, 0.0, 1.0)

        # Step 2: Evaluate rate at predicted state using end-of-step quantities
        z_new = 1.0 / new_a - 1.0
        nH_new = n_H_0_cgs / new_a**3
        H_new = new_H * _c_over_Mpc
        TR_new = T_cmb / new_a
        xe_pred = xHII_pred + 0.25 * Y_He / (1.0 - Y_He) * (xHeII + 2.0 * xHeIII)
        dxHII_2 = _recfast_dxHII_dlna(
            xe_pred, xHII_pred, nH_new, H_new, z_new, new_tb, TR_new)

        # Step 3: Corrector (trapezoidal average)
        new_xHII_recfast = jnp.clip(
            xHII + 0.5 * dlna_step * (dxHII_1 + dxHII_2), 0.0, 1.0)

        # For z > 1600: hydrogen Saha equilibrium (CLASS thermodynamics.c:4074-4081).
        # x_H*(x_H + xHeII)/(1 - x_H) = rhs, solved via quadratic formula.
        # rhs = (CR*Tmat)^{3/2} * exp(-CB1/Tmat) / nH_physical
        # where CB1 = hc*L_H_ion/kB ~ 157807 K (ground-state ionization temp).
        T_saha = jnp.maximum(tbhalf, 100.0)  # clamp to avoid exp(-huge)
        nH_saha = n_H_0_cgs / ahalf**3
        # Clamp exponent to avoid underflow producing 0*inf NaN in gradients
        saha_exp = jnp.exp(jnp.maximum(-_CB1 / T_saha, -500.0))
        saha_rhs = jnp.maximum(
            (_CR_CGS * T_saha)**1.5 * saha_exp / nH_saha, 1e-300)
        # Helium electron contribution per H nucleus
        xHeII_contrib = 0.25 * Y_He / (1.0 - Y_He) * xHeII
        # Quadratic: x_H = 2/(1 + xHeII/rhs + sqrt((1+xHeII/rhs)^2 + 4/rhs))
        inv_rhs = jnp.minimum(1.0 / saha_rhs, 1e100)  # clamp to avoid sqrt overflow
        v_over_rhs = xHeII_contrib * inv_rhs
        xHII_saha = 2.0 / (1.0 + v_over_rhs + jnp.sqrt(
            (1.0 + v_over_rhs)**2 + 4.0 * inv_rhs))
        # Stop gradient through Saha: at z>1600, x_HII ≈ 1 and the exact value
        # barely affects C_l. Gradients should flow through the RECFAST ODE
        # (z<1600) where recombination physics actually matters.
        xHII_saha = jax.lax.stop_gradient(xHII_saha)

        # Below z~1600: use RECFAST Peebles ODE (accurate through recombination).
        # Above z~1600: use Saha equilibrium (hydrogen fully ionized).
        # Smooth sigmoid blend for differentiability (width=50, matching CLASS delta_z).
        w_saha = jax.nn.sigmoid(0.1 * (z_half - 1600.0))
        new_xHII = w_saha * xHII_saha + (1.0 - w_saha) * new_xHII_recfast

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

    # --- Derived quantities (n_H_0, kappa_dot prefactor) ---
    z_grid = 1.0 / a_grid - 1.0
    mu_H = 1.0 / (1.0 - Y_He)
    _bigH = 3.2407792902755102e-18  # H0=100 km/s/Mpc in s^-1
    n_H_0 = 3.0 * (_bigH * params.h)**2 / (8.0 * math.pi * const.G_SI * 1.67353284e-27 * mu_H) * Omega_b
    # kappa_dot prefactor: kappa_dot(z) = xe * n_H_0 * (1+z)^2 * sigma_T * c/Mpc
    kd_prefactor = n_H_0 * (1.0 + z_grid)**2 * const.sigma_T * const.Mpc_over_m
    dtau_grid = jnp.diff(tau_grid)

    # --- Reionization: find z_reio self-consistently to match tau_reio ---
    # Step 1: Compute kappa from recombination only (xe_raw)
    kd_raw = xe_raw_grid * kd_prefactor
    kappa_raw_integ = 0.5 * (kd_raw[:-1] + kd_raw[1:]) * dtau_grid
    kappa_raw_total = jnp.sum(kappa_raw_integ)  # total optical depth without reionization

    # Step 2: Find z_reio such that the reionization contribution = tau_reio - kappa_raw_residual
    # We use a scan over candidate z_reio values and select the one giving the right tau.
    # The reionization optical depth for a given z_reio is:
    #   tau_reio_model(z_reio) = ∫ max(xe_reio(z,z_reio) - xe_raw(z), 0) * kd_prefactor * dtau
    # We precompute this for several z_reio candidates and interpolate.
    z_reio = _find_z_reio(
        params.tau_reio, xe_raw_grid, kd_prefactor, dtau_grid,
        z_grid, Y_He, kappa_raw_total)

    xe_reio_grid = jax.vmap(lambda z: _reionization_xe(z, z_reio, Y_He))(z_grid)
    xe_grid = jnp.maximum(xe_raw_grid, xe_reio_grid)

    # --- Optical depth ---
    kappa_dot_grid = xe_grid * kd_prefactor

    # κ = ∫_τ^τ_0 κ'(τ') dτ' (integrate backwards from today)
    kappa_integrand = 0.5 * (kappa_dot_grid[:-1] + kappa_dot_grid[1:]) * dtau_grid
    kappa_cumulative = jnp.cumsum(kappa_integrand[::-1])[::-1]
    kappa_grid = jnp.concatenate([kappa_cumulative, jnp.array([0.0])])

    exp_m_kappa_grid = jnp.exp(-kappa_grid)
    g_grid = kappa_dot_grid * exp_m_kappa_grid

    # --- g' = dg/dτ analytically (CLASS thermodynamics.c:3482-3483) ---
    # g = κ̇ e^{-κ},  g' = (κ̈ + κ̇²) e^{-κ}
    # where κ̈ = d(κ̇)/dτ.
    # Compute κ̈ using spline derivative for accuracy (not finite differences).
    # Build a temporary spline of κ̇(loga), then evaluate its derivative.
    # dκ̇/dτ = (dκ̇/d(loga)) * (d(loga)/dτ) = (dκ̇/d(loga)) * (a'/a) / a
    # But a'/a = aH, so d(loga)/dτ = (1/a)(da/dτ) = H (physical Hubble, not conformal).
    # Actually: d(loga)/dτ = d(ln a)/dτ = (1/a)(da/dτ) = a'/a² ... no.
    # loga = ln(a), d(loga)/dτ = (da/dτ)/a = a'/a = aH (conformal Hubble).
    # Wait: a' = da/dτ (conformal time), so d(ln a)/dτ = a'/a = aH. Yes.
    # So dκ̇/dτ = (dκ̇/d(loga)) * aH  where aH = a'(τ)/a(τ)
    kd_spline_tmp = CubicSpline(loga_grid, kappa_dot_grid)
    dkd_dloga_grid = jax.vmap(kd_spline_tmp.derivative)(loga_grid)
    # a'/a = aH at each grid point
    a_grid_loc = jnp.exp(loga_grid)
    H_grid_loc = jax.vmap(bg.H_of_loga.evaluate)(loga_grid)
    aH_grid = a_grid_loc * H_grid_loc
    ddkappa_grid = dkd_dloga_grid * aH_grid
    g_prime_grid = (ddkappa_grid + kappa_dot_grid**2) * exp_m_kappa_grid

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
    g_prime_of_loga = CubicSpline(loga_grid, g_prime_grid)
    cs2_of_loga = CubicSpline(loga_grid, cs2_grid)

    return ThermoResult(
        xe_of_loga=xe_of_loga,
        Tb_of_loga=Tb_of_loga,
        kappa_dot_of_loga=kappa_dot_of_loga,
        exp_m_kappa_of_loga=exp_m_kappa_of_loga,
        g_of_loga=g_of_loga,
        g_prime_of_loga=g_prime_of_loga,
        cs2_of_loga=cs2_of_loga,
        z_star=z_star,
        z_rec=z_rec,
        tau_star=tau_star,
        rs_star=rs_star,
        z_reio=z_reio,
    )


def _find_z_reio(tau_reio_target, xe_raw_grid, kd_prefactor, dtau_grid,
                 z_grid, Y_He, kappa_raw_total):
    """Find z_reio such that the reionization optical depth matches tau_reio.

    The reionization optical depth is the EXTRA optical depth from
    reionization above the recombination baseline:
        tau_reio = ∫ max(xe_reio - xe_raw, 0) * kd_prefactor * dtau

    Uses bisection (20 iterations → accuracy ~0.001 in z_reio).
    Memory-efficient: only one _reionization_xe evaluation per iteration.
    """
    def _tau_reio_for_zreio(z_reio_cand):
        """Compute reionization-only optical depth for a given z_reio."""
        xe_reio = jax.vmap(lambda z: _reionization_xe(z, z_reio_cand, Y_He))(z_grid)
        xe_extra = jnp.maximum(xe_reio - xe_raw_grid, 0.0)
        kd_extra = xe_extra * kd_prefactor
        kappa_integ = 0.5 * (kd_extra[:-1] + kd_extra[1:]) * dtau_grid
        return jnp.sum(kappa_integ)

    # Bisection: find z_reio in [4, 25] where tau_reio = tau_reio_target
    def bisect_step(carry, _):
        z_lo, z_hi = carry
        z_mid = 0.5 * (z_lo + z_hi)
        tau_mid = _tau_reio_for_zreio(z_mid)
        # If tau_mid < target, need higher z_reio (more reionization)
        z_lo = jnp.where(tau_mid < tau_reio_target, z_mid, z_lo)
        z_hi = jnp.where(tau_mid < tau_reio_target, z_hi, z_mid)
        return (z_lo, z_hi), None

    (z_lo, z_hi), _ = jax.lax.scan(bisect_step, (4.0, 25.0), jnp.arange(20))
    return 0.5 * (z_lo + z_hi)


def _estimate_z_reio(tau_reio_target):
    """Rough estimate of z_reio from tau_reio (legacy, kept for reference)."""
    return jnp.clip(2.0 + 150.0 * tau_reio_target, 4.0, 30.0)


# Convenience
def xe_of_z(th: ThermoResult, z: float) -> float:
    loga = jnp.log(1.0 / (1.0 + z))
    return th.xe_of_loga.evaluate(loga)
