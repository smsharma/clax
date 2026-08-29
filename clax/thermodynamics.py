"""Thermodynamics module for clax.

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

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from clax import constants as const
from clax.background import BackgroundResult
from clax.interpolation import CubicSpline
from clax.params import CosmoParams, PrecisionParams

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

# --- RECFAST/CLASS helium recombination constants ---
# CLASS wrap_recfast.h uses SI units (m^3/s) for rates. We convert to CGS (*1e6).
_L_He1_ion = 1.98310772e7     # HeI ionization [m^{-1}]
_L_He_2s = 1.66277434e7       # He 2s [m^{-1}]
_L_He_2p = 1.71134891e7       # He 2p [m^{-1}]
_Lalpha_He_cm = 100.0 / _L_He_2p  # He Lyman-alpha wavelength [cm]
_CDB_He = _hP_SI * _c_SI * (_L_He1_ion - _L_He_2s) / _kB_SI   # He 2s ionization [K]
_CL_He = _c_SI * _hP_SI * _L_He_2s / _kB_SI                    # He 2s energy [K]
_CK_He_CGS = _Lalpha_He_cm**3 / (8.0 * math.pi)                # He K prefactor [cm^3]
_CDB_He2s2p = _hP_SI * _c_SI * (_L_He_2p - _L_He_2s) / _kB_SI  # 2p-2s splitting [K]
_Lambda_He = 51.3              # He 2s→1s transition rate [s^{-1}]
_A2P_s_He = 1.798287e9        # He 2^1P singlet radiative rate [s^{-1}]
_sigma_He_2Ps = 1.436289e-22  # cross section for 2^1P He singlet [m^2] → [cm^2] = 1.436289e-18
_sigma_He_2Ps_cgs = 1.436289e-22 * 1e4  # [cm^2]
_not4 = 3.9715               # mHe / mH (not exactly 4)
_RECFAST_FUDGE_He = 0.86     # helium fudge factor (CLASS default)
# Verner & Ferland (1996) He case-B: convert from SI (m^3/s) to CGS (cm^3/s)
# CLASS: _a_VF_ = 10^{-16.744} [m^3/s]; CGS = SI * 1e6
_a_VF_He_CGS = 10.0**(-16.744) * 1e6  # ≈ 1.803e-11 cm^3/s
_b_VF_He = 0.711
_T_0_He = 10.0**0.477121      # ≈ 3.0 K
_T_1_He = 10.0**5.114         # ≈ 1.3e5 K

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
    dkappa_dot_dloga_of_loga: CubicSpline
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
            self.kappa_dot_of_loga, self.dkappa_dot_dloga_of_loga,
            self.exp_m_kappa_of_loga,
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


def _recfast_dxHe_dlna(xe, xHe, nH, Hz, z, TM, TR, fHe):
    """Helium Peebles equation: dxHe/d(lna).

    cf. CLASS wrap_recfast.c:198-349 (Heflag=0 mode, simplest K_He).
    All inputs in CGS. Returns dxHe/dlna.
    """
    TM_safe = jnp.maximum(TM, 1e-30)
    Hz_safe = jnp.maximum(Hz, 1e-30)
    n_He = fHe * nH

    # Verner & Ferland case-B He recombination (CGS)
    sq_0 = jnp.sqrt(TM_safe / _T_0_He)
    sq_1 = jnp.sqrt(TM_safe / _T_1_He)
    Rdown_He = _a_VF_He_CGS / (sq_0 * (1.0 + sq_0)**(1.0 - _b_VF_He)
                                * (1.0 + sq_1)**(1.0 + _b_VF_He))

    # Photoionization (detailed balance, Tmat mode)
    Rup_He = 4.0 * Rdown_He * (_CR_CGS * TM_safe)**1.5 * jnp.exp(
        jnp.maximum(-_CDB_He / TM_safe, -500.0))

    # K_He with Sobolev escape probability (Heflag>=2 in CLASS)
    # cf. wrap_recfast.c:259-274
    # Basic: K_He = CK_He / Hz
    # Sobolev: τ_s = A2P_s * CK_He * 3 * n_He * (1-xHe) / Hz
    #          p_s = (1 - exp(-τ_s)) / τ_s
    #          K_He = 1 / (A2P_s * p_s * 3 * n_He * (1-xHe))
    n_1s_He = jnp.maximum(n_He * (1.0 - xHe), 1e-30)
    # K_He with Sobolev escape (CLASS Heflag=0)
    # cf. wrap_recfast.c:254-260
    tauHe_s = _A2P_s_He * _CK_He_CGS * 3.0 * n_1s_He / Hz_safe
    tauHe_s_safe = jnp.maximum(tauHe_s, 1e-30)
    pHe_s = (1.0 - jnp.exp(-tauHe_s_safe)) / tauHe_s_safe
    K_He = 1.0 / jnp.maximum(_A2P_s_He * pHe_s * 3.0 * n_1s_He, 1e-30)

    # Peebles C_He with Boltzmann 2s-2p factor
    # Reformulated as (inv_A + Lambda) / (inv_A + Lambda + Rup) where A = K*n*B,
    # to avoid inf-inf NaN in JVP when He_Boltz = exp(CDB/T) is huge at low T.
    He_Boltz = jnp.exp(jnp.minimum(_CDB_He2s2p / TM_safe, 500.0))
    A_He = K_He * n_1s_He * He_Boltz
    inv_A_He = 1.0 / jnp.maximum(A_He, 1e-300)
    C_He = (inv_A_He + _Lambda_He) / (inv_A_He + _Lambda_He + Rup_He)
    C_He = jnp.where((xHe < 1e-15) | (xHe > 0.999), 1.0, C_He)

    dxHe_dlna = -(
        xe * xHe * nH * Rdown_He
        - Rup_He * (1.0 - xHe) * jnp.exp(jnp.maximum(-_CL_He / TM_safe, -500.0))
    ) * C_He / Hz_safe

    return dxHe_dlna


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

_NOT4 = 3.9715  # He-4/H mass ratio, cf. CLASS thermodynamics.h:707


def _reionization_xe_fraction(z, z_reio):
    """Compute the reionization FRACTION (0 to 1) from tanh profile.

    Returns the fraction of full reionization completed at redshift z.
    cf. CLASS thermodynamics.c: reio_camb tanh formula.
    """
    reio_exponent = 1.5
    reio_width = 0.5

    argument = (
        ((1.0 + z_reio) ** reio_exponent - (1.0 + z) ** reio_exponent)
        / (reio_exponent * (1.0 + z_reio) ** (reio_exponent - 1.0))
        / reio_width
    )
    return (jnp.tanh(argument) + 1.0) / 2.0


def _reionization_xe(z, z_reio, Y_He, xe_before=0.0):
    """Tanh reionization following CLASS reio_camb convention.

    CLASS formula: xe = xe_before + (xe_after - xe_before) * tanh_fraction
    This smoothly transitions from xe_before to xe_after, avoiding the
    kink from max(xe_raw, xe_reio).

    cf. CLASS thermodynamics.c: thermodynamics_reionization_function()
    """
    # fHe = n_He/n_H; CLASS uses _not4_=3.9715 instead of exactly 4
    # cf. CLASS thermodynamics.c:1207
    fHe = Y_He / (_NOT4 * (1.0 - Y_He))
    xe_after = 1.0 + fHe  # H + singly-ionized He

    # Hydrogen reionization (CLASS reio_camb)
    frac_H = _reionization_xe_fraction(z, z_reio)
    xe_reio = xe_before + (xe_after - xe_before) * frac_H

    # Helium double reionization at z ~ 3.5
    # cf. CLASS thermodynamics.c:1338-1358
    arg_He = (3.5 - z) / 0.5
    frac_He = (jnp.tanh(arg_He) + 1.0) / 2.0
    xe_reio += fHe * frac_He

    return xe_reio


@jax.custom_jvp
def _solve_hydrogen_saha(rhs, xHeII_contrib):
    """Solve the hydrogen Saha quadratic for ``x_HII``.

    The primal uses the stable positive-root form of

        x_HII^2 + (xHeII_contrib + rhs) * x_HII - rhs = 0.

    The JVP follows from implicit differentiation of this polynomial rather
    than differentiating the closed-form root expression directly, which is
    numerically fragile in the fully ionized Saha regime.
    """
    discriminant = jnp.sqrt((rhs + xHeII_contrib) ** 2 + 4.0 * rhs)
    return 2.0 * rhs / (rhs + xHeII_contrib + discriminant)


@_solve_hydrogen_saha.defjvp
def _solve_hydrogen_saha_jvp(primals, tangents):
    rhs, xHeII_contrib = primals
    drhs, dxHeII_contrib = tangents
    xHII = _solve_hydrogen_saha(rhs, xHeII_contrib)
    dF_dx = jnp.maximum(2.0 * xHII + xHeII_contrib + rhs, 1e-30)
    xHII_tangent = ((1.0 - xHII) * drhs - xHII * dxHeII_contrib) / dF_dx
    return xHII, xHII_tangent


def _first_derivative_table(x, y):
    """Return a differentiable first-derivative table on a monotonic grid.

    Uses centered differences in the interior and one-sided differences at the
    boundaries. On the dense thermodynamics grid this is sufficiently accurate
    for the opacity-derivative quantities consumed by perturbations while
    avoiding the unstable parameter-AD of differentiating the cubic-spline
    coefficients directly.
    """
    dydx = jnp.empty_like(y)
    dydx = dydx.at[0].set((y[1] - y[0]) / (x[1] - x[0]))
    dydx = dydx.at[-1].set((y[-1] - y[-2]) / (x[-1] - x[-2]))
    dydx = dydx.at[1:-1].set((y[2:] - y[:-2]) / (x[2:] - x[:-2]))
    return dydx


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def _thermodynamics_solve_impl(
    params: CosmoParams,
    prec: PrecisionParams,
    bg: BackgroundResult,
) -> ThermoResult:
    """Undecorated thermodynamics solver body (MB95 semi-implicit method).

    This is the single implementation behind the public
    ``thermodynamics_solve`` wrapper below.  It is deliberately left without
    ``jax.jit`` or any custom differentiation rule so that

    * the ``th_grad_mode="stable"`` path can differentiate it in FORWARD mode
      (``jax.jacfwd``) inside the custom VJP's backward pass, and
    * the ``th_grad_mode="native"`` path exposes plain JAX derivatives for
      forward-mode users (``jax.jvp``).

    Args:
        params: cosmological parameters
        prec: precision parameters (static)
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

        # --- RK4 (4th order) for hydrogen RECFAST ---
        # Heun (2nd order) leaves ~0.15% x_e residual. RK4 converges as O(h^4).
        z_new = 1.0 / new_a - 1.0
        nH_new = n_H_0_cgs / new_a**3
        H_new = new_H * _c_over_Mpc
        TR_new = T_cmb / new_a

        k1 = _recfast_dxHII_dlna(xe, xHII, n_H_cgs, H_cgs, z_half, tbhalf, TR_half)
        xHII_2 = jnp.clip(xHII + 0.5 * dlna_step * k1, 0.0, 1.0)
        xe_2 = xHII_2 + 0.25 * Y_He / (1.0 - Y_He) * (xHeII + 2.0 * xHeIII)
        k2 = _recfast_dxHII_dlna(xe_2, xHII_2, n_H_cgs, H_cgs, z_half, tbhalf, TR_half)
        xHII_3 = jnp.clip(xHII + 0.5 * dlna_step * k2, 0.0, 1.0)
        xe_3 = xHII_3 + 0.25 * Y_He / (1.0 - Y_He) * (xHeII + 2.0 * xHeIII)
        k3 = _recfast_dxHII_dlna(xe_3, xHII_3, n_H_cgs, H_cgs, z_half, tbhalf, TR_half)
        xHII_4 = jnp.clip(xHII + dlna_step * k3, 0.0, 1.0)
        xe_4 = xHII_4 + 0.25 * Y_He / (1.0 - Y_He) * (xHeII + 2.0 * xHeIII)
        k4 = _recfast_dxHII_dlna(xe_4, xHII_4, nH_new, H_new, z_new, new_tb, TR_new)
        new_xHII_recfast = jnp.clip(
            xHII + dlna_step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)

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
        xHII_saha = _solve_hydrogen_saha(saha_rhs, xHeII_contrib)
        # Below z~1600: use RECFAST Peebles ODE (accurate through recombination).
        # Above z~1600: use Saha equilibrium (hydrogen fully ionized).
        # Smooth sigmoid blend for differentiability (width=50, matching CLASS delta_z).
        w_saha = jax.nn.sigmoid(0.1 * (z_half - 1600.0))
        new_xHII = w_saha * xHII_saha + (1.0 - w_saha) * new_xHII_recfast

        # Helium: Peebles ODE for HeII→HeI (replaces Saha at z<5000)
        # HeIII→HeII uses Saha (accurate at z>5000)
        fHe = Y_He / (4.0 * (1.0 - Y_He))
        _, new_xHeIII = _ionHe(
            new_tb, new_a, new_xHII, xHeII, xHeIII, Y_He, H0_kmsMpc, Omega_b
        )

        # HeII Peebles ODE with sub-stepped Heun (10 sub-steps for stability)
        # The helium Peebles rate can be ~200/efold, requiring dlna << 0.005.
        # Main steps have dlna~0.02, so we sub-step to avoid overshoot.
        n_He_sub = 10
        dlna_He_sub = dlna_step / n_He_sub
        def he_sub_step(i, xHe_i):
            xe_i = new_xHII + 0.25 * Y_He / (1.0 - Y_He) * (xHe_i + 2.0 * new_xHeIII)
            dxHe_1 = _recfast_dxHe_dlna(
                xe_i, xHe_i, n_H_cgs, H_cgs, z_half, tbhalf, TR_half, fHe)
            xHe_pred = jnp.clip(xHe_i + dlna_He_sub * dxHe_1, 0.0, 1.0)
            xe_pred = new_xHII + 0.25 * Y_He / (1.0 - Y_He) * (xHe_pred + 2.0 * new_xHeIII)
            dxHe_2 = _recfast_dxHe_dlna(
                xe_pred, xHe_pred, nH_new, H_new, z_new, new_tb, TR_new, fHe)
            return jnp.clip(xHe_i + 0.5 * dlna_He_sub * (dxHe_1 + dxHe_2), 0.0, 1.0)
        new_xHeII_peebles = jax.lax.fori_loop(0, n_He_sub, he_sub_step, xHeII)

        # Saha fallback for z>5000 (HeIII→HeII epoch, Peebles not needed)
        new_xHeII_saha, _ = _ionHe(
            new_tb, new_a, new_xHII, xHeII, xHeIII, Y_He, H0_kmsMpc, Omega_b
        )

        # Blend: Saha at z>5000, Peebles below
        w_saha_He = jax.nn.sigmoid(0.05 * (z_half - 5000.0))
        new_xHeII = w_saha_He * new_xHeII_saha + (1.0 - w_saha_He) * new_xHeII_peebles

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

    # --- Extend the table below its first RECFAST/Saha knot (loga_start) with a
    # closed-form, fully-ionized kappa_dot ~ a^-2 solution.
    #
    # Why: perturbations.py starts scalar-mode integration at
    # tau_ini = min(0.5, 0.01/k) (clax/perturbations.py, _matter_delta_m_single_k_impl
    # and friends), which for most k of interest is EARLIER than tau_grid[0] above
    # (e.g. at th_z_max=5e3, tau_grid[0] ~ 80.7 Mpc). Because CubicSpline.evaluate()
    # CLIPS below its first knot (clax/interpolation.py:67, jnp.clip(x_eval, x[0],
    # x[-1])), kappa_dot would otherwise FREEZE at the table-boundary value instead
    # of continuing to scale as a^-2. That corrupts _compute_tca_criterion at
    # tau_ini (is_tca -> 0 for all k, i.e. the solver thinks the fully-ionized
    # early-radiation-domination plasma is free-streaming), and makes th_z_max a
    # physics knob (11 orders of magnitude in P(k)) instead of a numerical one.
    #
    # This does NOT reintroduce the RECFAST-integration instability that motivated
    # starting the scan at a_start (see the comment above `a_start = ...`): nothing
    # is integrated here, we tabulate the closed form directly. The plasma is fully
    # ionized at every z above the table's own first knot by construction (that is
    # exactly the "Initial conditions (early radiation domination, fully ionized)"
    # used to seed the scan above: xHII0=1, xHeIII0=1), so x_e is held fixed at
    # xe_raw_grid[0] and kappa_dot follows from the SAME closed form the module
    # already uses below (kd_prefactor = n_H_0*(1+z)^2*sigma_T*Mpc_over_m); we get
    # that automatically by extending a_grid/xe_raw_grid here, before z_grid and
    # kd_prefactor are derived. T_b = T_cmb/a and cs2 = (4/3)*barssc*T_b reuse the
    # same closed forms as the tb0/cs20 initial conditions above, so xe_of_loga,
    # Tb_of_loga and cs2_of_loga (all consumed by perturbations.py) are extended
    # consistently too, not just kappa_dot.
    #
    # Point count: kappa_dot ~ exp(-2*loga) is smooth and monotonic in loga, so a
    # natural-cubic-spline interpolation error estimate (~h^4/24, relative, for
    # f''''=16f) gives ~9e-8 relative error at h=0.038 -- i.e. N_PREPEND=200 points
    # spread over the worst case in this codebase (th_z_max=5e3: loga range
    # bg.loga_table[0] (~log(bg_a_ini_default)=1e-7) to loga_start is ~7.6) is far
    # more than enough (verified directly by tests/test_thermodynamics.py, which
    # asserts kappa_dot*a^2 is constant to 1e-6 in this regime). `endpoint=False`
    # keeps the prepended grid strictly below loga_start so the combined grid stays
    # strictly increasing (required by CubicSpline).
    N_PREPEND = 200
    loga_prepend = jnp.linspace(bg.loga_table[0], loga_start, N_PREPEND, endpoint=False)
    a_prepend = jnp.exp(loga_prepend)
    tau_prepend = bg.tau_of_loga.evaluate(loga_prepend)
    tb_prepend = T_cmb / a_prepend
    xe_prepend = xe_raw_grid[0] * jnp.ones_like(a_prepend)
    barssc_prepend = _barssc_raw * (1.0 - 0.75 * Y_He + (1.0 - Y_He) * xe_prepend)
    cs2_prepend = 4.0 / 3.0 * barssc_prepend * tb_prepend

    a_grid = jnp.concatenate([a_prepend, a_grid])
    tau_grid = jnp.concatenate([tau_prepend, tau_grid])
    tb_grid = jnp.concatenate([tb_prepend, tb_grid])
    xe_raw_grid = jnp.concatenate([xe_prepend, xe_raw_grid])
    cs2_grid = jnp.concatenate([cs2_prepend, cs2_grid])
    loga_grid = jnp.concatenate([loga_prepend, loga_grid])

    # --- Derived quantities (n_H_0, kappa_dot prefactor) ---
    z_grid = 1.0 / a_grid - 1.0
    mu_H = 1.0 / (1.0 - Y_He)
    _bigH = 3.2407792902755102e-18  # H0=100 km/s/Mpc in s^-1
    # n_H_0 = (1-Y_He) * rho_b / m_H, where the appropriate mass is the hydrogen
    # atom mass (m_H), NOT the proton mass (m_p).  CLASS does the same at
    # thermodynamics.c:812 using _m_H_ = 1.673575e-27 kg.
    n_H_0 = 3.0 * (_bigH * params.h)**2 / (8.0 * math.pi * const.G_SI * const.m_H_kg * mu_H) * Omega_b
    # kappa_dot prefactor: kappa_dot(z) = xe * n_H_0 * (1+z)^2 * sigma_T * c/Mpc
    kd_prefactor = n_H_0 * (1.0 + z_grid)**2 * const.sigma_T * const.Mpc_over_m
    dtau_grid = jnp.diff(tau_grid)

    # --- Reionization: find z_reio self-consistently to match tau_reio ---
    # Forward solve stays with the robust bounded bisection below. The reverse
    # pass for z_reio(theta) is supplied by _find_z_reio's custom VJP so AD
    # does not inherit the zero-sensitivity jnp.where updates from bisection.
    z_reio = _find_z_reio(
        params.tau_reio, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
    )

    # CLASS-style additive reionization: xe = xe_before + (xe_after - xe_before) * frac
    xe_grid = jax.vmap(
        lambda z, xe_raw: _reionization_xe(z, z_reio, Y_He, xe_before=xe_raw)
    )(z_grid, xe_raw_grid)

    # --- Optical depth ---
    kappa_dot_grid = xe_grid * kd_prefactor

    # κ = ∫_τ^τ_0 κ'(τ') dτ' (integrate backwards from today)
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
    loga_grid_sg = jax.lax.stop_gradient(loga_grid)
    xe_of_loga = CubicSpline(loga_grid, xe_grid)
    Tb_of_loga = CubicSpline(loga_grid, tb_grid)

    # --- AD-safe n_H_0 rescaling for kappa_dot, kappa (-> exp_m_kappa), g ---
    # kappa_dot_grid and kappa_grid carry a large spurious accumulated gradient
    # from the Friedmann scan (d(a_grid[i])/d(omega_b) grows as an eigenvalue
    # product; ~10^8x FD blowup). Both are ∝ n_H_0 at fixed x_e and a, so: stop
    # all accumulated gradient, then restore only the n_H_0 ∝ omega_b path.
    # Exact where x_e ~ const (loga < -8); 10-30% residual near recombination.
    _kd_safe = (
        jax.lax.stop_gradient(kappa_dot_grid)
        * (n_H_0 / jax.lax.stop_gradient(n_H_0))
    )
    _kappa_safe = (
        jax.lax.stop_gradient(kappa_grid)
        * (n_H_0 / jax.lax.stop_gradient(n_H_0))
    )
    _exp_m_kappa_safe = jnp.exp(-_kappa_safe)
    _g_safe = _kd_safe * _exp_m_kappa_safe
    kappa_dot_of_loga = CubicSpline(loga_grid_sg, _kd_safe)
    exp_m_kappa_of_loga = CubicSpline(loga_grid_sg, _exp_m_kappa_safe)
    g_of_loga = CubicSpline(loga_grid_sg, _g_safe)
    cs2_of_loga = CubicSpline(loga_grid, cs2_grid)

    # dκ̇/dloga: stop all accumulated Friedmann-scan gradient in kappa_dot_grid,
    # then restore only the n_H_0 ∝ omega_b path by multiplying by n_H_0 / sg(n_H_0).
    # Every element of kappa_dot ∝ n_H_0 (linear), so this factoring is exact.
    # Gradient per knot: kd[i] * d(n_H_0)/dp / n_H_0 = kd[i] / omega_b.
    # Finite-difference in derivative formula: (kd[608]-kd[607])/(h*omega_b) =
    # dkd/dloga / omega_b ≈ -335, matching FD to <1%.
    _kd_deriv_spline = CubicSpline(loga_grid_sg, _kd_safe)
    dkd_dloga_grid = jax.vmap(_kd_deriv_spline.derivative)(loga_grid_sg)
    dkappa_dot_dloga_of_loga = CubicSpline(loga_grid_sg, dkd_dloga_grid)

    # g' = dg/dτ analytically (CLASS thermodynamics.c:3482-3483)
    # g = κ̇ e^{-κ},  g' = (κ̈ + κ̇²) e^{-κ},  κ̈ = (dκ̇/dloga) * aH
    _aH_grid = jnp.exp(loga_grid_sg) * jax.vmap(bg.H_of_loga.evaluate)(loga_grid_sg)
    g_prime_grid = (dkd_dloga_grid * _aH_grid + _kd_safe**2) * _exp_m_kappa_safe
    g_prime_of_loga = CubicSpline(loga_grid_sg, g_prime_grid)

    return ThermoResult(
        xe_of_loga=xe_of_loga,
        Tb_of_loga=Tb_of_loga,
        kappa_dot_of_loga=kappa_dot_of_loga,
        dkappa_dot_dloga_of_loga=dkappa_dot_dloga_of_loga,
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


# ---------------------------------------------------------------------------
# Reverse-mode stabilization: vjp-through-jvp custom rule (issue #30)
# ---------------------------------------------------------------------------
# Native reverse-mode AD through the solver above carries a ~2% error on
# h-like parameters (issue #30): the Peebles/RECFAST recombination rates
# contain Boltzmann-exponential ratios (exp(B/kT) ~ e^52), so AD
# intermediates reach ~1e13.  Forward mode pairs huge x tiny factors locally
# per grid point (the exponentials cancel element-wise before anything large
# is formed), which is why jax.jvp through this solver is FD-exact.  Reverse
# mode must contract thousands of cotangent terms through shared scalar
# intermediates, summing +/-1e13-scale terms whose true total is ~1e-3 (or
# exactly 0); float64 keeps a deterministic ULP residue (e.g. 2^-9 -- one
# ULP at magnitude ~1e13), localized in the recombination-era middle region
# of the tables.
#
# The custom VJP below is NOT an approximation and contains no fudge
# factors: it evaluates the SAME chain-rule contraction
#     params_bar[i] = <ct, d th / d params_i>
# in a different association order -- per-parameter forward-mode columns
# first (each numerically clean), then one well-conditioned inner product
# with the output cotangent -- instead of the native
# transpose-then-accumulate order that forms the ~1e13 partial sums.
# Mathematically identical arithmetic; only the floating-point evaluation
# order changes.
#
# The BackgroundResult cotangent keeps the NATIVE reverse rule restricted to
# the bg argument ("hybrid" backward): bg enters the scan only through
# smooth spline evaluations of H(loga) and tau(loga), not through the
# n_H_0-scaled Boltzmann funnel that owns the params-channel disease.
#
# Cost of one backward pass: one jacfwd over the ~20 traced CosmoParams
# leaves (a single vmapped forward pass, ~2-4x one thermo evaluation) plus
# one native vjp w.r.t. bg.  Memory is trivial (the tables are ~20k
# float64 per leaf).


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _thermodynamics_solve_stable(
    params: CosmoParams,
    prec: PrecisionParams,
    bg: BackgroundResult,
) -> ThermoResult:
    """``thermodynamics_solve`` with the numerically stable reverse rule.

    Primal values are bit-identical to ``_thermodynamics_solve_impl`` (it IS
    the same function); only ``jax.grad``/``jax.vjp`` behavior differs.
    Forward-mode (``jax.jvp``) through this wrapper raises ``TypeError`` by
    JAX design (custom_vjp blocks jvp); use ``th_grad_mode="native"``.
    """
    return _thermodynamics_solve_impl(params, prec, bg)


def _thermodynamics_solve_stable_fwd(params, prec, bg):
    """Forward pass: primal plus residuals (the primal inputs).

    Residuals are just ``(params, bg)``: the backward pass re-solves the
    thermodynamics in forward mode, which is what makes it stable -- storing
    native intermediates is exactly what we must avoid.
    """
    th = _thermodynamics_solve_impl(params, prec, bg)
    return th, (params, bg)


def _stable_bwd_params_cotangent(prec, params, bg, ct):
    """CosmoParams cotangent via a batched forward (jacfwd) basis.

    Computes ``params_bar[i] = <ct, d th / d params_i>`` where the Jacobian
    columns come from forward-mode AD (proven exact for this solver, see
    issue #30) and the contraction is a single well-conditioned inner
    product per parameter.  ``jax.jacfwd`` batches all ~20 scalar CosmoParams
    leaves into ONE vmapped forward pass.
    """
    jac = jax.jacfwd(lambda p: _thermodynamics_solve_impl(p, prec, bg))(params)
    # ``jac`` mirrors the ThermoResult output structure, with every output
    # leaf replaced by a CosmoParams-structured pytree of Jacobian columns
    # d(out_leaf)/d(param); each column has out_leaf.shape because every
    # traced CosmoParams leaf is a scalar.
    ct_leaves = jax.tree_util.tree_leaves(ct)
    jac_blocks = jax.tree_util.tree_structure(ct).flatten_up_to(jac)
    params_bar = None
    for ct_leaf, block in zip(ct_leaves, jac_blocks):
        contrib = jax.tree_util.tree_map(
            lambda col: jnp.sum(ct_leaf * col), block)
        params_bar = contrib if params_bar is None else jax.tree_util.tree_map(
            jnp.add, params_bar, contrib)
    return params_bar


def _thermodynamics_solve_stable_bwd(prec, residuals, ct):
    """Backward pass: jacfwd basis for params, native vjp for bg."""
    params, bg = residuals

    # (1) CosmoParams cotangent: vjp-through-jvp (the issue #30 fix).
    params_bar = _stable_bwd_params_cotangent(prec, params, bg, ct)

    # (2) BackgroundResult cotangent: native reverse rule restricted to bg.
    #     params is captured concrete-per-trace here, so no cotangent flows
    #     onto the params leaves through this call.
    _, pullback = jax.vjp(
        lambda b: _thermodynamics_solve_impl(params, prec, b), bg)
    (bg_bar,) = pullback(ct)

    return (params_bar, bg_bar)


_thermodynamics_solve_stable.defvjp(
    _thermodynamics_solve_stable_fwd, _thermodynamics_solve_stable_bwd)


@functools.partial(jax.jit, static_argnums=(1,))
def thermodynamics_solve(
    params: CosmoParams,
    prec: PrecisionParams,
    bg: BackgroundResult,
) -> ThermoResult:
    """Solve the thermodynamics using the MB95 semi-implicit method.

    Reverse-mode differentiation is gated by the static
    ``prec.th_grad_mode`` (see ``PrecisionParams``):

    * ``"stable"`` (default): custom VJP computing the CosmoParams cotangent
      via a batched forward-mode basis ("vjp-through-jvp") and the
      BackgroundResult cotangent via the native reverse rule.  Fixes the ~2%
      reverse-mode h-gradient error from catastrophic cancellation in the
      recombination-era backward pass (issue #30).  ``jax.jvp`` through this
      mode raises ``TypeError`` (JAX: custom_vjp blocks forward mode).
    * ``"native"``: plain JAX-derived derivatives; required for forward-mode
      (``jax.jvp``/``jax.jacfwd``) users, mirroring ``ode_adjoint="direct"``.

    Both modes produce bit-identical primal values.

    Args:
        params: cosmological parameters
        prec: precision parameters
        bg: background result from background_solve()

    Returns:
        ThermoResult with all thermodynamic spline tables
    """
    if prec.th_grad_mode == "stable":
        return _thermodynamics_solve_stable(params, prec, bg)
    if prec.th_grad_mode == "native":
        return _thermodynamics_solve_impl(params, prec, bg)
    raise ValueError(
        f"PrecisionParams.th_grad_mode must be 'stable' or 'native', "
        f"got {prec.th_grad_mode!r}")


def _tau_reio_for_zreio(
    z_reio_cand,
    xe_raw_grid,
    kd_prefactor,
    dtau_grid,
    z_grid,
    Y_He,
):
    """Return the reionization-only optical depth for a given ``z_reio``.

    Uses CLASS-style additive reionization:
        xe_total = xe_before + (xe_after - xe_before) * frac
    and integrates only the extra optical depth above the recombination baseline.
    """
    xe_total = jax.vmap(
        lambda z, xe_raw: _reionization_xe(z, z_reio_cand, Y_He, xe_before=xe_raw)
    )(z_grid, xe_raw_grid)
    diff = xe_total - xe_raw_grid
    # safe-mask: avoids NaN tangents from jnp.maximum at the boundary
    xe_extra = jnp.where(diff > 0.0, diff, jnp.zeros_like(diff))
    kd_extra = xe_extra * kd_prefactor
    kappa_integ = 0.5 * (kd_extra[:-1] + kd_extra[1:]) * dtau_grid
    return jnp.sum(kappa_integ)


def _find_z_reio_impl(
    tau_reio_target,
    xe_raw_grid,
    kd_prefactor,
    dtau_grid,
    z_grid,
    Y_He,
):
    """Find ``z_reio`` such that the reionization optical depth matches ``tau_reio``.

    The reionization optical depth is the EXTRA optical depth from
    reionization above the recombination baseline:
        tau_reio = ∫ max(xe_reio - xe_raw, 0) * kd_prefactor * dtau

    Uses bounded bisection in the forward pass.
    Memory-efficient: only one _reionization_xe evaluation per iteration.
    """
    # Bisection: find z_reio in [4, 25] where tau_reio = tau_reio_target
    def bisect_step(carry, _):
        z_lo, z_hi = carry
        z_mid = 0.5 * (z_lo + z_hi)
        tau_mid = _tau_reio_for_zreio(
            z_mid, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
        )
        # If tau_mid < target, need higher z_reio (more reionization)
        z_lo = jnp.where(tau_mid < tau_reio_target, z_mid, z_lo)
        z_hi = jnp.where(tau_mid < tau_reio_target, z_hi, z_mid)
        return (z_lo, z_hi), None

    (z_lo, z_hi), _ = jax.lax.scan(bisect_step, (4.0, 25.0), jnp.arange(40))
    return 0.5 * (z_lo + z_hi)


@jax.custom_jvp
def _find_z_reio(
    tau_reio_target,
    xe_raw_grid,
    kd_prefactor,
    dtau_grid,
    z_grid,
    Y_He,
):
    """Differentiable z_reio solve: primal uses bounded bisection in
    _find_z_reio_impl; JVP applies the implicit function theorem to
    F(z, inputs) = tau_reio_model(z, inputs) - tau_reio_target.
    JAX derives VJP via transposition, preserving reverse-mode behaviour."""
    return _find_z_reio_impl(
        tau_reio_target, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
    )


@_find_z_reio.defjvp
def _find_z_reio_jvp(primals, tangents):
    tau_reio_target, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He = primals
    (
        tau_reio_target_dot,
        xe_raw_grid_dot,
        kd_prefactor_dot,
        dtau_grid_dot,
        z_grid_dot,
        Y_He_dot,
    ) = tangents

    z_reio = _find_z_reio_impl(
        tau_reio_target, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
    )
    # IFT: treat the solution z_reio as a constant when JAX derives VJP by
    # transposition. Without stop_gradient, JAX adds spurious second-order
    # terms d(dF_dz)/d(inputs) that violate the IFT formula.
    z_reio_sg = jax.lax.stop_gradient(z_reio)

    # dF/dz at the solution: F(z) = tau_reio_model(z) - target
    _, dF_dz = jax.jvp(
        lambda z_: _tau_reio_for_zreio(
            z_, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
        ),
        (z_reio_sg,),
        (jnp.ones_like(z_reio_sg),),
    )
    # The IFT denominator is a constant at the solution — stop_gradient
    # prevents JAX from differentiating through it when deriving VJP.
    dF_dz = jax.lax.stop_gradient(dF_dz)
    # Guard against near-zero denominator
    dF_dz = jnp.where(
        jnp.abs(dF_dz) < 1e-12,
        jnp.where(dF_dz >= 0.0, 1e-12, -1e-12),
        dF_dz,
    )

    # Tangent of F w.r.t. inputs at fixed z_reio (implicit function theorem)
    _, tau_jvp = jax.jvp(
        lambda xe, kd, dt, zg, yh: _tau_reio_for_zreio(
            z_reio_sg, xe, kd, dt, zg, yh
        ),
        (xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He),
        (xe_raw_grid_dot, kd_prefactor_dot, dtau_grid_dot, z_grid_dot, Y_He_dot),
    )
    F_dot = tau_jvp - tau_reio_target_dot
    z_reio_dot = -F_dot / dF_dz
    return z_reio, z_reio_dot


def _estimate_z_reio(tau_reio_target):
    """Rough estimate of z_reio from tau_reio (legacy, kept for reference)."""
    return jnp.clip(2.0 + 150.0 * tau_reio_target, 4.0, 30.0)


# Convenience
def xe_of_z(th: ThermoResult, z: float) -> float:
    loga = jnp.log(1.0 / (1.0 + z))
    return th.xe_of_loga.evaluate(loga)
