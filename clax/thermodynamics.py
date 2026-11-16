"""Thermodynamics module for clax.

Computes the ionization history x_e(z), matter temperature T_m(z),
visibility function g(τ), optical depth κ(τ), and baryon sound speed c_s²(τ).

The baryon temperature uses the Ma & Bertschinger (1995) semi-implicit update
(as in DISCO-EB's thermodynamics_mb95.py). Everything is advanced by one
jax.lax.scan over a grid uniform in ln tau, which is JAX-compatible and
differentiable in both modes.

Recombination (smsharma/clax#48; docs/adr/0003-recombination-staging.md).
Helium and hydrogen follow CLASS's RECFAST with recfast_Heswitch = 1 (singlet-only
helium) and CLASS's approximation staging, so x_e agrees with that CLASS run to
< 1e-3 over z in [800, 5000] (gated contiguously in tests/test_recomb_thermo.py,
with window bars of 2x the measured error down to 1e-8), and, the reason for the
design, its derivatives with respect to h, omega_b, omega_cdm, Y_He, N_eff and
T_cmb are smooth:
    - HeIII -> HeII: the MB95 Saha solve (_ionHe), unchanged apart from D4's f_He. Above z ~ 5000 the
      same Saha solve also supplies x_HeII, sigmoid-blended into the staged
      HeII -> HeI value below.
    - HeII -> HeI: Saha equilibrium above z = 2870 (CLASS thermodynamics.c:4053-4067),
      the RECFAST helium Peebles equation below it (wrap_recfast.c:346-349); the
      output is blended from Saha to the ODE by CLASS's smoothstep f2 over
      z in [2770, 2870] (thermodynamics.h:49, thermodynamics.c:2973-2987).
    - Hydrogen: Saha with the x_HeII correction above z = 1600
      (thermodynamics.c:4070-4087), the RECFAST Peebles equation below it
      (wrap_recfast.c:110-174), f2 blend over z in [1500, 1600].
    - Numerics: each Peebles equation is advanced by one L-stable, second-order
      linearly implicit step (ROS2, _ros2_advance; 10 sub-steps per grid step) and
      NO clip to [0, 1]. L-stable means a stiff relaxation lands on its equilibrium
      without overshoot or oscillation, which is why no clip is used. The Jacobian
      holds x_e fixed; why that is accurate enough, with the measured one-step
      factors, is in the _ros2_scalar_step docstring and ADR 0003 (D1). The
      stage-switch redshifts are fixed numbers, as in CLASS, so no derivative
      passes through a switch location (the one state-dependent switch kept,
      RECFAST's C = 1 at x_H >= 0.995, is under Known limitations).
    - He/H number ratio: every number-ratio site uses
      f_He = Y_He / (3.9715 (1 - Y_He)), CLASS's _not4_ (thermodynamics.h:707), via
      _f_He. The four "1 - 0.75 Y_He" mean-molecular-weight sites (c_s^2 prefactor
      x3, T_b Compton-coupling fraction x1) still use exactly 4; they belong to the
      c_s^2-versus-CLASS-c_b^2 follow-up and leave c_s^2 2.6e-4 to 5.4e-4 (relative)
      from CLASS's form.
Why: the earlier explicit steps (Heun for helium, RK4 for hydrogen, both clipped,
plus a hard x_He > 0.999 switch and Saha/ODE sigmoids) ran through the ranges
where CLASS uses Saha. x_e looked fine (0.35% wiggles), but d ln x_e / d theta
spiked to ~5e12 at isolated grid points; those reached delta_m through
c_s^2 k^2 delta_b depending on where the ODE solver happened to step, and on the
fit_cl 3000-point grid x_e collapsed to 0.035 at z ~ 1559.

Known limitations:
    - Heswitch = 1 helium only. CLASS's default (Heswitch = 6, Doppler and
      triplet terms) differs in x_e by 1.5% at z = 2000 and 4.2% at z ~ 1771.
    - RECFAST's Peebles C = 1 while x_H >= 0.995 (wrap_recfast.c:164) is kept, and
      it depends on the state. Where the Saha x_H at the hydrogen ODE start
      (z = 1600) exceeds 0.995 (omega_b <~ 0.020 at fiducial T_cmb, or
      T_cmb >~ 2.735 K; 0.9954 at omega_b - 20%, 0.9943 at fiducial) the ODE starts
      with C = 1 and crosses the switch a few grid steps later, at a point that
      moves with the parameters. Measured at omega_b - 20% (ADR 0003): no visible
      effect, because x_H sits at its equilibrium while C = 1, so the switch
      rescales a rate that is already ~0.
    - c_s^2 is 1.3% from CLASS's c_b^2 at z ~ 632 (the T_b update's former hard
      Taylor switch at f_e < 1e-3, which put a spike of ratio 20 in the T_b
      tangent at z ~ 624 on a 1500-point grid and a staircase in the primal's
      omega_b dependence, is gone: the update uses the exact expm1 form).
    - Reverse mode through _ros2_advance needs a static Python-int sub-step count.
    - No test gates the C_l shift caused by this change; ADR 0003 records the
      measured planck_cl and fit_cl shifts against the unfixed code.

Inputs/outputs: CosmoParams, PrecisionParams and a BackgroundResult in; a
ThermoResult out, whose tables are CubicSplines in ln a.

Key function:
    thermodynamics_solve(params, prec, bg) -> ThermoResult

References:
    - Ma & Bertschinger (1995) ApJ 455, 7 for the baryon temperature update
    - Verwer et al. (1999) for the ROS2 step
    - DISCO-EB: src/discoeb/thermodynamics_mb95.py
    - CLASS: source/thermodynamics.c, external/RecfastCLASS/wrap_recfast.c
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace as dataclass_replace

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from clax import constants as const
from clax.background import BackgroundResult, background_solve
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
_NOT4 = 3.9715               # mHe / mH (not exactly 4), CLASS _not4_ (thermodynamics.h:707); used only via _f_He
_RECFAST_FUDGE_He = 0.86     # helium fudge factor (CLASS default)
# Verner & Ferland (1996) He case-B: convert from SI (m^3/s) to CGS (cm^3/s)
# CLASS: _a_VF_ = 10^{-16.744} [m^3/s]; CGS = SI * 1e6
_a_VF_He_CGS = 10.0**(-16.744) * 1e6  # ≈ 1.803e-11 cm^3/s
_b_VF_He = 0.711
_T_0_He = 10.0**0.477121      # ≈ 3.0 K
_T_1_He = 10.0**5.114         # ≈ 1.3e5 K
# HeI ionization temperature, cf. wrap_recfast.c:78 and thermodynamics.c:823
# (const_Tion_HeI = h c L_He1_ion / k_B), ~285,325 K.
_CB1_He1 = _hP_SI * _c_SI * _L_He1_ion / _kB_SI
# CLASS approximation-stage boundaries (smsharma/clax#48):
# start of regime "H" (x_He evolved below it): precisions.h:201-202,
# thermodynamics.c:868 (ap_z_limits) and :894 (smoothing delta);
# start of regime "frec" (x_H evolved below it): precisions.h:204-205,
# thermodynamics.c:870 and :895.
_Z_EARLY_H_RECOMB = 2870.0
_DZ_EARLY_H_RECOMB = 50.0
_Z_FULL_H_RECOMB = 1600.0
_DZ_FULL_H_RECOMB = 50.0

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
    """Output of the thermodynamics module.

    ``tau_free_streaming`` [Mpc] is CLASS's ``pth->tau_free_streaming``
    (thermodynamics.c): the first conformal time after recombination at which
    the Thomson mean free path tau_c = 1/kappa' exceeds
    ``PrecisionParams.pt_rsa_trigger_tau_c_over_tau`` (= 5) conformal times.
    Together with k*tau > ``pt_rsa_trigger_tau_over_tau_k`` it switches the
    photon radiation streaming approximation on in perturbations.py. It
    carries no gradient (``stop_gradient``): like CLASS's table point, the
    switch location is not differentiated (the dropped term is <= 3e-5 on
    d ln C_l / d ln theta; see the comment where it is computed).

    Derived epochs, all sub-grid and differentiable (smsharma/clax#44):
    ``z_star``, ``tau_star`` [Mpc], ``rs_star`` [Mpc] at the maximum of the
    visibility g by a three-point parabola in ln a (CLASS's ``z_rec`` /
    ``tau_rec`` / ``rs_rec``, thermodynamics.c
    thermodynamics_calculate_recombination_quantities); ``z_rec`` where the
    optical depth crosses one, kappa(z_rec) = 1, by linear interpolation (no
    CLASS counterpart); ``z_d``, ``tau_d`` [Mpc], ``rs_d`` [Mpc] at the baryon
    drag epoch, kappa_b(tau_d) = 1 with dkappa_b/dtau = kappa'/R and
    R = 3 rho_b / 4 rho_gamma, by linear interpolation (CLASS's ``z_d`` /
    ``tau_d`` / ``rs_d``, thermodynamics_calculate_drag_quantities). rs_d is
    the BAO ruler and the EPT IR-resummation damping scale.
    """
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
    tau_free_streaming: float
    z_d: float
    tau_d: float
    rs_d: float

    def tree_flatten(self):
        return [
            self.xe_of_loga, self.Tb_of_loga,
            self.kappa_dot_of_loga, self.dkappa_dot_dloga_of_loga,
            self.exp_m_kappa_of_loga,
            self.g_of_loga, self.g_prime_of_loga, self.cs2_of_loga,
            self.z_star, self.z_rec, self.tau_star, self.rs_star, self.z_reio,
            self.tau_free_streaming, self.z_d, self.tau_d, self.rs_d,
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

    cf. CLASS wrap_recfast.c:198-349. K_He is the Sobolev form of CLASS's Heflag >= 1
    branch (wrap_recfast.c:258-261), which Heswitch = 1 selects for x_He between the
    small trigger and 0.995 at z >= 800 (wrap_recfast.c:243-248); outside that range
    CLASS uses Heflag = 0, K_He = CK_He/Hz. clax uses the Sobolev form at every x_He:
    the two agree to machine precision whenever tau_s >> 1, which holds in both
    Heflag = 0 ranges (tau_s >~ 3e5 at x_He > 0.995).
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

    # K_He with Sobolev escape probability (Heflag >= 1 in CLASS)
    # cf. wrap_recfast.c:258-261
    # Basic: K_He = CK_He / Hz
    # Sobolev: τ_s = A2P_s * CK_He * 3 * n_He * (1-xHe) / Hz
    #          p_s = (1 - exp(-τ_s)) / τ_s
    #          K_He = 1 / (A2P_s * p_s * 3 * n_He * (1-xHe))
    n_1s_He = jnp.maximum(n_He * (1.0 - xHe), 1e-30)
    # K_He with Sobolev escape (CLASS Heflag >= 1)
    # cf. wrap_recfast.c:258-261
    tauHe_s = _A2P_s_He * _CK_He_CGS * 3.0 * n_1s_He / Hz_safe
    tauHe_s_safe = jnp.maximum(tauHe_s, 1e-30)
    # p_s = (1 - e^{-tau})/tau, wrap_recfast.c:260, cancellation-free (expm1),
    # series below tau = 1e-4 (truncation tau^4/120 < 1e-18); double-where
    # keeps both branches finite for AD.
    _small = tauHe_s_safe < 1e-4
    _t_big = jnp.where(_small, 1.0, tauHe_s_safe)
    pHe_s = jnp.where(_small,
                      1.0 - tauHe_s_safe / 2.0 + tauHe_s_safe**2 / 6.0 - tauHe_s_safe**3 / 24.0,
                      -jnp.expm1(-_t_big) / _t_big)
    K_He = 1.0 / jnp.maximum(_A2P_s_He * pHe_s * 3.0 * n_1s_He, 1e-30)

    # Peebles C_He with Boltzmann 2s-2p factor
    # Reformulated as (inv_A + Lambda) / (inv_A + Lambda + Rup) where A = K*n*B,
    # to avoid inf-inf NaN in JVP when He_Boltz = exp(CDB/T) is huge at low T.
    He_Boltz = jnp.exp(jnp.minimum(_CDB_He2s2p / TM_safe, 500.0))
    A_He = K_He * n_1s_He * He_Boltz
    inv_A_He = 1.0 / jnp.maximum(A_He, 1e-300)
    C_He = (inv_A_He + _Lambda_He) / (inv_A_He + _Lambda_He + Rup_He)
    # CLASS has no C_He switch near x_He = 1: its x_He switches are the K_He form
    # (Heflag, wrap_recfast.c:243-244), and C_He is always wrap_recfast.c:346.
    # For x_He < 1e-15 CLASS sets dx_He/dz = 0 (wrap_recfast.c:328-329); clax
    # instead keeps the C_He = 1 guard from the pre-#48 code (where it was part of
    # the deleted `(xHe < 1e-15) | (xHe > 0.999)` switch) and still evaluates the rate.
    # The effect on x_e is at most fHe * 1e-15 (x_He itself is below 1e-15).
    C_He = jnp.where(xHe < 1e-15, 1.0, C_He)

    dxHe_dlna = -(
        xe * xHe * nH * Rdown_He
        - Rup_He * (1.0 - xHe) * jnp.exp(jnp.maximum(-_CL_He / TM_safe, -500.0))
    ) * C_He / Hz_safe

    return dxHe_dlna


# ---------------------------------------------------------------------------
# Shared L-stable stiff step for the recombination Peebles equations (#48)
# ---------------------------------------------------------------------------

_ROS2_GAMMA = 1.0 + 1.0 / math.sqrt(2.0)   # ROS2, Verwer et al. 1999 (L-stable)


def _smoothstep(s):
    """CLASS's stage-transition weight f2(s) = 3 s^2 - 2 s^3 (thermodynamics.h:49),
    with s clipped to [0, 1] so it is exactly 0 or 1 outside the window."""
    s = jnp.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _ros2_scalar_step(rate, xe_of, x, h, env0, env1):
    """One two-stage linearly implicit (ROS2) step of dx/dlna = rate(x_e, x, *env).

    W = 1 - gamma h J,  k1 = h f(s_j, x_j)/W,  k2 = (h f(s_j+1, x_j + k1) - 2 k1)/W,
    x_j+1 = x_j + 3/2 k1 + 1/2 k2, with gamma = 1 + 1/sqrt(2). J = d rate/d x at
    the sub-step start with x_e held fixed (ROS2 stays second order for any J).
    L-stability, the reason no clip is used, holds for the exact J (x_e moving
    with x). Measured: wherever the step is stiff (|h J| > 0.5) this J is within
    0.5-1.2% of the exact one, because photoionisation dominates the hydrogen
    rate near its switch, so the one-step factor stays within 0.006 of the
    exact-J one up to the largest |h J| reached (2.35 on the 3000-point grid at
    10 cosmologies, 4.0 on 1500 points). Against the code's own 100000-point
    solution this J is also the more accurate one on the 3000-point grid (x_e
    1.3e-4 vs 2.3e-4 for the exact J), so it is kept (ADR 0003, D1).
    env0/env1: environment tuples at the sub-step's two ends.
    """
    xe_j = xe_of(x)
    f0, J = jax.jvp(lambda y: rate(xe_j, y, *env0), (x,), (jnp.ones_like(x),))
    W = 1.0 - _ROS2_GAMMA * h * J
    k1 = h * f0 / W
    x2 = x + k1
    f1 = rate(xe_of(x2), x2, *env1)
    k2 = (h * f1 - 2.0 * k1) / W
    return x + 1.5 * k1 + 0.5 * k2


def _ros2_advance(rate, xe_of, x0, envs, h_total):
    """Advance x over one grid step of length h_total (in ln a) with
    n_sub = len(envs[0]) - 1 ROS2 sub-steps. envs holds the environment at the
    n_sub + 1 sub-step boundaries, one array per rate argument, so it is built
    once per grid step and shared by hydrogen and helium; n_sub comes from its
    static shape."""
    n_sub = envs[0].shape[0] - 1
    h = h_total / n_sub

    def body(j, x):
        return _ros2_scalar_step(rate, xe_of, x, h,
                                 tuple(e[j] for e in envs), tuple(e[j + 1] for e in envs))

    return jax.lax.fori_loop(0, n_sub, body, x0)


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

    c = _f_He(Y_He)

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

def _f_He(Y_He):
    """Helium-to-hydrogen number ratio n_He/n_H = Y/(_not4_ (1-Y)), CLASS
    thermodynamics.h:707. The one definition used by every x_e sum and helium
    rate in this module (smsharma/clax#48)."""
    return Y_He / (_NOT4 * (1.0 - Y_He))


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
    fHe = _f_He(Y_He)
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

@functools.partial(jax.jit, static_argnums=(1,))
def thermodynamics_solve(
    params: CosmoParams,
    prec: PrecisionParams,
    bg: BackgroundResult,
) -> ThermoResult:
    """Solve the thermodynamics: MB95 semi-implicit baryon temperature and
    CLASS-staged recombination (Saha -> ROS2 ODE; see the module docstring and
    docs/adr/0003-recombination-staging.md).

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
    xe0 = xHII0 + _f_He(Y_He) * (xHeII0 + 2.0 * xHeIII0)
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
        'cs2': cs20, 'xHeII_ode': xHeII0, 'xHII_ode': xHII0,
    }
    keys = ('a', 'adot', 'tau', 'tb', 'xHII', 'xe', 'xHeII', 'xHeIII', 'cs2')

    def scan_step(carry, i):
        a = carry['a']
        adot = carry['adot']
        tau = carry['tau']
        tb = carry['tb']
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
        da = new_a - a
        x = thomc * da
        # a^2 T_b' = a^2 (T_b - T_g) e^{-x} - T_cmb (1 - e^{-x}) / thomc, with
        # (1 - e^{-x}) / thomc = da * phi(x), phi(x) = -expm1(-x) / x. expm1 keeps
        # phi exact down to x ~ 1e-12, where the series 1 - x/2 equals it to
        # 1e-24, so the jnp.where below carries no jump. Until Oct 22, 2026
        # (CHANGELOG) the code switched to a first-order Taylor branch at
        # f_e < 1e-3 to dodge the cancellation in 1 - e^{-x}. That branch was
        # used where x is not small, so its truncation error grew with the grid
        # step (T_b 2.7e-4 / 1.8e-3 from CLASS at z 300-700 on 20000 / 3000
        # points, x_e 1.1e-4 / 8.7e-4 at z 100-500), and the switch moved with
        # the parameters and crossed knots as they changed, which left the
        # primal a staircase below z ~ 620 (ADR 0004: finite differences along
        # omega_b biased by 1.7e-3 of the derivative on 10000 points) and a
        # spike of ratio 20 in the T_b tangent at z ~ 624 on a 1500-point grid.
        x_safe = jnp.maximum(x, 1e-12)
        phi = jnp.where(x > 1e-12, -jnp.expm1(-x_safe) / x_safe, 1.0 - 0.5 * x)
        a2t = a**2 * (tb - tg0) * jnp.exp(-x) - T_cmb * da * phi

        new_tb = T_cmb / new_a + a2t / new_a**2

        # Ionization step
        z_half = 1.0 / ahalf - 1.0
        z_new = 1.0 / new_a - 1.0
        nH_new = n_H_0_cgs / new_a**3

        # Sub-step environment for the recombination ROS2 steps (#48): at
        # fraction s of this grid step, ln a is linear and ln H, T_b are linear
        # in ln a. Returns the rate arguments (n_H, H, z, T_b, T_r) in CGS.
        lna0 = jnp.log(a)
        dlna_main = new_loga - lna0
        lnH0 = jnp.log(adot / a**2)          # a' = a^2 H -> H at the step start
        lnH1 = jnp.log(new_H)
        fHe = _f_He(Y_He)  # n_He/n_H with CLASS's _not4_: every x_e sum and the helium rate below

        def sub_env(s):
            a_s = jnp.exp(lna0 + s * dlna_main)
            return (n_H_0_cgs / a_s**3,
                    jnp.exp(lnH0 + s * (lnH1 - lnH0)) * _c_over_Mpc,
                    1.0 / a_s - 1.0,
                    tb + s * (new_tb - tb),
                    T_cmb / a_s)

        # The 11 boundaries of the 10 ROS2 sub-steps, built once and shared by
        # the hydrogen and helium advances below.
        envs = sub_env(jnp.linspace(0.0, 1.0, 11))

        # Hydrogen (smsharma/clax#48): the Peebles equation (wrap_recfast.c:110-174,
        # dx_H/dz -> d/dlna) advanced from the ODE state by the shared L-stable
        # ROS2 step, 10 sub-steps, no clip. It replaces an explicit RK4 whose
        # h*J = -3.2 at z = 1600 lies outside RK4's stability interval [-2.79, 0].
        # Above z = 1600 the jnp.where below discards this advance, so lax.cond
        # skips it (identical output; the grid is uniform in ln tau, so the
        # index of z = 1600 moves with the cosmology and the scan cannot be split
        # at a fixed index). Under vmap with a batched predicate both branches run.
        new_xHII_peebles = jax.lax.cond(
            z_new > _Z_FULL_H_RECOMB, lambda: carry['xHII_ode'],
            lambda: _ros2_advance(
                _recfast_dxHII_dlna,
                lambda x_: x_ + fHe * (xHeII + 2.0 * xHeIII),
                carry['xHII_ode'], envs, dlna_main))

        # For z > 1600: hydrogen Saha equilibrium (CLASS thermodynamics.c:4074-4081).
        # x_H*(x_H + xHeII)/(1 - x_H) = rhs, solved via quadratic formula.
        # rhs = (CR*Tmat)^{3/2} * exp(-CB1/Tmat) / nH_physical
        # where CB1 = hc*L_H_ion/kB ~ 157807 K (ground-state ionization temp).
        # T_b and n_H are taken at the step end, like the regime switch, the blend
        # weight w_H and the helium Saha below, so x_HII is the value at the new
        # grid point. The x_HeII correction is the step-start value (the helium
        # update needs x_HII first): a one-step lag in a term of ~0.2 f_He near
        # z = 1600, estimated to move x_H by ~1e-6 on the 3000-point grid, not
        # measured separately.
        T_saha = jnp.maximum(new_tb, 100.0)  # clamp to avoid exp(-huge)
        nH_saha = nH_new
        # Clamp exponent to avoid underflow producing 0*inf NaN in gradients
        saha_exp = jnp.exp(jnp.maximum(-_CB1 / T_saha, -500.0))
        saha_rhs = jnp.maximum(
            (_CR_CGS * T_saha)**1.5 * saha_exp / nH_saha, 1e-300)
        # Helium electron contribution per H nucleus
        xHeII_contrib = fHe * xHeII
        xHII_saha = _solve_hydrogen_saha(saha_rhs, xHeII_contrib)
        # CLASS staging for hydrogen: regime "H" (z > 1600) is Saha with the x_HeII
        # correction (thermodynamics.c:4070-4087, the quadratic above). The ODE
        # state is the Saha value while z > 1600 (initial condition of regime
        # "frec", thermodynamics.c:2093), the Peebles step below; the output is
        # smoothed by f2 over z in [1500, 1600] (thermodynamics.c:2973-2987).
        new_xHII_ode = jnp.where(z_new > _Z_FULL_H_RECOMB, xHII_saha, new_xHII_peebles)
        w_H = _smoothstep((_Z_FULL_H_RECOMB - z_new) / (2.0 * _DZ_FULL_H_RECOMB))
        new_xHII = w_H * new_xHII_ode + (1.0 - w_H) * xHII_saha

        # Helium. HeIII->HeII: Saha (_ionHe). HeII->HeI: the MB95 Saha above
        # z ~ 5000, then the CLASS-style HeI Saha down to z = 2870, then the
        # Peebles ODE below it, blended over z 2770-2870 (staging block below).
        # The one _ionHe solve gives both x_HeIII and the MB95 x_HeII used above
        # z ~ 5000 (HeIII -> HeII epoch, Peebles not needed).
        new_xHeII_saha, new_xHeIII = _ionHe(
            new_tb, new_a, new_xHII, xHeII, xHeIII, Y_He, H0_kmsMpc, Omega_b
        )

        # HeII -> HeI (smsharma/clax#48): CLASS staging. Regime "He2" (z > 2870)
        # is Saha; below, the Peebles equation (wrap_recfast.c:346-349, dx_He/dz
        # -> d/dlna) advanced by the shared L-stable ROS2 step, 10 sub-steps,
        # no clip.
        # Skipped above z = 2870, where the jnp.where below discards it (as for
        # hydrogen above).
        new_xHeII_peebles = jax.lax.cond(
            z_new > _Z_EARLY_H_RECOMB, lambda: carry['xHeII_ode'],
            lambda: _ros2_advance(
                lambda xe_, x_, *env: _recfast_dxHe_dlna(xe_, x_, *env, fHe),
                lambda x_: new_xHII + fHe * (x_ + 2.0 * new_xHeIII),
                carry['xHeII_ode'], envs, dlna_main))
        # Saha HeII <-> HeI with H fully ionized, thermodynamics.c:4053-4067:
        #   rhs = 4 (C_R T_b)^{3/2} exp(-T_HeI/T_b) / n_H(z)          (:4057)
        #   x_e = 0.5 (sqrt((rhs-1)^2 + 4(1+fHe) rhs) - (rhs-1)),  x_He = (x_e-1)/fHe
        # taken in the cancellation-free root of y = fHe x_He:
        #   y^2 + (1+rhs) y - fHe rhs = 0.
        # The root assumes no HeIII. Below z ~ 5000 the MB95 x_HeIII from _ionHe is
        # still nonzero and enters new_xe below, so the HeI Saha splits only the
        # helium not in HeIII. Without the (1 - x_HeIII) share, x_HeII + x_HeIII
        # reached 1.0007 near z = 4970 (fiducial, both grids) and x_e was 5.5e-5
        # off CLASS over z 2870-5000; with it, 1.1e-6.
        rhs_He = 4.0 * (_CR_CGS * new_tb)**1.5 * jnp.exp(
            jnp.maximum(-_CB1_He1 / new_tb, -500.0)) / nH_new
        xHeII_saha_he2 = (1.0 - new_xHeIII) * 2.0 * rhs_He / (
            (1.0 + rhs_He) + jnp.sqrt((1.0 + rhs_He)**2 + 4.0 * fHe * rhs_He))
        # ODE state: the Saha value while z > 2870 (initial condition on entering
        # regime "H", thermodynamics.c:2059-2065), the Peebles step below.
        new_xHeII_ode = jnp.where(z_new > _Z_EARLY_H_RECOMB, xHeII_saha_he2,
                                  new_xHeII_peebles)
        # Output smoothing after the switch, thermodynamics.c:2973-2987:
        # f2((z_lim - z)/(2 Dz)), exactly 0 or 1 outside z in [2770, 2870].
        w_He = _smoothstep((_Z_EARLY_H_RECOMB - z_new) / (2.0 * _DZ_EARLY_H_RECOMB))
        xHeII_staged = w_He * new_xHeII_ode + (1.0 - w_He) * xHeII_saha_he2

        # Blend: the MB95 Saha above z ~ 5000; below it the staged value (HeI Saha
        # down to z = 2870, the Peebles ODE below, blended over z 2770-2870).
        w_saha_He = jax.nn.sigmoid(0.05 * (z_half - 5000.0))
        new_xHeII = w_saha_He * new_xHeII_saha + (1.0 - w_saha_He) * xHeII_staged

        # Total ionization fraction
        new_xe = new_xHII + fHe * (new_xHeII + 2.0 * new_xHeIII)

        # Sound speed squared (over c^2)
        dtbdla = -2.0 * new_tb - thomc * a2t / new_a
        barssc = _barssc_raw * (1.0 - 0.75 * Y_He + (1.0 - Y_He) * new_xe)
        new_cs2 = barssc * new_tb * (1.0 - dtbdla / new_tb / 3.0)

        new_carry = {
            'a': new_a, 'adot': new_adot, 'tau': new_tau, 'tb': new_tb,
            'xHII': new_xHII, 'xe': new_xe, 'xHeII': new_xHeII,
            'xHeIII': new_xHeIII, 'cs2': new_cs2,
            'xHeII_ode': new_xHeII_ode, 'xHII_ode': new_xHII_ode,
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

    # --- Derived epochs (smsharma/clax#44): sub-grid, so they are continuous
    # and differentiable in the parameters. Until #44 all four were grid-point
    # values, piecewise constant in theta (one 20000-point step is 5e-4 in z at
    # recombination), and tau_star sets the perturbation save grid.
    #
    # z_star: maximum of g by a three-point parabola through the knots around
    # argmax g, in ln a (CLASS, thermodynamics.c
    # thermodynamics_calculate_recombination_quantities, does the same in z on
    # its equally spaced table). The vertex of the parabola through
    # (x0, g0), (x1, g1), (x2, g2) with unequal spacing.
    idx_star = jnp.argmax(g_grid)
    i1 = jnp.clip(idx_star, 1, loga_grid.shape[0] - 2)
    x0, x1, x2 = loga_grid[i1 - 1], loga_grid[i1], loga_grid[i1 + 1]
    g0, g1, g2 = g_grid[i1 - 1], g_grid[i1], g_grid[i1 + 1]
    _num = (x1 - x0) ** 2 * (g1 - g2) - (x1 - x2) ** 2 * (g1 - g0)
    _den = (x1 - x0) * (g1 - g2) - (x1 - x2) * (g1 - g0)
    loga_star = x1 - 0.5 * _num / jnp.where(_den == 0.0, 1.0, _den)
    z_star = 1.0 / jnp.exp(loga_star) - 1.0
    tau_star = bg.tau_of_loga.evaluate(loga_star)
    rs_star = bg.rs_of_loga.evaluate(loga_star)

    def _crossing_loga(f_grid, level):
        """ln a where the decreasing table f_grid crosses `level`, linearly
        interpolated between the two bracketing knots (CLASS's recipe for z_d)."""
        j = jnp.argmax(f_grid < level)                  # first knot below the level
        j = jnp.clip(j, 1, f_grid.shape[0] - 1)
        f0, f1 = f_grid[j - 1], f_grid[j]
        w = (level - f0) / jnp.where(f1 == f0, 1.0, f1 - f0)
        return loga_grid[j - 1] + w * (loga_grid[j] - loga_grid[j - 1])

    # z_rec: optical depth one, kappa(z_rec) = 1.
    z_rec = 1.0 / jnp.exp(_crossing_loga(kappa_grid, 1.0)) - 1.0

    # Baryon drag epoch: kappa_b(tau_d) = 1 with -dkappa_b/dtau = kappa'/R,
    # R = 3 rho_b / 4 rho_gamma (thermodynamics.c, thermodynamics_calculate_
    # drag_quantities); integrated backwards from today like kappa, then z_d by
    # linear interpolation, tau_d and rs_d from the background at z_d.
    _R_grid = 0.75 * jax.vmap(bg.rho_b_of_loga.evaluate)(loga_grid) \
        / jax.vmap(bg.rho_g_of_loga.evaluate)(loga_grid)
    _kappa_b_dot_grid = kappa_dot_grid / _R_grid
    _kappa_b_integrand = 0.5 * (_kappa_b_dot_grid[:-1] + _kappa_b_dot_grid[1:]) * dtau_grid
    kappa_b_grid = jnp.concatenate([jnp.cumsum(_kappa_b_integrand[::-1])[::-1], jnp.array([0.0])])
    loga_d = _crossing_loga(kappa_b_grid, 1.0)
    z_d = 1.0 / jnp.exp(loga_d) - 1.0
    tau_d = bg.tau_of_loga.evaluate(loga_d)
    rs_d = bg.rs_of_loga.evaluate(loga_d)

    # --- Build splines on loga grid ---
    # Every table, the opacity ones included, is a spline on the live ln a knots
    # with the live values: knots and values carry their tangents together
    # (smsharma/clax#55). The knots move with the parameters (the scan is
    # uniform in ln tau), so the tangent of a value at fixed knot index is not
    # the tangent at fixed a; it differs by the transport term (table slope times
    # d ln a/d theta). The pre-#55 code froze the knots and values of kappa_dot,
    # exp(-kappa), g, dkappa_dot/dlna and g' under stop_gradient and restored only
    # the n_H_0 ∝ omega_b path (d127731, against a "10^8x" blow-up that was this
    # transport term read against a vanishing true derivative): along omega_cdm
    # and N_ur their tangents were exactly zero, along omega_b and Y_He the x_e
    # response was missing, and dP(k)/d omega_b came out 4% off finite differences.
    # With the full tangents the five tables match Richardson finite differences
    # at fixed z to the FD's own precision at 11 cosmologies (tests/
    # test_recomb_thermo.py::test_opacity_tangents_match_converged_fd).
    xe_of_loga = CubicSpline(loga_grid, xe_grid)
    Tb_of_loga = CubicSpline(loga_grid, tb_grid)
    kappa_dot_of_loga = CubicSpline(loga_grid, kappa_dot_grid)
    exp_m_kappa_of_loga = CubicSpline(loga_grid, exp_m_kappa_grid)
    g_of_loga = CubicSpline(loga_grid, g_grid)
    cs2_of_loga = CubicSpline(loga_grid, cs2_grid)

    # dκ̇/dloga: the derivative of the kappa_dot spline at its own knots.
    dkd_dloga_grid = jax.vmap(kappa_dot_of_loga.derivative)(loga_grid)
    dkappa_dot_dloga_of_loga = CubicSpline(loga_grid, dkd_dloga_grid)

    # g' = dg/dτ analytically (CLASS thermodynamics.c:3482-3483)
    # g = κ̇ e^{-κ},  g' = (κ̈ + κ̇²) e^{-κ},  κ̈ = (dκ̇/dloga) * aH
    _aH_grid = jnp.exp(loga_grid) * jax.vmap(bg.H_of_loga.evaluate)(loga_grid)
    g_prime_grid = (dkd_dloga_grid * _aH_grid + kappa_dot_grid**2) * exp_m_kappa_grid
    g_prime_of_loga = CubicSpline(loga_grid, g_prime_grid)

    # --- tau_free_streaming: second trigger of the radiation streaming
    # approximation. CLASS thermodynamics.c ("find time (always after
    # recombination) at which tau_c/tau falls below some threshold, defining
    # tau_free_streaming"): the first table time after the visibility peak at
    # which the Thomson mean free path tau_c = 1/kappa' exceeds
    # radiation_streaming_trigger_tau_c_over_tau (= 5) conformal times. CLASS
    # takes the table point itself; here the crossing is interpolated linearly
    # in log a between the two bracketing points (364.77 Mpc, z = 760 at the
    # fiducial cosmology, visibility at 2.4% of its peak; CLASS's table point
    # is 364.767 Mpc). The switch time is held fixed under AD (stop_gradient),
    # like CLASS's table point, which has no derivative. What would otherwise
    # reach it: the partials through kappa_dot and through the background time
    # tau(z) of the two bracketing points. With d ln C_l/d ln tau_fs ~ 1e-4 and
    # d ln tau_fs/d ln theta <= 0.26 (omega_cdm) the dropped term is <= 3e-5 on
    # d ln C_l/d ln theta.
    # tau from the background spline, as for tau_star. If no crossing exists
    # (never for a recombining universe) the switch is set to the end of the
    # table.
    _tau_grid_bg = jax.vmap(bg.tau_of_loga.evaluate)(loga_grid)
    _tau_c_over_tau = 1.0 / jnp.maximum(kappa_dot_grid * _tau_grid_bg, 1e-300)
    _after_peak = jnp.arange(loga_grid.shape[0]) > idx_star
    _crossed = _after_peak & (_tau_c_over_tau >= prec.pt_rsa_trigger_tau_c_over_tau)
    _i1 = jnp.argmax(_crossed)
    _i0 = jnp.maximum(_i1 - 1, 0)
    _f0 = _tau_c_over_tau[_i0]
    _f1 = _tau_c_over_tau[_i1]
    _w = jnp.clip((prec.pt_rsa_trigger_tau_c_over_tau - _f0)
                  / jnp.where(_f1 == _f0, 1.0, _f1 - _f0), 0.0, 1.0)
    _loga_fs = loga_grid[_i0] + _w * (loga_grid[_i1] - loga_grid[_i0])
    tau_free_streaming = jax.lax.stop_gradient(
        jnp.where(jnp.any(_crossed), bg.tau_of_loga.evaluate(_loga_fs),
                  _tau_grid_bg[-1]))

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
        tau_free_streaming=tau_free_streaming,
        z_d=z_d,
        tau_d=tau_d,
        rs_d=rs_d,
    )


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


# ---------------------------------------------------------------------------
# Fused background + thermodynamics solve with reverse-mode-stable AD
# (issue #30, fix option 2: "vjp-through-jvp")
# ---------------------------------------------------------------------------
#
# Why this exists — https://github.com/smsharma/clax/issues/30:
# reverse-mode AD (jax.grad) through thermodynamics_solve carries a ~2% error
# on h-like parameters. The Peebles/RECFAST recombination rates contain
# Boltzmann-exponential ratios (exp(B/kT) ~ e^52, cf. _recfast_dxHII_dlna:
# Rup = Rdown*(CR*TM)^{3/2}*exp(-CDB/TM), wrap_recfast.c:133-134), so AD
# intermediates reach ~1e13. Forward mode pairs huge x tiny factors locally
# per grid point — nothing large is ever formed, and jax.jvp was verified
# exact against converged finite differences six independent ways. Reverse
# mode must contract thousands of +-1e13-scale cotangent terms through shared
# scalars whose true total is ~1e-3 (or exactly 0); float64 keeps a
# deterministic ULP residue (measured: xe.y full-table reverse "derivative"
# = 2^-9 exactly; thermo-chain reverse 8.66e7 where the truth is 1.16e5).
#
# The fix: a custom reverse rule that computes the SAME mathematical object,
# J^T ct, by re-association: (J^T ct)_i = <ct, J e_i>, evaluating the columns
# J e_i with a batched forward-mode basis (jax.jacfwd) over the ~20 traced
# CosmoParams leaves. This replaces the numerically pathological contraction
# order with the proven-exact forward order — identical arithmetic content,
# no approximation, no fudge factors. Cost: one extra primal + ~20 fused
# forward tangent passes in the backward (a few times one bg+thermo solve;
# negligible next to any perturbation solve).
#
# Both bg and th cotangents are handled here (hence the FUSED entry point):
# the fused function's only differentiable input is CosmoParams, so the
# entire bg -> th chain — including the bg-mediated channel whose reverse
# health was never established — is covered by the forward basis.


def _solve_bg_th_native(
    params: CosmoParams,
    prec: PrecisionParams,
) -> tuple[BackgroundResult, ThermoResult]:
    """Plain (native-AD) fused solve: background then thermodynamics.

    Exactly the two-call sequence every pipeline entry point used before
    issue #30; kept as a separate function so the stable custom_vjp wrapper
    and the "native" mode share one primal implementation.
    """
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    return bg, th


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _solve_bg_th_stable(
    params: CosmoParams,
    prec: PrecisionParams,
) -> tuple[BackgroundResult, ThermoResult]:
    """Fused solve wrapped in jax.custom_vjp (reverse-mode-stable rule).

    Primal is identical to the native path. Only the reverse rule differs:
    see _solve_bg_th_stable_bwd. NOTE: jax.custom_vjp blocks forward-mode
    (jax.jvp raises TypeError); forward-mode users must select
    th_grad_mode="native" (cf. tests/test_pk_forward_mode.py).
    """
    return _solve_bg_th_native(params, prec)


def _solve_bg_th_stable_fwd(params, prec):
    """custom_vjp forward rule: run the primal, save params as residual.

    The backward pass re-derives everything it needs from params via the
    forward basis, so params is the only residual (the primal outputs are
    recomputed inside the batched jacfwd pass, sharing one primal across
    all ~20 tangent directions).
    """
    return _solve_bg_th_native(params, prec), params


def _solve_bg_th_stable_bwd(prec, params, cotangents):
    """custom_vjp backward rule: params cotangent via a forward-mode basis.

    Computes bar_params = J^T ct where J = d(bg, th)/d(params), as
        bar_theta_i = d/dtheta_i <ct, (bg, th)(theta)>,
    i.e. the gradient of the scalar contraction of the (constant) output
    cotangent with the fused solve, evaluated by jax.jacfwd over the stacked
    CosmoParams leaves theta. This is mathematically identical to the native
    VJP — the associativity of the chain-rule contraction is the ONLY thing
    that changes — but every product huge*tiny is formed locally per grid
    point (forward mode), so no +-1e13-scale partial sums ever exist. The
    final <ct, tangent> reduction only sums output-sized, output-scaled
    terms — the same benign reduction any VJP must perform.

    ode_adjoint="direct" is forced INSIDE the basis only: background_solve's
    production default RecursiveCheckpointAdjoint is an eqx.filter_custom_vjp
    (diffrax/_adjoint.py:538) which blocks jvp, while DirectAdjoint supports
    both AD modes. This does not alter caller-visible semantics: the primal
    result the caller sees was produced under the caller's own prec, and the
    adjoint choice only selects how derivatives are propagated, not the
    solution. thermodynamics_solve itself is a semi-implicit lax.scan with
    no diffrax solve, so it is jvp-transparent under any prec.

    Verified (2026-08-29, CPU probe on the exact construction below):
    grad(stable) matches jvp to 1.2e-16..8.1e-15 relative across
    {sum(xe^2), random-linear, sum(g^2)} x {h, omega_b}, where the native
    reverse showed 5.1e-11 (well-scaled) up to 5.2e+1 (near-zero-derivative)
    relative error on the same block.
    """
    prec_fwd = dataclass_replace(prec, ode_adjoint="direct")
    leaves, treedef = jax.tree_util.tree_flatten(params)
    theta = jnp.stack([jnp.asarray(leaf) for leaf in leaves])

    def contracted(theta_vec):
        p = jax.tree_util.tree_unflatten(
            treedef, [theta_vec[i] for i in range(len(leaves))]
        )
        out = _solve_bg_th_native(p, prec_fwd)
        # <ct, out>: all BackgroundResult/ThermoResult leaves are float
        # arrays or scalars (no integer leaves), so a plain vdot per leaf
        # is well-defined. ct is treated as a constant here — its gradient
        # contribution is exactly the VJP row we want.
        products = jax.tree_util.tree_map(
            lambda ct_leaf, out_leaf: jnp.vdot(ct_leaf, out_leaf),
            cotangents, out,
        )
        return jax.tree_util.tree_reduce(lambda a, b: a + b, products)

    grad_vec = jax.jacfwd(contracted)(theta)
    params_bar = jax.tree_util.tree_unflatten(
        treedef, [grad_vec[i] for i in range(len(leaves))]
    )
    return (params_bar,)


_solve_bg_th_stable.defvjp(_solve_bg_th_stable_fwd, _solve_bg_th_stable_bwd)


def solve_background_and_thermo(
    params: CosmoParams,
    prec: PrecisionParams = PrecisionParams(),
) -> tuple[BackgroundResult, ThermoResult]:
    """Fused background + thermodynamics solve, (params, prec) -> (bg, th).

    The ONLY differentiable input is params (prec is static), which lets a
    single custom reverse rule cover the whole bg -> th chain. Dispatch is
    on the static flag prec.th_grad_mode:

    - "stable" (default): jax.custom_vjp whose backward pass evaluates the
      params cotangent for BOTH outputs via a batched forward-mode basis
      ("vjp-through-jvp") — fixes the issue #30 recombination-era
      catastrophic-cancellation error in native reverse mode. jax.jvp
      through this mode raises TypeError (custom_vjp caveat).
    - "native": plain background_solve + thermodynamics_solve — bitwise the
      pre-issue-#30 behavior in both primal and AD; required for
      forward-mode (jax.jvp / jax.jacfwd) users.

    The primal outputs are identical in both modes. Existing public APIs
    background_solve(params, prec) and thermodynamics_solve(params, prec,
    bg) are unchanged and remain fully supported; this entry point is what
    the main pipeline functions (clax.compute, clax.compute_pk_table,
    clax.compute_pk) route through.

    Args:
        params: cosmological parameters (JAX-traced)
        prec: precision parameters (static)

    Returns:
        (BackgroundResult, ThermoResult)
    """
    if prec.th_grad_mode == "stable":
        return _solve_bg_th_stable(params, prec)
    if prec.th_grad_mode == "native":
        return _solve_bg_th_native(params, prec)
    raise ValueError(
        f"Unknown PrecisionParams.th_grad_mode: {prec.th_grad_mode!r} "
        f"(expected 'stable' or 'native')"
    )
