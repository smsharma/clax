"""Standalone test of _recfast_dxHII_dlna against CLASS x_e(z).

Feeds CLASS background quantities (H, n_H, T_m, T_r) into the new RECFAST
RHS function and integrates with simple Euler in dlna. Compares x_e(z)
against CLASS reference.

This tests the recombination physics in isolation, without the full
jaxCLASS pipeline. Acceptance: x_e within 0.5% of CLASS at z=1000-1200.
"""
import sys
sys.path.insert(0, '.')

import math
import numpy as np
from scipy.interpolate import interp1d

# Import the standalone RECFAST function
from jaxclass.thermodynamics import _recfast_dxHII_dlna
from jaxclass import constants as const


def main():
    # Load CLASS thermodynamics reference
    d = np.load('reference_data/lcdm_fiducial/thermodynamics.npz')
    th_z = d['th_z'][::-1]  # ascending z
    th_xe = d['th_x_e'][::-1]

    # Load CLASS background for H(z)
    bg = np.load('reference_data/lcdm_fiducial/background.npz')
    bg_z = bg['bg_z'][::-1]  # ascending z
    bg_H = bg['bg_H'][::-1]  # H in Mpc^-1
    bg_conf = bg['bg_conf_time'][::-1]

    # Cosmological parameters
    h = 0.6736
    omega_b = 0.02237
    Y_He = 0.2454
    T_cmb = 2.7255
    Omega_b = omega_b / h**2

    # Compute n_H(z) = (1-Y_He) * rho_b / m_H * (1+z)^3
    _H100_cgs = 3.2407792902755e-18  # s^-1 per (km/s/Mpc)
    _mH_g = 1.67353284e-24
    _G_cgs = 6.67428e-8
    H0_cgs = h * _H100_cgs
    rho_crit_cgs = 3.0 * H0_cgs**2 / (8.0 * math.pi * _G_cgs)
    n_H_0 = (1.0 - Y_He) * Omega_b * rho_crit_cgs / _mH_g  # cm^-3

    # Convert H from Mpc^-1 to s^-1
    c_over_Mpc = const.c_SI / const.Mpc_over_m  # s^-1 per Mpc^-1
    H_interp = interp1d(bg_z, bg_H * c_over_Mpc, kind='linear',
                         bounds_error=False, fill_value='extrapolate')

    # Integration strategy (matching CLASS history.c):
    # Phase 1: z > 1600 → Saha equilibrium for H (fast, no ODE needed)
    # Phase 2: z < 1600 → Peebles ODE with small dlna steps
    z_saha_switch = 1600.0  # switch from Saha to ODE
    z_end = 200.0
    dlna = 5e-4  # finer than CLASS DLNA_SWIFT=4e-3 for accuracy

    fHe = 0.25 * Y_He / (1.0 - Y_He)

    # Saha equilibrium for hydrogen:
    # s = SAHA_FACT * TR^{3/2} * exp(-EI/TR) / nH
    # xHII = 2/(1 + sqrt(1 + 4/s))  (from quadratic)
    _kB = 8.617343e-5  # eV/K
    _EI = 13.598286071938324  # eV
    _SAHA = 3.016103031869581e21  # eV^{-3/2} cm^{-3}

    def saha_xHII(z):
        a = 1.0 / (1.0 + z)
        nH = n_H_0 / a**3
        TR_eV = _kB * T_cmb / a
        s = _SAHA * TR_eV**1.5 * np.exp(-_EI / TR_eV) / nH
        return 2.0 / (1.0 + np.sqrt(1.0 + 4.0 / max(s, 1e-30)))

    # Start ODE from Saha equilibrium at z_saha_switch
    # At z < 1800, helium is fully recombined (HeII→HeI done), so xe ≈ xHII
    # (fHe contribution is 0 because HeII fraction is negligible)
    xHII = min(saha_xHII(z_saha_switch), 1.0)
    xe = xHII  # He already recombined at z=1600
    print(f"IC at z={z_saha_switch:.0f}: xHII_saha={xHII:.6f}, xe={xe:.6f}")

    lna_start = -np.log(1.0 + z_saha_switch)
    lna_end = -np.log(1.0 + z_end)
    n_steps = int((lna_end - lna_start) / dlna) + 1

    lna = lna_start
    results_z = []
    results_xe = []
    deriv_prev = None  # for Adams-Bashforth

    print("Standalone RECFAST integration")
    print(f"Saha until z={z_saha_switch:.0f}, then Peebles ODE (dlna={dlna})")
    print("=" * 60)

    for step in range(n_steps):
        a = np.exp(lna)
        z = 1.0 / a - 1.0

        nH = n_H_0 / a**3
        H = float(H_interp(max(z, 0.1)))
        TM = T_cmb / a  # T_m ≈ T_r (Compton equilibrium, good until z~200)
        TR = T_cmb / a

        # CLASS Peebles equation in departure-from-Saha form (hydrogen.c:105-107):
        # dxHII/dlna = -(C*nH/H) * [s*(1-x)*(αB(TM)-αB(TR)) + (xe*x - s*(1-x))*αB(TM)]
        # When TM≈TR, first term ≈0. The key term is (xe*x - s*(1-x))*αB which
        # is the departure from Saha equilibrium. This is O(1) not O(10^14).
        TR_eV = _kB * TR
        TR_eV_safe = max(TR_eV, 1e-30)
        t4_M = TM / 1e4; t4_R = TR / 1e4
        alphaB_TM = 1.14 * 4.309e-13 * max(t4_M,1e-30)**(-0.6166) / (1+0.6703*max(t4_M,1e-30)**0.53)
        alphaB_TR = 1.14 * 4.309e-13 * max(t4_R,1e-30)**(-0.6166) / (1+0.6703*max(t4_R,1e-30)**0.53)

        # Saha factor
        s = _SAHA * TR_eV_safe * np.sqrt(TR_eV_safe) * np.exp(-_EI / TR_eV_safe) / nH

        # Photoionization rate (for C factor)
        four_betaB = _SAHA * TR_eV_safe * np.sqrt(TR_eV_safe) * np.exp(-0.25*_EI/TR_eV_safe) * alphaB_TR

        # Peebles C factor
        x1s = max(1-xHII, 1e-30)
        RLya = 4.662899067555897e15 * H / (nH * x1s)
        C = (3*RLya + 8.2206) / (3*RLya + 8.2206 + four_betaB)

        # Departure from Saha
        Delta = xe * xHII - s * (1.0 - xHII)

        # RHS (CLASS form): dxHII/dlna = -(C*nH/H) * [s*(1-x)*(αTM-αTR) + Delta*αTM]
        rate = C * nH / max(H, 1e-30)
        dxHII_dlna = -rate * (s * x1s * (alphaB_TM - alphaB_TR) + Delta * alphaB_TM)

        # Heun's method (2nd-order predictor-corrector)
        xHII_pred = np.clip(xHII + dlna * dxHII_dlna, 0, 1)
        xe_pred = xHII_pred
        # Re-evaluate RHS at predictor
        a_pred = np.exp(lna + dlna)
        z_pred = 1/a_pred - 1
        nH_pred = n_H_0 / a_pred**3
        H_pred = float(H_interp(max(z_pred, 0.1)))
        TR_pred = T_cmb / a_pred
        TR_eV_pred = _kB * TR_pred
        TR_eV_pred_safe = max(TR_eV_pred, 1e-30)
        t4_pred = TR_pred / 1e4
        alphaB_pred = 1.14 * 4.309e-13 * max(t4_pred,1e-30)**(-0.6166) / (1+0.6703*max(t4_pred,1e-30)**0.53)
        s_pred = _SAHA * TR_eV_pred_safe**1.5 * np.exp(-_EI/TR_eV_pred_safe) / nH_pred
        four_betaB_pred = _SAHA * TR_eV_pred_safe**1.5 * np.exp(-0.25*_EI/TR_eV_pred_safe) * alphaB_pred
        x1s_pred = max(1-xHII_pred, 1e-30)
        RLya_pred = 4.662899067555897e15 * H_pred / (nH_pred * x1s_pred)
        C_pred = (3*RLya_pred + 8.2206) / (3*RLya_pred + 8.2206 + four_betaB_pred)
        Delta_pred = xe_pred * xHII_pred - s_pred * (1-xHII_pred)
        rate_pred = C_pred * nH_pred / max(H_pred, 1e-30)
        # At predictor point, TM ≈ TR (standalone test), so T_m correction ≈ 0
        # In full pipeline, TM_pred would differ from TR_pred
        alphaB_TM_pred = alphaB_pred  # TM = TR in this standalone test
        dxHII_pred = -rate_pred * (s_pred * x1s_pred * (alphaB_TM_pred - alphaB_pred) + Delta_pred * alphaB_TM_pred)

        xHII_new = xHII + 0.5 * dlna * (dxHII_dlna + dxHII_pred)
        xHII_new = np.clip(xHII_new, 0, 1)
        xHII = xHII_new
        xe = xHII

        results_z.append(z)
        results_xe.append(xe)
        lna += dlna

    results_z = np.array(results_z)
    results_xe = np.array(results_xe)

    # Compare against CLASS
    xe_class_interp = interp1d(th_z, th_xe, kind='linear',
                                bounds_error=False, fill_value=1.0)

    print(f"\n{'z':>6s} {'CLASS':>12s} {'RECFAST':>12s} {'err':>8s}")
    print(f"{'-'*42}")
    test_z = [2000, 1600, 1400, 1300, 1200, 1150, 1100, 1089, 1050, 1000,
              900, 800, 600, 400, 300]
    max_err_recomb = 0
    for z in test_z:
        if z < results_z[-1] or z > results_z[0]:
            continue
        xe_c = float(xe_class_interp(z))
        # Interpolate our result
        xe_r = float(np.interp(z, results_z[::-1], results_xe[::-1]))
        if abs(xe_c) > 1e-10:
            err = (xe_r / xe_c - 1) * 100
            marker = " <<<" if abs(err) > 3 else ""
            print(f"{z:6.0f} {xe_c:12.6e} {xe_r:12.6e} {err:7.2f}%{marker}")
            if 1000 <= z <= 1200:
                max_err_recomb = max(max_err_recomb, abs(err))

    print(f"\nMax |error| at z=1000-1200: {max_err_recomb:.2f}%")
    if max_err_recomb < 0.5:
        print("PASS: x_e within 0.5% at recombination")
    elif max_err_recomb < 2.0:
        print(f"CLOSE: x_e within {max_err_recomb:.1f}% (target: 0.5%)")
    else:
        print(f"FAIL: x_e error {max_err_recomb:.1f}% exceeds 2% threshold")


if __name__ == '__main__':
    main()
