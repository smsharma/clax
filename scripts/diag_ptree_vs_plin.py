#!/usr/bin/env python3
"""Diagnostic: Compare CLASS-PT's internal P_tree (pk_mult[14]) against cosmo.pk_lin().

If P_tree from CLASS-PT ≠ pk_lin we feed to JAX, that would explain the scale-dependent
P_loop discrepancy (since P_loop = P22 + P13 both depend on P_lin via FFTLog input).

Also tests: what sigma_v does CLASS-PT use for the UV counterterm?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from clax.ept import ept_kgrid, EPTPrecisionParams, _load_matrices, NMAX_EPT, B_MATTER, KMIN_H, KMAX_H

Z = 0.61

PARAMS = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'A_s': 2.089e-9, 'n_s': 0.9649, 'tau_reio': 0.052,
    'YHe': 0.2425, 'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
}

def main():
    from classy import Class

    # Run CLASS-PT
    params = dict(PARAMS)
    params.update({
        'output': 'mPk',
        'non linear': 'PT',
        'IR resummation': 'No',
        'Bias tracers': 'No',
        'cb': 'No',
        'RSD': 'No',
        'P_k_max_h/Mpc': 100.0,
        'z_pk': Z,
    })
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    h = cosmo.h()
    prec = EPTPrecisionParams()
    k_h = ept_kgrid(prec)     # h/Mpc, N=256
    k_mpc = k_h * h            # 1/Mpc

    # --- Compare P_lin sources ---
    # Source 1: cosmo.pk_lin(k_1/Mpc, z) * h^3 → (Mpc/h)^3
    pk_lin_pkfunc = np.array([cosmo.pk_lin(k, Z) for k in k_mpc]) * h**3

    # Source 2: cosmo.pk (non-linear) at z — is it the same as pk_lin for PT mode?
    # CLASS-PT doesn't modify pk, so this should equal pk_lin_pkfunc

    # Source 3: get pk_mult[14] = P_tree via get_pk_mult
    # initialize_output first
    cosmo.initialize_output(k_mpc, Z, len(k_mpc))

    # Attempt to get pk_mult array per k
    # pk_mult is an array of 30+ components per k; index 14 = P_tree
    try:
        # Try calling get_pk_mult at single k values
        # classy.pyx: get_pk_mult(double k, double z, int k_size) → array
        # Actually signature may differ by version. Try one k first.
        k_test = k_mpc[120]  # roughly k=0.046 h/Mpc
        mult = cosmo.get_pk_mult(k_test, Z, 1)
        print(f"pk_mult has {len(mult)} components")
        ptree_test = mult[14] * h**3
        plin_test = cosmo.pk_lin(k_test, Z) * h**3
        print(f"At k={k_h[120]:.4f} h/Mpc:")
        print(f"  P_tree (pk_mult[14]*h^3) = {ptree_test:.6f}")
        print(f"  pk_lin(k_mpc, z) * h^3  = {plin_test:.6f}")
        print(f"  ratio P_tree/pk_lin      = {ptree_test/plin_test:.8f}")
    except Exception as e:
        print(f"get_pk_mult single-k failed: {e}")
        # Try with the array version
        try:
            mult_arr = cosmo.get_pk_mult(k_mpc, Z, len(k_mpc))
            print(f"get_pk_mult(array): shape={np.array(mult_arr).shape}")
        except Exception as e2:
            print(f"get_pk_mult array also failed: {e2}")

    # --- Check sigma_v from CLASS-PT ---
    # CLASS-PT computes sigma_v on its own kdisc grid (0.00005*h to 100*h Mpc^-1)
    # Let's compute sigma_v on that exact grid
    kmin_disc_mpc = 0.00005 * h   # Mpc^-1
    kmax_disc_mpc = 100.0 * h      # Mpc^-1
    # N=256 log-spaced grid
    k_disc_mpc = np.exp(np.linspace(np.log(kmin_disc_mpc), np.log(kmax_disc_mpc), 256))
    pk_disc = np.array([cosmo.pk_lin(k, Z) for k in k_disc_mpc]) * h**3  # (Mpc/h)^3

    lnk_disc = np.log(k_disc_mpc / h)  # log of k in h/Mpc
    sigma2_v_disc = np.trapz(pk_disc * (k_disc_mpc/h), lnk_disc) / (6.0 * np.pi**2)
    print(f"\nsigma_v^2 on kdisc grid (h-units integral): {sigma2_v_disc:.6f} (Mpc/h)^2")

    # Also compute on our k_h grid (should match)
    lnk_h = np.log(k_h)
    sigma2_v_kh = np.trapz(pk_lin_pkfunc * k_h, lnk_h) / (6.0 * np.pi**2)
    print(f"sigma_v^2 on k_h grid (same range):         {sigma2_v_kh:.6f} (Mpc/h)^2")

    # Compare P_tree and pk_lin
    print("\n=== P_tree vs pk_lin comparison (check if identical) ===")
    print(f"{'k(h/Mpc)':>10} {'pk_lin_h':>14} {'ratio':>12}")
    test_idx = [100, 110, 120, 130, 140, 150, 160, 170, 180]
    for i in test_idx:
        print(f"{k_h[i]:>10.4f} {pk_lin_pkfunc[i]:>14.4f}")

    # --- Now run the FFTLog and compute P13, P22 exactly as CLASS-PT does ---
    # CLASS-PT feeds Pbin = P_lin (with IR=No) to FFTLog
    # The kdisc grid points ARE k_h * h in 1/Mpc, but the FFTLog runs in h-units

    # Check: does CLASS-PT internally use k in h/Mpc or 1/Mpc for the FFTLog matrix product?
    # From nonlinear_pt.c:
    #   kmin = pnlpt->k[0] (in 1/Mpc)
    #   kmax = pnlpt->k[pnlpt->k_size-1] (in 1/Mpc)
    #   In the cmsym formula: kmin^{-etam} → this k_min is in 1/Mpc!
    # But in the P1loop output:
    #   P1loop[j] is stored in Mpc^3 units
    # Then in classy.pyx: pk_mm_real returns P * h^3 → (Mpc/h)^3

    print("\n=== FFTLog in 1/Mpc units test ===")
    # Redo the FFTLog in 1/Mpc units (as CLASS-PT does)
    nmax = NMAX_EPT
    kmin_mpc_disc = kmin_disc_mpc  # 1/Mpc
    kmax_mpc_disc = kmax_disc_mpc  # 1/Mpc

    step_mpc = np.log(kmax_mpc_disc / kmin_mpc_disc) / (nmax - 1)
    m = np.arange(nmax + 1)
    j_m = m - nmax // 2
    etam_mpc = B_MATTER + 2j * np.pi * j_m / (nmax * step_mpc)

    # P_lin on kdisc: in Mpc^3 (NOT h-units!) — CLASS-PT internal units
    pk_disc_mpc3 = np.array([cosmo.pk_lin(k, Z) for k in k_disc_mpc])  # Mpc^3

    j_idx = np.arange(nmax)
    input_mpc = pk_disc_mpc3 * np.exp(-j_idx * B_MATTER * step_mpc)
    cm_mpc = np.fft.fft(input_mpc)
    nmax2 = nmax // 2
    cm_sym_mpc = np.concatenate([np.conj(cm_mpc[nmax2 - np.arange(nmax2)]),
                                   cm_mpc[np.arange(nmax2+1)]]) / nmax
    cm_sym_mpc *= np.exp(-etam_mpc * np.log(kmin_mpc_disc))
    cm_sym_mpc[0] *= 0.5; cm_sym_mpc[-1] *= 0.5

    # Load matrices
    mats = _load_matrices(nmax)
    M13  = mats["M13"]
    M22  = mats["M22"]

    # Evaluate x at k in 1/Mpc
    x_mpc = cm_sym_mpc[None, :] * np.exp(etam_mpc[None, :] * np.log(k_disc_mpc)[:, None])

    # P22 in Mpc^3 units
    y_mpc = x_mpc @ M22
    f22_mpc = np.sum(x_mpc * y_mpc, axis=-1)
    cutoff_mpc = 10.0 * h  # 10 h/Mpc in 1/Mpc
    uv_mpc = np.exp(-(k_disc_mpc / cutoff_mpc)**6)
    P22_mpc3 = np.real(k_disc_mpc**3 * f22_mpc) * uv_mpc

    # P13 in Mpc^3 units
    f13_mpc = np.sum(x_mpc * M13[None, :], axis=-1)
    P13_raw_mpc3 = np.real(k_disc_mpc**3 * f13_mpc * pk_disc_mpc3)
    # sigma_v in 1/Mpc units: sigma_v^2 = (1/6pi^2) int P(k) dk = ...
    lnk_disc_mpc = np.log(k_disc_mpc)
    sigma2_v_mpc = np.trapz(pk_disc_mpc3 * k_disc_mpc, lnk_disc_mpc) / (6.0 * np.pi**2)
    P13_UV_mpc3 = -(61.0/105.0) * sigma2_v_mpc * k_disc_mpc**2 * pk_disc_mpc3
    P13_mpc3 = P13_raw_mpc3 + P13_UV_mpc3

    Ploop_mpc3 = P13_mpc3 + P22_mpc3
    Ploop_h3 = Ploop_mpc3 * h**3  # convert to (Mpc/h)^3

    print(f"sigma_v^2 (1/Mpc units, integral over kdisc in 1/Mpc) = {sigma2_v_mpc:.6f} Mpc^2")
    print(f"sigma_v^2 in (Mpc/h)^2 = {sigma2_v_mpc * h**(-2):.6f}   (should equal {sigma2_v_kh:.6f})")
    # Note: sigma_v in Mpc^2 × 1/h^2 = sigma_v in (Mpc/h)^2

    # Get CLASS-PT reference P_loop
    pk_mm_arr = np.array(cosmo.pk_mm_real(0.0))  # already at k_mpc grid
    pk_lin_arr = pk_lin_pkfunc  # at k_h grid (same k)
    Ploop_classpt = pk_mm_arr - pk_lin_arr

    # Compare 1/Mpc computation vs CLASS-PT
    print("\n=== 1/Mpc-units numpy vs CLASS-PT ===")
    print(f"{'k(h/Mpc)':>10} {'Ploop_ref':>12} {'Ploop_mpc_np':>14} {'err%':>8} {'P22(h)':>12} {'P13(h)':>12}")
    print("─" * 74)
    for i in [100, 110, 120, 125, 130, 135, 140, 145, 150, 155, 160]:
        if i < len(k_h):
            ref = Ploop_classpt[i]
            val = Ploop_h3[i]
            err = (val - ref) / (abs(ref) + 1e-10) * 100
            print(f"{k_h[i]:>10.4f} {ref:>12.4f} {val:>14.4f} {err:>8.2f}% "
                  f"{P22_mpc3[i]*h**3:>12.4f} {P13_mpc3[i]*h**3:>12.4f}")

    # Now compare 1/Mpc units vs h-unit computation
    print("\n=== h-unit numpy vs 1/Mpc-unit numpy ===")
    # Redo h-unit computation
    step_h = np.log(KMAX_H / KMIN_H) / (nmax - 1)
    etam_h = B_MATTER + 2j * np.pi * j_m / (nmax * step_h)
    pk_h_arr = pk_lin_pkfunc   # (Mpc/h)^3
    inp_h = pk_h_arr * np.exp(-j_idx * B_MATTER * step_h)
    cm_h  = np.fft.fft(inp_h)
    cm_sym_h = np.concatenate([np.conj(cm_h[nmax2 - np.arange(nmax2)]),
                                cm_h[np.arange(nmax2+1)]]) / nmax
    cm_sym_h *= np.exp(-etam_h * np.log(KMIN_H))
    cm_sym_h[0] *= 0.5; cm_sym_h[-1] *= 0.5

    x_h = cm_sym_h[None, :] * np.exp(etam_h[None, :] * np.log(k_h)[:, None])
    y_h = x_h @ M22
    f22_h = np.sum(x_h * y_h, axis=-1)
    P22_h_np = np.real(k_h**3 * f22_h) * np.exp(-(k_h / 10.0)**6)

    f13_h = np.sum(x_h * M13[None, :], axis=-1)
    P13_raw_h = np.real(k_h**3 * f13_h * pk_h_arr)
    sigma2_v_h = np.trapz(pk_h_arr * k_h, np.log(k_h)) / (6.0 * np.pi**2)
    P13_UV_h = -(61.0/105.0) * sigma2_v_h * k_h**2 * pk_h_arr
    P13_h_np = P13_raw_h + P13_UV_h
    Ploop_h_np = P13_h_np + P22_h_np

    print(f"{'k(h/Mpc)':>10} {'Ploop_1/Mpc':>14} {'Ploop_h-unit':>14} {'diff%':>8}")
    for i in [100, 110, 120, 125, 130, 135, 140, 145, 150, 155, 160]:
        if i < len(k_h):
            v_mpc = Ploop_h3[i]
            v_h   = Ploop_h_np[i]
            diff  = (v_h - v_mpc) / (abs(v_mpc) + 1e-10) * 100
            print(f"{k_h[i]:>10.4f} {v_mpc:>14.4f} {v_h:>14.4f} {diff:>8.3f}%")

    cosmo.struct_cleanup()
    cosmo.empty()
    print("\nDone.")

if __name__ == "__main__":
    main()
