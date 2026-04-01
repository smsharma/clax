#!/usr/bin/env python3
"""Diagnostic: P_loop component comparison (P13 vs P22) against CLASS-PT.

Runs CLASS-PT with IR=No, cb=No, RSD=No, then compares P_lin, P13, P22
step by step to isolate the source of the ~10-30% scale-dependent error.

Usage:
    python scripts/diag_ploop_components.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp

from clax.ept import (
    _fftlog_decompose, _compute_p13, _compute_p22, _x_at_k,
    _load_matrices, ept_kgrid, EPTPrecisionParams, NMAX_EPT,
    B_MATTER, KMIN_H, KMAX_H, CUTOFF,
)

PYTHON = "/Users/nguyenmn/miniconda3/envs/sbi_pytorch_osx-arm64-py310forge/bin/python3"

# ─── Cosmological parameters ────────────────────────────────────────────────
PARAMS = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'A_s': 2.089e-9, 'n_s': 0.9649, 'tau_reio': 0.052,
    'YHe': 0.2425, 'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
}
Z = 0.61

def run_classpt():
    """Run CLASS-PT and return pk_lin and pk_loop on the JAX k-grid."""
    from classy import Class

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
    f = cosmo.scale_independent_growth_factor_f(Z)

    prec = EPTPrecisionParams()
    k_h = ept_kgrid(prec)   # h/Mpc, N=256
    k_mpc = k_h * h         # 1/Mpc — CLASS expects this

    # Get P_lin on our k-grid
    pk_lin_mpc3 = np.array([cosmo.pk_lin(k, Z) for k in k_mpc])
    pk_lin_h = pk_lin_mpc3 * h**3   # (Mpc/h)^3

    # Get CLASS-PT P_loop via initialize_output + pk_mm_real(cs=0)
    # initialize_output takes k in 1/Mpc
    cosmo.initialize_output(k_mpc, Z, len(k_mpc))
    pk_mm = np.array(cosmo.pk_mm_real(0.0))  # cs=0 → P_tree + P_loop

    # pk_mm_real = (pk_mult[0] + pk_mult[14] + 2*cs*pk_mult[10]/h²) * h³
    # cs=0 → pk_mm = (P_loop + P_tree) * h³
    # But these are already in h-units per classy.pyx — let me check by
    # computing P_tree manually and subtracting.

    # Also get what CLASS gives as pk_lin at these k values via pk_lin separately
    # to compare against what CLASS-PT uses internally.

    cosmo.struct_cleanup()
    cosmo.empty()

    return k_h, k_mpc, pk_lin_h, pk_mm, float(h), float(f)


def run_jax_ept_no_ir(pk_lin_h, k_h, h, f):
    """Run JAX EPT with IR resummation disabled."""
    prec = EPTPrecisionParams(ir_resummation=False)

    nmax = prec.nmax
    kmin = prec.kmin_h
    kmax = prec.kmax_h
    b    = prec.b_matter
    cutoff_h = prec.cutoff_h

    mats = _load_matrices(nmax)
    M13  = jnp.array(mats["M13"])
    M22  = jnp.array(mats["M22"])

    pk = jnp.array(pk_lin_h)
    k  = jnp.array(k_h)
    lnk = jnp.log(k)

    cmsym, etam = _fftlog_decompose(pk, kmin, kmax, nmax, b)
    x = _x_at_k(cmsym, etam, k)

    P22 = _compute_p22(x, k, M22, cutoff_h)
    P13 = _compute_p13(x, k, pk, M13, lnk)

    return np.array(P13), np.array(P22)


def numpy_p13_p22_reference(pk_lin_h, k_h):
    """Pure numpy reference computation matching CLASS-PT exactly."""
    nmax = NMAX_EPT
    kmin = KMIN_H
    kmax = KMAX_H
    b    = B_MATTER
    cutoff_h = CUTOFF

    # Load matrices
    mats = _load_matrices(nmax)
    M13 = mats["M13"]
    M22 = mats["M22"]

    step = np.log(kmax / kmin) / (nmax - 1)

    # --- FFTLog decomposition (pure numpy) ---
    m = np.arange(nmax + 1)
    j_m = m - nmax // 2
    etam = b + 2j * np.pi * j_m / (nmax * step)

    j_idx = np.arange(nmax)
    input_arr = pk_lin_h * np.exp(-j_idx * b * step)
    cm_fft = np.fft.fft(input_arr)

    nmax2 = nmax // 2
    idx_low  = nmax2 - np.arange(nmax2)
    cm_sym_low  = np.conj(cm_fft[idx_low])
    idx_high = np.arange(nmax2 + 1)
    cm_sym_high = cm_fft[idx_high]
    cm_raw = np.concatenate([cm_sym_low, cm_sym_high]) / nmax
    cmsym = cm_raw * np.exp(-etam * np.log(kmin))
    cmsym[0]  *= 0.5
    cmsym[-1] *= 0.5

    # Evaluate x at k
    x = cmsym[None, :] * np.exp(etam[None, :] * np.log(k_h)[:, None])

    # --- P22 ---
    y   = x @ M22
    f22 = np.sum(x * y, axis=-1)
    uv  = np.exp(-(k_h / cutoff_h)**6)
    P22_np = np.real(k_h**3 * f22) * uv

    # --- P13 ---
    f13    = np.sum(x * M13[None, :], axis=-1)
    P13_raw = np.real(k_h**3 * f13 * pk_lin_h)
    lnk = np.log(k_h)
    sigma2_v = np.trapz(pk_lin_h * k_h, lnk) / (6.0 * np.pi**2)
    P13_UV  = -(61.0/105.0) * sigma2_v * k_h**2 * pk_lin_h
    P13_np  = P13_raw + P13_UV

    return P13_np, P22_np, sigma2_v


def main():
    print("Loading CLASS-PT...", flush=True)
    k_h, k_mpc, pk_lin_h, pk_mm_classpt, h, f = run_classpt()

    print(f"h = {h:.4f},  f = {f:.4f},  z = {Z}")
    print(f"k range: [{k_h.min():.4e}, {k_h.max():.4e}] h/Mpc, N={len(k_h)}")

    # Run JAX and numpy reference
    P13_jax, P22_jax = run_jax_ept_no_ir(pk_lin_h, k_h, h, f)
    P13_np,  P22_np, sigma2_v = numpy_p13_p22_reference(pk_lin_h, k_h)
    Ploop_jax = P13_jax + P22_jax

    # CLASS-PT gives pk_mm_real(cs=0) = P_tree + P_loop (in h-units)
    # We need P_tree to extract P_loop. With IR=No, P_tree = P_lin.
    Ploop_classpt = pk_mm_classpt - pk_lin_h   # P_loop = pk_mm - P_tree

    print(f"\nsigma_v^2 (JAX integrand over our grid) = {sigma2_v:.6f} (Mpc/h)^2")

    # Compare JAX vs numpy (should be identical)
    print("\n=== JAX vs numpy P13/P22 (should be 0%) ===")
    for i in [10, 50, 100, 150, 200, 250]:
        if i < len(k_h):
            print(f"k={k_h[i]:.4f}  P13: jax={P13_jax[i]:.4f} np={P13_np[i]:.4f} "
                  f"  P22: jax={P22_jax[i]:.4f} np={P22_np[i]:.4f}")

    # Main comparison
    test_idx = [10, 30, 50, 70, 100, 120, 140, 160, 180, 200]
    print(f"\n{'k(h/Mpc)':>10} {'Ploop_ref':>12} {'Ploop_jax':>12} {'err%':>8} "
          f"{'P22_jax':>12} {'P13_jax':>12} {'P22_ref?':>10}")
    print("─" * 90)

    for i in test_idx:
        if i < len(k_h):
            ref = Ploop_classpt[i]
            jax = Ploop_jax[i]
            err = (jax - ref) / (abs(ref) + 1e-10) * 100
            print(f"{k_h[i]:>10.4f} {ref:>12.4f} {jax:>12.4f} {err:>8.2f}% "
                  f"{P22_jax[i]:>12.4f} {P13_jax[i]:>12.4f}")

    # Check P_lin consistency
    print("\n=== P_lin sanity check ===")
    print(f"k=0.1 h/Mpc: pk_lin_h={pk_lin_h[100]:.4f} (Mpc/h)^3")
    print(f"k=0.3 h/Mpc: pk_lin_h={pk_lin_h[140]:.4f}")

    # Now try: what if we use P_lin from CLASS at z=0 × D(z)^2 vs pk_lin(k,z)?
    # Let's check if the issue is in sigma_v
    print(f"\nsigma_v^2 (our grid, full range 0.00005..100 h/Mpc) = {sigma2_v:.6f}")
    # Estimate sigma_v on a restricted range like CLASS-PT default P_k_max=10
    mask10 = k_h <= 10.0
    sigma2_v_10 = np.trapz(pk_lin_h[mask10] * k_h[mask10], np.log(k_h[mask10])) / (6.0 * np.pi**2)
    print(f"sigma_v^2 (k <= 10 h/Mpc)  = {sigma2_v_10:.6f}")
    mask1 = k_h <= 1.0
    sigma2_v_1 = np.trapz(pk_lin_h[mask1] * k_h[mask1], np.log(k_h[mask1])) / (6.0 * np.pi**2)
    print(f"sigma_v^2 (k <= 1.0 h/Mpc) = {sigma2_v_1:.6f}")

    # Check: what does CLASS-PT use for sigma_v?
    # nonlinear_pt.c line 5301: integral over pnlpt->k[] with P_k_max_h/Mpc=100
    # Let's compute what sigma_v would be on CLASS's internal k grid
    # (which uses 100 points from 0.0001 to P_k_max * h, roughly)
    # Try compute P_loop with sigma_v restricted to k <= P_k_max_1/Mpc
    print("\n=== P13 sensitivity to sigma_v truncation ===")
    for k_cut in [1.0, 5.0, 10.0, 50.0, 100.0]:
        mask = k_h <= k_cut
        sv2 = np.trapz(pk_lin_h[mask] * k_h[mask], np.log(k_h[mask])) / (6.0 * np.pi**2)
        # Recompute P13 with this sigma_v
        mats = _load_matrices(NMAX_EPT)
        M13 = mats["M13"]
        step = np.log(KMAX_H / KMIN_H) / (NMAX_EPT - 1)
        etam_arr = B_MATTER + 2j * np.pi * (np.arange(NMAX_EPT+1) - NMAX_EPT//2) / (NMAX_EPT * step)
        j_idx = np.arange(NMAX_EPT)
        inp = pk_lin_h * np.exp(-j_idx * B_MATTER * step)
        cm = np.fft.fft(inp)
        nmax2 = NMAX_EPT // 2
        cm_sym = np.concatenate([np.conj(cm[nmax2 - np.arange(nmax2)]),
                                   cm[np.arange(nmax2+1)]]) / NMAX_EPT
        cm_sym *= np.exp(-etam_arr * np.log(KMIN_H))
        cm_sym[0] *= 0.5; cm_sym[-1] *= 0.5
        x_tmp = cm_sym[None,:] * np.exp(etam_arr[None,:] * np.log(k_h)[:,None])
        f13 = np.sum(x_tmp * M13[None,:], axis=-1)
        P13r = np.real(k_h**3 * f13 * pk_lin_h)
        P13_UV_cut = -(61.0/105.0) * sv2 * k_h**2 * pk_lin_h
        P13_cut = P13r + P13_UV_cut

        # Compare at k≈0.1 and k≈0.3
        i01 = np.argmin(np.abs(k_h - 0.1))
        i03 = np.argmin(np.abs(k_h - 0.3))
        ref01 = Ploop_classpt[i01] - P22_jax[i01]  # P13 from CLASS-PT = Ploop - P22
        ref03 = Ploop_classpt[i03] - P22_jax[i03]
        print(f"  k_cut={k_cut:5.1f}: sv2={sv2:.4f}  "
              f"P13(0.1)={P13_cut[i01]:.3f}[ref≈{ref01:.3f}]  "
              f"P13(0.3)={P13_cut[i03]:.3f}[ref≈{ref03:.3f}]")

    # Check if P22 is correct (independent of sigma_v)
    print("\n=== P22 vs inferred from CLASS-PT ===")
    # P22_classpt ≈ Ploop - P13  (but P13 from CLASS-PT is unknown)
    # Instead, let's look at total P_loop ratio
    print(f"{'k(h/Mpc)':>10} {'Ploop_ref':>12} {'P22_jax':>12} {'P13_jax':>12} "
          f"{'P22/Ploop%':>12} {'P13/Ploop%':>12}")
    for i in test_idx[:8]:
        if i < len(k_h):
            ref = Ploop_classpt[i]
            print(f"{k_h[i]:>10.4f} {ref:>12.4f} {P22_jax[i]:>12.4f} {P13_jax[i]:>12.4f} "
                  f"{100*P22_jax[i]/(ref+1e-10):>12.1f}% {100*P13_jax[i]/(ref+1e-10):>12.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
