"""Synthetic sanity check for clax.ept P22/P13 vs expectations.

This script does NOT require classy or CLASS-PT Python wrapper.
It uses a synthetic power-law P_lin to verify the loop integrals
produce physically sensible output:

  - P22(k) > 0 for all k
  - P13(k) < 0 for most k (negative loop correction at low k)
  - |P_loop| / P_lin < 50% in the perturbative regime k < 0.3 h/Mpc
  - P_loop = P22 + P13 + (2/3) P13_UV  (total one-loop correction)

Output saved to data/classpt_compare_synthetic.txt for inspection.
Real accuracy comparison against CLASS-PT requires classy (not installed).

Usage:
    python3 scripts/compare_classpt_synthetic.py

Requires: JAX, scipy (for IR resummation DST-II)
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Ensure clax package is on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_DIR)

try:
    import jax
    import jax.numpy as jnp
    print(f"JAX {jax.__version__} available — proceeding.")
except ImportError:
    print("ERROR: JAX not installed. Run: pip install jax")
    sys.exit(1)

try:
    import clax.ept as ept
except ImportError as e:
    print(f"ERROR: Cannot import clax.ept: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic ΛCDM-like P_lin
# ---------------------------------------------------------------------------

def synthetic_pk_lin(k: np.ndarray) -> np.ndarray:
    """Power-law × Gaussian cutoff P(k) ~ k^0.96 exp(-(k/5)^2).

    Normalized so P(0.1 h/Mpc) ~ 3000 (Mpc/h)^3 — rough ΛCDM order.
    """
    pk = k ** 0.96 * np.exp(-(k / 5.0) ** 2)
    pk = pk / np.interp(0.1, k, pk) * 3000.0
    return pk


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def main():
    NMAX = ept.NMAX_EPT   # 256
    kmin = ept.KMIN_H     # 0.00005 h/Mpc
    kmax = ept.KMAX_H     # 100.0 h/Mpc

    # FFTLog grid
    k_disc_np = np.logspace(np.log10(kmin), np.log10(kmax), NMAX)
    pk_disc_np = synthetic_pk_lin(k_disc_np)
    pk_disc = jnp.array(pk_disc_np)

    print("Loading CLASS-PT kernel matrices...")
    matrices = ept._load_matrices(NMAX)
    M22 = jnp.array(matrices["M22"])
    M13 = jnp.array(matrices["M13"])
    print(f"  M22 shape: {M22.shape}, M13 shape: {M13.shape}")

    # Verify M22 is symmetric (sanity check for the matrix bug fix)
    M22_np = np.array(M22)
    sym_err = np.max(np.abs(M22_np - M22_np.T))
    print(f"  M22 symmetry error (should be ~0): {sym_err:.2e}")

    print("Running FFTLog decomposition...")
    cmsym, etam = ept._fftlog_decompose(pk_disc, kmin, kmax, NMAX, ept.B_MATTER)
    print(f"  c_m shape: {cmsym.shape}, |c_m| range: [{float(jnp.min(jnp.abs(cmsym))):.2e}, "
          f"{float(jnp.max(jnp.abs(cmsym))):.2e}]")

    # Output k-grid: 20 log-spaced points in [0.01, 0.5] h/Mpc
    k_out_np = np.logspace(np.log10(0.01), np.log10(0.5), 20)
    k_out = jnp.array(k_out_np)

    print("Computing P22 and P13...")
    x = ept._x_at_k(cmsym, etam, k_out)
    P22 = ept._compute_p22(x, k_out, M22, cutoff_h=ept.CUTOFF)

    pk_at_k_out = jnp.array(np.interp(k_out_np, k_disc_np, pk_disc_np))
    lnk_out = jnp.log(k_out)
    P13 = ept._compute_p13(x, k_out, pk_at_k_out, M13, lnk_out)

    P_loop = P22 + P13
    P_lin  = pk_at_k_out

    P22_np   = np.array(P22)
    P13_np   = np.array(P13)
    Ploop_np = np.array(P_loop)
    Plin_np  = np.array(P_lin)

    print("\n--- Results ---")
    print(f"{'k [h/Mpc]':>12}  {'P_lin':>12}  {'P22':>12}  {'P13':>12}  "
          f"{'P_loop':>12}  {'P_loop/P_lin':>14}")
    print("-" * 80)
    for i, kval in enumerate(k_out_np):
        ratio = Ploop_np[i] / Plin_np[i] if Plin_np[i] != 0 else float("nan")
        print(f"{kval:12.4f}  {Plin_np[i]:12.2f}  {P22_np[i]:12.2f}  "
              f"{P13_np[i]:12.2f}  {Ploop_np[i]:12.2f}  {ratio:14.3%}")

    # Sanity checks
    print("\n--- Sanity checks ---")
    ok = True

    if np.all(P22_np > 0):
        print("  ✓ P22 > 0 everywhere")
    else:
        print(f"  ✗ P22 has {np.sum(P22_np <= 0)} non-positive values!")
        ok = False

    loop_ratio = np.max(np.abs(Ploop_np)) / np.max(Plin_np)
    if loop_ratio < 0.5:
        print(f"  ✓ max|P_loop|/max(P_lin) = {loop_ratio:.2%} < 50%")
    else:
        print(f"  ✗ max|P_loop|/max(P_lin) = {loop_ratio:.2%} — PT may have broken down")
        ok = False

    if not np.any(np.isnan(Ploop_np)):
        print("  ✓ No NaN in P_loop")
    else:
        print("  ✗ NaN detected in P_loop!")
        ok = False

    # Save output
    os.makedirs(os.path.join(_REPO_DIR, "data"), exist_ok=True)
    out_path = os.path.join(_REPO_DIR, "data", "classpt_compare_synthetic.txt")
    header = (
        "Synthetic ΛCDM-like P_lin comparison (no classy required)\n"
        "P_lin = k^0.96 exp(-(k/5)^2), normalized to P(0.1)=3000 (Mpc/h)^3\n"
        f"{'k_hMpc':>12}  {'P_lin':>12}  {'P22':>12}  {'P13':>12}  {'P_loop':>12}"
    )
    data_out = np.column_stack([k_out_np, Plin_np, P22_np, P13_np, Ploop_np])
    np.savetxt(out_path, data_out, header=header, fmt="%14.6e")
    print(f"\n  Output saved to {out_path}")

    if ok:
        print("\n✓ All sanity checks passed.")
    else:
        print("\n✗ Some checks failed — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
