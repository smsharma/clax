#!/usr/bin/env python3
"""Validate clax.ept against CLASS-PT reference and generate all figures.

Runs all 9 spectra, reports max errors, generates figures in notebooks/figures/.
Marks failing spectra (>1%) with FAIL in titles.

Usage:
    ~/miniconda3/envs/sbi_pytorch_osx-arm64-py310forge/bin/python3 scripts/validate_and_plot.py
"""

from __future__ import annotations
import os, sys
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from clax.ept import (
    compute_ept, EPTPrecisionParams, ept_kgrid,
    pk_mm_real, pk_gg_real, pk_gm_real,
    pk_mm_l0, pk_mm_l2, pk_mm_l4,
    pk_gg_l0, pk_gg_l2, pk_gg_l4,
)

# ── Load reference ────────────────────────────────────────────────────────────
# Use full-range 256-pt reference (same k-grid as EPT internal, no extrapolation needed)
REF = os.path.join(_REPO, "reference_data", "classpt_z0.38_fullrange.npz")
if not os.path.isfile(REF):
    print(f"ERROR: reference not found: {REF}")
    print("Run: python3 scripts/gen_reference_fullrange.py")
    sys.exit(1)
ref = np.load(REF)

k_h_ref = ref["k_h"]        # 256-point, [5e-5, 100] h/Mpc — matches EPT internal grid
pk_lin_ref = ref["pk_lin"]
k_h = k_h_ref  # use reference grid everywhere
h  = float(ref["h"])
f  = float(ref["fz"])  # same key in both reference files
b1 = float(ref["bias_b1"])
b2     = float(ref["bias_b2"])
bG2    = float(ref["bias_bG2"])
bGamma3 = float(ref["bias_bGamma3"])
cs     = float(ref["bias_cs"])
cs0    = float(ref["bias_cs0"])
cs2    = float(ref["bias_cs2"])
cs4    = float(ref["bias_cs4"])
Pshot  = float(ref["bias_Pshot"])
b4     = float(ref["bias_b4"])

print(f"Reference: z={float(ref['z']):.2f}, h={h:.4f}, f={f:.4f}")
print(f"Ref k grid: {len(k_h_ref)} pts [{k_h_ref.min():.2e}, {k_h_ref.max():.2e}] h/Mpc")
print(f"Bias: b1={b1}, b2={b2}, bG2={bG2}, bGamma3={bGamma3}, b4={b4}")
print()

# ── Run compute_ept on the same 256-pt grid as the reference ─────────────────
# k_h_ref = ept_kgrid(prec) — exact match, no extrapolation needed
prec = EPTPrecisionParams()
k_h_int = k_h_ref   # same grid

print(f"Running compute_ept on 256-point reference grid ...")
ept_out = compute_ept(jnp.array(pk_lin_ref), jnp.array(k_h_int), h=h, f=f, prec=prec)
print(f"  Pk_tree range: [{float(ept_out.Pk_tree.min()):.1f}, {float(ept_out.Pk_tree.max()):.1f}]")

def _interp_to_ref(arr_int):
    """Identity — already on reference grid."""
    return np.array(arr_int)

clax_spectra = {
    "pk_mm_real": _interp_to_ref(pk_mm_real(ept_out, cs0=cs0)),
    "pk_gg_real": _interp_to_ref(pk_gg_real(ept_out, b1, b2, bG2, bGamma3, cs=cs, cs0=cs0, Pshot=Pshot)),
    "pk_gm_real": _interp_to_ref(pk_gm_real(ept_out, b1, b2, bG2, bGamma3, cs0=cs0, cs=cs)),
    "pk_mm_l0":   _interp_to_ref(pk_mm_l0(ept_out, cs0=cs0)),
    "pk_mm_l2":   _interp_to_ref(pk_mm_l2(ept_out, cs2=cs2)),
    "pk_mm_l4":   _interp_to_ref(pk_mm_l4(ept_out, cs4=cs4)),
    "pk_gg_l0":   _interp_to_ref(pk_gg_l0(ept_out, b1, b2, bG2, bGamma3, cs0=cs0, Pshot=Pshot, b4=b4)),
    "pk_gg_l2":   _interp_to_ref(pk_gg_l2(ept_out, b1, b2, bG2, bGamma3, cs2=cs2, b4=b4)),
    "pk_gg_l4":   _interp_to_ref(pk_gg_l4(ept_out, b1, b2, bG2, bGamma3, cs4=cs4, b4=b4)),
}

ref_spectra = {
    "pk_mm_real": np.squeeze(ref["pk_mm_real"]),
    "pk_gg_real": np.squeeze(ref["pk_gg_real"]),
    "pk_gm_real": np.squeeze(ref["pk_mg_real"]),   # CLASS-PT stores as pk_mg_real
    "pk_mm_l0":   np.squeeze(ref["pk_mm_l0"]),
    "pk_mm_l2":   np.squeeze(ref["pk_mm_l2"]),
    "pk_mm_l4":   np.squeeze(ref["pk_mm_l4"]),
    "pk_gg_l0":   np.squeeze(ref["pk_gg_l0"]),
    "pk_gg_l2":   np.squeeze(ref["pk_gg_l2"]),
    "pk_gg_l4":   np.squeeze(ref["pk_gg_l4"]),
}

# ── Accuracy summary ─────────────────────────────────────────────────────────
K_MAX = 0.3
mask = k_h < K_MAX

TOLS = {
    "pk_mm_real": 0.005,   # 0.5% — tight (matter)
    "pk_gg_real": 0.01,
    "pk_gm_real": 0.01,
    "pk_mm_l0":   0.01,
    "pk_mm_l2":   0.02,    # looser for multipoles
    "pk_mm_l4":   0.05,
    "pk_gg_l0":   0.01,
    "pk_gg_l2":   0.02,
    "pk_gg_l4":   0.10,    # hexadecapole: 10%
}

print(f"\nAccuracy at k < {K_MAX} h/Mpc ({mask.sum()} modes):\n")
HEADER = f"{'Spectrum':<16} {'max_err':>9} {'mean_err':>9} {'tol':>6}  {'k@max':>8}  {'status':>6}"
print(HEADER)
print("-" * len(HEADER))

results = {}
all_pass = True
for name in clax_spectra:
    c = clax_spectra[name][mask]
    r = ref_spectra[name][mask]
    abs_ref = np.abs(r)
    valid = abs_ref > 0.01 * abs_ref.max()
    if valid.sum() < 3:
        print(f"  {name:<16}  (skipped)")
        continue
    rel_err = np.abs(c[valid] - r[valid]) / abs_ref[valid]
    max_err  = float(rel_err.max())
    mean_err = float(rel_err.mean())
    k_at_max = float(k_h[mask][valid][rel_err.argmax()])
    tol = TOLS[name]
    passed = max_err < tol
    results[name] = dict(max_err=max_err, mean_err=mean_err, k_at_max=k_at_max, passed=passed)
    sym = "PASS" if passed else "FAIL"
    print(f"  {name:<16} {max_err:>8.2%} {mean_err:>9.2%} {tol:>5.0%}  {k_at_max:>8.4f}  {sym}")
    if not passed:
        all_pass = False

# ── Generate figures ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(_REPO, "notebooks", "figures")
os.makedirs(FIGDIR, exist_ok=True)

SPECTRUM_LABELS = {
    "pk_mm_real": r"$P_{mm}^{\rm real}(k)$",
    "pk_gg_real": r"$P_{gg}^{\rm real}(k)$",
    "pk_gm_real": r"$P_{gm}^{\rm real}(k)$",
    "pk_mm_l0":   r"$P_{mm}^{\ell=0}(k)$",
    "pk_mm_l2":   r"$P_{mm}^{\ell=2}(k)$",
    "pk_mm_l4":   r"$P_{mm}^{\ell=4}(k)$",
    "pk_gg_l0":   r"$P_{gg}^{\ell=0}(k)$",
    "pk_gg_l2":   r"$P_{gg}^{\ell=2}(k)$",
    "pk_gg_l4":   r"$P_{gg}^{\ell=4}(k)$",
}

FIGNAMES = {
    "pk_mm_real": "fig3_pmm_real_validation.png",
    "pk_gg_real": "fig4_pgg_real_validation.png",
    "pk_gm_real": "fig5_pgm_real_validation.png",
    "pk_mm_l0":   "fig6_pmm_l0_validation.png",
    "pk_mm_l2":   "fig7_pmm_l2_validation.png",
    "pk_mm_l4":   "fig8_pmm_l4_validation.png",
    "pk_gg_l0":   "fig9_pgg_l0_validation.png",
    "pk_gg_l2":   "fig10_pgg_l2_validation.png",
    "pk_gg_l4":   "fig11_pgg_l4_validation.png",
}

def _y_range(r_arr):
    """Auto-set physically meaningful y-range for P(k)."""
    pos = r_arr[r_arr > 0]
    if len(pos) == 0:
        return None, None
    pmin = max(pos.min(), pos.max() * 1e-4)
    pmax = pos.max() * 3.0
    return pmin, pmax

def _resid_ylim(max_err, tol):
    """Symmetric y-limits for residual panel."""
    # Show at least ±1%, round up to nearest 0.5%
    span = max(max_err * 1.3, tol * 1.5, 0.01)
    tick = 0.005
    span = np.ceil(span / tick) * tick
    return -span * 100, span * 100

for name in clax_spectra:
    c = clax_spectra[name]
    r = ref_spectra[name]
    res = results.get(name, {})
    passed = res.get("passed", False)
    max_err = res.get("max_err", 0.0)
    tol = TOLS[name]

    label = SPECTRUM_LABELS[name]
    status_str = "PASS" if passed else f"FAIL (max {max_err:.1%})"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # Top panel: spectra
    ax1.loglog(k_h, r, "k-", lw=1.5, label="CLASS-PT reference")
    ax1.loglog(k_h, np.abs(c), "--", color="tab:blue", lw=1.2, label="clax.ept")
    ax1.axvline(K_MAX, color="gray", ls=":", lw=0.8)
    ymin, ymax = _y_range(r)
    if ymin is not None:
        ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel(r"$P(k)$ $[({\rm Mpc}/h)^3]$")
    ax1.legend(fontsize=9)
    ax1.set_title(f"{label}  [{status_str}]", fontsize=11)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    # Bottom panel: relative error
    abs_ref = np.abs(r)
    valid2 = abs_ref > 0.01 * abs_ref.max()
    rel_err_full = np.where(valid2, (c - r) / np.where(abs_ref > 0, abs_ref, 1), np.nan)

    ylo, yhi = _resid_ylim(max_err, tol)
    ax2.plot(k_h, rel_err_full * 100, "b-", lw=1.0)
    ax2.axhline(0, color="k", ls="-", lw=0.5)
    ax2.axhline(tol * 100, color="r", ls="--", lw=0.8, label=f"±{tol*100:.0f}% target")
    ax2.axhline(-tol * 100, color="r", ls="--", lw=0.8)
    ax2.axvline(K_MAX, color="gray", ls=":", lw=0.8)
    ax2.set_ylim(ylo, yhi)
    ax2.set_ylabel(r"$\Delta/P_{\rm ref}$ [%]")
    ax2.set_xlabel(r"$k$ $[h/{\rm Mpc}]$")
    ax2.legend(fontsize=8)
    ax2.grid(True, ls=":", alpha=0.4)

    figpath = os.path.join(FIGDIR, FIGNAMES[name])
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figpath}")

# ── fig1: P_mm comparison (loop breakdown) ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex=True)
fig.subplots_adjust(hspace=0.08)

pk_tree_ref  = _interp_to_ref(ept_out.Pk_tree)
pk_1loop_ref = _interp_to_ref(ept_out.Pk_loop)
ref_mm   = ref_spectra["pk_mm_real"]

ax1.loglog(k_h, ref_mm, "k-", lw=1.5, label="CLASS-PT")
ax1.loglog(k_h, clax_spectra["pk_mm_real"], "--", color="tab:blue", lw=1.2, label="clax total")
ax1.loglog(k_h, np.abs(pk_tree_ref), ":", color="tab:orange", lw=1.2, label="Tree (clax)")
ax1.loglog(k_h, np.abs(pk_1loop_ref), "-.", color="tab:green", lw=1.0, label="|1-loop| (clax)")
ax1.axvline(K_MAX, color="gray", ls=":", lw=0.8)
ymin, ymax = _y_range(ref_mm)
if ymin is not None:
    ax1.set_ylim(ymin, ymax)
ax1.set_ylabel(r"$P_{mm}(k)$ $[({\rm Mpc}/h)^3]$")
ax1.legend(fontsize=9)
ax1.set_title(r"$P_{mm}$ real-space loop breakdown", fontsize=11)
ax1.grid(True, which="both", ls=":", alpha=0.4)

res_mm = results.get("pk_mm_real", {})
max_err_mm = res_mm.get("max_err", 0.0)
tol_mm = TOLS["pk_mm_real"]
rel_mm = (clax_spectra["pk_mm_real"] - ref_mm) / ref_mm
ax2.plot(k_h, rel_mm * 100, "b-", lw=1.0)
ax2.axhline(0, color="k", ls="-", lw=0.5)
ylo, yhi = _resid_ylim(max_err_mm, tol_mm)
ax2.axhline(tol_mm * 100, color="r", ls="--", lw=0.8)
ax2.axhline(-tol_mm * 100, color="r", ls="--", lw=0.8)
ax2.axvline(K_MAX, color="gray", ls=":", lw=0.8)
ax2.set_ylim(ylo, yhi)
ax2.set_ylabel(r"$\Delta/P_{\rm ref}$ [%]")
ax2.set_xlabel(r"$k$ $[h/{\rm Mpc}]$")
ax2.grid(True, ls=":", alpha=0.4)
fig.savefig(os.path.join(FIGDIR, "fig1_pmm_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved fig1_pmm_comparison.png")

# ── fig2: loop component breakdown ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(k_h, _interp_to_ref(ept_out.Pk_0_vv1) / ref_mm, "-", lw=1.2, label=r"$P^{(0)}_{vv,1\ell}$")
ax.semilogx(k_h, _interp_to_ref(ept_out.Pk_0_vd1) / ref_mm, "--", lw=1.2, label=r"$P^{(0)}_{vd,1\ell}$")
ax.semilogx(k_h, _interp_to_ref(ept_out.Pk_0_dd1) / ref_mm, "-.", lw=1.2, label=r"$P^{(0)}_{dd,1\ell}$")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(K_MAX, color="gray", ls=":", lw=0.8)
ax.set_xlabel(r"$k$ $[h/{\rm Mpc}]$")
ax.set_ylabel(r"$P^{(0)}_{xy,1\ell} / P^{\rm ref}_{mm}$")
ax.set_title(r"Monopole 1-loop component fractions", fontsize=11)
ax.set_ylim(-0.5, 0.5)
ax.legend(fontsize=9)
ax.grid(True, ls=":", alpha=0.4)
fig.savefig(os.path.join(FIGDIR, "fig2_loop_breakdown.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved fig2_loop_breakdown.png")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*52}")
if all_pass:
    print("ALL SPECTRA PASS within tolerance  ✓")
else:
    failing = [n for n, r in results.items() if not r["passed"]]
    passing = [n for n, r in results.items() if r["passed"]]
    print(f"PASS ({len(passing)}): {', '.join(passing)}")
    print(f"FAIL ({len(failing)}): {', '.join(failing)}")
print(f"{'='*52}")

sys.exit(0 if all_pass else 1)
