#!/usr/bin/env python3
"""Generate P_mm validation figures: clax.ept vs CLASS-PT.

Creates three figures in notebooks/figures/:
  fig1_pmm_comparison.png   — P_mm clax vs CLASS-PT with residuals
  fig2_loop_breakdown.png   — P_13, P_22, P_loop components
  fig3_ir_resummation.png   — IR resummation vs no-IR comparison

Usage:
    ~/miniconda3/envs/sbi_pytorch_osx-arm64-py310forge/bin/python3 \
        scripts/generate_validation_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError:
    print("ERROR: JAX not found. Use sbi_pytorch_osx-arm64-py310forge env.")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clax.ept import (
    compute_ept, EPTPrecisionParams, ept_kgrid,
    pk_mm_real as clax_pk_mm_real,
)

# ── Load CLASS-PT reference data ────────────────────────────────────────────
REF_FILE = os.path.join(_REPO, "docs", "classpt_reference_table.npz")
if not os.path.isfile(REF_FILE):
    print(f"ERROR: reference file not found: {REF_FILE}")
    print("Run: python3 scripts/generate_classpt_reference_v2.py")
    sys.exit(1)

ref = np.load(REF_FILE)
k_ref   = ref["k_hMpc"]          # (60,) h/Mpc
pk_lin  = ref["pk_lin"]          # (60,) (Mpc/h)^3
pk_mm_classpt = ref["pk_mm_real"] # (60,) (Mpc/h)^3
z_pk    = float(ref["z_pk"])
h       = float(ref["h"])
fz      = float(ref["fz"])

print(f"Reference: z={z_pk:.2f}, h={h:.4f}, f(z)={fz:.4f}")
print(f"k range: [{k_ref[0]:.4f}, {k_ref[-1]:.3f}] h/Mpc, N={len(k_ref)}")

# ── Compute pk_lin on the full EPT k-grid via CLASS ──────────────────────────
# The reference file has pk_lin only on 60 points (0.005–0.30 h/Mpc);
# compute_ept needs pk_lin on the EPT k-grid (0.00005–100 h/Mpc, 256 pts).
# Use CLASS directly to get pk_lin at the correct resolution.
from classy import Class as _Class

prec = EPTPrecisionParams()
k_ept_np = ept_kgrid(prec)           # 256-point internal grid, h/Mpc
k_ept = jnp.array(k_ept_np)
k_ept_1Mpc = k_ept_np * h           # convert to 1/Mpc for CLASS

print("Computing pk_lin on EPT grid with CLASS ...")
_M = _Class()
_M.set({
    'A_s': float(ref['A_s']),
    'n_s': float(ref['n_s']),
    'tau_reio': float(ref['tau_reio']),
    'omega_b': float(ref['omega_b']),
    'omega_cdm': float(ref['omega_cdm']),
    'h': h,
    'output': 'mPk',
    'P_k_max_1/Mpc': 200.0,
    'z_max_pk': z_pk + 0.1,
})
_M.compute()
pk_lin_ept_Mpc3 = np.array([_M.pk_lin(ki, z_pk) for ki in k_ept_1Mpc])
pk_lin_ept = pk_lin_ept_Mpc3 * h**3  # convert to (Mpc/h)^3
_M.struct_cleanup(); _M.empty()
print(f"  pk_lin at k=0.1 h/Mpc: {pk_lin_ept[np.argmin(np.abs(k_ept_np - 0.1))]:.3e} (Mpc/h)^3")

pk_lin_ept_jnp = jnp.array(pk_lin_ept)

print("Running compute_ept (IR resummation ON) ...")
ept_ir = compute_ept(pk_lin_ept_jnp, k_ept, h=h, f=fz, prec=prec)

# Also run without IR resummation
prec_noir = EPTPrecisionParams(ir_resummation=False)
print("Running compute_ept (IR resummation OFF) ...")
ept_noir = compute_ept(pk_lin_ept_jnp, k_ept, h=h, f=fz, prec=prec_noir)

# ── Extract spectra on EPT grid ─────────────────────────────────────────────
pmm_clax_ir   = np.array(clax_pk_mm_real(ept_ir,   cs0=0.0))
pmm_clax_noir = np.array(clax_pk_mm_real(ept_noir, cs0=0.0))
ploop_clax = np.array(ept_ir.Pk_loop)
ptree_clax = np.array(ept_ir.Pk_tree)

# Interpolate CLASS-PT spectra onto EPT grid (for component comparisons)
# For P_mm we compare on the reference grid (both evaluated there)
pmm_classpt_ept = np.exp(np.interp(np.log(k_ept_np), np.log(k_ref), np.log(np.abs(pk_mm_classpt))))

# For residuals, evaluate clax on the reference k-grid via interpolation
pmm_clax_on_ref = np.exp(np.interp(np.log(k_ref), np.log(k_ept_np), np.log(np.abs(pmm_clax_ir))))

# P_loop proxy from CLASS-PT: pm_mm - pk_lin (using reference grid)
ploop_classpt = pk_mm_classpt - pk_lin   # loop = total - tree (tree ≈ pk_lin in CLASS-PT)

# Restrict comparison to k < 0.3 h/Mpc (PT validity)
K_MAX = 0.30
mask_ref = k_ref < K_MAX
rel_err = (pmm_clax_on_ref[mask_ref] - pk_mm_classpt[mask_ref]) / pk_mm_classpt[mask_ref] * 100
max_err = np.max(np.abs(rel_err))
rms_err = np.sqrt(np.mean(rel_err**2))
print(f"\nP_mm accuracy (k < {K_MAX} h/Mpc):")
print(f"  max |err| = {max_err:.3f}%,  rms = {rms_err:.3f}%")

# ── Figure 1: P_mm comparison ───────────────────────────────────────────────
print("\nGenerating fig1_pmm_comparison.png ...")
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                 gridspec_kw={"height_ratios": [2.5, 1]},
                                 sharex=True)

# Top: power spectra
ax1.loglog(k_ept_np, pmm_clax_ir,    "b-",  lw=2.0, label="clax P_mm (IR on)")
ax1.loglog(k_ref,    pk_mm_classpt,  "r--", lw=1.8, label="CLASS-PT P_mm")
ax1.loglog(k_ref,    pk_lin,         color="0.5", ls=":",  lw=1.4, label=r"$P_\mathrm{lin}$ (tree)")
ax1.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6, label="k=0.30 h/Mpc")
ax1.set_ylabel(r"$P(k)\ [(\mathrm{Mpc}/h)^3]$", fontsize=12)
ax1.set_xlim(k_ref[0], k_ref[-1])
ax1.legend(fontsize=10)
ax1.set_title(f"$P_{{mm}}$: clax vs CLASS-PT  (max err = {max_err:.2f}%)", fontsize=13)
ax1.grid(True, which="both", alpha=0.2)

# Bottom: residuals
ax2.axhline(0,    color="k",   lw=1.0)
ax2.axhline(+1.0, color="r",   lw=0.8, ls="--", alpha=0.7)
ax2.axhline(-1.0, color="r",   lw=0.8, ls="--", alpha=0.7, label="±1%")
ax2.axhline(+0.5, color="darkorange", lw=0.8, ls=":", alpha=0.7)
ax2.axhline(-0.5, color="darkorange", lw=0.8, ls=":", alpha=0.7, label="±0.5%")
ax2.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
# Compute residuals on full ref grid
rel_err_full = (pmm_clax_on_ref - pk_mm_classpt) / pk_mm_classpt * 100
ax2.semilogx(k_ref, rel_err_full, "b-", lw=1.5, label="(clax − CLASS-PT)/CLASS-PT")
ax2.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$", fontsize=12)
ax2.set_ylabel(r"$\Delta P/P\ [\%]$", fontsize=11)
ax2.set_ylim(-3.0, 3.0)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, which="both", alpha=0.2)

fig1.tight_layout()
out1 = os.path.join(_REPO, "notebooks", "figures", "fig1_pmm_comparison.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved: {out1}")

# ── Figure 2: Loop breakdown ─────────────────────────────────────────────────
print("Generating fig2_loop_breakdown.png ...")

# CLASS-PT P_loop proxy: P_mm - P_lin (CLASS-PT's P_mm includes tree + loop)
# pk_mult[0] = Pd1d1 (P_1loop matter), available from the raw output array
pk_mult = ref["pk_mult"]   # shape (96, 60) -- raw CLASS-PT loop components
p1loop_classpt = pk_mult[0]   # Pd1d1: the 1-loop matter power spectrum component

# Interpolate clax loop component onto reference grid
ploop_on_ref = np.interp(k_ref, k_ept_np, ploop_clax)
ptree_on_ref = np.interp(k_ref, k_ept_np, ptree_clax)

# vv, vd, dd sub-components available from EPTComponents for RSD loop breakdown
# For real-space P_loop = P_tree + P_loop components
# Approximate P_13 ~ from vd component, P_22 ~ from vv, but simpler to use pk_mult indices
# pk_mult: index 5 ≈ P22 (Pdd22), index 6 ≈ P13 (Pdd13) in CLASS-PT convention
# Use CLASS-PT indices as labeled: indices vary; use p1loop proxy instead
p22_classpt = pk_mult[5] if pk_mult.shape[0] > 6 else None  # approximate
p13_classpt = pk_mult[6] if pk_mult.shape[0] > 7 else None  # approximate

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                  gridspec_kw={"height_ratios": [2.5, 1]},
                                  sharex=True)

# Top: compare clax P_loop with CLASS-PT P_loop proxies
ax1.semilogy(k_ref, np.abs(ploop_on_ref), "b-", lw=2.0,
             label=r"$|P_\mathrm{loop}|$ clax ($P_{13}+P_{22}$)")
ax1.semilogy(k_ref, np.abs(p1loop_classpt), "r--", lw=1.8,
             label=r"$|P_{1\mathrm{loop}}|$ CLASS-PT (pk_mult[0], Pd1d1)")
ax1.semilogy(k_ref, np.abs(ploop_classpt), "g-.", lw=1.5,
             label=r"$|P_{mm} - P_\mathrm{lin}|$ CLASS-PT (proxy)")
ax1.semilogy(k_ref, pk_lin, color="0.5", ls=":", lw=1.2, label=r"$P_\mathrm{lin}$ (reference)")
ax1.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax1.set_ylabel(r"$|P(k)|\ [(\mathrm{Mpc}/h)^3]$", fontsize=12)
ax1.set_xlim(k_ref[0], k_ref[-1])
ax1.legend(fontsize=9)
ax1.set_title(r"Loop power spectrum: clax vs CLASS-PT", fontsize=13)
ax1.grid(True, which="both", alpha=0.2)

# Bottom: residuals vs P_loop proxy
valid = np.abs(ploop_classpt) > 1.0   # skip near-zero
rel_loop_err = np.where(valid,
    (ploop_on_ref - ploop_classpt) / np.abs(ploop_classpt) * 100,
    np.nan)
# Also compare with Pd1d1 proxy
valid1 = np.abs(p1loop_classpt) > 0.01 * np.max(np.abs(p1loop_classpt))
rel_loop1_err = np.where(valid1,
    (ploop_on_ref - p1loop_classpt) / np.abs(p1loop_classpt) * 100,
    np.nan)
ax2.axhline(0,    color="k",   lw=1.0)
ax2.axhline(+5.0, color="r",   lw=0.8, ls="--", alpha=0.7, label="±5%")
ax2.axhline(-5.0, color="r",   lw=0.8, ls="--", alpha=0.7)
ax2.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax2.semilogx(k_ref, rel_loop_err, "b-", lw=1.5,
             label=r"clax vs $P_{mm}-P_\mathrm{lin}$")
ax2.semilogx(k_ref, rel_loop1_err, "g--", lw=1.3,
             label=r"clax vs Pd1d1 (pk_mult[0])")
ax2.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$", fontsize=12)
ax2.set_ylabel(r"$\Delta P_\mathrm{loop}/|P_\mathrm{loop}|\ [\%]$", fontsize=10)
ax2.set_ylim(-30, 30)
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.2)

fig2.tight_layout()
out2 = os.path.join(_REPO, "notebooks", "figures", "fig2_loop_breakdown.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {out2}")

# ── Figure 3: IR resummation effect ─────────────────────────────────────────
print("Generating fig3_ir_resummation.png ...")

# Need CLASS-PT reference without IR resummation — regenerate on the fly
try:
    from classy import Class
    print("  Generating CLASS-PT reference without IR resummation ...")
    M = Class()
    cosmo_params = dict(A_s=2.0989e-9, n_s=0.9649, tau_reio=0.0544,
                        omega_b=0.02237, omega_cdm=0.1200, h=h)
    settings_noir = dict(output="mPk", **{"non linear": "PT",
                         "IR resummation": "No",
                         "Bias tracers": "Yes", "RSD": "Yes",
                         "z_pk": z_pk, "P_k_max_h/Mpc": 100.})
    M.set({**cosmo_params, **settings_noir})
    M.compute()
    M.initialize_output(k_ref * h, z_pk, len(k_ref))
    pk_mm_classpt_noir = M.pk_mm_real(cs=0.0)
    M.struct_cleanup(); M.empty()
    have_classpt_noir = True
    print("  CLASS-PT (no IR) done.")
except Exception as exc:
    print(f"  WARNING: Could not generate CLASS-PT no-IR reference: {exc}")
    have_classpt_noir = False

fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                  gridspec_kw={"height_ratios": [2.5, 1]},
                                  sharex=True)

# Top: P_mm with and without IR
ax1.loglog(k_ept_np, pmm_clax_ir,    "b-",  lw=2.0, label="clax: with IR resummation")
ax1.loglog(k_ept_np, pmm_clax_noir,  "g-",  lw=2.0, label="clax: no IR resummation")
ax1.loglog(k_ref,    pk_mm_classpt,  "r--", lw=1.6, label="CLASS-PT: with IR")
if have_classpt_noir:
    ax1.loglog(k_ref, np.abs(pk_mm_classpt_noir), "orange", ls="--", lw=1.6,
               label="CLASS-PT: no IR")
ax1.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax1.set_ylabel(r"$P_{mm}(k)\ [(\mathrm{Mpc}/h)^3]$", fontsize=12)
ax1.set_xlim(k_ref[0], k_ref[-1])
ax1.legend(fontsize=9)
ax1.set_title(r"IR resummation effect on $P_{mm}$", fontsize=13)
ax1.grid(True, which="both", alpha=0.2)

# Bottom: ratio (with IR)/(without IR) showing BAO damping
ratio_clax = pmm_clax_ir / pmm_clax_noir
ax2.axhline(1.0, color="k", lw=1.0)
ax2.semilogx(k_ept_np, ratio_clax, "b-", lw=1.8, label="clax: P(IR on)/P(IR off)")
if have_classpt_noir:
    ratio_classpt_on_ref = pk_mm_classpt / np.abs(pk_mm_classpt_noir)
    ax2.semilogx(k_ref, ratio_classpt_on_ref, "r--", lw=1.6,
                 label="CLASS-PT: P(IR on)/P(IR off)")
ax2.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax2.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$", fontsize=12)
ax2.set_ylabel(r"$P_\mathrm{IR}/P_\mathrm{no-IR}$", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.2)

fig3.tight_layout()
out3 = os.path.join(_REPO, "notebooks", "figures", "fig3_ir_resummation.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved: {out3}")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"P_mm accuracy summary (k < {K_MAX} h/Mpc, {mask_ref.sum()} modes):")
print(f"  max |err| = {max_err:.3f}%")
print(f"  rms err   = {rms_err:.3f}%")
print(f"  PASS      = {'YES' if max_err < 1.0 else 'NO'} (threshold 1%)")
print(f"{'='*55}")
print("Figures written to notebooks/figures/")
