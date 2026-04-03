#!/usr/bin/env python3
"""Create notebooks/pm_mm_validation.ipynb programmatically using nbformat."""

import os
import nbformat

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(_REPO, "notebooks"), exist_ok=True)

nb = nbformat.v4.new_notebook()

# ── Cell 1: Title + Setup ────────────────────────────────────────────────────
cell1 = nbformat.v4.new_markdown_cell("""\
# P_mm Validation: clax.ept vs CLASS-PT

Validates the one-loop matter power spectrum `P_mm(k)` from `clax.ept`
against CLASS-PT reference data at **z = 0.38** (BOSS CMASS redshift).

**Cosmology**: Planck 2018 best-fit ΛCDM
**Bias**: b₁=2, all other bias/EFT params = 0 (tests pure matter P_mm)
**Reference**: `docs/classpt_reference_table.npz` generated with `generate_classpt_reference_v2.py`

## Figures
1. `fig1_pmm_comparison.png` — P_mm clax vs CLASS-PT with residuals (max 0.33%)
2. `fig2_loop_breakdown.png` — Loop power spectrum P_loop comparison
3. `fig3_ir_resummation.png` — IR resummation effect (BAO suppression)
""")

# ── Cell 2: Setup imports + params ──────────────────────────────────────────
cell2 = nbformat.v4.new_code_cell("""\
import os, sys
import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 11})

# Ensure clax is importable from repo root
_REPO = os.path.abspath("..")
sys.path.insert(0, _REPO)

from clax.ept import (
    compute_ept, EPTPrecisionParams, ept_kgrid,
    pk_mm_real as clax_pk_mm_real,
)
from classy import Class

# ── Cosmological parameters (Planck 2018) ────────────────────────────────────
z_pk   = 0.38
cosmo  = dict(A_s=2.0989e-9, n_s=0.9649, tau_reio=0.0544,
              omega_b=0.02237, omega_cdm=0.1200, h=0.6736)
h      = cosmo["h"]

# Load CLASS-PT reference data (z=0.38, 60-point k-grid 0.005–0.30 h/Mpc)
REF_FILE = os.path.join(_REPO, "docs", "classpt_reference_table.npz")
ref        = np.load(REF_FILE)
k_ref      = ref["k_hMpc"]         # (60,)  h/Mpc
pk_lin_ref = ref["pk_lin"]          # (60,)  (Mpc/h)^3
pk_mm_cpt  = ref["pk_mm_real"]      # (60,)  (Mpc/h)^3
fz         = float(ref["fz"])

print(f"Loaded reference: z={z_pk}, h={h}, f(z)={fz:.4f}")
print(f"k range: [{k_ref[0]:.4f}, {k_ref[-1]:.3f}] h/Mpc  (N={len(k_ref)})")
print(f"P_mm(k=0.1) = {pk_mm_cpt[np.argmin(np.abs(k_ref - 0.1))]:.3e}  (Mpc/h)^3  [CLASS-PT]")
""")

# ── Cell 3: CLASS-PT computation (reference overview) ────────────────────────
cell3 = nbformat.v4.new_code_cell("""\
# ── CLASS-PT reference computation (IR on, no-IR) ────────────────────────────
# The docs/ reference table was generated with IR resummation ON.
# We also compute the no-IR reference for Figure 3.

print("Computing CLASS-PT reference without IR resummation ...")
M_noir = Class()
M_noir.set({**cosmo,
            "output": "mPk",
            "non linear": "PT",
            "IR resummation": "No",
            "Bias tracers": "Yes",
            "RSD": "Yes",
            "z_pk": z_pk,
            "P_k_max_h/Mpc": 100.})
M_noir.compute()
M_noir.initialize_output(k_ref * h, z_pk, len(k_ref))
pk_mm_cpt_noir = M_noir.pk_mm_real(cs=0.0)
M_noir.struct_cleanup(); M_noir.empty()

print(f"P_mm(no-IR) at k=0.1: {pk_mm_cpt_noir[np.argmin(np.abs(k_ref - 0.1))]:.3e}  (Mpc/h)^3")
print(f"P_mm(with-IR) at k=0.1: {pk_mm_cpt[np.argmin(np.abs(k_ref - 0.1))]:.3e}  (Mpc/h)^3")
""")

# ── Cell 4: clax EPT computation ─────────────────────────────────────────────
cell4 = nbformat.v4.new_code_cell("""\
# ── clax EPT computation ──────────────────────────────────────────────────────
# Need pk_lin on the full EPT k-grid (0.00005–100 h/Mpc, 256 pts).
# Compute via CLASS directly to avoid extrapolation errors.

prec      = EPTPrecisionParams()
k_ept_np  = ept_kgrid(prec)          # 256 log-spaced pts, h/Mpc
k_ept     = jnp.array(k_ept_np)

print(f"EPT k-grid: [{k_ept_np[0]:.5f}, {k_ept_np[-1]:.1f}] h/Mpc  (N={len(k_ept_np)})")

# pk_lin on EPT grid from CLASS
M_lin = Class()
M_lin.set({**cosmo,
           "output": "mPk",
           "P_k_max_1/Mpc": 200.0,
           "z_max_pk": z_pk + 0.1})
M_lin.compute()
pk_lin_ept = np.array([M_lin.pk_lin(ki * h, z_pk) for ki in k_ept_np]) * h**3
M_lin.struct_cleanup(); M_lin.empty()

print(f"pk_lin(k=0.1 h/Mpc) = {pk_lin_ept[np.argmin(np.abs(k_ept_np - 0.1))]:.3e} (Mpc/h)^3")

# Run clax EPT (with + without IR resummation)
print("Running compute_ept (IR on) ...")
ept_ir   = compute_ept(jnp.array(pk_lin_ept), k_ept, h=h, f=fz, prec=prec)

prec_nr  = EPTPrecisionParams(ir_resummation=False)
print("Running compute_ept (IR off) ...")
ept_noir = compute_ept(jnp.array(pk_lin_ept), k_ept, h=h, f=fz, prec=prec_nr)

pmm_clax_ir   = np.array(clax_pk_mm_real(ept_ir,   cs0=0.0))
pmm_clax_noir = np.array(clax_pk_mm_real(ept_noir, cs0=0.0))

# Interpolate clax output onto reference k-grid for comparison
pmm_clax_on_ref = np.exp(np.interp(np.log(k_ref), np.log(k_ept_np),
                                    np.log(np.abs(pmm_clax_ir))))

# Accuracy metrics
K_MAX = 0.30
mask = k_ref < K_MAX
rel_err = (pmm_clax_on_ref[mask] - pk_mm_cpt[mask]) / pk_mm_cpt[mask] * 100
print(f"\\nP_mm accuracy (k < {K_MAX} h/Mpc):")
print(f"  max |err| = {np.max(np.abs(rel_err)):.3f}%")
print(f"  rms err   = {np.sqrt(np.mean(rel_err**2)):.3f}%")
""")

# ── Cell 5: Figure 1 — P_mm comparison ──────────────────────────────────────
cell5 = nbformat.v4.new_code_cell("""\
# ── Figure 1: P_mm comparison ────────────────────────────────────────────────
K_MAX = 0.30
mask  = k_ref < K_MAX
rel_err_full = (pmm_clax_on_ref - pk_mm_cpt) / pk_mm_cpt * 100
max_err = np.max(np.abs(rel_err_full[mask]))

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                  gridspec_kw={"height_ratios": [2.5, 1]},
                                  sharex=True)

ax1.loglog(k_ept_np, pmm_clax_ir,   "b-",  lw=2.0, label="clax P_mm (IR on)")
ax1.loglog(k_ref,    pk_mm_cpt,     "r--", lw=1.8, label="CLASS-PT P_mm (IR on)")
ax1.loglog(k_ref,    pk_lin_ref,    color="0.5", ls=":", lw=1.4,
           label=r"$P_\\mathrm{lin}$ (tree)")
ax1.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6, label="k=0.30 h/Mpc")
ax1.set_ylabel(r"$P(k)\\ [(\\mathrm{Mpc}/h)^3]$", fontsize=12)
ax1.set_xlim(k_ref[0], k_ref[-1])
ax1.legend(fontsize=10)
ax1.set_title(f"$P_{{mm}}$: clax vs CLASS-PT  (max err = {max_err:.2f}%)", fontsize=13)
ax1.grid(True, which="both", alpha=0.2)

ax2.axhline(0,    color="k",   lw=1.0)
ax2.axhline(+1.0, color="r",   lw=0.8, ls="--", alpha=0.7, label="±1%")
ax2.axhline(-1.0, color="r",   lw=0.8, ls="--", alpha=0.7)
ax2.axhline(+0.5, color="darkorange", lw=0.8, ls=":", alpha=0.7, label="±0.5%")
ax2.axhline(-0.5, color="darkorange", lw=0.8, ls=":", alpha=0.7)
ax2.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax2.semilogx(k_ref, rel_err_full, "b-", lw=1.5)
ax2.set_xlabel(r"$k\\ [h/\\mathrm{Mpc}]$", fontsize=12)
ax2.set_ylabel(r"$\\Delta P/P\\ [\\%]$", fontsize=11)
ax2.set_ylim(-3.0, 3.0)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, which="both", alpha=0.2)

fig1.tight_layout()
fig1.savefig("figures/fig1_pmm_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figures/fig1_pmm_comparison.png  (max err = {max_err:.3f}%)")
""")

# ── Cell 6: Figure 2 — Loop breakdown ───────────────────────────────────────
cell6 = nbformat.v4.new_code_cell("""\
# ── Figure 2: Loop breakdown ──────────────────────────────────────────────────
pk_mult   = ref["pk_mult"]             # (96, 60)  raw CLASS-PT loop components
p1loop_cpt  = pk_mult[0]               # Pd1d1: 1-loop matter spectrum
ploop_classpt = pk_mm_cpt - pk_lin_ref  # proxy: P_mm - P_lin

ploop_on_ref = np.exp(np.interp(np.log(k_ref), np.log(k_ept_np),
                                  np.log(np.abs(np.array(ept_ir.Pk_loop)))))
ptree_on_ref = np.exp(np.interp(np.log(k_ref), np.log(k_ept_np),
                                  np.log(np.abs(np.array(ept_ir.Pk_tree)))))

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                  gridspec_kw={"height_ratios": [2.5, 1]},
                                  sharex=True)

ax1.semilogy(k_ref, np.abs(ploop_on_ref),  "b-",  lw=2.0,
             label=r"$|P_\\mathrm{loop}|$ clax ($P_{13}+P_{22}$)")
ax1.semilogy(k_ref, np.abs(p1loop_cpt),    "r--", lw=1.8,
             label=r"$|P_\\mathrm{1loop}|$ CLASS-PT (Pd1d1)")
ax1.semilogy(k_ref, np.abs(ploop_classpt), "g-.", lw=1.5,
             label=r"$|P_{mm} - P_\\mathrm{lin}|$ CLASS-PT (proxy)")
ax1.semilogy(k_ref, pk_lin_ref, color="0.5", ls=":", lw=1.2,
             label=r"$P_\\mathrm{lin}$")
ax1.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax1.set_ylabel(r"$|P(k)|\\ [(\\mathrm{Mpc}/h)^3]$", fontsize=12)
ax1.set_xlim(k_ref[0], k_ref[-1])
ax1.legend(fontsize=9)
ax1.set_title(r"Loop power spectrum: clax vs CLASS-PT", fontsize=13)
ax1.grid(True, which="both", alpha=0.2)

valid = np.abs(ploop_classpt) > 1.0
rel_loop_err = np.where(valid,
    (ploop_on_ref - ploop_classpt) / np.abs(ploop_classpt) * 100, np.nan)
valid1 = np.abs(p1loop_cpt) > 0.01 * np.max(np.abs(p1loop_cpt))
rel_loop1_err = np.where(valid1,
    (ploop_on_ref - p1loop_cpt) / np.abs(p1loop_cpt) * 100, np.nan)

ax2.axhline(0,    color="k",   lw=1.0)
ax2.axhline(+5.0, color="r",   lw=0.8, ls="--", alpha=0.7, label="±5%")
ax2.axhline(-5.0, color="r",   lw=0.8, ls="--", alpha=0.7)
ax2.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax2.semilogx(k_ref, rel_loop_err,  "b-",  lw=1.5, label=r"vs $P_{mm}-P_\\mathrm{lin}$")
ax2.semilogx(k_ref, rel_loop1_err, "g--", lw=1.3, label="vs Pd1d1")
ax2.set_xlabel(r"$k\\ [h/\\mathrm{Mpc}]$", fontsize=12)
ax2.set_ylabel(r"$\\Delta P_\\mathrm{loop}/|P_\\mathrm{loop}|\\ [\\%]$", fontsize=10)
ax2.set_ylim(-30, 30)
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.2)

fig2.tight_layout()
fig2.savefig("figures/fig2_loop_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ── Cell 7: Figure 3 — IR resummation ───────────────────────────────────────
cell7 = nbformat.v4.new_code_cell("""\
# ── Figure 3: IR resummation effect ──────────────────────────────────────────
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                  gridspec_kw={"height_ratios": [2.5, 1]},
                                  sharex=True)

ax1.loglog(k_ept_np, pmm_clax_ir,    "b-",  lw=2.0, label="clax: with IR resummation")
ax1.loglog(k_ept_np, pmm_clax_noir,  "g-",  lw=2.0, label="clax: no IR resummation")
ax1.loglog(k_ref,    pk_mm_cpt,      "r--", lw=1.6, label="CLASS-PT: with IR")
ax1.loglog(k_ref, np.abs(pk_mm_cpt_noir), "orange", ls="--", lw=1.6,
           label="CLASS-PT: no IR")
ax1.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax1.set_ylabel(r"$P_{{mm}}(k)\\ [(\\mathrm{Mpc}/h)^3]$", fontsize=12)
ax1.set_xlim(k_ref[0], k_ref[-1])
ax1.legend(fontsize=9)
ax1.set_title(r"IR resummation effect on $P_{{mm}}$", fontsize=13)
ax1.grid(True, which="both", alpha=0.2)

# Ratio (with IR) / (without IR) — shows BAO suppression
ratio_clax   = pmm_clax_ir / pmm_clax_noir
ratio_classpt = pk_mm_cpt / np.abs(pk_mm_cpt_noir)
ax2.axhline(1.0, color="k", lw=1.0)
ax2.semilogx(k_ept_np, ratio_clax,   "b-",  lw=1.8, label="clax")
ax2.semilogx(k_ref,    ratio_classpt, "r--", lw=1.6, label="CLASS-PT")
ax2.axvline(K_MAX, color="gray", ls="-.", lw=1.0, alpha=0.6)
ax2.set_xlabel(r"$k\\ [h/\\mathrm{Mpc}]$", fontsize=12)
ax2.set_ylabel(r"$P_\\mathrm{IR}/P_\\mathrm{no-IR}$", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.2)

fig3.tight_layout()
fig3.savefig("figures/fig3_ir_resummation.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ── Cell 8: Summary accuracy table ───────────────────────────────────────────
cell8 = nbformat.v4.new_code_cell("""\
# ── Summary accuracy table ────────────────────────────────────────────────────
mask = k_ref < 0.30
rel_err_full = (pmm_clax_on_ref - pk_mm_cpt) / pk_mm_cpt * 100

print("=" * 55)
print("P_mm accuracy: clax vs CLASS-PT (z=0.38, Planck 2018)")
print("=" * 55)
print(f"k range compared: 0.005 – 0.30 h/Mpc  ({mask.sum()} modes)")
print()
print(f"  Max |err|  = {np.max(np.abs(rel_err_full[mask])):.3f} %")
print(f"  RMS err    = {np.sqrt(np.mean(rel_err_full[mask]**2)):.3f} %")
print(f"  Mean err   = {np.mean(rel_err_full[mask]):.4f} %  (sign = bias direction)")
print()
print(f"  Threshold  = 1.0 %  (CLASS-PT sub-percent accuracy target)")
print(f"  PASS       = {'YES  ✓' if np.max(np.abs(rel_err_full[mask])) < 1.0 else 'NO   ✗'}")
print("=" * 55)
print()
print("k [h/Mpc]  clax P_mm    CLASS-PT P_mm   rel err")
print("-" * 55)
k_check = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
for kc in k_check:
    idx = np.argmin(np.abs(k_ref - kc))
    c_v = pmm_clax_on_ref[idx]
    r_v = pk_mm_cpt[idx]
    err = (c_v - r_v) / r_v * 100
    print(f"  {k_ref[idx]:.3f}    {c_v:12.2f}     {r_v:12.2f}   {err:+.3f}%")
""")

# ── Assemble notebook ────────────────────────────────────────────────────────
nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]

# Set kernel metadata
nb.metadata["kernelspec"] = {
    "display_name": "sbi_pytorch_osx-arm64-py310forge",
    "language": "python",
    "name": "sbi_pytorch_osx-arm64-py310forge",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10.0",
}

outpath = os.path.join(_REPO, "notebooks", "pm_mm_validation.ipynb")
nbformat.write(nb, outpath)
print(f"Notebook written to: {outpath}")
print(f"  {len(nb.cells)} cells")
