"""Generate publication-quality comparison figure: jaxCLASS vs CLASS."""
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication style
plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

DOUBLE_COL = 6.75
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'gray': '#BBBBBB',
    'cyan': '#33BBEE',
}

# ── Load CLASS reference data ──────────────────────────────────────────
ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ref_pk = np.load('reference_data/lcdm_fiducial/pk.npz')

ell_ref = ref_cls['ell']
# D_l = l(l+1)/(2pi) * C_l
fac_ref = ell_ref * (ell_ref + 1) / (2 * np.pi)

# ── Compute jaxCLASS spectra ──────────────────────────────────────────
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxclass import CosmoParams, PrecisionParams
from jaxclass.background import background_solve
from jaxclass.thermodynamics import thermodynamics_solve
from jaxclass.perturbations import perturbations_solve
from jaxclass.harmonic import compute_cl_tt_interp, compute_cl_ee_interp, compute_cl_te_interp
from jaxclass.primordial import primordial_scalar_pk

print("Computing jaxCLASS spectra...", flush=True)
params = CosmoParams()

# P(k) — use default precision
prec_pk = PrecisionParams()
bg = background_solve(params, prec_pk)
th = thermodynamics_solve(params, prec_pk, bg)
pt_pk = perturbations_solve(params, prec_pk, bg, th)

# P(k) at z=0: P(k) = (2pi^2/k^3) * P_R(k) * |delta_m(k,z=0)|^2
k_pk = np.array(pt_pk.k_grid)
P_R = np.array(primordial_scalar_pk(pt_pk.k_grid, params))
delta_m = np.array(pt_pk.delta_m[:, -1])  # last tau = today
pk_jax = (2 * np.pi**2 / k_pk**3) * P_R * delta_m**2

# C_l — use science_cl with source interpolation (faster than planck_cl on CPU)
prec_cl = PrecisionParams(
    pt_k_max_cl=0.35, pt_k_per_decade=60,
    pt_tau_n_points=4000, pt_l_max_g=50, pt_l_max_pol_g=50, pt_l_max_ur=50,
    pt_ode_rtol=1e-6, pt_ode_atol=1e-11, ode_max_steps=131072,
)
bg_cl = background_solve(params, prec_cl)
th_cl = thermodynamics_solve(params, prec_cl, bg_cl)
pt_cl = perturbations_solve(params, prec_cl, bg_cl, th_cl)

ells_compute = list(range(2, 31)) + list(range(35, 101, 5)) + list(range(120, 501, 20))
print(f"Computing C_l at {len(ells_compute)} multipoles...", flush=True)

cl_tt_jax = np.array(compute_cl_tt_interp(pt_cl, params, bg_cl, ells_compute, n_k_fine=3000))
cl_ee_jax = np.array(compute_cl_ee_interp(pt_cl, params, bg_cl, ells_compute, n_k_fine=3000))
cl_te_jax = np.array(compute_cl_te_interp(pt_cl, params, bg_cl, ells_compute, n_k_fine=3000))

ells_jax = np.array(ells_compute)
fac_jax = ells_jax * (ells_jax + 1) / (2 * np.pi)

print("Computing done. Making figure...", flush=True)

# ── CLASS reference at same ells ──────────────────────────────────────
cl_tt_class = np.array([ref_cls['tt'][l] for l in ells_compute])
cl_ee_class = np.array([ref_cls['ee'][l] for l in ells_compute])
cl_te_class = np.array([ref_cls['te'][l] for l in ells_compute])

# P(k) reference — interpolate to our k values
k_ref = ref_pk['k']
pk_ref = ref_pk['pk_z0.0']
pk_class_interp = np.interp(k_pk, k_ref, pk_ref)

# ── Residuals ─────────────────────────────────────────────────────────
def safe_residual(ours, theirs):
    """(ours - theirs) / |theirs| * 100, avoiding division by tiny values."""
    mask = np.abs(theirs) > np.max(np.abs(theirs)) * 1e-6
    res = np.full_like(ours, np.nan)
    res[mask] = (ours[mask] - theirs[mask]) / np.abs(theirs[mask]) * 100
    return res

res_pk = safe_residual(pk_jax, pk_class_interp)
res_tt = safe_residual(cl_tt_jax, cl_tt_class)
res_ee = safe_residual(cl_ee_jax, cl_ee_class)
res_te = safe_residual(cl_te_jax, cl_te_class)

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(DOUBLE_COL, 5.5), constrained_layout=True)

# Create 4x2 grid: each panel gets a main plot + residual subplot
gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.08)

panels = [
    # (row_main, row_res, col, title, x, y_class, y_jax, residual, xlabel, ylabel, xlog, ylog)
    (0, 1, 0, r'$P(k)$', k_pk, pk_class_interp, pk_jax, res_pk,
     r'$k$ [Mpc$^{-1}$]', r'$P(k)$ [Mpc$^3$]', True, True),
    (0, 1, 1, r'$D_\ell^{TT}$', ells_jax, cl_tt_class * fac_jax, cl_tt_jax * fac_jax, res_tt,
     r'$\ell$', r'$\ell(\ell+1)C_\ell^{TT}/2\pi$', False, False),
    (2, 3, 0, r'$D_\ell^{EE}$', ells_jax, cl_ee_class * fac_jax, cl_ee_jax * fac_jax, res_ee,
     r'$\ell$', r'$\ell(\ell+1)C_\ell^{EE}/2\pi$', False, False),
    (2, 3, 1, r'$D_\ell^{TE}$', ells_jax, cl_te_class * fac_jax, cl_te_jax * fac_jax, res_te,
     r'$\ell$', r'$\ell(\ell+1)C_\ell^{TE}/2\pi$', False, False),
]

for row_m, row_r, col, title, x, y_class, y_jax, residual, xlabel, ylabel, xlog, ylog in panels:
    ax_main = fig.add_subplot(gs[row_m, col])
    ax_res = fig.add_subplot(gs[row_r, col], sharex=ax_main)

    # Main panel
    ax_main.plot(x, y_class, color=COLORS['blue'], lw=1.2, label='CLASS', zorder=2)
    ax_main.plot(x, y_jax, color=COLORS['orange'], lw=0.9, ls='--', label='jaxCLASS', zorder=3)

    if xlog:
        ax_main.set_xscale('log')
    if ylog:
        ax_main.set_yscale('log')

    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    ax_main.legend(loc='best', frameon=True, fancybox=False, edgecolor='none',
                   facecolor='white', framealpha=0.8)
    ax_main.tick_params(labelbottom=False)

    # Residual panel
    ax_res.axhline(0, color='black', lw=0.5, zorder=1)
    ax_res.fill_between(x, -1, 1, color=COLORS['teal'], alpha=0.12, zorder=0, label='Sub-percent')
    ax_res.plot(x, residual, color=COLORS['red'], lw=0.8, zorder=2)

    if xlog:
        ax_res.set_xscale('log')

    ax_res.set_xlabel(xlabel)
    ax_res.set_ylabel(r'$\Delta$ [%]')

    # Set residual y-limits
    valid = residual[np.isfinite(residual)]
    if len(valid) > 0:
        ymax = min(max(np.percentile(np.abs(valid), 95) * 1.5, 2), 20)
        ax_res.set_ylim(-ymax, ymax)

# Save
fig.savefig('figures/comparison_class_jaxclass.pdf', dpi=300)
fig.savefig('figures/comparison_class_jaxclass.png', dpi=200)
print("Saved to figures/comparison_class_jaxclass.{pdf,png}")
