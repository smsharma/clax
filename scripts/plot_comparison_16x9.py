"""jaxCLASS vs CLASS comparison — 4 panels with residuals (16:9 presentation)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'teal': '#009988',
    'red': '#CC3311',
}

# ── Load data ──────────────────────────────────────────────────────────
d = np.load('figures/jaxclass_spectra_h100.npz')
k_pk, pk_jax = d['k_pk'], d['pk_jax']
ells = d['ells']
cl_tt_jax, cl_ee_jax, cl_te_jax = d['cl_tt'], d['cl_ee'], d['cl_te']
fac = ells * (ells + 1) / (2 * np.pi)

ref_cls = np.load('reference_data/lcdm_fiducial/cls.npz')
ref_pk = np.load('reference_data/lcdm_fiducial/pk.npz')
cl_tt_ref = np.array([ref_cls['tt'][int(l)] for l in ells])
cl_ee_ref = np.array([ref_cls['ee'][int(l)] for l in ells])
cl_te_ref = np.array([ref_cls['te'][int(l)] for l in ells])
pk_ref = np.interp(k_pk, ref_pk['k'], ref_pk['pk_z0.0'])


def residual(ours, theirs):
    """Percentage residual — no masking, let ylim handle zero crossings."""
    return (ours - theirs) / np.abs(theirs) * 100


# ── Figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 2.8))
gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.08, wspace=0.08)

panels = [
    dict(col=0, title=r'Matter power spectrum',
         x=k_pk, y_ref=pk_ref, y_jax=pk_jax, res=residual(pk_jax, pk_ref),
         xlabel=r'$k$ [Mpc$^{-1}$]', ylabel=r'$P(k)$ [Mpc$^3$]',
         xlog=True, ylog=True, xlim=(1e-4, 1.0), reslim=(-5, 5)),
    dict(col=1, title=r'Temperature $C_\ell^{TT}$',
         x=ells, y_ref=cl_tt_ref * fac, y_jax=cl_tt_jax * fac,
         res=residual(cl_tt_jax, cl_tt_ref),
         xlabel=r'$\ell$', ylabel=r'$\ell(\ell+1)C_\ell^{TT}/2\pi$',
         xlog=False, ylog=False, xlim=(20, 1000), reslim=(-5, 5)),
    dict(col=2, title=r'$E$-mode $C_\ell^{EE}$',
         x=ells, y_ref=cl_ee_ref * fac, y_jax=cl_ee_jax * fac,
         res=residual(cl_ee_jax, cl_ee_ref),
         xlabel=r'$\ell$', ylabel=r'$\ell(\ell+1)C_\ell^{EE}/2\pi$',
         xlog=False, ylog=True, xlim=(20, 1000), reslim=(-5, 5)),
    dict(col=3, title=r'Cross $C_\ell^{TE}$',
         x=ells, y_ref=cl_te_ref * fac, y_jax=cl_te_jax * fac,
         res=residual(cl_te_jax, cl_te_ref),
         xlabel=r'$\ell$', ylabel=r'$\ell(\ell+1)C_\ell^{TE}/2\pi$',
         xlog=False, ylog=False, xlim=(20, 1000), reslim=(-5, 5)),
]

for p in panels:
    c = p['col']
    ax_top = fig.add_subplot(gs[0, c])
    ax_bot = fig.add_subplot(gs[1, c], sharex=ax_top)

    # ── Top: spectra ──
    ax_top.plot(p['x'], p['y_ref'], color=COLORS['blue'], lw=1.2,
                label='CLASS')
    ax_top.plot(p['x'], p['y_jax'], color=COLORS['orange'], ls='--', lw=1.2,
                label='jaxCLASS')

    if p['xlog']:
        ax_top.set_xscale('log')
    if p['ylog']:
        ax_top.set_yscale('log')
    ax_top.set_xlim(*p['xlim'])
    ax_top.set_title(p['title'])
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel(p['ylabel'])
    ax_top.legend(loc='upper right')

    # ── Bottom: residuals ──
    ax_bot.axhline(0, color='black', lw=0.4)
    ax_bot.fill_between(p['x'], -1, 1, color=COLORS['teal'], alpha=0.2)
    ax_bot.plot(p['x'], p['res'], color=COLORS['red'], lw=0.8)

    if p['xlog']:
        ax_bot.set_xscale('log')
    ax_bot.set_xlim(*p['xlim'])
    ax_bot.set_ylim(*p['reslim'])
    ax_bot.set_xlabel(p['xlabel'])
    ax_bot.set_ylabel(r'$\Delta$ [\%]')

fig.savefig('figures/comparison_class_jaxclass_h100.pdf')
fig.savefig('figures/comparison_class_jaxclass_h100.png', dpi=200)
print("Saved to figures/comparison_class_jaxclass_h100.{pdf,png}")
