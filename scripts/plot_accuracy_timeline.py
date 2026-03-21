"""clax accuracy timeline — from first commit to science-grade C_l."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime, timezone, timedelta

plt.style.use(Path("~/.claude/skills/matplotlib-publication/matplotlibrc").expanduser())

EST = timezone(timedelta(hours=-5))

def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=EST)

def to_hours(dt, t0):
    return (dt - t0).total_seconds() / 3600

t0 = parse_dt("2026-02-07 09:00")

# ── Accuracy data (|error| at l=100 unless noted) ─────────────────────
# Each entry: (datetime_str, abs_error_percent)

tt_l100 = [
    ("2026-02-07 09:04", 1000),     # initial: ~10x off
    ("2026-02-07 10:46", 1.0),      # IBP source + 4pi norm (lucky l)
    ("2026-02-07 11:09", 2.0),      # other l still 20-80%
    ("2026-02-07 15:36", 4.0),      # GPU, 40 k/dec
    ("2026-02-07 19:13", 5.1),      # CPU verified
    ("2026-02-07 23:21", 17),       # extreme k-res reveals 17% floor
    ("2026-02-08 01:38", 17.5),     # nonIBP Doppler, same floor
    ("2026-02-08 13:55", 8.2),      # theta_b' fix (bugs 16-18)
    ("2026-02-08 15:36", 8.0),      # RECFAST, marginal help for TT
    ("2026-02-09 23:22", 0.73),     # source interp + T012 + thermo fix
    ("2026-02-10 01:14", 0.57),     # planck_cl preset
    ("2026-02-10 16:05", 0.57),     # RSA damping (no effect)
    # Agent work (Feb 11)
    ("2026-02-11 03:17", 0.57),     # agent session 1 (g' fix, RSA sources)
    ("2026-02-11 14:01", 0.45),     # Bug 22: RSA in source Einstein eqs
    ("2026-02-11 16:58", 0.23),     # k-integration fix + hierarchy truncation
    ("2026-02-11 21:34", 0.02),     # ncdm (rho+p) correction
    # Agent work (Feb 12, H100)
    ("2026-02-12 10:39", 0.02),     # full ncdm hierarchy Psi_l(q)
]

ee_l100 = [
    ("2026-02-07 13:55", 49),       # source_E normalization fix
    ("2026-02-07 15:36", 13),       # Bessel fix + k-resolution
    ("2026-02-07 23:21", 5),        # extreme k-res: 3-8%
    ("2026-02-08 13:55", 5),        # diagnostics, no EE change
    ("2026-02-08 15:36", 5),        # RECFAST, modest help
    ("2026-02-09 23:22", 0.11),     # source interp + thermo fix
    ("2026-02-10 01:14", 0.17),     # planck_cl
    ("2026-02-10 16:05", 0.17),     # unchanged
    # Agent work
    ("2026-02-11 14:01", 0.15),     # Bug 22 fix
    ("2026-02-11 21:34", 0.04),     # ncdm correction
    ("2026-02-12 10:39", 0.02),     # full ncdm hierarchy
]

te_l100 = [
    ("2026-02-10 01:14", 0.5),      # first TE measurement
    ("2026-02-10 16:05", 0.5),      # unchanged
    ("2026-02-11 14:01", 0.38),     # Bug 22 fix
    ("2026-02-11 21:34", 0.04),     # ncdm correction
    ("2026-02-12 10:39", 0.03),     # full ncdm hierarchy
]

# ── Colors ─────────────────────────────────────────────────────────────
c_tt = '#CC3311'
c_ee = '#0077BB'
c_te = '#009988'
c_dead = '#AA3377'

bg_day = '#fafafa'
bg_night = '#f0f4f8'

# ── Annotation style ──────────────────────────────────────────────────
ann_fs = 7.5  # annotation font size
ann_bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='#ddd', alpha=0.88, lw=0.4)
ann_arrow = dict(arrowstyle='-', color='#aaa', lw=0.5)
dead_bbox = dict(boxstyle='round,pad=0.15', fc='#fff5f5', ec='#AA3377', alpha=0.85, lw=0.4)
dead_arrow = dict(arrowstyle='-', color='#AA3377', lw=0.4, ls='--')

# ── Figure ─────────────────────────────────────────────────────────────
xmax = 128  # ~5.3 days
fig, ax = plt.subplots(figsize=(7, 3.5))

# Day/night shading
for s, e in [(-1, 16), (24, 40), (48, 64), (72, 88), (96, 112)]:
    ax.axvspan(s, e, color=bg_day, zorder=0)
for s, e in [(16, 24), (40, 48), (64, 72), (88, 96), (112, xmax)]:
    ax.axvspan(s, e, color=bg_night, zorder=0)

# Plot lines
for data, color, label, marker in [
    (tt_l100, c_tt, r'$C_\ell^{TT}$', 'o'),
    (ee_l100, c_ee, r'$C_\ell^{EE}$', 's'),
    (te_l100, c_te, r'$C_\ell^{TE}$', 'D'),
]:
    h = [to_hours(parse_dt(d), t0) for d, _ in data]
    v = [v for _, v in data]
    ax.semilogy(h, v, f'{marker}-', color=color, ms=3.5, lw=1.0,
                label=f'{label} ($\\ell{{=}}100$)', zorder=5,
                markeredgecolor='white', mew=0.4)

# Dead ends
dead_ends = [
    ("2026-02-07 21:12", 5.1, "Sign flip attempt\n(reverted)", (18, -16)),
    ("2026-02-10 08:55", 0.57, "Radiation damping\n(no effect)", (24, 26)),
]
for ds, y, txt, ofs in dead_ends:
    h = to_hours(parse_dt(ds), t0)
    ax.plot(h, y, 'X', color=c_dead, ms=7, zorder=10,
            markeredgecolor='white', mew=0.8)
    ax.annotate(txt, (h, y), textcoords='offset points', xytext=ofs,
                fontsize=ann_fs, ha='center', color=c_dead, style='italic',
                arrowprops=dead_arrow, bbox=dead_bbox)

# Reference lines
ax.axhline(1, color='k', ls='--', lw=0.5, alpha=0.4, zorder=1)
ax.axhline(0.1, color='k', ls=':', lw=0.5, alpha=0.3, zorder=1)
trans_yr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.995, 1.15, r'$1\%$', fontsize=7, color='k', alpha=0.5, va='bottom',
        ha='right', transform=trans_yr)
ax.text(0.995, 0.115, r'$0.1\%$', fontsize=7, color='k', alpha=0.5, va='bottom',
        ha='right', transform=trans_yr)

# Annotations for key milestones
annots = [
    ("2026-02-07 10:46", 1.0,
     "Source function\nnormalization fix", (24, -16)),
    ("2026-02-07 15:36", 4.0,
     "Bessel functions\n+ $k$-grid resolution", (18, 52)),
    ("2026-02-07 23:21", 17,
     "17\\% floor discovered\n(source function bug)", (18, 46)),
    ("2026-02-08 13:55", 8.2,
     "Baryon velocity\n+ tight-coupling fix", (22, 22)),
    ("2026-02-09 23:22", 0.73,
     "Source interpolation\n+ recombination rewrite", (-28, 28)),
    ("2026-02-11 14:01", 0.45,
     "Radiation streaming\nin Einstein eqs", (18, 28)),
    ("2026-02-11 21:34", 0.02,
     "Neutrino energy\u2013\nmomentum fix", (-40, -8)),
    ("2026-02-12 10:39", 0.02,
     "Full neutrino\nhierarchy", (-34, 28)),
]
for ds, v, txt, ofs in annots:
    h = to_hours(parse_dt(ds), t0)
    ax.annotate(txt, (h, v), textcoords='offset points', xytext=ofs,
                fontsize=ann_fs, ha='center', color='#333',
                arrowprops=ann_arrow, bbox=ann_bbox)

# Day labels as x-axis ticks
day_positions = [8, 32, 56, 77, 102, 120]
day_labels = ['Fri\nFeb 7', 'Sat\nFeb 8', 'Sun\nFeb 9', 'Mon\nFeb 10',
              'Tue\nFeb 11', 'Wed\nFeb 12']

# Legend
dead_patch = plt.Line2D([0], [0], marker='X', color='w', markerfacecolor=c_dead,
                        markeredgecolor='white', ms=6, label='Dead end / revert')
handles, labels = ax.get_legend_handles_labels()
handles.append(dead_patch)
ax.legend(handles=handles, loc='upper right', fontsize=8.5,
          bbox_to_anchor=(1.0, 0.92))

# Axes
ax.set_ylabel(r'$|C_\ell^\mathrm{clax}/C_\ell^\mathrm{CLASS} - 1|$ [\%]', fontsize=9)
ax.set_xlim(-1, xmax)
ax.set_ylim(0.008, 5000)
ax.set_xticks(day_positions)
ax.set_xticklabels(day_labels)
ax.xaxis.set_minor_locator(mticker.MultipleLocator(4))
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='x', labelsize=7)

fig.savefig('figures/accuracy_timeline.pdf', bbox_inches='tight')
fig.savefig('figures/accuracy_timeline.png', dpi=450, bbox_inches='tight')
print("Saved to figures/accuracy_timeline.{pdf,png}")
