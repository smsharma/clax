"""Generate progress timeline plots for jaxCLASS development talk."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
from datetime import datetime, timezone, timedelta

EST = timezone(timedelta(hours=-5))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ─── Data from git history and commit messages ───────────────────────────

# C_l^TT error at l=100 (absolute %, from commit messages)
tt_l100 = [
    ("2026-02-07 09:04", 1000),     # ~10x off
    ("2026-02-07 10:46", 1.0),      # IBP form, one lucky l
    ("2026-02-07 11:09", 2.0),      # but other l still 20-80%
    ("2026-02-07 15:36", 4.0),      # GPU, 40 k/dec
    ("2026-02-07 19:13", 5.1),      # CPU verified
    ("2026-02-07 23:21", 17),       # extreme k-res reveals floor
    ("2026-02-08 01:38", 17.5),     # nonIBP, same floor
    ("2026-02-08 13:55", 8.2),      # theta_b' fix (bugs 16-18)
    ("2026-02-08 15:36", 8.0),      # RECFAST, marginal help here
    ("2026-02-09 23:22", 0.73),     # source interp + T012 + thermo fix
    ("2026-02-10 01:14", 0.57),     # planck_cl preset
    ("2026-02-10 16:05", 0.57),     # final (RSA had no effect)
]

# C_l^EE error at l=100 (absolute %)
ee_l100 = [
    ("2026-02-07 13:55", 49),       # after source_E normalization fix
    ("2026-02-07 15:36", 13),       # Bessel fix + k-resolution
    ("2026-02-07 23:21", 5),        # extreme k-res: 3-8%
    ("2026-02-08 13:55", 5),        # diagnostics, no EE change
    ("2026-02-08 15:36", 5),        # RECFAST, modest help
    ("2026-02-09 23:22", 0.11),     # source interp + thermo fix
    ("2026-02-10 01:14", 0.17),     # planck_cl
    ("2026-02-10 16:05", 0.17),     # final
]

# g(tau_star) error (absolute %)
g_tau = [
    ("2026-02-07 09:04", 2.6),      # estimated, MB95 only
    ("2026-02-08 13:55", 2.6),      # measured: MB95 recombination
    ("2026-02-08 15:36", 1.66),     # RECFAST added
    ("2026-02-09 23:22", 0.04),     # fudge fix + reionization bisection
    ("2026-02-10 16:05", 0.04),     # unchanged
]

# Tests passing
tests_data = [
    ("2026-02-07 09:04", 49),
    ("2026-02-07 11:51", 67),
    ("2026-02-07 13:55", 95),
    ("2026-02-07 17:09", 103),
    ("2026-02-08 13:55", 95),
    ("2026-02-09 23:22", 100),
    ("2026-02-10 16:05", 100),
]

# Lines of code (source, tests) at each commit
loc_data = [
    ("2026-02-07 09:04", 3109, 813),
    ("2026-02-07 11:51", 3611, 1183),
    ("2026-02-07 12:31", 4488, 1183),
    ("2026-02-07 13:55", 4511, 1755),
    ("2026-02-07 15:36", 4586, 1755),
    ("2026-02-07 17:05", 4612, 1755),
    ("2026-02-07 17:09", 4612, 1967),
    ("2026-02-07 19:57", 4660, 1967),
    ("2026-02-07 21:54", 4689, 1967),
    ("2026-02-08 01:38", 4752, 1967),
    ("2026-02-08 13:55", 4812, 1967),
    ("2026-02-08 15:36", 4902, 1967),
    ("2026-02-09 23:22", 5126, 1967),
    ("2026-02-10 01:14", 5242, 1967),
    ("2026-02-10 16:05", 5295, 1967),
]


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=EST)


def to_hours(dt, t0):
    return (dt - t0).total_seconds() / 3600


t0 = parse_dt("2026-02-07 09:00")
xmax = 82

# ─── Figure (3 panels: C_l, g(tau), LOC+tests) ──────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), sharex=True,
                         gridspec_kw={"hspace": 0.07,
                                      "height_ratios": [1.5, 0.9, 0.8]})

# Color palette
c_tt = "#c62828"
c_ee = "#1565c0"
c_g = "#2e7d32"
c_src = "#616161"
c_tst = "#e65100"
c_dead = "#b71c1c"

bg_day = "#fafafa"
bg_night = "#f0f4f8"

# ── Day/night shading (all panels) ──────────────────────────────────────
for ax_i in axes:
    for s, e in [(-1, 16), (24, 40), (48, 64), (72, xmax)]:
        ax_i.axvspan(s, e, color=bg_day, zorder=0)
    for s, e in [(16, 24), (40, 48), (64, 72)]:
        ax_i.axvspan(s, e, color=bg_night, zorder=0)

# ── Panel 1: C_l accuracy (log scale) ────────────────────────────────────
ax = axes[0]

h_tt = [to_hours(parse_dt(d), t0) for d, _ in tt_l100]
v_tt = [v for _, v in tt_l100]
ax.semilogy(h_tt, v_tt, "o-", color=c_tt, ms=6, lw=2,
            label=r"$C_\ell^{TT}$ ($\ell{=}100$)", zorder=5,
            markeredgecolor="white", mew=0.6)

h_ee = [to_hours(parse_dt(d), t0) for d, _ in ee_l100]
v_ee = [v for _, v in ee_l100]
ax.semilogy(h_ee, v_ee, "s-", color=c_ee, ms=6, lw=2,
            label=r"$C_\ell^{EE}$ ($\ell{=}100$)", zorder=5,
            markeredgecolor="white", mew=0.6)

# Dead ends
dead_ends = [
    ("2026-02-07 21:12", 5.1),
    ("2026-02-10 08:55", 0.57),
]
for ds, y in dead_ends:
    h = to_hours(parse_dt(ds), t0)
    ax.plot(h, y, "X", color=c_dead, ms=11, zorder=10,
            markeredgecolor="white", mew=1.2)

# Reference lines
ax.axhline(1, color="k", ls="--", lw=0.7, alpha=0.4, zorder=1)
ax.axhline(0.1, color="k", ls=":", lw=0.7, alpha=0.3, zorder=1)
trans_yr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.995, 1.15, "1%", fontsize=10, color="k", alpha=0.5, va="bottom",
        ha="right", transform=trans_yr)
ax.text(0.995, 0.115, "0.1%", fontsize=10, color="k", alpha=0.5, va="bottom",
        ha="right", transform=trans_yr)

ax.set_ylabel(r"$|C_\ell^{\rm jax}/C_\ell^{\rm CLASS} - 1|$  [%]", fontsize=13)
ax.set_ylim(0.03, 3000)

# Legend
dead_patch = plt.Line2D([0], [0], marker="X", color="w", markerfacecolor=c_dead,
                        markeredgecolor="white", ms=9, label="Dead end / revert")
handles, labels = ax.get_legend_handles_labels()
handles.append(dead_patch)
ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.95,
          edgecolor="#ccc")

ax.set_title(r"jaxCLASS: from first commit to sub-percent $C_\ell$ in one weekend",
             fontsize=16, fontweight="bold", pad=14)

# Day labels
trans_top = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for h_mid, label in [(8, "Fri Feb 7"), (32, "Sat Feb 8"),
                      (56, "Sun Feb 9"), (77, "Mon Feb 10")]:
    ax.text(h_mid, 0.95, label, fontsize=11, ha="center", color="#777",
            style="italic", transform=trans_top, fontweight="medium")

# Annotations
annots = [
    ("2026-02-07 10:46", 1.0, "IBP source\n+ 4\u03c0 norm", (-30, 22), 9.5),
    ("2026-02-07 15:36", 4.0, "Hybrid Bessel\n+ GPU k-res", (-40, 14), 9.5),
    ("2026-02-07 23:21", 17, "17% floor discovered\n(source fn bug)", (20, 12), 9.5),
    ("2026-02-08 13:55", 8.2, "Bugs #16\u201318\n(\u03b8\u2032_b + TCA)", (20, 12), 9.5),
    ("2026-02-09 23:22", 0.73, "Source interp +\nRECFAST + T\u2080+T\u2081+T\u2082", (-10, -32), 9.5),
    ("2026-02-10 01:14", 0.57, "planck_cl\npreset", (22, 14), 9),
]
for ds, v, txt, ofs, fs in annots:
    h = to_hours(parse_dt(ds), t0)
    ax.annotate(txt, (h, v), textcoords="offset points", xytext=ofs,
                fontsize=fs, ha="center", color="#333",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.7),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd",
                          alpha=0.92, lw=0.5))

# Dead end annotations
ax.annotate("T1 sign flip\n(reverted)", (to_hours(parse_dt("2026-02-07 21:12"), t0), 5.1),
            textcoords="offset points", xytext=(16, 20), fontsize=9, ha="center",
            color=c_dead, style="italic",
            arrowprops=dict(arrowstyle="-", color=c_dead, lw=0.6, ls="--"),
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff5f5", ec=c_dead, alpha=0.9, lw=0.5))
ax.annotate("RSA revert\n(minimal impact)", (to_hours(parse_dt("2026-02-10 08:55"), t0), 0.57),
            textcoords="offset points", xytext=(0, 24), fontsize=9, ha="center",
            color=c_dead, style="italic",
            arrowprops=dict(arrowstyle="-", color=c_dead, lw=0.6, ls="--"),
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff5f5", ec=c_dead, alpha=0.9, lw=0.5))

# ── Panel 2: g(tau_star) error ────────────────────────────────────────────
ax = axes[1]
h_g = [to_hours(parse_dt(d), t0) for d, _ in g_tau]
v_g = [v for _, v in g_tau]
ax.semilogy(h_g, v_g, "D-", color=c_g, ms=7, lw=2, zorder=5,
            markeredgecolor="white", mew=0.6)
ax.axhline(0.1, color="k", ls=":", lw=0.7, alpha=0.3, zorder=1)
trans_yr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(0.995, 0.115, "0.1%", fontsize=10, color="k", alpha=0.5, va="bottom",
        ha="right", transform=trans_yr)
ax.set_ylabel(r"Visibility $g(\tau_\star)$" "\n" "error [%]", fontsize=13)
ax.set_ylim(0.015, 10)

annots_g = [
    ("2026-02-08 15:36", 1.66, "RECFAST\nadded", (18, 12)),
    ("2026-02-09 23:22", 0.04, r"Fudge fix + bisection $z_{\rm reio}$"
     "\n" r"(70$\times$ improvement)", (28, -12)),
]
for ds, v, txt, ofs in annots_g:
    h = to_hours(parse_dt(ds), t0)
    ax.annotate(txt, (h, v), textcoords="offset points", xytext=ofs,
                fontsize=9.5, ha="center", color="#333",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.7),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd",
                          alpha=0.92, lw=0.5))

ax.text(0.015, 0.92, r"Visibility function $g(\tau_\star)$ error",
        fontsize=11, color=c_g, fontweight="medium",
        transform=ax.transAxes, va="top")

# ── Panel 3: Lines of code + tests ───────────────────────────────────────
ax = axes[2]
h_loc = [to_hours(parse_dt(d), t0) for d, _, _ in loc_data]
v_src = [s for _, s, _ in loc_data]
v_tst = [t for _, _, t in loc_data]
v_tot = [s + t for _, s, t in loc_data]

h_loc_ext = h_loc + [xmax]
v_src_ext = v_src + [v_src[-1]]
v_tot_ext = v_tot + [v_tot[-1]]

ax.fill_between(h_loc_ext, 0, v_src_ext, color=c_src, alpha=0.18, step="post",
                label="Source code")
ax.fill_between(h_loc_ext, v_src_ext, v_tot_ext, color=c_tst, alpha=0.18,
                step="post", label="Tests")
ax.plot(h_loc_ext, v_src_ext, color=c_src, lw=1.5, drawstyle="steps-post")
ax.plot(h_loc_ext, v_tot_ext, color=c_tst, lw=1.5, drawstyle="steps-post")

# Tests passing on secondary axis
ax2 = ax.twinx()
h_tests = [to_hours(parse_dt(d), t0) for d, _ in tests_data]
v_tests = [v for _, v in tests_data]
ax2.plot(h_tests, v_tests, "^-", color=c_tst, ms=6, lw=1.5,
         label="Tests passing", zorder=5, markeredgecolor="white", mew=0.6)
ax2.set_ylabel("Tests passing", fontsize=12, color=c_tst)
ax2.tick_params(axis="y", colors=c_tst, labelsize=11)
ax2.set_ylim(0, 130)
ax2.spines["right"].set_color(c_tst)

ax.set_ylabel("Lines of code", fontsize=13)
ax.set_ylim(0, 8500)
ax.yaxis.set_major_locator(mticker.MultipleLocator(2000))
ax.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#ccc")
ax2.legend(loc="center right", fontsize=10, framealpha=0.9, edgecolor="#ccc")

# Feature annotations
feat_annots = [
    ("2026-02-07 12:31", 5700, "+EE/TE, tensors,\nlensing (+900 LoC)", (18, 8)),
    ("2026-02-09 23:22", 7100, "Source interp,\nRECFAST rewrite", (18, 5)),
]
for ds, v, txt, ofs in feat_annots:
    h = to_hours(parse_dt(ds), t0)
    ax.annotate(txt, (h, v), textcoords="offset points", xytext=ofs,
                fontsize=9, ha="left", color="#555",
                arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ddd",
                          alpha=0.9, lw=0.4))

# ── Shared x-axis ────────────────────────────────────────────────────────
axes[-1].set_xlabel("Hours since project start (Fri Feb 7, 9 AM EST)", fontsize=13)
axes[-1].set_xlim(-1, xmax)

for ax_i in axes:
    ax_i.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax_i.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax_i.grid(axis="x", alpha=0.15, which="major")
    ax_i.grid(axis="y", alpha=0.12)
    ax_i.spines["top"].set_visible(False)
    ax_i.spines["right"].set_visible(False)

axes[2].spines["right"].set_visible(True)

fig.subplots_adjust(hspace=0.08, left=0.09, right=0.91, top=0.94, bottom=0.06)

plt.savefig("scripts/jaxclass_progress.pdf", bbox_inches="tight", dpi=150)
plt.savefig("scripts/jaxclass_progress.png", bbox_inches="tight", dpi=200)
print("Saved to scripts/jaxclass_progress.pdf and .png")
