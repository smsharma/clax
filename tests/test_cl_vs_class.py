"""V1 (smsharma/clax#54): C_l (unlensed TT/EE/TE, C_l^phiphi, lensed TT/EE/TE/BB)
at fit_cl and planck_cl against CLASS is no worse than the pre-#54 code (5ac3df3)
in any l band. GPU. Multi-cosmology: cl_value_cosmology.

This is an acceptance gate tied to the stored baseline's _provenance.base_commit
(5ac3df3), not a standing regression gate: it compares live spectra with spectra
stored from that commit, so once upstream physics changes the fit_cl or planck_cl spectra
(#58, #60 and #61 have, since) it measures those changes, not this branch. Regenerate
the baseline on the new base (or retire the test) at that point.

The criterion (the slack 1e-4 is the plan's; docs/adr/0005-cl-differentiability.md,
Decision 8). At a multipole (or, for the band-maximum form below, over a band) the
fix passes if EITHER
  (1) fix_err <= base_err + 1e-4: the error against CLASS, |C/C_CLASS - 1|
      (TE and lensed TE over sqrt(TT EE) of the reference), grew by at most
      the plan's slack; OR
  (2) |C_fix - C_base| / |C_base| <= 1e-4 (TE and lensed TE over sqrt(TT EE)
      of the reference, since TE crosses zero): the fix changed the spectrum by
      no more than the slack.
Bands starting at l >= 30 (30-100, 100-500, 500-1500, 1500-2500) are checked l
by l: the band passes only if EVERY l in it passes (1) or (2), so a
worsening that stays below the band's worst error (at fit_cl 500-1500 that is the
l = 1500 edge, 11%) cannot hide. The test prints, per band, the worst l (the l
with the least margin on its better branch), its fix_err, base_err, shift and
which branch passed. Bands 2-30 keep the band-maximum form (max over the band of
the error, of the shift): at l < 30 the CLASS reference has low-l noise and fit_cl
is unstable, so an l-by-l comparison there says nothing; KNOWN_SHIFTS below
holds the allowances. The last line each case prints, "V1 SUMMARY ...", counts
the bands per branch and names the tightest one with its margin on the better
branch (see _tally), so `grep 'V1 SUMMARY'` gives the verdicts of a run.
Base errors (band maxima): tests/data/cl_vs_class_baseline.json; base spectra,
for the l-by-l form: tests/data/cl_vs_class_baseline.npz.

Why two branches: the cubic Hermite Bessel lookup makes two intended changes.
(a) It raises TT near l ~ 40-50 by 9.6e-5 to 9.9e-5 of itself at every
cosmology: the removal of the linear 30-point table's amplitude bias,
-(dx)^2/12 ~ -9.3e-5 at table spacing dx = 1/30. As a difference of errors against a reference the base misses
by 4.94% (massive_nu_015 at fit_cl, #57) that reads 1.0086e-4, so branch (1)
alone would flag the removed bias; branch (2) passes it. The TT 30-100 margin
is about 1e-6 by construction (the slack equals the removed bias): the tightest
band of each case has 0.75-2.3e-6 left on its better branch (3.8e-6 at
massive_nu_015 fit_cl, on branch (2)), least at planck_cl omega_b_high l = 47
(7.5e-7; GPU jobs 30237-30241). So a rebase onto #58/#60/#61 re-measures it; so does
a JAX/XLA upgrade
(measured with JAX 0.9.2 on a V100) or a CPU run, because Rodas5 step sequences
depend on compile order (job 27675).
(b) It lowers EE at l = 2 by 0.7-0.9% (and TE by up to 4.5e-3 of sqrt(TT EE)):
the removal of the linear table's l = 2 bias, which improves planck_cl EE 2-4
against CLASS from 0.596% to 0.151%. At fit_cl the base already sits 1.2-4.5%
below CLASS at l = 2, so the same shift moves C_2^EE a further 0.74-0.88% AWAY
from CLASS at all five cosmologies. fit_cl EE 2-30 nevertheless passes branch
(1) (+1.49e-5 at massive_nu_015, -5.7e-5 to -6.0e-5 at the other four), because
the error metric is the band maximum, which sits at l = 10-15, not at l = 2. In
the same way planck_cl EE (1.3e-4 to 1.6e-4) and TE (1.0e-4 to 1.2e-4) at l = 3
move away from CLASS inside bands whose maximum is at l = 13-23. l < 30 is
outside every gradient functional, and cosmic variance at l = 2 is 63%. Where
the l = 2 TE shift sets fit_cl's band maximum, KNOWN_SHIFTS holds the documented
allowance (fix_err <= base_err + allowance), and the test prints the measured
shift every run.

Like with like: the base is measured at the fix's k-grid (fit_cl at n_k_fine
20000; the base preset's own 5000 is recorded under "fit_cl_nk5000" in the JSON
as documentation and is not gated). The fiducial, omega_b_high and h_high
cls.npz / cls_lensed.npz predate the CLASS build of cl_pp.npz (up to ~2.5e-3);
fix and base are compared against the same files, so that offset cancels."""
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from clax import PrecisionParams
from clax.harmonic import compute_cls_all_fast
from clax.lensing import compute_cl_pp, lens_cls
from clax.perturbations import perturbations_solve
from clax.thermodynamics import solve_background_and_thermo
from dataclasses import replace

pytestmark = pytest.mark.slow
HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "..", "reference_data")
with open(os.path.join(HERE, "data", "cl_vs_class_baseline.json")) as _f:
    BASE = json.load(_f)
BASE_CL = np.load(os.path.join(HERE, "data", "cl_vs_class_baseline.npz"))
PRESETS = {"fit_cl": PrecisionParams.fit_cl, "planck_cl": PrecisionParams.planck_cl}
BANDS = {"2-30": (2, 30), "30-100": (30, 100), "100-500": (100, 500),
         "500-1500": (500, 1500), "1500-2500": (1500, 2500)}
TOL = 1e-4
PERL_FROM = 30  # bands starting at l >= PERL_FROM are checked l by l
_TE_L2 = ("the cubic Hermite Bessel lookup moves fit_cl TE at l = 2 by -4.0e-3 of sqrt(TT EE) "
          "against the linear 30-point table at equal n_k_fine (job 27208); fit_cl at "
          "l < 30 is outside every gradient functional")
# (preset, spectrum, band) -> (allowance over the base error, reason); recorded in ADR
# 0005 (Consequences). No entry may be added without a reason recorded there.
KNOWN_SHIFTS = {
    ("fit_cl", "te", "2-30"): (5e-3, _TE_L2),
    ("fit_cl", "lte", "2-30"): (
        5e-3, _TE_L2 + "; lensing leaves TE at l = 2 unchanged, and the lensed shift equals the "
        "unlensed one to 1.4e-7 (jobs 30140-30150)"),
}


def _errors(name, p, prec):
    """Per band: max |C/C_CLASS - 1| (TE: over sqrt(TT EE) of the reference); and the spectra."""
    prec = replace(prec, th_grad_mode="native")
    bg, th = solve_background_and_thermo(p, prec)
    pt = perturbations_solve(p, prec, bg, th)
    lmax = prec.hr_l_max
    cl = compute_cls_all_fast(pt, p, bg, l_max=lmax, n_k_fine=prec.hr_n_k_fine, prec=prec)
    pp = compute_cl_pp(pt, p, bg, th, lmax, prec=prec)
    lt, le, lte, lb = lens_cls(cl["tt"], cl["ee"], cl["te"], jnp.zeros_like(cl["tt"]), pp, l_max=lmax)
    u = np.load(os.path.join(REF, name, "cls.npz"))
    ln = np.load(os.path.join(REF, name, "cls_lensed.npz"))
    rp = np.load(os.path.join(REF, name, "cl_pp.npz"))
    spectra = {
        "tt": (np.asarray(cl["tt"]), u["tt"]), "ee": (np.asarray(cl["ee"]), u["ee"]),
        "te": (np.asarray(cl["te"]), u["te"], np.sqrt(u["tt"] * u["ee"])),
        "pp": (np.asarray(pp), rp["pp"]),
        "ltt": (np.asarray(lt), ln["tt"]), "lee": (np.asarray(le), ln["ee"]),
        "lte": (np.asarray(lte), ln["te"], np.sqrt(ln["tt"] * ln["ee"])),
        "lbb": (np.asarray(lb), ln["bb"]),
    }
    out = {}
    for s, tup in spectra.items():
        mine, ref = tup[0], tup[1]
        norm = tup[2] if len(tup) == 3 else ref
        out[s] = {}
        for band, (lo, hi) in BANDS.items():
            hi = min(hi, lmax)
            if lo >= hi:
                continue
            sl = slice(lo, hi + 1)
            out[s][band] = float(np.max(np.abs(mine[sl] - ref[sl]) / np.abs(norm[sl])))
    return out, spectra, lmax


def _vs_base(name, preset, spectra, lmax):
    """Per (spectrum, band), against the saved base spectra:
    "shift" = max over the band of |C_fix - C_base| / |C_base| (TE, lensed TE: over sqrt(TT EE) of
    the reference), "base_max" = the base error's band maximum recomputed from the npz (checked
    against the JSON in the test); and, for bands with lo >= PERL_FROM, the l-by-l form: the band
    is "ok" only if every l has fix_err(l) <= base_err(l) + TOL or shift(l) <= TOL (a NaN fails).
    "l" is the worst multipole, the one with the least margin on its better branch
    (the largest min(fix_err - base_err - TOL, shift - TOL)); fe/be/sh are the values there and
    "n_spec" counts the multipoles that pass on the spectrum branch only."""
    out = {}
    for s, tup in spectra.items():
        mine, ref = tup[0], tup[1]
        norm = tup[2] if len(tup) == 3 else ref
        base = BASE_CL[f"{name}__{preset}__{s}"]
        assert base.shape == (lmax + 1,), (s, base.shape, lmax)
        snorm = tup[2] if len(tup) == 3 else base
        out[s] = {}
        for band, (lo, hi) in BANDS.items():
            hi = min(hi, lmax)
            if lo >= hi:
                continue
            sl = slice(lo, hi + 1)
            fe = np.abs(mine[sl] - ref[sl]) / np.abs(norm[sl])
            be = np.abs(base[sl] - ref[sl]) / np.abs(norm[sl])
            sh = np.abs(mine[sl] - base[sl]) / np.abs(snorm[sl])
            row = {"shift": float(np.max(sh)), "base_max": float(np.max(be))}
            if lo >= PERL_FROM:
                ex_err, ex_spec = fe - be - TOL, sh - TOL
                err_ok = ex_err <= 0  # comparisons written so that a NaN fails
                ok = err_ok | (ex_spec <= 0)
                slack = np.minimum(ex_err, ex_spec)
                i = int(np.argmax(np.where(np.isnan(slack), np.inf, slack)))
                row.update(ok=bool(np.all(ok)), l=lo + i, fe=float(fe[i]), be=float(be[i]),
                           sh=float(sh[i]), n=len(ok), n_spec=int(np.sum(ok & ~err_ok)),
                           why=("err within slack" if err_ok[i] else "spectrum within slack"
                                if ex_spec[i] <= 0 else "FAIL"))
            out[s][band] = row
    return out


def _tally(counts, tight, why, margin, label):
    """Count a band under its verdict (KNOWN entries as "KNOWN"); keep in ``tight`` the
    non-KNOWN band with the least margin to 1e-4 on its better branch (a NaN margin
    counts as the tightest), for the summary line. The printed margin is the larger
    of the two branches' margins; the label in parentheses is the branch the verdict
    names, which is the error branch whenever that one passes (at omega_b_high
    fit_cl TT 30-100: margin 1.2e-6 from the spectrum branch, labelled "err within
    slack", whose own margin is 3.2e-7)."""
    key = "KNOWN" if why.startswith("KNOWN") else why
    counts[key] = counts.get(key, 0) + 1
    if margin is not None:
        m = -np.inf if np.isnan(margin) else float(margin)
        if not tight or m < tight[0][0]:
            tight[:] = [(m, label, key)]


@pytest.mark.parametrize("preset", list(PRESETS))
def test_v1_no_worse_than_base_vs_class(cl_value_cosmology, preset):
    name, p = cl_value_cosmology
    errs, spectra, lmax = _errors(name, p, PRESETS[preset]())
    vs = _vs_base(name, preset, spectra, lmax)
    bad = []
    counts, tight = {}, []
    for s, bands in errs.items():
        for band, e in bands.items():
            b = BASE[name][preset][s][band]
            v = vs[s][band]
            # the JSON band error and the npz must stay the same record
            if not abs(v["base_max"] - b) <= 1e-9:
                bad.append(f"{s} {band}: baseline JSON {b:.6e} != baseline npz {v['base_max']:.6e}")
            d, r = e - b, v["shift"]
            known = KNOWN_SHIFTS.get((preset, s, band))
            if known is not None:  # comparisons written so that a NaN fails
                ok = d <= known[0]
                why = f"KNOWN (allowance {known[0]:.0e})" if ok else "FAIL"
                if not ok:
                    bad.append(f"{s} {band}: KNOWN allowance {known[0]:.0e} exceeded, measured shift "
                               f"{d:+.2e} (fix_err {e:.3e}, base_err {b:.3e})")
                _tally(counts, tight, why, None, None)
            elif "ok" in v:  # l by l
                why = v["why"]
                if not v["ok"]:
                    why = "FAIL"
                    bad.append(f"{s} {band} l={v['l']}: fix_err - base_err {v['fe'] - v['be']:+.2e} "
                               f"> {TOL:.0e} and shift {v['sh']:.2e} > {TOL:.0e}")
                print(f"{name} {preset} {s:3s} {band:9s} worst l={v['l']:<4d} fix_err {v['fe']:.3e} "
                      f"base_err {v['be']:.3e} diff {v['fe'] - v['be']:+.2e} shift {v['sh']:.2e} "
                      f"-> {why} ({v['n_spec']}/{v['n']} l on the spectrum branch only)")
                _tally(counts, tight, why, np.max([TOL - (v["fe"] - v["be"]), TOL - v["sh"]]),
                       f"{s} {band} l={v['l']}")
                continue
            elif d <= TOL:
                why = "err within slack"
            elif r <= TOL:
                why = "spectrum within slack"
            else:
                why = "FAIL"
                bad.append(f"{s} {band}: fix_err - base_err {d:+.2e} > {TOL:.0e} and band shift "
                           f"{r:.2e} > {TOL:.0e}")
            print(f"{name} {preset} {s:3s} {band:9s} band max fix_err {e:.3e} base_err {b:.3e} "
                  f"diff {d:+.2e} rel_shift {r:.2e} -> {why}")
            if known is None:
                _tally(counts, tight, why, np.max([TOL - d, TOL - r]), f"{s} {band}")
    for (pr, s, band), (allow, reason) in KNOWN_SHIFTS.items():
        if pr == preset and band in errs.get(s, {}):
            print(f"{name} {preset}: KNOWN shift {s} {band} "
                  f"{errs[s][band] - BASE[name][preset][s][band]:+.2e} (allowance {allow:.0e}: {reason})")
    m, label, key = tight[0] if tight else (np.nan, "-", "-")
    print(f"V1 SUMMARY {name} {preset}: {sum(counts.values())} bands: "
          f"{counts.get('err within slack', 0)} on the error branch, "
          f"{counts.get('spectrum within slack', 0)} on the spectrum branch, "
          f"{counts.get('KNOWN', 0)} KNOWN, {counts.get('FAIL', 0)} FAIL; tightest {label} "
          f"margin {m:.1e} ({key}); {len(bad)} failing check(s)")
    assert not bad, (f"ERROR {len(bad)} band(s) failing against the pre-#54 base ({name}, {preset}): "
                     + "; ".join(bad[:3]))
