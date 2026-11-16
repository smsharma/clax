"""Preset accuracy against CLASS (precision sweep, Oct 10-12, 2026).

The Oct 2026 sweep (docs/precision_sweep_2026-10-10.md) retuned planck_cl from
Kvaerno5 at l_max 50, rtol 1e-6, atol 1e-11 (4606 s per perturbation solve on
a V100) to l_max 17, rtol 1e-4, atol 1e-6 (100 s) with spectra within 0.1% of
the old preset, and fixed the fast C_l path (Bessel table 30 points per unit x,
n_k_fine 20000). Measured against the CLASS massless-RECFAST reference (sweep config P1,
Oct 12, 2026) the new preset reads TT within 0.09% (l < 30), 0.22% (l <= 500),
0.25% (l <= 1000); EE within 0.60% (l = 2-4), 0.48% (l = 5-500), 0.98% at a
single l near 526 (rms 0.28% over 500-1000); TE/sqrt(TT EE) within 0.37%.
The l ~ 526 EE excursion is the sparse l grid of the C_l stage, not the
preset: `sparse_l_grid` steps by 20 to l = 500 and by 50 from 550, and
splining CLASS's own spectra on the padded grid gives EE 1.13% at l = 526 and
TT 0.16% at l = 527 from the grid alone (CHANGELOG, Oct 12, 2026, follow-ups).
With the default 0.06 eV neutrino against the CLASS default (HyRec) fiducial
reference: TT 0.20 / 0.23 / 0.26%, EE 0.52 / 0.49 / 0.75%, TE 0.36%. These tests pin those numbers with margin and
guard the two failure classes the sweep found: a biased Bessel table (-1% at
l < 50, +20% in EE at l = 2) and a solver whose dense output corrupts the late
ISW (rodas5: -4% to -46% at l = 2).

Multi-cosmology rule: the fiducial massless-RECFAST comparison is the tight
gate; the LCDM grid (massive 0.06 eV neutrino, CLASS default HyRec
recombination) runs with looser bars that include the RECFAST-vs-HyRec and
ncdm-fluid systematics. fit_cl has the same grid test, one on the massive-neutrino
grid and a hierarchy-truncation guard (Oct 20, 2026, with the 3 q-bin hierarchy
of #57; bars from jobs 27173 / 27187, docs/fitcl_qgrid_2026-10-20.md). All tests
are GPU-scale and marked slow.
"""
import dataclasses
import os

import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.harmonic import compute_cls_all_fast

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_data")


def _cls(params, prec, l_max=None):
    """C_l from the preset, computed to the preset's hr_l_max unless an l_max
    is passed: the preset is validated as shipped (its own l range, table rows
    and n_k_fine), with the bands read well inside that range. The sparse-l
    spline's end condition, which once put ~2% into the last ~50 multipoles
    below the requested l_max, is padded out of the returned range by
    harmonic.L_SPLINE_PAD."""
    l_max = prec.hr_l_max if l_max is None else l_max
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    cls = compute_cls_all_fast(pt, params, bg, l_max=l_max, n_k_fine=prec.hr_n_k_fine)
    return {k: np.asarray(cls[k]) for k in ("tt", "ee", "te")}


def _band_max(num, ref, lo, hi):
    """max |num/ref - 1| over lo <= l <= hi; ref may extend beyond len(num)."""
    n = len(num)
    ref = np.asarray(ref)[:n]
    ell = np.arange(n)
    m = (ell >= lo) & (ell <= hi)
    return float(np.max(np.abs(num[m] / ref[m] - 1.0)))


def _band_rms(num, ref, lo, hi):
    """rms of num/ref - 1 over lo <= l <= hi."""
    n = len(num)
    ref = np.asarray(ref)[:n]
    ell = np.arange(n)
    m = (ell >= lo) & (ell <= hi)
    return float(np.sqrt(np.mean((num[m] / ref[m] - 1.0) ** 2)))


def _te_norm_max(cls, ref, lo, hi):
    """max |(TE - TE_ref)| / sqrt(TT_ref EE_ref) over lo <= l <= hi."""
    n = len(cls["te"])
    ell = np.arange(n)
    m = (ell >= lo) & (ell <= hi)
    tt, ee, te = (np.asarray(ref[k])[:n] for k in ("tt", "ee", "te"))
    d = (cls["te"][m] - te[m]) / np.sqrt(tt[m] * ee[m])
    return float(np.max(np.abs(d)))


def _report(label, items):
    # concise: one line per spectrum/band
    print(label + ": " + ", ".join(f"{k} {100 * v:.3f}%" for k, v in items.items()))


@pytest.mark.slow
class TestPlanckClPreset:
    """planck_cl must hold the sweep's measured accuracy with margin."""

    def test_fiducial_massless_recfast(self):
        ref = dict(np.load(os.path.join(REFERENCE_DIR, "cls_massless_recfast.npz")))
        params = CosmoParams(m_ncdm=0.0)
        prec = dataclasses.replace(PrecisionParams.planck_cl(), ncdm_q_size=0)
        cls = _cls(params, prec)
        got = {
            "TT 2-30": _band_max(cls["tt"], ref["tt"], 2, 30),
            "TT 2-500": _band_max(cls["tt"], ref["tt"], 2, 500),
            "TT 2-1000": _band_max(cls["tt"], ref["tt"], 2, 1000),
            "EE 2-4": _band_max(cls["ee"], ref["ee"], 2, 4),
            "EE 5-500": _band_max(cls["ee"], ref["ee"], 5, 500),
            "EE 5-1000": _band_max(cls["ee"], ref["ee"], 5, 1000),
            "TEn 2-1000": _te_norm_max(cls, ref, 2, 1000),
        }
        _report("planck_cl vs CLASS massless RECFAST", got)
        # measured 0.091 / 0.220 / 0.254 / 0.60 / 0.48 / 0.98 / 0.37 percent (P1); bars ~1.5x
        # (job 27000, after #53: 0.086 / 0.238 / 0.271 / 0.596 / 0.455 / 0.981 / 0.327;
        # job 24668, before #53: 0.086 / 0.233 / 0.262 / 0.60 / 0.46 / 0.98 / 0.33;
        # the 0.98 is the l ~ 526 sparse-l-grid point, see the module docstring)
        assert got["TT 2-30"] < 0.0015
        assert got["TT 2-500"] < 0.0035
        # With the CLASS radiation-streaming trigger (this branch) TT 2-1000 is
        # 0.370% after #53 (job 27001; 0.36% before): modes with k = 0.12-0.15/Mpc
        # also feed l ~ 1000 through the Bessel tail. Bar 1.5x that.
        assert got["TT 2-1000"] < 0.0055
        assert got["EE 2-4"] < 0.010
        assert got["EE 5-500"] < 0.0075
        assert got["EE 5-1000"] < 0.015
        assert got["TEn 2-1000"] < 0.006

    def test_fiducial_massless_recfast_high_l(self):
        """The damping tail, l = 1000-2500 (docs/highl_residual_2026-10-12.md). Before the RSA-trigger fix the preset
        read TT rms 1.7% / 2.0% and EE 3.0% / 8.2% over l = 1000-2000 /
        2000-2500 (the photon quadrupole source was zeroed at tau ~ 296 Mpc for
        k > 0.15/Mpc); with CLASS's tau_free_streaming criterion, 60 k per
        decade: TT 0.38% / 0.49%, EE 0.39% / 0.52% before #53; after #53 (job
        27001) TT 0.407% / 0.544%, EE 0.405% / 0.599%, TT max 0.686% (bars
        1.5-1.7x). Computed to
        the preset's hr_l_max (2500) with the sparse-l spline padded past it
        (L_SPLINE_PAD), so the single-l maximum covers the whole band. What
        remains is a smooth +0.3% to +0.5% from the helium-era opacity
        (k_D +0.12%).

        Fiducial massless-RECFAST only: it is the one RECFAST-matched C_l
        reference; the grid cls.npz are HyRec + BBN Y_He and would need looser
        bars above l = 1000 from the recombination-code offset. The trigger's
        multi-cosmology coverage is tests/test_sources_vs_class.py::test_lcdm_grid
        and TestThermoScalars::test_tau_free_streaming_grid."""
        ref = dict(np.load(os.path.join(REFERENCE_DIR, "cls_massless_recfast.npz")))
        params = CosmoParams(m_ncdm=0.0)
        prec = dataclasses.replace(PrecisionParams.planck_cl(), ncdm_q_size=0)
        cls = _cls(params, prec)
        got = {
            "TT rms 1000-2000": _band_rms(cls["tt"], ref["tt"], 1000, 2000),
            "TT rms 2000-2500": _band_rms(cls["tt"], ref["tt"], 2000, 2500),
            "EE rms 1000-2000": _band_rms(cls["ee"], ref["ee"], 1000, 2000),
            "EE rms 2000-2500": _band_rms(cls["ee"], ref["ee"], 2000, 2500),
            "TT max 1000-2500": _band_max(cls["tt"], ref["tt"], 1000, 2500),
        }
        _report("planck_cl vs CLASS massless RECFAST, high l", got)
        assert got["TT rms 1000-2000"] < 0.006
        assert got["TT rms 2000-2500"] < 0.008
        assert got["EE rms 1000-2000"] < 0.007
        assert got["EE rms 2000-2500"] < 0.010
        assert got["TT max 1000-2500"] < 0.010

    def test_lcdm_grid_massive(self, lcdm_cosmology):
        """Grid points use the default massive neutrino and CLASS's default
        (HyRec) reference, so the bars include the recombination and ncdm
        systematics (measured ~0.3% at l <= 1000 at fiducial)."""
        name, params = lcdm_cosmology
        ref_dir = os.path.join(REFERENCE_DIR, name)
        if not os.path.isdir(ref_dir):
            pytest.skip(f"no CLASS reference for {name}")
        ref = dict(np.load(os.path.join(ref_dir, "cls.npz")))
        cls = _cls(params, PrecisionParams.planck_cl())
        got = {
            "TT 2-1000": _band_max(cls["tt"], ref["tt"], 2, 1000),
            "EE 5-1000": _band_max(cls["ee"], ref["ee"], 5, 1000),
            "TEn 2-1000": _te_norm_max(cls, ref, 2, 1000),
        }
        _report(f"planck_cl vs CLASS ({name}, massive, HyRec)", got)
        # fiducial measured 0.26 / 0.75 / 0.36 percent (P1m). Over the five grid
        # points after #53 (job 27000): TT 0.228-0.424%, EE 0.755-1.038%, TEn
        # 0.297-0.417% (bars 1.9x / 1.45x / 1.9x); before #53
        # (jobs 24639 / 24668): TT <= 0.43% (omega_b_high),
        # EE <= 1.03% (h_high), TE <= 0.44% (omega_b_high, omega_cdm_low); the
        # bars 0.8 / 1.5 / 0.8% are 1.9x / 1.45x / 1.8x those maxima.
        assert got["TT 2-1000"] < 0.008
        assert got["EE 5-1000"] < 0.015
        assert got["TEn 2-1000"] < 0.008


@pytest.mark.slow
class TestFitClPreset:
    """fit_cl is the HMC preset: coarse, but its low-l TT must not be corrupted
    by the solver's dense output (rodas5 gave -46% at l = 2 at these tolerances)
    or by its neutrino treatment (ncdm_q_size 0 gave -15% at l = 2, #57).

    Bars for the two grids below: TT 2-30 < 0.9%, TT 2-500 < 1.1%, EE 5-500
    < 2.5%, TE/sqrt(TT EE) 2-500 < 1.7%, set at 1.7x / 1.6x / 1.5x / 1.5x the
    maxima measured Oct 22, 2026 with the shipped 5 q-bins / l_max_ncdm 8 /
    no fluid approximation over the LCDM grid (job 27746) and the 0.15 and
    0.30 eV points (job 27711); the sweep is in docs/fitcl_qgrid_2026-10-20.md."""

    BARS = {"TT 2-30": 0.009, "TT 2-500": 0.011, "EE 5-500": 0.025, "TEn 2-500": 0.017}

    def test_low_l_within_fit_budget(self):
        ref = dict(np.load(os.path.join(REFERENCE_DIR, "cls_massless_recfast.npz")))
        params = CosmoParams(m_ncdm=0.0)
        prec = PrecisionParams.fit_cl()
        cls = _cls(params, prec)
        got = {
            "TT 2-30": _band_max(cls["tt"], ref["tt"], 2, 30),
            "TT 2-500": _band_max(cls["tt"], ref["tt"], 2, 500),
            "EE 5-500": _band_max(cls["ee"], ref["ee"], 5, 500),
        }
        _report("fit_cl vs CLASS massless RECFAST", got)
        # measured 0.482 / 0.482 / 1.208 percent with the shipped hierarchy (job 27746;
        # 0.461 / 0.461 / 1.194 with ncdm_q_size 0 after #53, job 27000); bars 3.1x / 4.1x / 1.7x
        assert got["TT 2-30"] < 0.015
        assert got["TT 2-500"] < 0.02
        assert got["EE 5-500"] < 0.02

    def _grid_point(self, name, params):
        ref_dir = os.path.join(REFERENCE_DIR, name)
        if not os.path.isdir(ref_dir):
            pytest.skip(f"no CLASS reference for {name}")
        ref = dict(np.load(os.path.join(ref_dir, "cls.npz")))
        cls = _cls(params, PrecisionParams.fit_cl())
        got = {
            "TT 2-30": _band_max(cls["tt"], ref["tt"], 2, 30),
            "TT 2-500": _band_max(cls["tt"], ref["tt"], 2, 500),
            "EE 5-500": _band_max(cls["ee"], ref["ee"], 5, 500),
            "TEn 2-500": _te_norm_max(cls, ref, 2, 500),
        }
        _report(f"fit_cl vs CLASS ({name}, massive, HyRec)", got)
        for key, bar in self.BARS.items():
            assert got[key] < bar, f"{key} {100 * got[key]:.3f}% >= {100 * bar:.2f}%"

    def test_lcdm_grid_massive(self, lcdm_cosmology):
        """fit_cl on the LCDM grid against CLASS's default references (HyRec,
        one 0.06 eV neutrino): the HMC preset is evaluated away from fiducial by
        construction, so it must be measured there too. Measured Oct 22, 2026
        with the shipped 5 q-bins / l_max_ncdm 8 / no fluid approximation (job
        27746, max |C_l / C_CLASS - 1| over fiducial / h_high / omega_b_high /
        omega_cdm_low / ns_high): TT 2-30 0.47-0.54%, TT 2-500 0.56-0.69%,
        EE 5-500 1.20-1.65%, TEn 2-500 0.87-1.13% (maxima at omega_b_high and
        omega_cdm_low); planck_cl on the same grid
        reads 0.23-0.42% / 0.76-1.04% / 0.30-0.42% to l = 1000. With ncdm_q_size 0
        (job 27008) the same bands read 14.6-15.4% / the same / 3.1-4.1% /
        16.8-19.4% (#57)."""
        name, params = lcdm_cosmology
        self._grid_point(name, params)

    def test_nulcdm_grid_massive(self, nulcdm_cosmology):
        """ncdm-sector rule: the massive-neutrino grid against CLASS references
        (fiducial 0.06 eV; massive_nu_015; massive_nu_030, generated Oct 20,
        2026; massive_nu_006 is the fiducial and skips). Measured Oct 22, 2026
        (job 27711, 5 q-bins / l_max_ncdm 8 / no fluid approximation): 0.15 eV
        TT 2-30 0.48%, TT 2-500 0.52%, EE 5-500 1.18%, TEn 2-500 0.87%; 0.30 eV
        0.52 / 0.64 / 1.16 / 0.86%; fiducial 0.51 / 0.60 / 1.20 / 0.87%; and
        0.53 / 0.63 / 1.19 / 0.85% at 0.50 eV, outside this grid. With the
        "class" fluid approximation the same hierarchy reads TT 0.71% at
        0.15 eV, 0.98% at 0.30 eV and 1.29% at 0.50 eV (its implementation,
        #46, biases the late ISW), which is why the preset runs without it."""
        name, params = nulcdm_cosmology
        self._grid_point(name, params)

    def test_lmax_ncdm_truncation(self, nulcdm_cosmology):
        """fit_cl truncates the massive-neutrino hierarchy at pt_l_max_ncdm 8
        (CLASS 17). Consistency guard at every massive-neutrino grid point, no
        reference needed: against the same preset at l_max_ncdm 17 the C_l
        differed by <= 0.011% in TT and EE to l = 500 at every measured point
        with the fluid approximation on (LCDM grid, massless, 0.15 and 0.30 eV;
        <= 0.083% to l = 1000; jobs 27173 / 27187, Oct 20, 2026); with it off,
        as shipped, TT <= 0.007% and EE <= 0.010% at fiducial, 0.15 and 0.30 eV
        (job 27766, Oct 22, 2026). Bar 0.1%. In linear P_m the same truncation is +0.9% at
        k = 1 Mpc^-1 (the species' relativistic-era free streaming; not a C_l
        matter, see the preset docstring)."""
        name, params = nulcdm_cosmology
        prec = PrecisionParams.fit_cl()
        a = _cls(params, prec)
        b = _cls(params, dataclasses.replace(prec, pt_l_max_ncdm=17))
        got = {
            "TT 2-500": _band_max(a["tt"], b["tt"], 2, 500),
            "EE 5-500": _band_max(a["ee"], b["ee"], 5, 500),
        }
        _report(f"fit_cl l_max_ncdm 8 vs 17 ({name})", got)
        assert got["TT 2-500"] < 0.001
        assert got["EE 5-500"] < 0.001
