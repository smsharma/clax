"""S1 (smsharma/clax#48): thermodynamics tables and their forward tangents, and
the opacity tables' tangents (smsharma/clax#55).
Primary oracle: CLASS RECFAST recfast_Heswitch=1 (reference_data/<name>/
thermo_recfast_hesw1.npz). Bars are the code-accuracy gates set for #48.
Multi-cosmology: recomb_cosmology (11 points; fiducial only under --fast)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve
from clax.interpolation import CubicSpline
from clax.thermodynamics import thermodynamics_solve
from tests.recomb_test_utils import (QS, QS_OPACITY, THERMO_GRIDS, THETAS, as_traced, load_drag, load_ref, spike_ratio, table_on_ref_grid, tables, tables_opacity, tangent_tables, thermo_prec, xe_on_ref_grid, z_of)

_CACHE = {}
# Under --fast the 3000-point (fit_cl) grid runs at fiducial; the default 20000-point
# grid and the FD tests (a jvp compile per cosmology and theta) are marked slow.
GRIDS = [pytest.param(n, marks=pytest.mark.slow, id=str(n)) if n > 3000 else n
         for n in THERMO_GRIDS]


_TH_CACHE = {}


def _thermo(name, p, n):
    """The ThermoResult itself (for its derived scalars), cached per (name, n)."""
    key = (name, n)
    if key not in _TH_CACHE:
        prec = thermo_prec(n)
        pt = as_traced(p)
        _TH_CACHE[key] = thermodynamics_solve(pt, prec, background_solve(pt, prec))
    return _TH_CACHE[key]


def _prim_tan(name, p, n, theta="h"):
    key = (name, n, theta)
    if key not in _CACHE:
        _CACHE[key] = tangent_tables(p, thermo_prec(n), theta)
    return _CACHE[key]


def _fd_check(recomb_cosmology, theta, zs, floor=1e-8, qs=QS, tables_fn=tables, b_bar=1e-6):
    """Forward tangent vs central finite differences of the tables qs (built by
    tables_fn), at the FIXED sample redshifts zs (smsharma/clax#48; the opacity
    tables of #55 go through the same check, some with their own floor and b_bar).

    The jvp is the tangent of the spline-evaluated values at ln a = log(1/(1+z)):
    the thermodynamics ln a knots move with theta, so the per-knot tangent table
    (`_prim_tan`) is d/dtheta at fixed knot index, not at fixed a, and is not what
    FD measures. f1, f3 are the central FDs at relative steps 1e-3 and 3e-3; Q is
    the primal value at the sample. Each (z, Q) point is in exactly one class:
      A (resolved):  |f1/Q| > floor, |f3/Q| > floor and |f1 - f3| <= 1e-4 |f1|;
                     assert |jvp/f1 - 1| <= 1e-3.
      B (at the table's rounding floor): |f1/Q| <= floor and |f3/Q| <= floor, i.e.
                     the true derivative vanishes to rounding; assert |jvp/Q| <= b_bar,
                     so a tangent spike where the derivative vanishes still fails.
      C: everything else, skipped and counted in the printout.
    """
    name, p = recomb_cosmology
    prec = thermo_prec(20000)
    th0 = float(getattr(p, theta))
    loga_s = jnp.log(1.0 / (1.0 + jnp.asarray(zs)))

    def sampled(x):
        tab = tables_fn(replace(as_traced(p), **{theta: x}), prec)
        return {q: CubicSpline(tab["loga"], tab[q]).evaluate(loga_s) for q in qs}

    prim, tan = jax.jit(lambda x: jax.jvp(sampled, (x,), (jnp.ones_like(x),)))(jnp.asarray(th0, jnp.float64))
    g = jax.jit(sampled)
    fds = []
    for rel_step in (1e-3, 3e-3):
        hstep = rel_step * abs(th0)
        fp = g(jnp.asarray(th0 + hstep, jnp.float64))
        fm = g(jnp.asarray(th0 - hstep, jnp.float64))
        fds.append({q: (np.asarray(fp[q]) - np.asarray(fm[q])) / (2 * hstep) for q in qs})
    n_a = n_b = n_c = 0
    worst_a, worst_b = 0.0, 0.0
    loc_b = None
    for q in qs:
        Q, jv = np.asarray(prim[q]), np.asarray(tan[q])
        f1, f3 = fds[0][q], fds[1][q]
        # Reject non-finite values first: Python max() drops NaN, so a NaN tangent would pass.
        assert np.all(np.isfinite(jv)) and np.all(np.isfinite(Q)), \
            f"ERROR non-finite {q} tangent or value at the sample z ({name}, {theta})"
        small1, small3 = np.abs(f1 / Q) <= floor, np.abs(f3 / Q) <= floor
        cls_a = ~small1 & ~small3 & (np.abs(f1 - f3) <= 1e-4 * np.abs(f1))
        cls_b = small1 & small3
        n_a += int(cls_a.sum())
        n_b += int(cls_b.sum())
        n_c += int((~cls_a & ~cls_b).sum())
        if cls_a.any():
            worst_a = max(worst_a, float(np.max(np.abs(jv[cls_a] / f1[cls_a] - 1.0))))
        for i in np.flatnonzero(cls_b):
            r = abs(jv[i] / Q[i])
            if r > worst_b:
                worst_b, loc_b = float(r), (q, float(zs[i]))
    print(f"{name} d/d{theta}: A {n_a} B {n_b} C {n_c} of {len(zs) * len(qs)}, "
          f"worst A |jvp/FD-1| {worst_a:.3e}, worst B |jvp/Q| {worst_b:.3e}")
    # Minimum resolved points: a third of the sample must be in class A, except
    # theta = h. At fixed z >= 400 with omega_b and omega_cdm fixed, h enters the
    # tables only through Lambda (~1e-9 of H), so d ln Q/dh is at the rounding
    # floor, class B, and the h tangent is checked there instead. A third of the
    # sample must still be classified (A or B), so an h test cannot pass with
    # every point skipped in class C.
    n_min = len(zs) * len(qs) // 3
    min_a = 0 if theta == "h" else n_min
    assert n_a >= min_a, f"ERROR resolved FD points {n_a} < {min_a} ({name}, {theta})"
    assert n_a + n_b >= n_min, f"ERROR classified FD points {n_a + n_b} < {n_min} ({name}, {theta})"
    assert worst_a <= 1e-3, f"ERROR tangent vs FD {worst_a:.3e} > 1e-3 ({name}, {theta})"
    assert worst_b <= b_bar, (f"ERROR tangent where the derivative vanishes: |jvp/Q| {worst_b:.3e} > {b_bar:.0e} "
                              f"at {loc_b} ({name}, {theta})")


@pytest.mark.parametrize("n", GRIDS)
def test_xe_helium_window_matches_class(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= 2870.0) & (ref["z"] <= 5000.0)
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    print(f"{name} n={n}: x_e vs CLASS z 2870-5000 max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f}")
    assert rel[i] <= 1e-3, f"ERROR x_e helium window {rel[i]:.3e} > 1e-3 ({name}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("theta", THETAS)
def test_no_derivative_spikes_helium_window(recomb_cosmology, n, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, theta)
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, {theta}, n={n})"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 1800.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/d{theta}: worst helium-window spike ratio {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR spike ratio {worst[0]:.3g} ({name}, {theta}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
def test_values_smooth_helium_window(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    z = z_of(prim["loga"])
    sel = (z >= 1800.0) & (z <= 6000.0)
    d2 = np.abs(np.diff(prim["xe"], 2))[sel[1:-1]]
    bar = 3e-5 * (20000.0 / n) ** 2      # second differences scale as spacing^2
    print(f"{name} n={n}: max |d2 x_e| over z 1800-6000 = {d2.max():.3e} (bar {bar:.2e})")
    assert d2.max() < bar, f"ERROR x_e not smooth: {d2.max():.3e} ({name}, n={n})"


ZS_HE = np.array([5500., 4500., 3800., 3400., 3200., 3000., 2800., 2600., 2400., 2200., 2000.])


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_tangents_match_converged_fd_helium(recomb_cosmology, theta):
    _fd_check(recomb_cosmology, theta, ZS_HE)


# ---- Hydrogen: CLASS staging, Saha -> Peebles ODE at z = 1600 ----

@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("window", [(800.0, 1650.0), (1650.0, 2870.0)])
def test_xe_hydrogen_windows_match_class(recomb_cosmology, n, window):
    # (800, 1650) is the hydrogen stage with no gap: the f2 blend (z 1500-1600) and
    # the Saha -> ODE hand-over included; the tighter per-window bars are in
    # test_hydrogen_staging_guard_xe_windows. (1650, 2870) is the helium ODE's x_e;
    # helium staging alone could not pass it, because its error sits at the
    # z ~ 1650 hydrogen edge, which only hydrogen staging fixes.
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= window[0]) & (ref["z"] <= window[1])
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    print(f"{name} n={n}: x_e vs CLASS z {window} max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f}")
    assert rel[i] <= 1e-3, f"ERROR x_e {window} {rel[i]:.3e} > 1e-3 ({name}, n={n})"


# ---- Below z = 800: the freeze-out tail and the Compton-coupling era of T_b ----
# x_e and T_b against CLASS RECFAST where the T_b update's former Taylor switch
# (f_e < 1e-3, z ~ 620) sat. That branch was a first-order expansion in
# x = thomc * da used where x was not small, an error that grew with the grid step:
# T_b was 2.7e-4 (20000 points) / 1.8e-3 (3000) from CLASS at z 300-700 and x_e
# 1.1e-4 / 8.7e-4 at z 100-500. With the exact expm1 form the bars below are twice
# the worst over the 11 cosmologies (job 27897, Oct 22, 2026; the worst is
# massive_nu_015 everywhere), ceiled to two figures; the old code misses every one
# of them by 2x to 12x (job 27896). Above z = 700 nothing changed.
LOW_Z_CLASS_BARS = {
    # (z_lo, z_hi): {n: (x_e bar, T_b bar)}
    (100.0, 300.0): {20000: (3.6e-5, 1.3e-4), 3000: (7.9e-5, 3.4e-4)},
    (300.0, 500.0): {20000: (3.9e-5, 6.4e-5), 3000: (7.1e-5, 2.2e-4)},
    (500.0, 700.0): {20000: (3.9e-5, 2.2e-5), 3000: (1.5e-4, 1.1e-4)},
}


@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("window", list(LOW_Z_CLASS_BARS), ids=lambda w: f"z{w[0]:g}-{w[1]:g}")
def test_xe_tb_low_z_match_class(recomb_cosmology, n, window):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= window[0]) & (ref["z"] <= window[1])
    bar_xe, bar_tb = LOW_Z_CLASS_BARS[window][n]
    failures = []
    for q, rq, bar in (("xe", "x_e", bar_xe), ("tb", "Tb", bar_tb)):
        rel = np.abs(table_on_ref_grid(prim, q, ref["z"][sel]) / ref[rq][sel] - 1.0)
        i = int(np.argmax(rel))
        print(f"{name} n={n}: {q} vs CLASS z {window} max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f} (bar {bar:.1e})")
        if rel[i] > bar:
            failures.append(f"ERROR {q} {window} {rel[i]:.3e} > {bar:.1e} ({name}, n={n})")
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("n", GRIDS)
def test_visibility_peak_matches_class(recomb_cosmology, n):
    from tests.recomb_test_utils import visibility_peak
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    dz = abs(visibility_peak(z_of(prim["loga"]), prim["g"]) - visibility_peak(ref["z"], ref["g"]))
    bar = 0.01 if n == 20000 else 0.03          # per-grid bars
    print(f"{name} n={n}: visibility-peak |dz| = {dz:.4f} (bar {bar})")
    assert dz < bar, f"ERROR visibility peak shift {dz:.4f} ({name}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("theta", THETAS)
def test_no_derivative_spikes_full_range(recomb_cosmology, n, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, theta)
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, {theta}, n={n})"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 300.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/d{theta}: worst spike ratio z 300-6000 {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR spike ratio {worst[0]:.3g} ({name}, {theta}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
def test_values_smooth_full_range(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    z = z_of(prim["loga"])
    sel = (z >= 1500.0) & (z <= 6000.0)
    d2 = np.abs(np.diff(prim["xe"], 2))[sel[1:-1]]
    bar = 3e-5 * (20000.0 / n) ** 2
    print(f"{name} n={n}: max |d2 x_e| over z 1500-6000 = {d2.max():.3e} (bar {bar:.2e})")
    assert d2.max() < bar, f"ERROR x_e not smooth: {d2.max():.3e} ({name}, n={n})"


# 1600 and 1550: the hydrogen Saha -> ODE hand-over and the middle of its f2 blend.
ZS_H = np.array([1800., 1600., 1550., 1300., 1100., 800., 400.])


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_tangents_match_converged_fd_hydrogen(recomb_cosmology, theta):
    _fd_check(recomb_cosmology, theta, ZS_H)


# ---- Derived epochs: the baryon drag epoch (smsharma/clax#44) and the visibility peak ----
# Oracle: CLASS RECFAST (Heswitch = 1) with the same setup as the x_e references,
# drag_recfast_hesw1.json (scripts/generate_class_reference.py --recfast-hesw1-drag).
# z_d is where kappa_b = 1, dkappa_b/dtau = kappa'/R with R = 3 rho_b / 4 rho_gamma,
# found by linear interpolation on the table as thermodynamics.c does; tau_d and
# rs_d are the background's conformal time and sound horizon there. z_rec is CLASS's
# visibility maximum by a three-point parabola on its table, which clax calls
# z_star; tau_rec and rs_rec likewise. Bars are twice the worst over the 11
# cosmologies on the fixed code, ceiled to two figures (job 28139, Oct 23, 2026:
# z_d 1.5e-6 / 3.6e-5, rs_d 8.6e-7 / 2.4e-5, z_star 1.1e-6 / 2.9e-5 on 20000 /
# 3000 points). The Aubourg et al. fit this replaces was 1.3e-4 off at fiducial
# and 1.5e-3 at omega_b +20%; the grid-point z_star was up to 1.8e-4 / 1.1e-3 off.
DRAG_EPOCH_BARS = {
    20000: {"z_d": 3.1e-6, "tau_d": 2.0e-6, "rs_d": 1.8e-6},
    3000: {"z_d": 7.3e-5, "tau_d": 5.4e-5, "rs_d": 4.8e-5},
}
VISIBILITY_PEAK_BARS = {
    20000: {"z_star": 2.2e-6, "tau_star": 1.4e-6, "rs_star": 1.2e-6},
    3000: {"z_star": 5.8e-5, "tau_star": 4.3e-5, "rs_star": 3.7e-5},
}


@pytest.mark.parametrize("n", GRIDS)
def test_drag_epoch_matches_class(recomb_cosmology, n):
    name, p = recomb_cosmology
    th = _thermo(name, p, n)
    ref = load_drag(name)
    failures = []
    for q in ("z_d", "tau_d", "rs_d"):
        rel = abs(float(getattr(th, q)) / ref[q] - 1.0)
        bar = DRAG_EPOCH_BARS[n][q]
        print(f"{name} n={n}: {q} {float(getattr(th, q)):.6f} vs CLASS {ref[q]:.6f} rel {rel:.3e} (bar {bar:.0e})")
        if rel > bar:
            failures.append(f"ERROR {q} {rel:.3e} > {bar:.0e} ({name}, n={n})")
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("n", GRIDS)
def test_visibility_peak_scalars_match_class(recomb_cosmology, n):
    """z_star, tau_star, rs_star against CLASS's parabolic visibility peak. The
    grid-point values the code took until #44 were off by up to one thermodynamics
    step (5e-4 in z on 20000 points, 3e-3 on 3000) and piecewise constant in the
    parameters; tau_star sets the perturbation save grid, so that was a primal jump."""
    name, p = recomb_cosmology
    th = _thermo(name, p, n)
    ref = load_drag(name)
    failures = []
    for q, rq in (("z_star", "z_rec"), ("tau_star", "tau_rec"), ("rs_star", "rs_rec")):
        rel = abs(float(getattr(th, q)) / ref[rq] - 1.0)
        bar = VISIBILITY_PEAK_BARS[n][q]
        print(f"{name} n={n}: {q} {float(getattr(th, q)):.6f} vs CLASS {rq} {ref[rq]:.6f} rel {rel:.3e} (bar {bar:.0e})")
        if rel > bar:
            failures.append(f"ERROR {q} {rel:.3e} > {bar:.0e} ({name}, n={n})")
    assert not failures, "; ".join(failures)


# ---- Opacity tables: kappa_dot, exp(-kappa), g, dkappa_dot/dlna, g' (smsharma/clax#55) ----
# The tables perturbations.py reads. Until #55 they carried a stop_gradient with an
# n_H_0 rescale (d127731): along omega_cdm and N_ur their tangents were exactly zero at
# every z, and along omega_b and Y_He only the n_H_0 path survived, which is what put
# the P(k) derivatives 4% (omega_b) and +0.66% (omega_cdm) off converged finite
# differences. The samples span the fully ionised plasma, recombination, freeze-out
# and reionisation; the check is the one x_e, c_s^2 and T_b pass above.
# 1650 and 1550 straddle the hydrogen hand-over at z = 1600 (ZS_H samples it): the
# derivative tables dkappa_dot/dlna and g' have a kink at each end of the smoothstep
# window, where a finite difference measures the grid's position against the kink.
ZS_OP = np.array([3000., 2000., 1650., 1550., 1400., 1200., 1100., 1000., 900., 800., 600.,
                  400., 200., 100., 30., 10., 7., 3., 1., 0.3, 0.])
# omega_b: every sample. With the T_b update's former Taylor switch (f_e < 1e-3,
# z ~ 620) the primal's omega_b dependence was a dense staircase below the switch,
# one step per knot it crossed, and a finite difference measured the staircase's
# mean slope: 1e-3 of kappa_dot and g, 1.1e-2 of exp(-kappa) at its zero crossing
# near z = 400 at omega_b + 20% (ADR 0004), which kept this gate at z >= 900.
ZS_OP_OMEGA_B = ZS_OP
# h at fixed omega_b, omega_cdm: the tables respond at z <= 10 (Lambda in H(z) and the
# reionisation epoch), where AD and FD agree to 1e-6. Above z = 1400 exp(-kappa) is
# below 1e-5 and its h tangent is the trapezoid sum's sensitivity to knot phase
# (1e-5 to 1e-3 of the value, step-unstable in FD too), so those samples are dropped;
# between z = 30 and 600 the derivatives are 1e-7 to 1e-5 of the value and sit under
# the staircase that the former T_b switch made (now gone), so the floor is 1e-5;
# class B then asserts |jvp/Q| <= 1.9e-5, twice the worst measured over the 11
# cosmologies (9.5e-6 at omega_cdm - 20%, job 27941, Oct 23, 2026; 6.7e-6 with the
# switch, job 27184).
ZS_OP_H = ZS_OP[ZS_OP <= 1200.0]


# The derivative tables dkappa_dot/dlna and g' are spline derivatives of cumulative
# sums and carry rounding at the 1e-10 level of their value, so a derivative below
# 1e-6 of the value per unit theta cannot be resolved to 1e-3 by a 1e-3 step: the
# floor is 1e-6 for the opacity tables (1e-8 is the x_e tables'), and class B asserts
# |jvp/Q| <= OPACITY_B_BAR. With the x_e floor one such point, dkappa_dot/dlna at
# z = 1 along omega_cdm at N_eff = 2 (1.5e-7 of the value), landed in class A at
# 2.8e-3 once the T_b staircase no longer unsettled its finite difference. The bar
# is twice the worst class-B value over the 11 cosmologies and four parameters
# (9.5e-7, Y_He at omega_b - 20%, job 27941, Oct 23, 2026); worst resolved errors in
# that run: 1.2e-5 (omega_b, at every redshift), 1.7e-4 (omega_cdm), 6.8e-5 (Y_He),
# 2.0e-5 (N_ur), 1.8e-4 (h).
OPACITY_B_BAR = 2.0e-6


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_opacity_tangents_match_converged_fd(recomb_cosmology, theta):
    if theta == "h":
        _fd_check(recomb_cosmology, theta, ZS_OP_H, floor=1e-5, qs=QS_OPACITY,
                  tables_fn=tables_opacity, b_bar=1.9e-5)
    elif theta == "omega_b":
        _fd_check(recomb_cosmology, theta, ZS_OP_OMEGA_B, floor=1e-6, qs=QS_OPACITY,
                  tables_fn=tables_opacity, b_bar=OPACITY_B_BAR)
    else:
        _fd_check(recomb_cosmology, theta, ZS_OP, floor=1e-6, qs=QS_OPACITY,
                  tables_fn=tables_opacity, b_bar=OPACITY_B_BAR)


# ---- Robustness checks: bounds, T_cmb, coarser grid, jit vs eager, vmap ----

# Also on the 1500-point grid of the coarse-grid spike test and PREC_FAST's 1200
# points, where |h J| reaches ~4-5 and an overshoot of the unclipped step would show
# first (the unstaged L-stable variant drove x_e to -988, ADR 0003). The spike window
# runs down to z = 300, through the T_b update's former Taylor switch at f_e < 1e-3
# (z ~ 620), which on this grid put a spike of ratio 20 in the T_b tangent.
@pytest.mark.parametrize("n", GRIDS + [1500, 1200])
def test_fractions_stay_physical_without_clips(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    xe = np.asarray(prim["xe"])
    z = z_of(prim["loga"])
    y = float(p.Y_He)
    upper = 1.0 + 2.0 * y / (3.9715 * (1.0 - y))     # >= the exactly-4 value too
    print(f"{name} n={n}: x_e in [{xe.min():.3e}, {xe.max():.6f}], upper {upper:.6f}")
    assert xe.min() >= 0.0 and xe.max() <= upper + 1e-12, f"ERROR x_e out of bounds ({name}, n={n})"
    # For 2400 <= z <= 2870 hydrogen is fully ionised (hydrogen Saha at the
    # fiducial cosmology: 1 - x_HII ~ 2e-9 at z = 2870, ~ 7e-8 at z = 2400)
    # and HeIII is negligible, so x_e <= 1 + f_He bounds the helium fraction
    # x_HeII <= 1. ThermoResult does not expose x_HeII; the clips are gone, so
    # this is the guard against the ROS2 step overshooting. Its margin is
    # 3.6e-4 to 5.2e-4 (x_HeII ~ 0.995 at its maximum in this window, not 1), so
    # it catches an x_HeII overshoot only above about 5e-3.
    he = (z >= 2400.0) & (z <= 2870.0)
    he_upper = 1.0 + y / (3.9715 * (1.0 - y)) + 1e-10
    j = int(np.argmax(np.where(he, xe, -np.inf)))
    print(f"{name} n={n}: z 2400-2870 max x_e {xe[j]:.8f} at z={z[j]:.1f}, bound {he_upper:.8f}, "
          f"margin {he_upper - xe[j]:.3e}")
    assert xe[j] <= he_upper, (f"ERROR x_e {xe[j]:.10f} at z={z[j]:.1f} exceeds the helium bound "
                               f"{he_upper:.10f}, x_HeII > 1 ({name}, n={n})")


@pytest.mark.parametrize("n", GRIDS)
def test_helium_counted_once_z2870_5000(recomb_cosmology, n):
    """x_HeII + x_HeIII <= 1 between the HeIII -> HeII hand-over (z ~ 5000) and the
    HeI staging switch (z = 2870). ThermoResult does not expose the helium
    fractions. Hydrogen is fully ionised there (1 - x_HII < 1e-8), so
    (x_e - 1)/f_He = x_HeII + 2 x_HeIII, the electrons per helium nucleus, in clax
    (its own f_He) and in CLASS (_not4_); comparing those is independent of the
    helium-to-hydrogen ratio convention. The HeI Saha root once ignored the MB95
    HeIII still present below z ~ 5000: x_HeII + x_HeIII reached 1.0007 near
    z = 4970, and this difference was 5.7e-4 to 7.5e-4 at all 10 cosmologies on
    both grids. The bar, 1e-4, fails all of those and sits 6.6x above the
    corrected worst (1.5e-5). Above z = 5000 the MB95 HeIII stage is 2.0e-4 to 2.4e-4 from
    CLASS in x_e, a separate, pre-existing offset, so the window stops there."""
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= 2870.0) & (ref["z"] <= 5000.0)
    z = ref["z"][sel]
    y = float(p.Y_He)
    # Local import: the module then still collects on code without _f_He (the
    # unfixed tree), and goes red here by assertion rather than at import.
    from clax.thermodynamics import _f_He
    he_clax = (xe_on_ref_grid(prim, z) - 1.0) / float(_f_He(y))
    he_class = (ref["x_e"][sel] - 1.0) / (y / (3.9715 * (1.0 - y)))
    d = np.abs(he_clax - he_class)
    j = int(np.argmax(d))
    print(f"{name} n={n}: z 2870-5000 max |electrons per He nucleus - CLASS| = {d[j]:.3e} at z={z[j]:.0f}")
    assert d[j] < 1e-4, (f"ERROR helium counted twice: electrons per He nucleus {d[j]:.3e} "
                         f"off CLASS at z={z[j]:.0f}, bar 1e-4 ({name}, n={n})")


@pytest.mark.parametrize("n", GRIDS)
def test_no_spikes_for_T_cmb(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, "T_cmb")
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, T_cmb, n={n})"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 300.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/dT_cmb: worst spike ratio {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR T_cmb spike ratio {worst[0]:.3g} ({name}, n={n})"


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_coarse_grid_1500_helium_hydrogen_spike_free(recomb_cosmology, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, 1500, theta)
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, {theta}, n=1500)"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 300.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n=1500 d/d{theta}: worst spike ratio z 300-6000 {worst[0]:.3g}")
    assert worst[0] < 10.0, f"ERROR 1500-grid spike ratio {worst[0]:.3g} ({name}, {theta})"


@pytest.mark.slow
def test_jit_and_eager_primal_agree(recomb_cosmology):
    name, p = recomb_cosmology
    prec = thermo_prec(20000)
    pt = as_traced(p)
    # thermodynamics_solve is itself @jax.jit, so tables() always runs it compiled;
    # __wrapped__ is the uncompiled function.
    # 20000 points: the jit/eager rounding-level gap of the former RK4 code showed up on this
    # grid (measured during the design study, not in the repo); 3000 points does not amplify it.
    a = np.asarray(thermodynamics_solve.__wrapped__(pt, prec, background_solve(pt, prec)).xe_of_loga.y)
    b = np.asarray(jax.jit(lambda q: tables(q, prec))(pt)["xe"])
    d = float(np.max(np.abs(a - b)))
    print(f"{name}: max |x_e(jit) - x_e(eager)| = {d:.3e}")
    assert d < 1e-12, f"ERROR jit vs eager x_e differ by {d:.3e} ({name})"


@pytest.mark.slow
def test_vmap_over_h_matches_loop(recomb_cosmology):
    name, p = recomb_cosmology
    prec = thermo_prec(3000)
    hs = float(p.h) * jnp.array([0.98, 1.0, 1.02])
    f = jax.jit(lambda h: tables(replace(as_traced(p), h=h), prec)["xe"])
    batched = np.asarray(jax.vmap(f)(hs))
    looped = np.stack([np.asarray(f(h)) for h in hs])
    d = float(np.max(np.abs(batched - looped)))
    print(f"{name}: max |vmap - loop| = {d:.3e}")
    assert d < 1e-12, f"ERROR vmap vs loop differ by {d:.3e} ({name})"


# ---- One helium-to-hydrogen number ratio, CLASS's _not4_ ----

@pytest.mark.parametrize("n", GRIDS)
def test_fully_ionised_xe_uses_class_helium_mass_ratio(recomb_cosmology, n):
    """x_e at z = 2e4 (H, He fully ionised) = 1 + 2 f_He with f_He = Y/(3.9715 (1-Y)),
    CLASS thermodynamics.h:707 (_not4_). At z = 2e4 only the final x_e sum is
    sensitive to the helium ratio, so this guards that sum alone: it fails if the
    sum uses exactly 4 instead of 3.9715, a 1.2e-3 offset in x_e. The other
    number-ratio sites are guarded by test_helium_ratio_guard_xe_windows below, which
    lists the ones that stay unguarded. Parametrised over GRIDS: the value is
    grid-independent, so the 3000-point case runs routinely under --fast."""
    name, p = recomb_cosmology
    prec = thermo_prec(n)
    q = as_traced(p)
    th = thermodynamics_solve(q, prec, background_solve(q, prec))
    xe = float(th.xe_of_loga.evaluate(jnp.log(1.0 / 2.0001e4)))
    y = float(p.Y_He)
    expect = 1.0 + 2.0 * y / (3.9715 * (1.0 - y))
    print(f"{name} n={n}: x_e(z=2e4) = {xe:.8f}, expected {expect:.8f}, diff {xe - expect:+.3e}")
    assert abs(xe - expect) < 1e-6, f"ERROR fully ionised x_e {xe:.8f} vs {expect:.8f} ({name}, n={n})"


# Tight x_e-vs-CLASS windows that guard the helium ratio at every site they can see.
# Bar per (window, grid) = 2 x the worst |x_e/x_e,CLASS - 1| the code measures over
# the 11 recomb_cosmology points, rounded up to 2 significant figures (measured
# worst, 20000 / 3000 points: 2.35e-4 / 2.35e-4; 1.39e-6 / 1.39e-6; 1.16e-8 / 2.22e-8;
# 9.03e-8 / 3.21e-7; 3.86e-7 / 1.41e-6, in the order below). Re-derived Oct 19, 2026
# on the references regenerated at tol_thermo_integration = 1e-10 (job 26861).
HE_RATIO_GUARD_BARS = {
    (5000.0, 8000.0): {20000: 4.8e-4, 3000: 4.8e-4},   # HeIII -> HeII Saha, x_e sum
    (2870.0, 5000.0): {20000: 2.8e-6, 3000: 2.8e-6},   # HeII -> HeI Saha, x_e sum
    (2870.0, 3500.0): {20000: 2.4e-8, 3000: 4.5e-8},   # HeII Saha quadratic (clax = CLASS to ~1e-8)
    (2000.0, 2400.0): {20000: 1.9e-7, 3000: 6.5e-7},   # helium Peebles rate and x_e sum
    (1650.0, 2870.0): {20000: 7.8e-7, 3000: 2.9e-6},   # whole helium ODE window
}


@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("window", list(HE_RATIO_GUARD_BARS), ids=lambda w: f"z{w[0]:g}-{w[1]:g}")
def test_helium_ratio_guard_xe_windows(recomb_cosmology, n, window):
    """x_e against CLASS RECFAST (Heswitch=1) in five z windows, with bars of 2 x the
    worst error the code has at the 11 cosmologies (HE_RATIO_GUARD_BARS): 2 to 4e4
    times tighter than the 1e-3 gates above. It guards the CLASS He/H number ratio
    f_He = Y/(3.9715 (1-Y)) at every site these windows can see. Reverting one such
    site to exactly 4 pushes x_e past its bar (mutant error / bar, 20000 / 3000
    points, the smaller of fiducial and Y_He = 0.28; job 26861): HeII Saha quadratic
    7.8 / 4.0 (z 2870-3500); helium Peebles rate 360 / 105 and helium Peebles x_e
    sum 28 / 8.7 (both z 2000-2400); final x_e sum 2.1 to 2.3e4 (every window).

    Not guarded, in two kinds. (1) The reversion moves x_e by less than the error the
    code already has against CLASS in the same z range, so no bar can pass the current
    code with a 2x margin and fail the reversion. Best window searched, mutant error
    over its bar (fiducial and Y_He = 0.28, both grids, the 9 guard windows from
    z 800 to 8000): the HeIII Saha constant in `_ionHe` 0.47 (shift 1.9e-5 against a
    2.0e-4 offset from CLASS at z 5500-6500), the f_He in the hydrogen Peebles x_e sum
    0.85 and in the hydrogen Saha correction 0.83 (both z 1500-1650, 20000 points).
    (2) The site acts outside the
    references' z in [200, 8000]: the initial x_e
    of the scan, which sets the table only above z = 5e4, and `_reionization_xe`
    (z <~ 8); each shifts x_e by up to 1.0e-3 there. The four `1 - 0.75 Y_He`
    mean-molecular-weight sites keep exactly 4 by design (ADR 0003, D4) and are not
    guarded. Parametrised over GRIDS: the 3000-point fiducial case runs under --fast."""
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= window[0]) & (ref["z"] <= window[1])
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    bar = HE_RATIO_GUARD_BARS[window][n]
    print(f"{name} n={n}: x_e vs CLASS z {window} max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f} (bar {bar:.1e})")
    assert rel[i] <= bar, f"ERROR x_e helium-ratio guard {window} {rel[i]:.3e} > {bar:.1e} ({name}, n={n})"


# Hydrogen staging guard (PR #53 review, Oct 19, 2026): bars of 2 x the worst
# |x_e/x_e,CLASS - 1| over the 11 recomb_cosmology points on the tol 1e-10
# references, rounded up to 2 significant figures (measured worst, 20000 / 3000
# points: 7.05e-7 / 2.18e-6; 8.04e-7 / 2.60e-5; 2.12e-6 / 5.69e-5; 1.98e-5 / 1.14e-4,
# in the order below; job 26861).
H_STAGING_GUARD_BARS = {
    (1600.0, 1650.0): {20000: 1.5e-6, 3000: 4.4e-6},   # ODE start: the Saha value at the step end
    (1550.0, 1650.0): {20000: 1.7e-6, 3000: 5.2e-5},   # hand-over, incl. the former z 1550-1590 gap
    (1500.0, 1600.0): {20000: 4.3e-6, 3000: 1.2e-4},   # f2 blend window
    (800.0, 1550.0): {20000: 4.0e-5, 3000: 2.3e-4},    # Peebles ODE down to z = 800
}


@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("window", list(H_STAGING_GUARD_BARS), ids=lambda w: f"z{w[0]:g}-{w[1]:g}")
def test_hydrogen_staging_guard_xe_windows(recomb_cosmology, n, window):
    """x_e against CLASS RECFAST (Heswitch=1) over the hydrogen stage, z 800-1650, in
    four windows with bars of 2 x the worst error at the 11 cosmologies
    (H_STAGING_GUARD_BARS); the 1e-3 gate above cannot see the staging details. It
    guards the hydrogen Saha value taken at the step end (ADR 0003, D3): with it
    taken at the step start instead (the pre-review half-step lag), x_e exceeds its
    bar by 47 / 102 (z 1600-1650), 68 / 14 (z 1550-1650), 27 / 6.2 (z 1500-1600) and
    2.8 / 3.2 (z 800-1550) at 20000 / 3000 points, the smallest over fiducial,
    Y_He = 0.28 and omega_b -+ 20% (job 26861); that mutant also fails the
    helium-ratio guard's z 1650-2870 window by 31 / 54. The former gap between the
    1e-3 windows, z 1550-1590, where fit_cl's x_e once collapsed to 0.035, lies
    inside (1550, 1650). The references are CLASS at tol_thermo_integration = 1e-10:
    at CLASS's default 1e-6 its own integration error in these windows is 2e-6
    (2.3e-4 at omega_b - 20%, where the Peebles C = 1 switch fires inside the ODE),
    i.e. the size of these bars. Parametrised over GRIDS: the 3000-point fiducial
    case runs under --fast."""
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= window[0]) & (ref["z"] <= window[1])
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    bar = H_STAGING_GUARD_BARS[window][n]
    print(f"{name} n={n}: x_e vs CLASS z {window} max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f} (bar {bar:.1e})")
    assert rel[i] <= bar, f"ERROR x_e hydrogen-staging guard {window} {rel[i]:.3e} > {bar:.1e} ({name}, n={n})"
