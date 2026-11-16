"""G1-G4 (smsharma/clax#54): d/dtheta of F = sum_{l=30}^{1200} ln C_l^TT at fit_cl
is independent of solver step placement, agrees between forward and production
reverse mode, is smooth along h, and matches converged finite differences.
GPU (each point is a full fit_cl solve under AD). Multi-cosmology:
recomb_downstream_cosmology (3 points). Wall clock on a V100: about 6 h per cosmology for
the whole file, of which G4 is ~3 h (jobs 28321-28323) and each G2-direct case ~25 min of
reverse-program compile (job 27802); run one job per cosmology with -k.

Two perturbation solvers appear, for a reason that is about the finite-difference
reference and the step sequence, not about the physics:
- Kvaerno5 (implicit, Newton-iterated) reproduces its primal to ~1e-9 in every
  compilation, so it is the primary gate for the derivative-vs-finite-difference
  test (G4). Its derivative carries a small floor that is measured, not explained:
  reverse vs forward mode differ by 1.6-7.2e-6 at fit_cl tolerance and 1.5-2.7e-6 at
  100x tighter tolerance (jobs 27725/27726/27786), jit(grad) vs grad by 2.4e-9 to
  3.7e-8 (jobs 27739-27741), all far under the 1e-3 bar. The perturbation solve alone
  is an exact transpose (job 27848), so the floor arises upstream of it; its origin is
  not established.
- Rodas5 (linearly implicit; fit_cl's production solver) accepts or rejects steps
  by comparisons that compile-order roundoff can flip (jobs 27675/27676: primals
  4.7e-6 apart between two compilations of the same function; job 27641: an
  algebraically exact rewrite of the step flipped the decisions in 31 of 100
  k-modes). G1 and G3
  (forward mode) stay on it, and G4 and the full-pipeline RF1/RF4 each have a
  Rodas5 companion with a bar that includes that noise.

RF1 and RF4 (jit-of-grad equals grad; vmap equals loop) are harmonic-stage tests at
the 1e-10 bar: the ODE solve is held fixed, because neither solver's derivative is
compilation-independent (Rodas5 ~1e-5 from step-sequence changes, job 27675; Kvaerno5
1.6-7.2e-6 reverse vs forward and up to 3.7e-8 jit(grad) vs grad, origin not
established, see above), and what #54 changed is the Bessel table, the Hermite lookup
and the checkpointed l-scan.

G4 precision. G4 (both solvers) evaluates AD and the finite-difference stencil with
the perturbation ODE tolerance tightened 100x (rtol 1e-6, atol 1e-8, ode_max_steps
65536 so no k-mode starves), and prints AD at fit_cl's own tolerance beside AD at the
tight one (a convergence record, not gated). Why: at fit_cl tolerance (rtol 1e-4,
atol 1e-6) F = sum ln C_l wobbles along omega_b at ~1e-8 relative, from the solver's
step-sequence sensitivity (scan jobs 28117, 28245, 28262-28265: residual RMS about a
cubic 1.9e-8 at fit_cl, 1.6e-9 at 100x tighter, 1.5e-9 at 10^4x tighter), and a 1% stencil
turns that into ~1e-3 in the derivative, the size of the bar. The tolerance is a
precision setting of the reference, like the 8000-point background grid used by G3 and
G4; neither bar moves. G1-G3 stay at fit_cl tolerance. The wobble is not the
recombination epoch tau_star: holding tau_star fixed changes F by <= 9e-10 and leaves the
residual unchanged, and the background solve is excluded too (scan jobs 28117, 28133,
28168, 28190).

Reverse-mode tests (G2, RF1 companion). On the base of this branch every reverse-mode
gradient at fit_cl was NaN (vmapped k-batches run masked dt = 0 steps through Rodas5)
and, with that fixed, the production adjoint was 1% off (RecursiveCheckpointAdjoint at
diffrax's default checkpoint count recomputes steps whose Rodas5 trajectory does not
reproduce, and drops the end-state cotangent of a k-mode that finishes just before a
stored checkpoint). Both are fixed on main's side of this branch: smsharma/clax#63
(the 1/dt guard) and #64 (one checkpoint per step of the budget). The reverse-mode
tests below are therefore hard passes, measured on this tree:
- test_g2_direct_adjoint_reverse_matches_forward: jax.grad under the direct adjoint
  equals the forward-mode derivative, the statement that the C_l pipeline's reverse
  mode is correct (the thermodynamics gradient mode, stable or native, does not
  matter). Measured: TG2D_PENDING.
- test_g2_production_reverse_matches_forward: the production configuration (stable
  thermodynamics mode, recursive-checkpoint adjoint storing every step) against
  forward mode. Measured: TG2P_PENDING.
- test_rf1_jit_of_grad_equals_grad_rodas5_companion: eager and jitted reverse mode
  agree. Measured: TRF1R_PENDING.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.cl_test_utils import (
    FIT, G4_PT_RTOL, G4_TH, F_harm, F_tt, cubic_residual, fwd, g4_prec, harmonic_inputs,
    jvp_scalar, spread, stencil_fd)

pytestmark = pytest.mark.slow
THETAS = ("h", "omega_b", "omega_cdm")

# Harmonic-stage RF tests: parameter -> three values of x as multiples of x0. n_s only
# enters through the primordial spectrum; tau_0 (bg.conformal_age) enters the Bessel
# table's argument k*chi. The tau_0 multiples stay >= 1 so chi = tau_0 - tau >= 0 on
# the whole tau grid, which ends at tau_0.
RF_THETAS = ("n_s", "tau_0")
RF_SCALES = {"n_s": (0.99, 1.0, 1.01), "tau_0": (1.0, 1.005, 1.01)}



@pytest.mark.parametrize("theta", THETAS)
def test_g1_step_placement(recomb_downstream_cosmology, theta):
    """Forward-mode dF/dtheta does not depend on where the perturbation solver places
    its accepted steps. Step placement is moved two ways: a 0.1% jitter of the
    perturbation rtol (1.0 and 1.001 times fit_cl's 1e-4; atol unchanged) and eager vs
    jit compilation. The four derivatives must agree to 3e-4, spread = (max - min) /
    |median|. The k-mode chunk size is not an axis: each k-mode keeps its own adaptive
    step sequence inside a vmapped batch, so pt_k_chunk_size 8 and 64 give bit-identical
    derivatives in the C_l path (all 9 cases of jobs 27739-27741). A 0.1% tolerance
    jitter does change the accepted-step sequence while leaving the accuracy of F
    unchanged: with it the four derivatives are no longer bit-identical, and their
    spreads are 3.0e-7 / 6.5e-6 / 5.4e-7 (h), 1.65e-4 / 1.2e-5 / 8.0e-6 (omega_b) and
    1.3e-5 / 2.8e-6 / 4.4e-6 (omega_cdm) at lcdm_fiducial / omega_b_high / YHe_028
    (jobs 28454, 28487). The largest, dF/domega_b at lcdm_fiducial at 55% of the bar,
    is how that derivative depends on the fit_cl tolerance itself (690.198 at rtol x 1.0
    against 690.31 at x 1.001, jit and eager agreeing to <= 4.1e-6 at each), consistent
    with the G4 convergence record: AD at fit_cl tolerance and at 100x tighter tolerance
    differ by <= 2.8e-4 along omega_b (jobs 28321-28323)."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    v = [jvp_scalar(F_tt(replace(fwd(FIT), pt_ode_rtol=FIT.pt_ode_rtol * f), base, theta),
                    x0, jit=j)
         for f in (1.0, 1.001) for j in (True, False)]
    s = spread(v)
    print(f"G1 {name} dF/d{theta}: {v} spread {s:.3e}")
    assert s < 3e-4, f"ERROR C_l step-placement spread {s:.3e} ({name}, {theta})"


@pytest.mark.parametrize("theta", THETAS)
def test_g2_direct_adjoint_reverse_matches_forward(recomb_downstream_cosmology, theta):
    """Reverse mode under the direct adjoint (production thermodynamics mode kept)
    equals the forward-mode derivative under fwd(FIT), bar 3e-4. Each case compiles
    a reverse program of ~25 min on a V100 (job 27802: 1677 s for h)."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    f = jvp_scalar(F_tt(fwd(FIT), base, theta), x0)
    r = float(jax.grad(F_tt(replace(FIT, ode_adjoint="direct"), base, theta))(jnp.asarray(x0)))
    rel = abs(r / f - 1.0)
    print(f"G2-direct {name} dF/d{theta}: forward {f:.8e} reverse {r:.8e} rel {rel:.3e}")
    assert np.isfinite(r) and rel < 3e-4, f"ERROR reverse vs forward {rel:.3e} ({name}, {theta})"


@pytest.mark.parametrize("theta", THETAS)
def test_g2_production_reverse_matches_forward(recomb_downstream_cosmology, theta):
    """Production reverse mode (recursive-checkpoint adjoint storing every step, #64)
    against forward mode, bar 3e-4. Hard pass on this tree; see the module docstring."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    f = jvp_scalar(F_tt(fwd(FIT), base, theta), x0)
    r = float(jax.grad(F_tt(FIT, base, theta))(jnp.asarray(x0)))
    rel = abs(r / f - 1.0)
    print(f"G2 {name} dF/d{theta}: forward {f:.8e} reverse {r:.8e} rel {rel:.3e}")
    assert np.isfinite(r) and rel < 3e-4, f"ERROR reverse vs forward {rel:.3e} ({name}, {theta})"


def test_g3_smooth_along_h(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    hs = np.linspace(0.64, 0.70, 7)
    prec = replace(fwd(FIT), bg_n_points=8000)    # 800-point background-table AD noise (smsharma/clax#56)
    d = [jvp_scalar(F_tt(prec, replace(base, h=float(h)), "h"), float(h)) for h in hs]
    r = cubic_residual(hs, d)
    print(f"G3 {name}: dF/dh along h {d} cubic residual {r:.3e}")
    assert r < 3e-4, f"ERROR dF/dh not smooth along h: {r:.3e} ({name})"


@pytest.mark.parametrize("theta", THETAS)
def test_g4_ad_matches_converged_fd(recomb_downstream_cosmology, theta):
    """Primary gate: forward-mode AD against a stencil finite difference, on Kvaerno5
    (reproducible step sequence), bg_n_points=8000 (omega_b: th_n_points 20000 for both
    AD and FD, see G4_TH) and the perturbation ODE tolerance tightened 100x (rtol 1e-6,
    atol 1e-8; g4_prec). The FD is stencil_fd: a least-squares cubic through 9
    evaluations at x0 (1 + s), |s| <= 0.01; fd9 uses all 9, fd7 the inner 7, and
    unc = |fd9 - fd7| / |fd9| is the FD's own spread (a lower bound on its error, since fd7
    reuses 7 of the 9 points). Asserts unc <= 3e-4 and
    |AD/fd9 - 1| <= 1e-3, the spec bar. Also prints AD at fit_cl's own tolerance next
    to AD at the tight one (not gated): the record of how far fit_cl's derivative sits
    from the converged one.

    Why the tight tolerance: at fit_cl tolerance (rtol 1e-4, atol 1e-6) F = sum of ln C_l
    over 1171 multipoles wobbles along omega_b at ~1e-8 relative, from the perturbation
    solver's step-sequence sensitivity (scan jobs 28117 and 28245: residual about a cubic
    1.9e-8 at fit_cl, 1.6e-9 at 100x tighter). A 1% stencil turns that into ~1e-3
    in dF/domega_b, the size of the bar, so the 9-point FD along omega_b had uncertainty
    3.9e-4 / 3.4e-4 / 1.9e-3 (lcdm_fiducial / omega_b_high / YHe_028, jobs 27966-27968)
    and could not anchor AD. The tolerance is a precision setting of the reference, like
    the 8000-point background grid; the bars are unchanged. The wobble is not the
    recombination epoch tau_star (holding it fixed changes F by <= 9e-10; scan jobs
    28117, 28133, 28168, 28190).

    Why a stencil rather than two central differences that must agree to 1e-4: along h
    the two-step scatter did not move with the ODE tolerance (probe job 27869: 5.5e-4 at
    rtol 1e-4, 1e-6 and 1e-8), and a short step amplifies the wobble by 1/(2 dx). At
    lcdm_fiducial dF/dh was AD -251.41583, FD(1e-3) -251.19810, FD(3e-3) -251.33311
    (job 27739): the larger step is closer to AD, the two never agree to 1e-4, and the
    failure is the FD reference's, not AD's. The wide cubic fit averages the wobble."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    n_th = G4_TH[theta]
    F = F_tt(g4_prec("kvaerno5", n_th), base, theta)
    ad = jvp_scalar(F, x0)
    ad_fit = jvp_scalar(F_tt(g4_prec("kvaerno5", n_th, tight=False), base, theta), x0)
    fd9, fd7, unc = stencil_fd(F, x0)
    rel = abs(ad / fd9 - 1.0)
    print(f"G4 {name} dF/d{theta} [kvaerno5, th_n_points {n_th}, rtol {G4_PT_RTOL:g}]: "
          f"AD@FIT {ad_fit:.8e} AD@tight {ad:.8e} (FIT/tight-1 {ad_fit / ad - 1.0:.3e}) "
          f"fd9 {fd9:.8e} fd7 {fd7:.8e} unc {unc:.3e} AD/fd9-1 {rel:.3e} (AD/fd9 {ad / fd9:.6f})")
    assert unc <= 3e-4, f"ERROR FD uncertainty {unc:.3e} > 3e-4 ({name}, {theta})"
    assert rel <= 1e-3, f"ERROR AD vs FD {rel:.3e} ({name}, {theta})"


@pytest.mark.parametrize("theta", THETAS)
def test_g4_rodas5_companion(recomb_downstream_cosmology, theta):
    """Companion to the Kvaerno5 gate on fit_cl's production solver (Rodas5), under
    fwd(FIT) with bg_n_points=8000 (omega_b: th_n_points 20000), the perturbation ODE
    tolerance tightened 100x as in the gate (rtol 1e-6, atol 1e-8; g4_prec; the wobble
    that motivates it was measured on Rodas5, see the gate's docstring) and the same
    stencil_fd. Asserts |AD - fd9| <= 1e-3 |AD| + unc |fd9|: the spec bar plus the FD
    estimate's own spread, kept because Rodas5's accepted-step sequence is not
    reproducible (jobs 27675/27676: primals 4.7e-6 apart between compilations). Prints
    AD at fit_cl's own tolerance next to AD at the tight one (not gated), fd9, fd7 and
    unc."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    n_th = G4_TH[theta]
    F = F_tt(g4_prec(FIT.pt_ode_solver, n_th), base, theta)
    ad = jvp_scalar(F, x0)
    ad_fit = jvp_scalar(F_tt(g4_prec(FIT.pt_ode_solver, n_th, tight=False), base, theta), x0)
    fd9, fd7, unc = stencil_fd(F, x0)
    gap = abs(ad - fd9)
    allowed = 1e-3 * abs(ad) + unc * abs(fd9)
    print(f"G4-rodas5 {name} dF/d{theta} [th_n_points {n_th}, rtol {G4_PT_RTOL:g}]: "
          f"AD@FIT {ad_fit:.8e} AD@tight {ad:.8e} (FIT/tight-1 {ad_fit / ad - 1.0:.3e}) "
          f"fd9 {fd9:.8e} fd7 {fd7:.8e} unc {unc:.3e} |AD-fd9|/|AD| {gap / abs(ad):.3e} "
          f"allowed {allowed / abs(ad):.3e}")
    assert gap <= allowed, (f"ERROR |AD - fd9| {gap / abs(ad):.3e} of AD exceeds "
                            f"1e-3 + unc {allowed / abs(ad):.3e} ({name}, {theta})")


def _harmonic_x0(inputs, theta):
    p0, bg0, _ = inputs
    return float(p0.n_s) if theta == "n_s" else float(bg0.conformal_age)


@pytest.mark.parametrize("theta", RF_THETAS)
def test_rf1_harmonic_jit_of_grad_equals_grad(recomb_downstream_cosmology, theta):
    """jax.jit(jax.grad(F)) equals jax.grad(F) at the 1e-10 bar, for F = sum ln C_l^TT
    from compute_cls_all_fast alone. This tests the Bessel table, the Hermite lookup
    and the checkpointed l-scan (what #54 changed) under jit; the ODE solve is held
    fixed (computed once per cosmology at production fit_cl and closed over), because
    its derivatives carry a compilation-dependent floor (Rodas5 ~1e-5 from step-sequence
    changes, job 27675; Kvaerno5 1.6-7.2e-6 reverse vs forward, jobs 27725/27726, origin
    not established), so there is no ODE solve under the gradient. theta = n_s differentiates only the
    primordial spectrum; theta = tau_0 (bg.conformal_age) differentiates through the
    Bessel-table lookup and the checkpointed l-scan's backward pass."""
    name, base = recomb_downstream_cosmology
    inputs = harmonic_inputs(name, base)
    F = F_harm(inputs, theta)
    x0 = jnp.asarray(_harmonic_x0(inputs, theta))
    a, b = float(jax.grad(F)(x0)), float(jax.jit(jax.grad(F))(x0))
    print(f"RF1-harmonic {name} d/d{theta}: grad {a:.10e} jit(grad) {b:.10e} "
          f"rel {abs(a / b - 1.0):.3e}")
    assert np.isfinite(a) and np.isfinite(b), f"ERROR non-finite gradient ({name}, {theta})"
    assert abs(a / b - 1.0) < 1e-10, f"ERROR jit(grad) differs from grad ({name}, {theta})"


def test_rf1_jit_of_grad_equals_grad_rodas5_companion(recomb_downstream_cosmology):
    """Rodas5 companion of the RF1 gate: bar 3e-4 (Rodas5 re-integrates a different
    step sequence per compilation, 4.7e-6 in the primal), difference printed."""
    name, base = recomb_downstream_cosmology
    F = F_tt(FIT, base, "h")
    x0 = jnp.asarray(float(base.h))
    a, b = float(jax.grad(F)(x0)), float(jax.jit(jax.grad(F))(x0))
    print(f"RF1-rodas5 {name}: grad {a:.10e} jit(grad) {b:.10e} rel {abs(a / b - 1.0):.3e}")
    assert np.isfinite(a) and np.isfinite(b), f"ERROR non-finite gradient ({name})"
    assert abs(a / b - 1.0) < 3e-4, f"ERROR jit(grad) differs from grad ({name})"


def _vmap_vs_loop_rel(prec, base):
    F = jax.jit(F_tt(prec, base, "h"))
    hs = float(base.h) * jnp.array([0.99, 1.0, 1.01])
    a = np.asarray(jax.vmap(F)(hs))
    b = np.array([float(F(h)) for h in hs])
    return float(np.max(np.abs(a - b) / np.abs(b)))


@pytest.mark.parametrize("theta", RF_THETAS)
def test_rf4_harmonic_vmap_matches_loop(recomb_downstream_cosmology, theta):
    """vmap of jit(F) over three values of theta equals the loop of scalar calls at the
    1e-10 bar, for F = sum ln C_l^TT from compute_cls_all_fast alone (the same
    functional and the same fixed ODE solve as RF1-harmonic: the Bessel table, the
    Hermite lookup and the checkpointed l-scan under vmap, with no compilation-
    dependent ODE floor). Values of theta: x0 times RF_SCALES[theta]."""
    name, base = recomb_downstream_cosmology
    inputs = harmonic_inputs(name, base)
    F = jax.jit(F_harm(inputs, theta))
    xs = _harmonic_x0(inputs, theta) * jnp.array(RF_SCALES[theta])
    a = np.asarray(jax.vmap(F)(xs))
    b = np.array([float(F(x)) for x in xs])
    d = float(np.max(np.abs(a - b) / np.abs(b)))
    print(f"RF4-harmonic {name} theta={theta}: vmap vs loop max rel {d:.3e}")
    assert np.isfinite(a).all() and np.isfinite(b).all(), f"ERROR non-finite F ({name}, {theta})"
    assert d < 1e-10, f"ERROR vmap vs loop {d:.3e} ({name}, {theta})"


def test_rf4_vmap_matches_loop_rodas5_companion(recomb_downstream_cosmology):
    """Rodas5 companion of the RF4 gate: bar 3e-4 (step-sequence noise between the
    vmapped and the scalar compilation), difference printed."""
    name, base = recomb_downstream_cosmology
    d = _vmap_vs_loop_rel(fwd(FIT), base)
    print(f"RF4-rodas5 {name}: vmap vs loop max rel {d:.3e}")
    assert d < 3e-4, f"ERROR vmap vs loop {d:.3e} ({name})"
