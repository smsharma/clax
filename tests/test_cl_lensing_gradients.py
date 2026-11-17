"""G5 (smsharma/clax#54): derivatives of the lensing functionals at fit_cl,

    F_pp  = sum_{l=30}^{1200} ln C_l^phiphi,
    F_ltt = sum_{l=30}^{1200} ln (lensed C_l^TT),

for nonlinear = none / halofit / ept and theta = h / omega_b. GPU (each point is a
full fit_cl solve under AD, plus the tau0-complete matter solve for halofit/ept).
Multi-cosmology: recomb_downstream_cosmology (3 points). Run one job per cosmology
with -k (and per func if the 24 h limit requires it).

What #54 changed here. clax/lensing.py: the Halofit modulator's z-grid reads z_rec as a
traced value instead of float(th.z_rec), its extended k-grid is sized from the static
prec.pt_k_max_cl (compute_cl_pp(..., prec=)), and both modulators read P_lin with
validate=False (the k-support check needs concrete arrays); the EPT ratio
R0 = P_NL / P_lin is built on fixed shapes (P_NL splined over the fixed EPT nodes, read
at the exact perturbation k; R0 = 1 at the last pt node, where one-loop EFT does not
apply), so it is smooth in h. clax/ept.py: the IR-resummation DST grid (7e-5/h .. 7/h)
follows the traced h instead of a concrete float(stop_gradient(h)), so the EPT stage
runs under jit (primal change at fiducial: EPT P(k) <= 1.6e-12, C_l^phiphi-ept
<= 7.2e-16, job 28536). Before that, jit or AD of either nonlinear path raised
(float() of a traced value). clax/nonlinear.py: Halofit's nonlinear scale R_nl
(sigma(R_nl) = 1, found by bisection) carries the implicit-function tangent; before,
AD gave d R_nl / d theta = 0 and every Halofit derivative here missed FD by 11-31%
(C_l^phiphi) and 0.1-2.3% (lensed TT) (jobs 28542-28547; primal bit-identical, job
28762; unit test tests/test_nonlinear.py::TestHaloFitTangent).

Three tests, split so that one failing check cannot hide the others:
1. test_g5_forward_matches_fd_kvaerno5 -- the gate. Forward-mode AD against a stencil
   finite difference on Kvaerno5, whose step sequence reproduces between
   compilations (primals to ~1e-9, job 27726).
2. test_g5_forward_matches_fd_rodas5 -- companion on fit_cl's production solver,
   Rodas5, whose accepted-step sequence can change between two compilations of the
   same function (jobs 27675/27676: primals 4.7e-6 apart), so its bar adds the FD
   estimate's own spread to the spec bar.
3. test_g5_production_reverse_matches_forward -- production reverse mode against
   forward mode, a hard pass on this tree. On the base of this branch every
   reverse-mode gradient through Rodas5 at fit_cl was NaN (masked dt = 0 steps; job
   26946) and, with that fixed, the lensed_tt cases missed the bar by 3.4e-4 to 6.4e-3
   because the production RecursiveCheckpointAdjoint recomputed Rodas5 steps and, at the
   seam between a recomputed segment and a stored state, counted a k-mode's end value
   twice or not at all (the pp cases passed, 1.9e-9 to 5.9e-5; CHANGELOG, Nov 17, 2026).
   Both are fixed on main's side of this branch (smsharma/clax#63, #64). Measured here:
   TG5P_PENDING.

Finite-difference anchor (the same as the unlensed G4 test in tests/test_cl_gradients.py).
At fit_cl tolerance (rtol 1e-4, atol 1e-6) sum ln C_l wobbles along omega_b at ~1e-8
relative (and ~2e-9 along h) because the Rodas5 solve's accepted-step sequence changes
with the parameters (scan jobs 28117, 28245, 28262-28265: residual RMS about a cubic
1.9e-8 at rtol 1e-4, 1.6e-9 at rtol 1e-6). A 1% stencil turns that into ~1e-3 in the
derivative, the size of the bar. Therefore AD and FD alike are evaluated at g4_prec:
perturbation tolerance tightened 100x (rtol 1e-6, atol 1e-8, ode_max_steps 65536),
bg_n_points 8000 and, for omega_b, th_n_points 20000 (G4_TH; fit_cl's 3000-point
recombination grid makes F(omega_b) a staircase, PR #59 ADR 0004). The FD is
stencil_fd: a least-squares cubic through 9 evaluations at x0 (1 + s), |s| <= 0.01;
fd9 uses all 9, fd7 the inner 7, unc = |fd9 - fd7| / |fd9| (a lower bound on the
FD's error, since fd7 reuses 7 of the 9 points). Bars: unc <= 3e-4 and
|AD/fd9 - 1| <= 1e-3, the spec's. AD at fit_cl's own tolerance is printed beside
AD at the tight one (a convergence record, not gated).

Recombination epoch. On the base of this branch th.z_rec = z_grid[argmin |kappa - 1|] was
a knot of the thermodynamics grid, and the knots move with the parameters (the grid is
uniform in ln tau, and tau_0 depends on h). AD differentiated the knot's motion; finite
differences saw the motion plus a jump back by one index (a sawtooth: at omega_b_high
z_rec rose 0.32 per 0.25% step in h, then dropped 3.24). The physical kappa = 1 epoch
barely depends on h at fixed omega_b and omega_cdm. C_l^phiphi depends on z_rec through
the lensing kernel's start, so dF_pp/dh missed FD by 0.55-2.9% (jobs 28542/28544/28546,
28615/28617/28619) and dF_pp/domega_b at lcdm_fiducial by 2.0e-3 (a one-index jump at s =
-1%, job 28698); with z_rec held fixed, AD matched FD to 4.9e-5 (job 28583) and -2.3e-4
(unc 4.8e-5). smsharma/clax#61 (merged Oct 23, 2026; main 6ff1303) replaced the knot by
linear interpolation of kappa = 1 in ln a (clax/thermodynamics.py:1050); this tree carries
it, so the twelve pp cases that were xfail(strict) on the base, and the six lcdm_fiducial
pp Rodas5 companions that passed only because the knot widened their allowance, are plain
gates again. Measured on this tree: TG5F_PENDING. th.tau_star, also a knot, sets the
perturbation save grid; its effect on unlensed TT was a single step of +4.6e-9 of |F|
along h (scan job 28168).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.cl_test_utils import (
    FIT, G4_PT_RTOL, G4_TH, F_ltt, F_pp, fwd, g4_prec, jvp_scalar, stencil_fd)

pytestmark = pytest.mark.slow
FUNCS = {"pp": F_pp, "lensed_tt": F_ltt}
THETAS = ("h", "omega_b")

NONLINEAR = ("none", "halofit", "ept")
THETA_FUNC = [pytest.param(t, f, id=f"{t}-{f}") for t in THETAS for f in FUNCS]


@pytest.mark.parametrize("nonlinear", NONLINEAR)
@pytest.mark.parametrize("theta,func", THETA_FUNC)
def test_g5_forward_matches_fd_kvaerno5(recomb_downstream_cosmology, func, nonlinear,
                                        theta):
    """Gate: forward-mode AD of F against stencil_fd, on Kvaerno5 at g4_prec (see the
    module docstring). Asserts AD and FD finite, unc <= 3e-4 and |AD/fd9 - 1| <= 1e-3."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    n_th = G4_TH[theta]
    F = FUNCS[func](g4_prec("kvaerno5", n_th), base, theta, nonlinear)
    ad = jvp_scalar(F, x0)
    ad_fit = jvp_scalar(FUNCS[func](g4_prec("kvaerno5", n_th, tight=False), base, theta,
                                    nonlinear), x0)
    fd9, fd7, unc = stencil_fd(F, x0)
    rel = abs(ad / fd9 - 1.0)
    print(f"G5-K5 {name} {func}/{nonlinear} dF/d{theta} [th_n_points {n_th}, "
          f"rtol {G4_PT_RTOL:g}]: AD@FIT {ad_fit:.8e} AD@tight {ad:.8e} "
          f"(FIT/tight-1 {ad_fit / ad - 1.0:.3e}) fd9 {fd9:.8e} fd7 {fd7:.8e} "
          f"unc {unc:.3e} AD/fd9-1 {rel:.3e}")
    assert np.isfinite(ad) and np.isfinite(fd9), (
        f"ERROR non-finite AD or FD ({name}, {func}, {nonlinear}, {theta})")
    assert unc <= 3e-4, f"ERROR FD uncertainty {unc:.3e} > 3e-4 ({name}, {func}, {nonlinear}, {theta})"
    assert rel <= 1e-3, f"ERROR AD vs FD {rel:.3e} ({name}, {func}, {nonlinear}, {theta})"


@pytest.mark.parametrize("nonlinear", NONLINEAR)
@pytest.mark.parametrize("theta,func", THETA_FUNC)
def test_g5_forward_matches_fd_rodas5(recomb_downstream_cosmology, func, nonlinear,
                                      theta):
    """Companion on fit_cl's production solver (Rodas5) at g4_prec and the same
    stencil_fd. Asserts AD and FD finite and |AD - fd9| <= 1e-3 |AD| + unc |fd9|:
    the spec bar plus the FD estimate's own spread, kept because Rodas5's
    accepted-step sequence is not reproducible between compilations (jobs
    27675/27676: primals 4.7e-6 apart), so the FD reference carries that noise."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    n_th = G4_TH[theta]
    F = FUNCS[func](g4_prec(FIT.pt_ode_solver, n_th), base, theta, nonlinear)
    ad = jvp_scalar(F, x0)
    ad_fit = jvp_scalar(FUNCS[func](g4_prec(FIT.pt_ode_solver, n_th, tight=False), base,
                                    theta, nonlinear), x0)
    fd9, fd7, unc = stencil_fd(F, x0)
    gap = abs(ad - fd9)
    allowed = 1e-3 * abs(ad) + unc * abs(fd9)
    print(f"G5-R5 {name} {func}/{nonlinear} dF/d{theta} [th_n_points {n_th}, "
          f"rtol {G4_PT_RTOL:g}]: AD@FIT {ad_fit:.8e} AD@tight {ad:.8e} "
          f"(FIT/tight-1 {ad_fit / ad - 1.0:.3e}) fd9 {fd9:.8e} fd7 {fd7:.8e} "
          f"unc {unc:.3e} |AD-fd9|/|AD| {gap / abs(ad):.3e} allowed {allowed / abs(ad):.3e}")
    assert np.isfinite(ad) and np.isfinite(fd9), (
        f"ERROR non-finite AD or FD ({name}, {func}, {nonlinear}, {theta})")
    assert gap <= allowed, (f"ERROR |AD - fd9| {gap / abs(ad):.3e} of AD exceeds 1e-3 + unc "
                            f"{allowed / abs(ad):.3e} ({name}, {func}, {nonlinear}, {theta})")


@pytest.mark.parametrize("nonlinear", NONLINEAR)
@pytest.mark.parametrize("func", list(FUNCS))
@pytest.mark.parametrize("theta", THETAS)
def test_g5_production_reverse_matches_forward(recomb_downstream_cosmology, func, nonlinear,
                                               theta):
    """Production reverse mode (jax.grad under FIT: Rodas5, recursive-checkpoint
    adjoint storing every step) equals forward mode (jvp under fwd(FIT)), bar 3e-4.
    Hard pass on this tree: see the module docstring."""
    name, base = recomb_downstream_cosmology
    x0 = float(getattr(base, theta))
    f = jvp_scalar(FUNCS[func](fwd(FIT), base, theta, nonlinear), x0)
    r = float(jax.grad(FUNCS[func](FIT, base, theta, nonlinear))(jnp.asarray(x0)))
    rel = abs(r / f - 1.0)
    print(f"G5-rev {name} {func}/{nonlinear} dF/d{theta}: forward {f:.8e} reverse {r:.8e} "
          f"rel {rel:.3e}")
    assert np.isfinite(f) and np.isfinite(r), (
        f"ERROR non-finite derivative ({name}, {func}, {nonlinear}, {theta})")
    assert rel < 3e-4, f"ERROR reverse vs forward {rel:.3e} ({name}, {func}, {nonlinear}, {theta})"
