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

Three tests, split so that one failing check cannot hide the others (ruling R24):
1. test_g5_forward_matches_fd_kvaerno5 -- the gate. Forward-mode AD against a stencil
   finite difference on Kvaerno5, whose step sequence reproduces between
   compilations (primals to ~1e-9, Task 5).
2. test_g5_forward_matches_fd_rodas5 -- companion on fit_cl's production solver,
   Rodas5, whose accepted-step sequence can change between two compilations of the
   same function (jobs 27675/27676: primals 4.7e-6 apart), so its bar adds the FD
   estimate's own spread to the spec bar.
3. test_g5_production_reverse_matches_forward -- production reverse mode against
   forward mode. Every reverse-mode gradient through Rodas5 at fit_cl is NaN on main
   (vmapped k-batches run masked dt = 0 steps; Task 4, job 26946); the fix is its own
   PR (fix/rodas5-reverse-nan-v2), so this test is xfail(strict=True) here (XFAIL_REV).
   On the branch all 36 cases are NaN in reverse and xfail (jobs 28548/28550/28552,
   28690/28692/28694). With that fix (scratch overlay, --runxfail, jobs listed in
   XFAIL_REV) the 18 pp cases pass (1.9e-9 to 5.9e-5) and will XPASS, the strict
   signal to drop their marker; the 18 lensed_tt cases miss by 3.4e-4 to 6.4e-3
   because the production RecursiveCheckpointAdjoint through Rodas5 differs from
   forward mode (Task 5's G2-production defect), so their marker stays.

Finite-difference anchor (rulings R38, R43; Task 5's G4). At fit_cl tolerance
(rtol 1e-4, atol 1e-6) sum ln C_l wobbles along omega_b at ~1e-8 relative (and
~2e-9 along h) because the Rodas5 solve's accepted-step sequence changes with the
parameters (scan report taustar-scan-report.md: residual RMS 1.9e-8 at rtol 1e-4,
1.6e-9 at rtol 1e-6). A 1% stencil turns that into ~1e-3 in the derivative, the
size of the bar. Therefore AD and FD alike are evaluated at g4_prec: perturbation
tolerance tightened 100x (rtol 1e-6, atol 1e-8, ode_max_steps 65536),
bg_n_points 8000 and, for omega_b, th_n_points 20000 (G4_TH; fit_cl's 3000-point
recombination grid makes F(omega_b) a staircase, PR #59 ADR 0004). The FD is
stencil_fd: a least-squares cubic through 9 evaluations at x0 (1 + s), |s| <= 0.01;
fd9 uses all 9, fd7 the inner 7, unc = |fd9 - fd7| / |fd9| (a lower bound on the
FD's error, since fd7 reuses 7 of the 9 points). Bars: unc <= 3e-4 and
|AD/fd9 - 1| <= 1e-3, the spec's. AD at fit_cl's own tolerance is printed beside
AD at the tight one (a convergence record, not gated).

Recombination epoch (ZREC_K5, ZREC_R5). th.z_rec = z_grid[argmin |kappa - 1|]
(clax/thermodynamics.py:1004) is a knot of the thermodynamics grid, and the knots move
with the parameters (the grid is uniform in ln tau, and tau_0 depends on h). AD
differentiates the knot's motion; finite differences see the motion plus a jump back
by one index (a sawtooth: at omega_b_high z_rec rises 0.32 per 0.25% step in h, then
drops 3.24). The physical kappa = 1 epoch barely depends on h at fixed omega_b and
omega_cdm. C_l^phiphi depends on z_rec through the lensing kernel's start, so
dF_pp/dh misses FD by 0.55-2.9% (jobs 28542/28544/28546, 28615/28617/28619); with
z_rec held fixed, AD matches FD to 4.9e-5 (job 28583, omega_b_high). Along omega_b the
knot moves smoothly inside the 1% stencil at omega_b_high (job 28610), but at
lcdm_fiducial it jumps by one index at s = -1% (z_rec +0.54 against a -0.035 trend
per step, job 28698), so the FD spread is 2.2e-3; with z_rec frozen, AD/fd9 - 1 =
-2.3e-4, unc 4.8e-5. The cases this makes fail are xfail(strict) per cosmology
(ZREC_K5, ZREC_R5, applied at run time with request.applymarker, ruling R52) until PR
smsharma/clax#61 (z_rec by linear interpolation of kappa = 1). A case not listed
passes as it is: the Rodas5 companion at lcdm_fiducial, for instance, because the
knot's FD spread also widens its allowance. Lensed TT dilutes the channel below the
bar. th.tau_star, also a knot, sets the perturbation save grid; its effect on
unlensed TT was +4.6e-9 (Task 5 scan).
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

XFAIL_REV = pytest.mark.xfail(
    raises=AssertionError, strict=True,
    reason="reverse mode NaN through Rodas5 masked steps on main (fix: "
           "fix/rodas5-reverse-nan-v2, own PR). With that fix (overlay jobs 28549/28551/28553/"
           "28696, 28691/28693/28695, 28769-28771) pp passes (1.9e-9 to 5.9e-5, 18 cases) and "
           "lensed_tt still misses by 3.4e-4 to 6.4e-3 (18 cases): the "
           "RecursiveCheckpointAdjoint through Rodas5 differs from forward mode (Task 5's "
           "production-adjoint defect)")
ZREC_REASON = ("th.z_rec is a thermodynamics grid knot (thermodynamics.py:1004); continuous "
               "in PR #61")
NONLINEAR = ("none", "halofit", "ept")
THETA_FUNC = [pytest.param(t, f, id=f"{t}-{f}") for t in THETAS for f in FUNCS]
# The (cosmology, theta, func, nonlinear) cases the z_rec knot makes fail on this base, per
# test, with the measured miss (raw runs on the branch; see the module docstring). A case
# not listed passed its bar.
ZREC_K5 = {
    ("lcdm_fiducial", "h", "pp", "none"): "AD/fd9-1 6.8e-3, unc 6.8e-3 (job 28542)",
    ("omega_b_high", "h", "pp", "none"): "AD/fd9-1 2.0e-2, unc 2.9e-3 (job 28544)",
    ("YHe_028", "h", "pp", "none"): "AD/fd9-1 2.9e-2, unc 8.6e-3 (job 28546)",
    ("lcdm_fiducial", "omega_b", "pp", "none"): "AD/fd9-1 2.0e-3, unc 2.2e-3 (job 28542)",
    ("lcdm_fiducial", "h", "pp", "halofit"): "AD/fd9-1 5.6e-3, unc 5.5e-3 (job 28756)",
    ("omega_b_high", "h", "pp", "halofit"): "AD/fd9-1 1.6e-2, unc 2.4e-3 (job 28758)",
    ("YHe_028", "h", "pp", "halofit"): "AD/fd9-1 2.4e-2, unc 7.0e-3 (job 28760)",
    ("lcdm_fiducial", "omega_b", "pp", "halofit"): "AD/fd9-1 1.6e-3, unc 1.8e-3 (job 28756)",
    ("lcdm_fiducial", "h", "pp", "ept"): "AD/fd9-1 5.5e-3, unc 5.4e-3 (job 28615)",
    ("omega_b_high", "h", "pp", "ept"): "AD/fd9-1 1.6e-2, unc 2.3e-3 (job 28617)",
    ("YHe_028", "h", "pp", "ept"): "AD/fd9-1 2.4e-2, unc 6.9e-3 (job 28619)",
    ("lcdm_fiducial", "omega_b", "pp", "ept"): "AD/fd9-1 1.8e-3, unc 1.8e-3 (job 28615)",
}
# Not listed, and passing only because the z_rec knot widens the companion's FD-
# uncertainty allowance (1e-3 + unc): the six lcdm_fiducial pp cases. |AD-fd9|/|AD|
# against the allowance: (h, pp) none 6.9e-3/8.1e-3, halofit 5.5e-3/6.8e-3, ept
# 5.5e-3/6.7e-3 (82-85%); (omega_b, pp) none 2.1e-3/3.3e-3, halofit 1.4e-3/2.8e-3, ept
# 1.8e-3/3.0e-3 (50-65%). The gap is the knot artefact itself, not lensing. Not flaky:
# Rodas5 step noise at g4_prec is ~1e-5 in the derivative, against margins >= 1.2e-3.
# A change that moves thermodynamics grid knots before PR #61 could flip them to a
# FAIL, which should be read as the knot. On the #61 overlay they pass at <= 4.8e-4.
ZREC_R5 = {
    ("omega_b_high", "h", "pp", "none"): "|AD-fd9|/|AD| 2.0e-2 > allowed 3.9e-3 (job 28544)",
    ("YHe_028", "h", "pp", "none"): "|AD-fd9|/|AD| 3.0e-2 > allowed 1.0e-2 (job 28546)",
    ("omega_b_high", "h", "pp", "halofit"): "|AD-fd9|/|AD| 1.6e-2 > allowed 3.4e-3 (job 28758)",
    ("YHe_028", "h", "pp", "halofit"): "|AD-fd9|/|AD| 2.4e-2 > allowed 8.3e-3 (job 28760)",
    ("omega_b_high", "h", "pp", "ept"): "|AD-fd9|/|AD| 1.6e-2 > allowed 3.3e-3 (job 28617)",
    ("YHe_028", "h", "pp", "ept"): "|AD-fd9|/|AD| 2.4e-2 > allowed 8.3e-3 (job 28619)",
}


def _xfail_if_zrec(request, cases, name, theta, func, nonlinear):
    """Mark this case xfail(strict, AssertionError) if `cases` lists it (ruling R52)."""
    miss = cases.get((name, theta, func, nonlinear))
    if miss is not None:
        request.applymarker(pytest.mark.xfail(
            raises=AssertionError, strict=True, reason=f"{miss}; {ZREC_REASON}"))


@pytest.mark.parametrize("nonlinear", NONLINEAR)
@pytest.mark.parametrize("theta,func", THETA_FUNC)
def test_g5_forward_matches_fd_kvaerno5(request, recomb_downstream_cosmology, func, nonlinear,
                                        theta):
    """Gate: forward-mode AD of F against stencil_fd, on Kvaerno5 at g4_prec (see the
    module docstring). Asserts AD and FD finite, unc <= 3e-4 and |AD/fd9 - 1| <= 1e-3."""
    name, base = recomb_downstream_cosmology
    _xfail_if_zrec(request, ZREC_K5, name, theta, func, nonlinear)
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
def test_g5_forward_matches_fd_rodas5(request, recomb_downstream_cosmology, func, nonlinear,
                                      theta):
    """Companion on fit_cl's production solver (Rodas5) at g4_prec and the same
    stencil_fd. Asserts AD and FD finite and |AD - fd9| <= 1e-3 |AD| + unc |fd9|:
    the spec bar plus the FD estimate's own spread, kept because Rodas5's
    accepted-step sequence is not reproducible between compilations (jobs
    27675/27676: primals 4.7e-6 apart), so the FD reference carries that noise."""
    name, base = recomb_downstream_cosmology
    _xfail_if_zrec(request, ZREC_R5, name, theta, func, nonlinear)
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


@XFAIL_REV
@pytest.mark.parametrize("nonlinear", NONLINEAR)
@pytest.mark.parametrize("func", list(FUNCS))
@pytest.mark.parametrize("theta", THETAS)
def test_g5_production_reverse_matches_forward(recomb_downstream_cosmology, func, nonlinear,
                                               theta):
    """Production reverse mode (jax.grad under FIT: Rodas5, recursive-checkpoint
    adjoint) equals forward mode (jvp under fwd(FIT)), bar 3e-4. xfail(strict) on
    this branch: see the module docstring."""
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
