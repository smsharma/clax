"""Shared helpers for the C_l differentiability tests (smsharma/clax#54).
Functionals sum ln C_l over l in [30, 1200], avoiding fit_cl's low-l instability."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from clax import PrecisionParams
from clax.harmonic import compute_cls_all_fast
from clax.lensing import compute_cl_pp, lens_cls
from clax.perturbations import perturbations_solve, perturbations_solve_mpk
from clax.thermodynamics import solve_background_and_thermo
from tests.recomb_test_utils import as_traced

CL_LMIN, CL_LMAX = 30, 1200
FIT = PrecisionParams.fit_cl()


def fwd(prec):
    """Forward-mode settings: native thermodynamics AD, direct ODE adjoint."""
    return replace(prec, th_grad_mode="native", ode_adjoint="direct")


# Thermodynamics grid used in G4 (th_n_points; None keeps fit_cl's 3000). fit_cl's
# 3000-point recombination grid makes finite differences along omega_b a staircase
# (T_b Taylor-switch below z~620, biased ~1.7e-2 at 3000 points, 1.7e-3 at 10000; PR
# #59 ADR 0004), and AD itself moved 4.7e-4 between 10000 and 20000 points (job
# 27739), so omega_b uses the finest grid, for both AD and FD.
G4_TH = {"h": None, "omega_b": 20000, "omega_cdm": None}

# G4 perturbation ODE precision: 100x tighter than fit_cl (rtol 1e-4, atol 1e-6); see the
# module docstring of tests/test_cl_gradients.py. 65536 is the step cap the wobble scan
# used (ode_max_steps costs nothing unless hit: 26 s per evaluation at 8192, 16384 and
# 65536, job 28243).
G4_PT_RTOL, G4_PT_ATOL, G4_MAX_STEPS = 1e-6, 1e-8, 65536


def g4_prec(solver, th_n_points, tight=True):
    """fwd settings, the given perturbation solver, the 8000-point background (the
    800-point background-table AD noise, smsharma/clax#56, is not what limits the FD
    along h: at 8000 points the FD steps 1e-3 and 3e-3 still differ by 6.9e-4, job
    27227), optionally a denser thermodynamics grid (th_n_points; None keeps
    fit_cl's), and, if `tight`,
    the perturbation ODE tolerance 100x below fit_cl's (G4_PT_RTOL, G4_PT_ATOL;
    ode_max_steps G4_MAX_STEPS so no mode starves). The G4 tests and the lensing
    FD tests build their finite-difference precision here: tight=True for the gated AD
    and the FD stencil, tight=False (fit_cl's own tolerance and step cap) for the
    printed AD record."""
    prec = replace(fwd(FIT), pt_ode_solver=solver, bg_n_points=8000)
    if tight:
        prec = replace(prec, pt_ode_rtol=G4_PT_RTOL, pt_ode_atol=G4_PT_ATOL,
                       ode_max_steps=G4_MAX_STEPS)
    return prec if th_n_points is None else replace(prec, th_n_points=th_n_points)


def _solve(p, prec):
    bg, th = solve_background_and_thermo(p, prec)
    return bg, th, perturbations_solve(p, prec, bg, th)


def _unlensed(p, prec, bg, pt):
    return compute_cls_all_fast(pt, p, bg, l_max=CL_LMAX, n_k_fine=prec.hr_n_k_fine, prec=prec)


def F_tt(prec, base, theta):
    """F(theta) = sum_{l=30}^{1200} ln C_l^TT (unlensed)."""
    def F(x):
        p = replace(as_traced(base), **{theta: x})
        bg, th, pt = _solve(p, prec)
        return jnp.sum(jnp.log(_unlensed(p, prec, bg, pt)["tt"][CL_LMIN:CL_LMAX + 1]))
    return F


_HARMONIC_INPUTS = {}


def harmonic_inputs(name, base):
    """(params, bg, pt) at the cosmology `name`: background, thermodynamics and
    perturbation solve run once, eagerly, at production FIT (Rodas5), then reused by
    every harmonic-stage test at that cosmology."""
    if name not in _HARMONIC_INPUTS:
        p = as_traced(base)
        bg, _, pt = _solve(p, FIT)
        _HARMONIC_INPUTS[name] = (p, bg, pt)
    return _HARMONIC_INPUTS[name]


def F_harm(inputs, theta):
    """F(x) = sum_{l=30}^{1200} ln C_l^TT from compute_cls_all_fast alone: the ODE
    solve (pt, bg) is held fixed and closed over, so the derivative has no ODE
    step-sequence floor. compute_cls_all_fast reads `params` only through
    primordial_scalar_pk (n_s, ln10A_s, alpha_s, k_pivot) and `bg` only through
    conformal_age (tau_0 -> chi = tau_0 - tau -> the Bessel-table argument k*chi).
    theta = "n_s" differentiates the primordial tilt (the table and the l-scan carry
    no tangent); theta = "tau_0" differentiates through the Hermite lookup and the
    checkpointed l-scan."""
    p0, bg0, pt0 = inputs

    def F(x):
        p, bg = p0, bg0
        if theta == "n_s":
            p = replace(p0, n_s=x)
        elif theta == "tau_0":
            bg = replace(bg0, conformal_age=x)
        else:
            raise ValueError(f"F_harm: unknown theta {theta!r}")
        cl = compute_cls_all_fast(pt0, p, bg, l_max=CL_LMAX, n_k_fine=FIT.hr_n_k_fine, prec=FIT)
        return jnp.sum(jnp.log(cl["tt"][CL_LMIN:CL_LMAX + 1]))
    return F


def _cl_pp(p, prec, bg, th, pt, nonlinear):
    # F_pp and F_ltt serve the lensing tests (tests/test_cl_lensing_gradients.py);
    # the unlensed-C_l tests (tests/test_cl_gradients.py) never call them.
    # prec= is passed to compute_cl_pp, which needs it to jit or differentiate.
    pt_mpk = None if nonlinear == "none" else perturbations_solve_mpk(p, prec, bg, th)
    return compute_cl_pp(pt, p, bg, th, CL_LMAX, nonlinear=nonlinear, pt_mpk=pt_mpk, prec=prec)


def F_pp(prec, base, theta, nonlinear):
    """F(theta) = sum_{l=30}^{1200} ln C_l^phiphi."""
    def F(x):
        p = replace(as_traced(base), **{theta: x})
        bg, th, pt = _solve(p, prec)
        return jnp.sum(jnp.log(_cl_pp(p, prec, bg, th, pt, nonlinear)[CL_LMIN:CL_LMAX + 1]))
    return F


def F_ltt(prec, base, theta, nonlinear):
    """F(theta) = sum_{l=30}^{1200} ln (lensed C_l^TT)."""
    def F(x):
        p = replace(as_traced(base), **{theta: x})
        bg, th, pt = _solve(p, prec)
        cl = _unlensed(p, prec, bg, pt)
        pp = _cl_pp(p, prec, bg, th, pt, nonlinear)
        tt_l, _, _, _ = lens_cls(cl["tt"], cl["ee"], cl["te"], jnp.zeros_like(cl["tt"]), pp,
                                 l_max=CL_LMAX)
        return jnp.sum(jnp.log(tt_l[CL_LMIN:CL_LMAX + 1]))
    return F


def jvp_scalar(F, x0, jit=True):
    def g(x):
        return jax.jvp(F, (x,), (jnp.asarray(1.0),))[1]
    return float((jax.jit(g) if jit else g)(jnp.asarray(x0)))


def spread(values):
    v = np.asarray(values, dtype=float)
    return float((v.max() - v.min()) / abs(np.median(v)))


def cubic_residual(xs, ys):
    ys = np.asarray(ys, dtype=float)
    resid = ys - np.polyval(np.polyfit(xs, ys, 3), xs)
    return float(np.max(np.abs(resid)) / abs(np.median(ys)))


def fd_pair(F, x0):
    """Central FD of F at relative steps 1e-3 and 3e-3: (fd_1e-3, fd_3e-3)."""
    Fj = jax.jit(F)
    fds = []
    for rel in (1e-3, 3e-3):
        h = rel * abs(x0)
        fds.append((float(Fj(jnp.asarray(x0 + h))) - float(Fj(jnp.asarray(x0 - h)))) / (2 * h))
    return fds[0], fds[1]


def fd_agree(fd1, fd3, tol=1e-4):
    """True if the two central-FD steps agree to tol (relative to the first)."""
    return bool(abs(fd1 - fd3) <= tol * abs(fd1))


def converged_fd(F, x0):
    """Central FD at relative steps 1e-3 and 3e-3; converged if they agree to 1e-4.
    Returns (the 1e-3-step FD, converged flag); use fd_pair for both FD values.
    No test calls it now: the G4 tests (unlensed and lensed) use stencil_fd, because
    the two-step agreement test cannot be met at fit_cl (see test_g4_ad_matches_converged_fd).
    """
    fd1, fd3 = fd_pair(F, x0)
    return fd1, fd_agree(fd1, fd3)


def stencil_fd(F, x0, half_width=1e-2, n=9):
    """Derivative of F at x0 from a least-squares cubic through n jitted evaluations at
    x0 * (1 + s), s equally spaced on [-half_width, +half_width] (default 9 points:
    0, +-0.0025, +-0.005, +-0.0075, +-0.01). Returns (fd9, fd7, unc): fd9 from all n
    points, fd7 from the inner points |s| <= 0.75 half_width (7 of 9), and
    unc = |fd9 - fd7| / |fd9|, the estimate's own spread (a lower bound on its error,
    since fd7 reuses 7 of the 9 points). The fit uses a wide
    stencil because F's evaluation noise (~8e-9 relative along h at fit_cl) is
    amplified by 1/step in a two-point central difference but averaged by the fit; G4
    also tightens the ODE tolerance so that the noise itself is smaller (see
    test_g4_ad_matches_converged_fd)."""
    Fj = jax.jit(F)
    s = np.linspace(-half_width, half_width, n)
    ys = np.array([float(Fj(jnp.asarray(x0 * (1.0 + si)))) for si in s])

    def slope(mask):
        return float(np.polyfit(s[mask], ys[mask], 3)[-2]) / x0   # dF/ds at 0, then / x0

    fd9 = slope(np.ones(n, dtype=bool))
    fd7 = slope(np.abs(s) <= 0.75 * half_width + 1e-12)
    return fd9, fd7, float(abs(fd9 - fd7) / abs(fd9))
