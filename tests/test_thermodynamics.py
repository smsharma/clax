"""Tests thermodynamics-layer forward behavior and targeted gradient contracts.

Contract:
- Thermodynamics quantities and recombination-era functions match the documented CLASS-derived references.
- The repaired reionization and opacity-derivative AD paths remain consistent with finite differences.

Scope:
- Covers ``z_star``, ``z_rec``, ionization history, visibility behavior, and the repaired thermodynamics gradient subcontracts.
- Excludes background and perturbation-layer contracts owned elsewhere.

Notes:
- These tests use CLASS-generated reference data for forward assertions plus narrow FD spot checks for the repaired thermodynamics AD paths.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import _compute_tca_criterion
from clax.params import CosmoParams, PrecisionParams
from tests.pk_test_utils import PK_GRAD_PARAM_STEPS


PREC = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    # NOTE: intentionally 5e3, not the 5e4 floor (see PrecisionParams.th_z_max).
    # This module never runs perturbations, so the CubicSpline boundary-clip
    # bug that motivates the 5e4 floor cannot bite here. Raising to 5e4 was
    # tried and reverted: it breaks
    # TestThermoGradients::test_opacity_logderivative_gradient_matches_fd_for_omega_b
    # (rel err 8.14% vs 1% tolerance, dkappa_dot_dloga AD=-3.346399e+02 vs
    # FD=-3.642966e+02 at loga=-8) by shifting the th_n_points=10000 grid
    # spacing near recombination. Do not loosen the tolerance to compensate.
    th_n_points=10000, th_z_max=5e3,
)


@pytest.fixture(scope="module")
def bg():
    """Compute the fiducial background state once for this module."""
    return background_solve(CosmoParams(), PREC)


@pytest.fixture(scope="module")
def th(bg):
    """Compute the fiducial thermodynamics state once for this module."""
    return thermodynamics_solve(CosmoParams(), PREC, bg)


class TestThermoScalars:
    """Tests thermodynamics scalar quantities."""

    def test_z_star(self, th, lcdm_derived):
        """``z_star`` matches CLASS; expects <1% relative error."""
        ref = lcdm_derived['z_star']
        val = float(th.z_star)
        rel_err = abs(val - ref) / ref
        assert rel_err < 0.01, f"z_star: {val:.1f} vs CLASS {ref:.1f}, err={rel_err:.2%}"

    def test_z_rec(self, th, lcdm_derived):
        """``z_rec`` matches CLASS; expects <2% relative error."""
        ref = lcdm_derived['z_rec']
        val = float(th.z_rec)
        rel_err = abs(val - ref) / ref
        assert rel_err < 0.02, f"z_rec: {val:.1f} vs CLASS {ref:.1f}, err={rel_err:.2%}"


class TestIonizationFraction:
    """Tests ionization-fraction behavior."""

    def test_xe_high_z(self, th):
        """``x_e(z=3000)`` is near full ionization; expects a value near 1.08."""
        xe = float(th.xe_of_loga.evaluate(jnp.log(jnp.array(1.0/3001.0))))
        assert abs(xe - 1.08) < 0.05, f"xe(z=3000) = {xe:.4f}, expected ~1.08"

    def test_xe_recombination(self, th, lcdm_thermo_ref):
        """``x_e`` during recombination matches CLASS; expects <30% relative error at the probe redshifts."""
        ref_z = lcdm_thermo_ref['z']
        ref_xe = lcdm_thermo_ref['x_e']

        # Test at specific recombination redshifts
        for z_test in [1200.0, 1100.0, 1000.0, 800.0]:
            idx = np.argmin(np.abs(ref_z - z_test))
            xe_class = ref_xe[idx]
            la = float(jnp.log(1.0 / (1.0 + z_test)))
            xe_us = float(th.xe_of_loga.evaluate(jnp.array(la)))

            if xe_class > 0.001:
                rel_err = abs(xe_us - xe_class) / xe_class
                assert rel_err < 0.30, (
                    f"xe(z={z_test:.0f}): us={xe_us:.6f} CLASS={xe_class:.6f} "
                    f"err={rel_err:.1%}"
                )

    def test_xe_reionization(self, th):
        """``x_e(z=0)`` is fully reionized; expects a value near 1.16."""
        xe = float(th.xe_of_loga.evaluate(jnp.array(0.0)))
        assert abs(xe - 1.16) < 0.02, f"xe(z=0) = {xe:.4f}, expected ~1.16"


class TestVisibility:
    """Tests visibility-function behavior."""

    def test_visibility_peaks_at_recombination(self, th):
        """The visibility function peaks near recombination; expects ``z_star`` close to 1090."""
        assert abs(float(th.z_star) - 1090) < 30, f"z_star = {float(th.z_star):.1f}"


class TestEarlyTableExtension:
    """``th_z_max`` is a NUMERICAL knob, not a physics one (fix/thermo-early-extension).

    ``CubicSpline.evaluate`` clips below its first knot (clax/interpolation.py:67).
    The thermodynamics table used to start at ``a_start = 1/(1+th_z_max)`` (still
    ``tau_grid[0]`` on the fixed-up table -- see the "Extend the table..." comment
    in ``thermodynamics_solve``), which for small ``th_z_max`` (e.g. 5e3 -> table
    starts at tau=80.7 Mpc) is far LATER than ``tau_ini`` used by perturbations.py.
    Below the table start, kappa_dot used to FREEZE at the boundary value instead
    of scaling as a^-2, corrupting the TCA criterion at tau_ini (RED on main:
    is_tca=0.000000 for all k, i.e. the fully-ionized early-radiation-domination
    plasma looks free-streaming). GREEN after prepending an analytic a^-2
    extension covering the gap.

    Fixture ``th`` above uses PREC (th_z_max=5e3), the exact regime from the bug
    report, so no extra fixture is needed.
    """

    # Old table start at th_z_max=5e3: a_start = 1/(1+5e3), loga_start ~ -8.5174.
    _LOGA_START_OLD = float(jnp.log(1.0 / (1.0 + 5e3)))

    def test_kappa_dot_scales_as_a_minus_2_below_old_table_start(self, th):
        """``kappa_dot(a) * a**2`` is constant well below the pre-fix table start.

        Physically kappa_dot = x_e * n_H_0 * (1+z)^2 * sigma_T * c/Mpc with x_e
        exactly frozen at full ionization in this regime, so kappa_dot ~ a^-2
        is exact (not approximate) -- a tight tolerance is appropriate. Before
        the fix, the spline clips to the frozen boundary value here so
        kappa_dot*a^2 varies by orders of magnitude instead of being constant;
        1e-6 is >100x looser than the ~9e-8 spline-interpolation error the
        prepended analytic grid achieves (verified numerically at 8.6e-8).
        """
        loga_test = jnp.linspace(
            self._LOGA_START_OLD - 5.0, self._LOGA_START_OLD - 0.5, 20
        )
        kappa_dot = jax.vmap(th.kappa_dot_of_loga.evaluate)(loga_test)
        a_test = jnp.exp(loga_test)
        kappa_dot_a2 = kappa_dot * a_test**2

        rel_spread = float(
            (kappa_dot_a2.max() - kappa_dot_a2.min()) / kappa_dot_a2.mean()
        )
        assert rel_spread < 1e-6, (
            f"kappa_dot*a^2 relative spread = {rel_spread:.3e} "
            f"(values: {np.asarray(kappa_dot_a2)})"
        )

    @pytest.mark.parametrize("k", [0.01, 0.05, 0.1])
    def test_is_tca_near_one_at_tau_ini(self, bg, th, k):
        """``is_tca(tau_ini) ~ 1``: the plasma is tightly coupled at the start
        of scalar-mode integration (early radiation domination, fully ionized).

        ``tau_ini`` mirrors the ``compute_pk`` single-k path in
        ``_matter_delta_m_single_k_impl`` (clax/perturbations.py):
        ``tau_ini = min(0.5, 0.01/k)``. Before the fix this is 0.000000 for
        every k in this range (the smoking gun in the bug report) because
        kappa_dot is frozen far below its true early-time value.
        """
        tau_ini = min(0.5, 0.01 / k)
        loga_ini = bg.loga_of_tau.evaluate(jnp.array(tau_ini))
        a_ini = jnp.exp(loga_ini)
        H_ini = bg.H_of_loga.evaluate(loga_ini)
        a_prime_over_a = a_ini * H_ini
        kappa_dot_ini = th.kappa_dot_of_loga.evaluate(loga_ini)

        is_tca, _tau_c = _compute_tca_criterion(kappa_dot_ini, a_prime_over_a, k)
        assert float(is_tca) > 0.99, (
            f"is_tca(tau_ini={tau_ini}, k={k}) = {float(is_tca):.6f}, expected ~1 "
            f"(kappa_dot={float(kappa_dot_ini):.4e}, aH={float(a_prime_over_a):.4e})"
        )


def _thermo_ad_fd_pair(param_name, quantity_fn):
    """Return ``(ad, fd)`` for one scalar thermodynamics quantity."""
    params = CosmoParams()
    step = PK_GRAD_PARAM_STEPS[param_name]
    x0 = getattr(params, param_name)

    def wrapped(x):
        varied = params.replace(**{param_name: x})
        bg = background_solve(varied, PREC)
        th = thermodynamics_solve(varied, PREC, bg)
        return quantity_fn(th)

    ad = float(jax.grad(wrapped)(x0))
    fd = float((wrapped(x0 + step) - wrapped(x0 - step)) / (2.0 * step))
    return ad, fd


def _thermo_jvp_fd_pair(param_name, quantity_fn, eps=1e-3):
    """Return (jax.jvp tangent, centred-FD gradient) of quantity_fn w.r.t. param_name."""
    import dataclasses
    # Forward-mode needs a direct adjoint through background_solve
    # (RecursiveCheckpointAdjoint is reverse-mode only) and the native
    # thermo grad path (th_grad_mode="stable" is a custom_vjp, which JAX
    # forbids under jax.jvp -- see PrecisionParams.th_grad_mode).
    prec = dataclasses.replace(PREC, ode_adjoint="direct",
                               th_grad_mode="native")

    base = CosmoParams()
    p0 = float(getattr(base, param_name))

    def forward(pval):
        p = dataclasses.replace(base, **{param_name: pval})
        bg_ = background_solve(p, prec)
        th_ = thermodynamics_solve(p, prec, bg_)
        return quantity_fn(th_)

    _, tangent = jax.jvp(forward, (jnp.array(p0),), (jnp.array(1.0),))

    p_hi = dataclasses.replace(base, **{param_name: p0 * (1 + eps)})
    p_lo = dataclasses.replace(base, **{param_name: p0 * (1 - eps)})
    bg_hi = background_solve(p_hi, prec)
    bg_lo = background_solve(p_lo, prec)
    th_hi = thermodynamics_solve(p_hi, prec, bg_hi)
    th_lo = thermodynamics_solve(p_lo, prec, bg_lo)
    fd = float((quantity_fn(th_hi) - quantity_fn(th_lo)) / (2 * p0 * eps))

    return float(tangent), fd


class TestThermoGradients:
    """Tests thermodynamics gradient behavior at the repaired opacity branches."""

    @pytest.mark.parametrize("param_name", ["h", "omega_b"])
    def test_reionization_gradients_match_fd(self, param_name):
        """``z_reio`` and late-time ``x_e`` gradients match finite differences."""
        z_ad, z_fd = _thermo_ad_fd_pair(param_name, lambda th: th.z_reio)
        z_rel = abs(z_ad - z_fd) / (abs(z_fd) + 1e-30)
        assert z_rel < 0.01, (
            f"z_reio grad {param_name}: AD={z_ad:.6e} FD={z_fd:.6e} rel={z_rel:.2%}"
        )

        xe_ad, xe_fd = _thermo_ad_fd_pair(
            param_name, lambda th: th.xe_of_loga.evaluate(jnp.array(-2.0))
        )
        xe_rel = abs(xe_ad - xe_fd) / (abs(xe_fd) + 1e-30)
        assert xe_rel < 0.01, (
            f"xe(loga=-2) grad {param_name}: AD={xe_ad:.6e} FD={xe_fd:.6e} rel={xe_rel:.2%}"
        )

    def test_opacity_logderivative_gradient_matches_fd_for_omega_b(self):
        """Stored ``dκ̇/dloga`` table remains differentiable through recombination."""
        ad, fd = _thermo_ad_fd_pair(
            "omega_b", lambda th: th.dkappa_dot_dloga_of_loga.evaluate(jnp.array(-8.0))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"dkappa_dot_dloga(loga=-8) grad omega_b: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )

    def test_kappa_dot_gradient_matches_fd_for_omega_b(self):
        """AD gradient of kappa_dot_of_loga must not blow up from Friedmann scan."""
        ad, fd = _thermo_ad_fd_pair(
            "omega_b", lambda th: th.kappa_dot_of_loga.evaluate(jnp.array(-8.0))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"kappa_dot(loga=-8) grad omega_b: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )

    def test_exp_m_kappa_gradient_matches_fd_for_omega_b(self):
        """AD gradient of exp_m_kappa_of_loga must match FD at loga=-8."""
        ad, fd = _thermo_ad_fd_pair(
            "omega_b", lambda th: th.exp_m_kappa_of_loga.evaluate(jnp.array(-8.0))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.05, (
            f"exp_m_kappa(loga=-8) grad omega_b: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )

    def test_g_gradient_matches_fd_for_omega_b(self):
        """AD gradient of g_of_loga (visibility) must match FD at loga=-8.

        Tests at loga=-8 (early universe, x_e~const, kappa~0) where the
        n_H_0 rescaling approximation is exact. Near recombination (loga~-7)
        the approximation has 10-30% error from d(xe)/d(omega_b), but remains
        finite versus the current 10^8x blowup.
        """
        ad, fd = _thermo_ad_fd_pair(
            "omega_b", lambda th: th.g_of_loga.evaluate(jnp.array(-8.0))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"g(loga=-8) grad omega_b: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )


def test_find_z_reio_forward_mode_matches_fd():
    """jax.jvp through z_reio(h) is finite and matches centred FD to <1%.

    RED before converting _find_z_reio from custom_vjp to custom_jvp
    (raises TypeError: can't apply forward-mode autodiff to a custom_vjp function).
    GREEN after conversion.
    """
    import dataclasses

    PREC_JVP = PrecisionParams(
        bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
        th_n_points=10000, th_z_max=5e4,  # 5e4 floor: see PrecisionParams.th_z_max
        ode_adjoint="direct",
        th_grad_mode="native",  # jax.jvp cannot cross the "stable" custom_vjp
    )
    params = CosmoParams()

    def z_reio_of_h(h):
        p = dataclasses.replace(params, h=h)
        bg_ = background_solve(p, PREC_JVP)
        th_ = thermodynamics_solve(p, PREC_JVP, bg_)
        return th_.z_reio

    # Forward-mode AD
    primal, tangent = jax.jvp(z_reio_of_h, (params.h,), (jnp.asarray(1.0),))
    primal.block_until_ready()

    assert jnp.isfinite(tangent), f"jvp returned non-finite tangent: {tangent}"

    # Centred FD for ground truth
    eps = 1e-3
    fd = (z_reio_of_h(params.h + eps) - z_reio_of_h(params.h - eps)) / (2 * eps)
    rel = abs(float(tangent) - float(fd)) / (abs(float(fd)) + 1e-30)
    assert rel < 0.01, (
        f"jvp(z_reio, h)={float(tangent):.6e}  FD={float(fd):.6e}  rel={rel:.2%}"
    )




class TestThermoForwardModeAD:
    """Forward-mode AD (jax.jvp) through kappa_dot, exp_m_kappa, g splines.

    Complements TestThermoGradients (reverse-mode). Confirms that:
    1. jax.jvp runs without TypeError through the n_H_0-rescaled splines
    2. Tangents are finite (not 10^8x blown up)
    3. Tangents match centred FD to < 1% at loga=-8 (where x_e~const, rescaling exact)

    Both prerequisites (PR #21: _find_z_reio custom_jvp + direct-adjoint
    background_solve) are on main as of 2026-08-22, so these run un-marked.
    """

    def test_kappa_dot_forward_mode_matches_fd(self):
        """jax.jvp(kappa_dot, omega_b) is finite and matches FD to <1%."""
        jvp_val, fd = _thermo_jvp_fd_pair(
            "omega_b", lambda th: th.kappa_dot_of_loga.evaluate(jnp.array(-8.0))
        )
        assert jnp.isfinite(jvp_val), f"jvp(kappa_dot) non-finite: {jvp_val}"
        rel = abs(jvp_val - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, f"kappa_dot jvp={jvp_val:.6e} FD={fd:.6e} rel={rel:.2%}"

    def test_exp_m_kappa_forward_mode_matches_fd(self):
        """jax.jvp(exp_m_kappa, omega_b) is finite and matches FD to <5%."""
        jvp_val, fd = _thermo_jvp_fd_pair(
            "omega_b", lambda th: th.exp_m_kappa_of_loga.evaluate(jnp.array(-8.0))
        )
        assert jnp.isfinite(jvp_val), f"jvp(exp_m_kappa) non-finite: {jvp_val}"
        rel = abs(jvp_val - fd) / (abs(fd) + 1e-30)
        assert rel < 0.05, f"exp_m_kappa jvp={jvp_val:.6e} FD={fd:.6e} rel={rel:.2%}"

    def test_g_forward_mode_matches_fd(self):
        """jax.jvp(g, omega_b) is finite and matches FD to <1%."""
        jvp_val, fd = _thermo_jvp_fd_pair(
            "omega_b", lambda th: th.g_of_loga.evaluate(jnp.array(-8.0))
        )
        assert jnp.isfinite(jvp_val), f"jvp(g) non-finite: {jvp_val}"
        rel = abs(jvp_val - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, f"g jvp={jvp_val:.6e} FD={fd:.6e} rel={rel:.2%}"
