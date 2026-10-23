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

import dataclasses
import os

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
from tests.recomb_test_utils import load_drag


PREC = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    # NOTE: intentionally 5e3, not the 5e4 floor (see PrecisionParams.th_z_max).
    # This module never runs perturbations, so the CubicSpline boundary-clip
    # bug that motivates the 5e4 floor cannot bite here. Before #55 raising it
    # to 5e4 broke the omega_b dkappa_dot/dlna gradient check at loga = -8 by 8%
    # (AD -3.346399e+02 vs FD -3.642966e+02): that was the opacity tables'
    # stop_gradient cut, whose error changed with the grid spacing. With the
    # full tangents (ADR 0004) the same check agrees with FD to 4e-10 at both
    # 5e3 and 5e4 (probe of Oct 20, 2026, job 27170); 5e3 is kept as is.
    th_n_points=10000, th_z_max=5e3,
)


_REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_data")


def _class_tau_free_streaming(name):
    """CLASS 3.3.4 tau_free_streaming [Mpc] stored with the source references
    (scripts/generate_class_sources_reference.py), or None if absent."""
    path = os.path.join(_REFERENCE_DIR, name, "sources_k0.15.npz")
    if not os.path.exists(path):
        return None
    return float(np.load(path)["tau_free_streaming"])


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

    def test_z_star(self, th):
        """``z_star``, the visibility maximum, matches CLASS's parabolic estimate of
        it (CLASS calls that epoch z_rec) on the RECFAST oracle. Until #44 the code
        took the grid point (off by up to one step) and this test's bar was 1%."""
        ref = load_drag("lcdm_fiducial")["z_rec"]
        val = float(th.z_star)
        rel_err = abs(val - ref) / ref
        # 10000 points sit between the 20000-point (1.1e-6) and 3000-point (2.9e-5)
        # worst cases of tests/test_recomb_thermo.py; the grid-point value was off by
        # up to one step (9e-4 in z here), wherever the knot fell.
        assert rel_err < 1e-4, f"z_star: {val:.4f} vs CLASS {ref:.4f}, err={rel_err:.2e}"

    def test_z_rec(self, th):
        """``z_rec`` is where the optical depth crosses one: exp(-kappa(z_rec)) =
        e^-1. (CLASS has no such derived quantity; its "z_rec" is the visibility
        maximum, which this module calls z_star. Until #44 this test compared the
        two definitions with a 2% bar that hid their 0.4% difference.)"""
        val = float(th.exp_m_kappa_of_loga.evaluate(jnp.log(1.0 / (1.0 + th.z_rec))))
        rel_err = abs(val / np.exp(-1.0) - 1.0)
        assert rel_err < 1e-4, f"exp(-kappa(z_rec)) = {val:.6f} vs e^-1, err={rel_err:.2e}"

    def test_drag_epoch(self, th):
        """z_d, tau_d, rs_d (baryon drag, kappa_b = 1) match CLASS RECFAST (#44)."""
        ref = load_drag("lcdm_fiducial")
        for q in ("z_d", "tau_d", "rs_d"):
            val = float(getattr(th, q))
            rel_err = abs(val - ref[q]) / ref[q]
            # 10000 points: between the 20000-point (1.5e-6) and 3000-point (3.6e-5)
            # worst cases of tests/test_recomb_thermo.py; the fit was 1.3e-4 off here.
            assert rel_err < 1e-4, f"{q}: {val:.6f} vs CLASS {ref[q]:.6f}, err={rel_err:.2e}"

    def test_tau_free_streaming(self, th, bg):
        """``tau_free_streaming`` matches CLASS's definition to 0.05% (0.18 Mpc).

        Floor: the linear-in-log-a interpolation of the crossing (~0.01 Mpc on
        this 10000-point grid) and CLASS's 0.023 Mpc table-point ambiguity;
        measured 0.0003% at all five LCDM grid points after #53 (job 27002).

        CLASS (thermodynamics.c, ``tau_free_streaming``) switches the photon
        radiation-streaming approximation on at the first conformal time after
        recombination where the Thomson mean free path exceeds
        ``radiation_streaming_trigger_tau_c_over_tau`` = 5 conformal times,
        tau_c/tau = 1/(kappa' tau) >= 5. Reference: CLASS 3.3.4, RECFAST,
        Y_He = 0.2454006, fiducial with the 0.06 eV neutrino, interpolated
        between table points (scripts/generate_class_sources_reference.py,
        stored with the source references): 364.767 Mpc (z = 760), where the
        visibility function is 2.4% of its peak. The legacy clax criterion
        kappa'/(aH) < 5 corresponds to 296 Mpc (visibility at 69% of peak),
        see docs/highl_residual_2026-10-12.md (branch exp/highl-investigation).
        """
        ref = float(_class_tau_free_streaming('lcdm_fiducial'))
        val = float(th.tau_free_streaming)
        rel_err = abs(val - ref) / ref
        assert np.isfinite(val)
        assert val > float(th.tau_star), f"tau_free_streaming {val:.2f} must follow the visibility peak {float(th.tau_star):.2f}"
        assert rel_err < 0.0005, f"tau_free_streaming: {val:.3f} vs CLASS {ref:.3f} Mpc, err={rel_err:.3%}"

    def test_tau_free_streaming_no_gradient(self):
        """``tau_free_streaming`` carries no gradient: the switch location is
        held fixed under AD (stop_gradient), like CLASS's table point. Without
        it two partials would reach it: the one through the opacity table's
        n_H_0 rescale (+698 Mpc per unit omega_b against -545 by finite
        differences, the wrong sign) and the background-time partial through
        tau(z) (incomplete on its own); the dropped term is <= 3e-5 on
        d ln C_l / d ln theta. Forward mode (direct adjoint, as in
        tests/test_pk_forward_mode.py: the production checkpointed adjoint is
        a custom_vjp that jax.jvp cannot cross) and reverse mode (production
        adjoint, what inference runs) along omega_b must both return exactly
        zero. Exempt from the multi-cosmology rule: a stop_gradient is a
        structural property, identical at every cosmology."""
        def f(omega_b, prec):
            params = CosmoParams(omega_b=omega_b)
            return thermodynamics_solve(params, prec, background_solve(params, prec)).tau_free_streaming
        omega_b0 = jnp.asarray(CosmoParams().omega_b, dtype=jnp.float64)
        prec_fwd = dataclasses.replace(PREC, ode_adjoint="direct")
        val, dot = jax.jvp(lambda ob: f(ob, prec_fwd), (omega_b0,), (jnp.ones_like(omega_b0),))
        assert float(val) > 0.0
        assert float(dot) == 0.0, f"jvp: d tau_free_streaming/d omega_b = {float(dot):+.3e}, expected exactly 0"
        grad = float(jax.grad(lambda ob: f(ob, PREC))(omega_b0))
        assert grad == 0.0, f"grad: d tau_free_streaming/d omega_b = {grad:+.3e}, expected exactly 0"

    def test_tau_free_streaming_grid(self, lcdm_cosmology):
        """Multi-cosmology: ``tau_free_streaming`` within 0.05% of CLASS at every
        LCDM grid point (fiducial only under --fast), and it moves with the
        cosmology in the same direction as in CLASS (omega_cdm_low: later by
        19 Mpc; omega_b_high: earlier by 2.4 Mpc; h and n_s: unchanged)."""
        name, params = lcdm_cosmology
        ref = _class_tau_free_streaming(name)
        if ref is None:
            pytest.skip(f"no CLASS source/thermo reference for {name}")
        bg_i = background_solve(params, PREC)
        th_i = thermodynamics_solve(params, PREC, bg_i)
        val = float(th_i.tau_free_streaming)
        rel_err = abs(val - ref) / ref
        print(f"tau_free_streaming {name}: clax {val:.3f} vs CLASS {ref:.3f} Mpc ({100 * (val / ref - 1):+.3f}%)")
        assert rel_err < 0.0005, f"{name}: tau_free_streaming {val:.3f} vs CLASS {ref:.3f} Mpc, err={rel_err:.3%}"
        ref_fid = _class_tau_free_streaming('lcdm_fiducial')
        if abs(ref - ref_fid) > 1.0:
            bg_f = background_solve(CosmoParams(), PREC)
            val_fid = float(thermodynamics_solve(CosmoParams(), PREC, bg_f).tau_free_streaming)
            assert np.sign(val - val_fid) == np.sign(ref - ref_fid), (
                f"{name}: tau_free_streaming moves by {val - val_fid:+.2f} Mpc vs CLASS {ref - ref_fid:+.2f}")


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
    # (RecursiveCheckpointAdjoint is reverse-mode only).
    prec = dataclasses.replace(PREC, ode_adjoint="direct")

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

    # Reverse-mode AD of the opacity tables against finite differences at fixed
    # ln a (smsharma/clax#55). loga = -8 is the fully ionised plasma (z ~ 2980,
    # x_e constant), -7 is recombination (z ~ 1095), -6 the freeze-out tail
    # (z ~ 402) and -5 the residual-ionisation plateau (z ~ 147). Until #55 the
    # tables carried a stop_gradient with an n_H_0 rescale (d127731), exact
    # only where x_e is constant: along omega_b the recombination-era gradient
    # was 10-30% off and along omega_cdm it was exactly zero, so only the
    # loga = -8, omega_b case was tested.
    _OPACITY_SAMPLES = [(-8.0, "omega_b"), (-7.0, "omega_b"), (-6.0, "omega_b"),
                        (-7.0, "omega_cdm"), (-6.0, "omega_cdm")]
    # exp(-kappa) along omega_b at loga = -6 (z ~ 402), where its omega_b
    # derivative passes through zero (x_e n_H goes as H in the freeze-out tail and
    # the reionisation depth is pinned by tau_reio): with the T_b update's former
    # Taylor switch at f_e < 1e-3 the single-step finite difference there was set
    # by the switch's grid-scale jumps (FD -9.86e-5 against AD -6.87e-5 on 10000
    # points, job 27172, Oct 20, 2026), so the sample sat at loga = -5. It is at
    # -6 again, with -5 kept.
    _EXP_M_KAPPA_SAMPLES = [(-8.0, "omega_b"), (-7.0, "omega_b"), (-6.0, "omega_b"),
                            (-5.0, "omega_b"), (-7.0, "omega_cdm"), (-6.0, "omega_cdm")]

    # Derived epochs (smsharma/clax#44): rs_d is the BAO ruler and z_d, tau_star
    # feed the EPT damping scale and the perturbation save grid. Until #44 rs_d
    # came from a fitting formula and tau_star was a grid-point value, piecewise
    # constant in the parameters. Bars: twice the worst measured residual at this
    # precision block (4.6e-5, rs_d and z_d along h, where the epochs move only
    # through Lambda; 3.2e-6 or better along omega_b and omega_cdm; job 28147,
    # Oct 23, 2026), ceiled to one figure.
    @pytest.mark.parametrize("param_name", ["omega_b", "omega_cdm", "h"])
    def test_rs_d_gradient_matches_fd(self, param_name):
        ad, fd = _thermo_ad_fd_pair(param_name, lambda th: th.rs_d)
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 1e-4, f"rs_d grad {param_name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2e}"

    @pytest.mark.parametrize("param_name", ["omega_b", "omega_cdm", "h"])
    def test_tau_star_gradient_matches_fd(self, param_name):
        ad, fd = _thermo_ad_fd_pair(param_name, lambda th: th.tau_star)
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 1e-4, f"tau_star grad {param_name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2e}"

    @pytest.mark.parametrize("param_name", ["omega_b", "omega_cdm"])
    def test_z_d_gradient_matches_fd(self, param_name):
        ad, fd = _thermo_ad_fd_pair(param_name, lambda th: th.z_d)
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 1e-4, f"z_d grad {param_name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2e}"

    @pytest.mark.parametrize("loga,param_name", _OPACITY_SAMPLES)
    def test_opacity_logderivative_gradient_matches_fd(self, loga, param_name):
        """Stored ``dκ̇/dloga`` table is differentiable through recombination."""
        ad, fd = _thermo_ad_fd_pair(
            param_name, lambda th: th.dkappa_dot_dloga_of_loga.evaluate(jnp.array(loga))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"dkappa_dot_dloga(loga={loga}) grad {param_name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )

    @pytest.mark.parametrize("loga,param_name", _OPACITY_SAMPLES)
    def test_kappa_dot_gradient_matches_fd(self, loga, param_name):
        """AD gradient of kappa_dot_of_loga matches FD at fixed ln a, finite."""
        ad, fd = _thermo_ad_fd_pair(
            param_name, lambda th: th.kappa_dot_of_loga.evaluate(jnp.array(loga))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"kappa_dot(loga={loga}) grad {param_name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )

    @pytest.mark.parametrize("loga,param_name", _EXP_M_KAPPA_SAMPLES)
    def test_exp_m_kappa_gradient_matches_fd(self, loga, param_name):
        """AD gradient of exp_m_kappa_of_loga matches FD at fixed ln a."""
        ad, fd = _thermo_ad_fd_pair(
            param_name, lambda th: th.exp_m_kappa_of_loga.evaluate(jnp.array(loga))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"exp_m_kappa(loga={loga}) grad {param_name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
        )

    @pytest.mark.parametrize("loga,param_name", _OPACITY_SAMPLES)
    def test_g_gradient_matches_fd(self, loga, param_name):
        """AD gradient of g_of_loga (visibility) matches FD at fixed ln a."""
        ad, fd = _thermo_ad_fd_pair(
            param_name, lambda th: th.g_of_loga.evaluate(jnp.array(loga))
        )
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"g(loga={loga}) grad {param_name}: AD={ad:.6e} FD={fd:.6e} rel={rel:.2%}"
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
