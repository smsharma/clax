"""Tests for the public ``compute_cl_pp`` (source-Limber kernel).

The single public entry point is ``clax.lensing.compute_cl_pp(... nonlinear=...)``.
This file covers:

- Shape, sign, monotonicity contracts
- Accuracy of ``nonlinear="none"`` against an external CLASS reference
- Cross-check of ``nonlinear="none"`` against ``_compute_cl_pp_full_bessel``
  (the slow, dense-Bessel oracle kept private for tests)
- Smoke tests for ``nonlinear="halofit"`` (positivity, NL > linear at high l)
- JIT compilation and ``jax.grad`` differentiability

Accuracy of the Halofit-corrected C_l^pp against an external CLASS Halofit
reference lives in ``tests/test_clpp_halofit_ratio.py``.
"""

import os
import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dataclasses import replace as _replace


def _pipeline_prec():
    """The precision of the ``pipeline`` fixture (k_max=5 so Halofit can converge)."""
    from clax import PrecisionParams
    return _replace(PrecisionParams.fast_cl(), pt_k_max_cl=5.0, pt_k_chunk_size=20)


@pytest.fixture(scope="module")
def pipeline():
    """Run pipeline once for all tests (k_max=5 so Halofit can converge)."""
    from clax import CosmoParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve, perturbations_solve_mpk

    prec = _pipeline_prec()
    params = CosmoParams()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    pt_mpk = perturbations_solve_mpk(params, prec, bg, th)   # reaches today (#42)
    return params, bg, th, pt, pt_mpk


@pytest.fixture(scope="module")
def class_reference_linear():
    """CLASS linear C_l^pp reference (matching default CosmoParams)."""
    try:
        from classy import Class
    except ImportError:
        pytest.skip("CLASS Python wrapper not available")

    cosmo = Class()
    cosmo.set({
        'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.052,
        'omega_b': 0.02237, 'omega_cdm': 0.12, 'h': 0.6736,
        'YHe': 0.2425, 'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
        'output': 'lCl,tCl', 'lensing': 'Yes',
        'l_switch_limber': 9, 'non linear': 'none',
    })
    cosmo.compute()
    pp = cosmo.raw_cl(2500)['pp']
    cosmo.struct_cleanup()
    return pp


# -----------------------------------------------------------------------------
# Contract: shape, signature, sign
# -----------------------------------------------------------------------------

class TestContract:

    def test_import(self):
        """Public ``compute_cl_pp`` is importable from clax and clax.lensing."""
        from clax import compute_cl_pp  # noqa: F401
        from clax.lensing import compute_cl_pp  # noqa: F401

    def test_returns_correct_shape(self, pipeline):
        """Returns array of shape ``(l_max+1,)`` with l=0,1 zeroed."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        cl = compute_cl_pp(pt, params, bg, th, l_max=100)
        assert cl.shape == (101,), f"expected (101,), got {cl.shape}"
        assert float(cl[0]) == 0.0
        assert float(cl[1]) == 0.0
        assert float(cl[2]) > 0.0

    def test_unknown_nonlinear_raises(self, pipeline):
        """Unknown ``nonlinear`` value is rejected with ValueError."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        with pytest.raises(ValueError, match="unknown nonlinear"):
            compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear="bogus")

    def test_ept_accepted(self, pipeline):
        """``nonlinear='ept'`` is now an accepted value (no ValueError)."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        cl = compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear="ept", pt_mpk=pt_mpk)
        assert cl.shape == (11,)
        assert float(cl[2]) > 0


# -----------------------------------------------------------------------------
# Linear (nonlinear="none") accuracy
# -----------------------------------------------------------------------------

class TestLinearAccuracy:
    """Accuracy of the default (linear) source-Limber kernel."""

    def test_matches_class_at_low_l(self, pipeline, class_reference_linear):
        """Matches CLASS to <3% for l in {100, 200, 500}."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        cl = np.array(compute_cl_pp(pt, params, bg, th, l_max=500))

        for l in [100, 200, 500]:
            ratio = cl[l] / class_reference_linear[l]
            err = abs(ratio - 1.0)
            print(f"  l={l}: clax/CLASS = {ratio:.4f} ({err:.2%})")
            assert err < 0.03, f"l={l}: {err:.2%} error exceeds 3%"

    def test_matches_class_at_medium_l(self, pipeline, class_reference_linear):
        """Matches CLASS to <5% for l = 1000."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        cl = np.array(compute_cl_pp(pt, params, bg, th, l_max=1000))

        ratio = cl[1000] / class_reference_linear[1000]
        err = abs(ratio - 1.0)
        print(f"  l=1000: clax/CLASS = {ratio:.4f} ({err:.2%})")
        assert err < 0.05, f"l=1000: {err:.2%} error exceeds 5%"


class TestCrossImplAgreement:
    """Cross-check the public source-Limber path against the slow Bessel oracle.

    ``_compute_cl_pp_full_bessel`` uses ``clax.bessel.spherical_jl``, which
    relies on upward Bessel recurrence. That recurrence is reliable at low l
    (l <= ~100) but becomes unstable in the classically-forbidden region
    (x < 0.7l) at high l, so the oracle is NOT a trusted reference at
    l >= 200. The Limber approximation, in contrast, becomes more accurate
    as l increases — CLASS uses ``l_switch_limber`` between 9 and 40 for
    the lensing path because Limber is the correct answer there.

    This test therefore restricts the cross-check to l <= 100. At higher l,
    the linear-accuracy test against an external CLASS reference is the
    right verification.
    """

    def test_source_limber_vs_full_bessel_low_l(self, pipeline):
        """``compute_cl_pp(nonlinear="none")`` matches the full-Bessel oracle
        at low l where the oracle's upward Bessel recurrence is stable."""
        from clax.lensing import compute_cl_pp, _compute_cl_pp_full_bessel
        params, bg, th, pt, pt_mpk = pipeline

        l_probe = jnp.array([10, 20, 50, 100], dtype=jnp.float64)

        cl_public = np.array(compute_cl_pp(pt, params, bg, th, l_max=100))
        cl_oracle = np.array(
            _compute_cl_pp_full_bessel(pt, params, bg, th, l_probe))

        tols = {10: 0.05, 20: 0.03, 50: 0.005, 100: 0.005}
        for i, l_val in enumerate([10, 20, 50, 100]):
            ratio = cl_public[l_val] / cl_oracle[i]
            err = abs(ratio - 1.0)
            print(f"  l={l_val}: source_limber/full_bessel = {ratio:.4f} ({err:.2%})")
            assert err < tols[l_val], (
                f"l={l_val}: {err:.2%} exceeds tolerance {tols[l_val]:.1%}")


# -----------------------------------------------------------------------------
# Halofit smoke tests
# -----------------------------------------------------------------------------

class TestHalofit:
    """Smoke tests for ``nonlinear='halofit'``.

    Quantitative cross-check against a CLASS Halofit reference lives in
    ``tests/test_clpp_halofit_ratio.py``.
    """

    @pytest.fixture(scope="class")
    def cl_pair(self, pipeline):
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        cl_lin = np.array(compute_cl_pp(pt, params, bg, th, l_max=2000,
                                         nonlinear="none"))
        cl_nl = np.array(compute_cl_pp(pt, params, bg, th, l_max=2000,
                                        nonlinear="halofit", pt_mpk=pt_mpk))
        return cl_lin, cl_nl

    def test_positive(self, cl_pair):
        """Halofit C_l^pp is positive."""
        _, cl_nl = cl_pair
        assert np.all(cl_nl[2:] > 0), "C_l^pp Halofit must be positive for l>=2"

    def test_finite(self, cl_pair):
        """Halofit C_l^pp is finite."""
        _, cl_nl = cl_pair
        assert np.all(np.isfinite(cl_nl)), "C_l^pp Halofit has non-finite entries"

    def test_nl_boost_at_high_l(self, cl_pair):
        """NL/linear ratio is >1 at l>=500 (nonlinear power exceeds linear)."""
        cl_lin, cl_nl = cl_pair
        for l_val in [500, 1000, 1500, 2000]:
            ratio = cl_nl[l_val] / cl_lin[l_val]
            print(f"  l={l_val}: NL/lin = {ratio:.4f}")
            assert ratio > 1.005, (
                f"l={l_val}: NL/lin={ratio:.4f}, expected > 1.005")

    def test_no_boost_at_low_l(self, cl_pair):
        """NL/linear ratio is ~1 at l<=50 (Halofit irrelevant on largest scales)."""
        cl_lin, cl_nl = cl_pair
        for l_val in [10, 30, 50]:
            ratio = cl_nl[l_val] / cl_lin[l_val]
            print(f"  l={l_val}: NL/lin = {ratio:.4f}")
            assert abs(ratio - 1.0) < 0.02, (
                f"l={l_val}: NL/lin={ratio:.4f}, expected ~1 at low l")


class TestEPT:
    """Smoke tests for ``nonlinear='ept'`` (clax-pt one-loop EFT injection).

    The EPT path runs one ``compute_ept_from_clax`` at z=0 and growth-
    rescales to other redshifts via leading-order D^2 scaling. Quantitative
    accuracy vs CLASS-PT is documented in the EPT module.
    """

    @pytest.fixture(scope="class")
    def cl_pair(self, pipeline):
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        cl_lin = np.array(compute_cl_pp(pt, params, bg, th, l_max=2000,
                                         nonlinear="none"))
        cl_nl = np.array(compute_cl_pp(pt, params, bg, th, l_max=2000,
                                        nonlinear="ept", pt_mpk=pt_mpk))
        return cl_lin, cl_nl

    def test_positive(self, cl_pair):
        """EPT C_l^pp is positive."""
        _, cl_nl = cl_pair
        assert np.all(cl_nl[2:] > 0), "C_l^pp EPT must be positive for l>=2"

    def test_finite(self, cl_pair):
        """EPT C_l^pp is finite."""
        _, cl_nl = cl_pair
        assert np.all(np.isfinite(cl_nl)), "C_l^pp EPT has non-finite entries"

    def test_nl_boost_at_high_l(self, cl_pair):
        """NL/linear ratio is >1 at l>=500 (1-loop contribution positive)."""
        cl_lin, cl_nl = cl_pair
        for l_val in [500, 1000, 1500, 2000]:
            ratio = cl_nl[l_val] / cl_lin[l_val]
            print(f"  l={l_val}: NL/lin = {ratio:.4f}")
            assert ratio > 1.001, (
                f"l={l_val}: NL/lin={ratio:.4f}, expected > 1.001")

    def test_no_boost_at_low_l(self, cl_pair):
        """NL/linear ratio is ~1 at l<=50 (1-loop irrelevant on largest scales)."""
        cl_lin, cl_nl = cl_pair
        for l_val in [10, 30, 50]:
            ratio = cl_nl[l_val] / cl_lin[l_val]
            print(f"  l={l_val}: NL/lin = {ratio:.4f}")
            assert abs(ratio - 1.0) < 0.02, (
                f"l={l_val}: NL/lin={ratio:.4f}, expected ~1 at low l")


# -----------------------------------------------------------------------------
# JAX compatibility (linear path only — Halofit's vmap-over-z is heavier)
# -----------------------------------------------------------------------------

class TestJaxCompat:

    def test_jit_compatible(self, pipeline):
        """Function compiles under ``jax.jit``."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        cl_jit = jax.jit(
            compute_cl_pp, static_argnums=(4,), static_argnames=("nonlinear",)
        )(pt, params, bg, th, 50)
        assert cl_jit.shape == (51,)
        assert float(cl_jit[2]) > 0

    def test_grad_wrt_ln10As(self, pipeline):
        """``jax.grad`` through ``ln10A_s`` gives a finite, nonzero gradient."""
        from clax.lensing import compute_cl_pp
        _, bg, th, pt, _ = pipeline

        def objective(params):
            cl = compute_cl_pp(pt, params, bg, th, l_max=30)
            return jnp.sum(cl[2:])

        from clax import CosmoParams
        params = CosmoParams()
        grad = jax.grad(objective)(params)
        g_As = grad.ln10A_s
        print(f"  d(sum Cl)/d(ln10As) = {g_As:.6e}")
        assert jnp.isfinite(g_As), f"gradient is not finite: {g_As}"
        assert abs(g_As) > 0, "gradient is zero"


# -----------------------------------------------------------------------------
# pt_mpk plumbing (smsharma/clax#42)
# -----------------------------------------------------------------------------

class TestPtMpk:
    """compute_cl_pp's nonlinear corrections read P_lin at z~0, so they need
    the tau0-complete matter solve (smsharma/clax#42).

    These are cosmology-independent plumbing tests (wiring, raises,
    bit-identity), exempt from the multi-cosmology rule."""

    @pytest.mark.parametrize("nl", ["halofit", "ept"])
    def test_nonlinear_without_pt_mpk_raises(self, pipeline, nl):
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, _ = pipeline
        with pytest.raises(ValueError, match="pt_mpk"):
            compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear=nl)

    def test_linear_ignores_pt_mpk_bit_identically(self, pipeline):
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        a = np.asarray(compute_cl_pp(pt, params, bg, th, l_max=100))
        b = np.asarray(compute_cl_pp(pt, params, bg, th, l_max=100, pt_mpk=pt_mpk))
        assert np.array_equal(a, b)

    def test_both_ept_ratio_sides_read_pt_mpk(self, pipeline, monkeypatch):
        """Wiring: the P_NL numerator AND the P_lin denominator of the EPT
        correction both come from pt_mpk. A half migration puts the numerator
        and denominator on different solves, so the old second-order residual
        becomes first order."""
        import clax.ept
        import clax.transfer
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        seen = []
        real_ept = clax.ept.compute_ept_from_clax
        real_lin = clax.transfer.compute_linear_matter_pk_from_perturbations

        def spy_ept(params_, bg_, pt_, *a, **k):
            seen.append(("ept", pt_ is pt_mpk))
            return real_ept(params_, bg_, pt_, *a, **k)

        def spy_lin(pt_, *a, **k):
            seen.append(("lin", pt_ is pt_mpk))
            return real_lin(pt_, *a, **k)

        monkeypatch.setattr(clax.ept, "compute_ept_from_clax", spy_ept)
        monkeypatch.setattr(clax.transfer, "compute_linear_matter_pk_from_perturbations", spy_lin)
        compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear="ept", pt_mpk=pt_mpk)
        assert ("ept", True) in seen and ("lin", True) in seen, seen
        assert all(ok for _, ok in seen), seen

    def test_pt_mpk_with_narrower_k_range_raises(self, pipeline):
        """A pt_mpk solved at another precision must not silently misalign:
        P_lin is evaluated at explicit k, and k outside pt_mpk's support is
        rejected."""
        from dataclasses import replace
        from clax import PrecisionParams
        from clax.lensing import compute_cl_pp
        from clax.perturbations import perturbations_solve_mpk
        params, bg, th, pt, _ = pipeline
        narrow = replace(PrecisionParams.fast_cl(), pt_k_max_cl=1.0, pt_k_chunk_size=20)
        pt_narrow = perturbations_solve_mpk(params, narrow, bg, th)
        with pytest.raises(ValueError, match="solved perturbation grid"):
            compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear="halofit", pt_mpk=pt_narrow)

    @pytest.mark.parametrize("nl", ["halofit", "ept"])
    def test_pt_mpk_with_narrower_k_range_raises_under_jit(self, pipeline, nl):
        """Jitted twin of test_pt_mpk_with_narrower_k_range_raises (smsharma/clax#54):
        under jit the eager support check cannot read the traced pt_mpk grid, so the
        modulators' guard on the queried k (eqx.error_if) must raise at run time;
        without it the P_lin spline clamps at pt_mpk's k_max and R(k) is silently
        wrong."""
        from dataclasses import replace
        from clax import PrecisionParams
        from clax.lensing import compute_cl_pp
        from clax.perturbations import perturbations_solve_mpk
        params, bg, th, pt, _ = pipeline
        narrow = replace(PrecisionParams.fast_cl(), pt_k_max_cl=1.0, pt_k_chunk_size=20)
        pt_narrow = perturbations_solve_mpk(params, narrow, bg, th)
        prec = _pipeline_prec()
        f = jax.jit(lambda pm: compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear=nl,
                                             pt_mpk=pm, prec=prec))
        with pytest.raises(Exception, match="solved perturbation grid"):
            jax.block_until_ready(f(pt_narrow))

    def test_mismatched_prec_raises(self, pipeline):
        """A prec whose pt_k_max_cl is not pt.k_grid[-1] would silently resize the
        Halofit k-extension (it is sized from prec but starts at pt.k_grid[-1]);
        compute_cl_pp raises, eagerly and under jit (smsharma/clax#54)."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt, pt_mpk = pipeline
        wrong = _replace(_pipeline_prec(), pt_k_max_cl=1.0)
        with pytest.raises(Exception, match="pt_k_max_cl does not match"):
            compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear="halofit",
                          pt_mpk=pt_mpk, prec=wrong)
        # pt traced too, so the check runs as a run-time check (as inside a jitted
        # likelihood), not at trace time on a concrete grid.
        f = jax.jit(lambda p_, pm: compute_cl_pp(p_, params, bg, th, l_max=10,
                                                 nonlinear="halofit", pt_mpk=pm, prec=wrong))
        with pytest.raises(Exception, match="pt_k_max_cl does not match"):
            jax.block_until_ready(f(pt, pt_mpk))
