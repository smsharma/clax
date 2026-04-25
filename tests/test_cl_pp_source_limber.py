"""Tests for source-based Limber C_l^pp.

``compute_cl_pp_source_limber`` mirrors CLASS's Limber algorithm:
    source(k, tau) = (phi+psi)(k, tau) * W_lcmb(tau)
    T_l(k) = [S*chi]_interp * IPhiFlat / (l+0.5)       (Limber)
    C_l^pp = 4pi * integral d(lnk) * P_R(k) * T_l(k)^2

Accuracy: <3% for l <= 500, same as compute_cl_pp_limber at higher l.
The function provides cleaner architecture (no Poisson reconstruction)
and enables CLASS-style NL corrections via source multiplication.
"""

import os
import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dataclasses import replace as _replace


@pytest.fixture(scope="module")
def pipeline():
    """Run pipeline once for all tests."""
    from clax import CosmoParams, PrecisionParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve

    prec = _replace(PrecisionParams.fast_cl(),
                    pt_k_max_cl=5.0,
                    pt_k_chunk_size=20)
    params = CosmoParams()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, bg, th, pt


@pytest.fixture(scope="module")
def class_reference():
    """Generate CLASS linear C_l^pp with matching parameters."""
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


class TestSourceLimberExists:
    """Function exists and has the expected signature."""

    def test_import(self):
        """compute_cl_pp_source_limber is importable from clax.lensing."""
        from clax.lensing import compute_cl_pp_source_limber  # noqa: F401

    def test_returns_correct_shape(self, pipeline):
        """Returns array of shape (l_max+1,) with l=0,1 zeroed."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = compute_cl_pp_source_limber(pt, params, bg, th, l_max=100)
        assert cl.shape == (101,), f"Expected (101,), got {cl.shape}"
        assert float(cl[0]) == 0.0
        assert float(cl[1]) == 0.0
        assert float(cl[2]) > 0.0


class TestSourceLimberAccuracy:
    """Accuracy vs CLASS reference."""

    def test_matches_class_at_low_l(self, pipeline, class_reference):
        """Matches CLASS to <3% for l = 100, 200, 500."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=500))
        pp_class = class_reference

        for l_val in [100, 200, 500]:
            ratio = cl[l_val] / pp_class[l_val]
            err = abs(ratio - 1.0)
            print(f"  l={l_val}: source_limber/CLASS = {ratio:.4f} ({err:.1%})")
            assert err < 0.03, (
                f"l={l_val}: {err:.1%} error exceeds 3% "
                f"(ours={cl[l_val]:.4e}, CLASS={pp_class[l_val]:.4e})")

    def test_matches_class_at_medium_l(self, pipeline, class_reference):
        """Matches CLASS to <5% for l = 1000."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=1000))
        pp_class = class_reference

        ratio = cl[1000] / pp_class[1000]
        err = abs(ratio - 1.0)
        print(f"  l=1000: source_limber/CLASS = {ratio:.4f} ({err:.1%})")
        assert err < 0.05, (
            f"l=1000: {err:.1%} error exceeds 5% "
            f"(ours={cl[1000]:.4e}, CLASS={pp_class[1000]:.4e})")

    def test_consistent_with_poisson_limber(self, pipeline):
        """Source-based and Poisson-based Limber agree to <1% at all l."""
        from clax.lensing import compute_cl_pp_source_limber, compute_cl_pp_limber
        params, bg, th, pt = pipeline

        cl_source = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=2500))
        cl_poisson = np.array(compute_cl_pp_limber(
            pt, params, bg, th, l_max=2500, n_chi=500))

        max_err = 0.0
        for l_val in [100, 500, 1000, 1500, 2000, 2500]:
            ratio = cl_source[l_val] / cl_poisson[l_val]
            err = abs(ratio - 1.0)
            max_err = max(max_err, err)
            print(f"  l={l_val}: source/poisson = {ratio:.4f} ({err:.2%})")
        assert max_err < 0.01, (
            f"Source and Poisson Limber disagree by {max_err:.1%} — "
            f"should be <1% since both use equivalent formulas")


class TestSourceLimberPositivity:
    """Basic physical sanity checks."""

    def test_positive_for_l_ge_2(self, pipeline):
        """C_l^pp is positive for all l >= 2."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=100))
        for l in range(2, 101):
            assert cl[l] > 0, f"C_l^pp(l={l}) = {cl[l]:.4e} is not positive"

    def test_decreasing_with_l(self, pipeline):
        """C_l^pp decreases monotonically for l >= 10."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp_source_limber(
            pt, params, bg, th, l_max=500))
        for l in range(10, 500):
            assert cl[l] >= cl[l + 1], (
                f"Not decreasing: C_l({l})={cl[l]:.4e} < C_l({l+1})={cl[l+1]:.4e}")


class TestSourceLimberJaxCompat:
    """JIT compilation and automatic differentiation."""

    def test_jit_compatible(self, pipeline):
        """Function compiles under jax.jit."""
        from clax.lensing import compute_cl_pp_source_limber
        params, bg, th, pt = pipeline
        cl_jit = jax.jit(compute_cl_pp_source_limber, static_argnums=(4,))(
            pt, params, bg, th, 50)
        assert cl_jit.shape == (51,)
        assert float(cl_jit[2]) > 0

    def test_grad_wrt_ln10As(self, pipeline):
        """jax.grad through ln10A_s gives finite nonzero gradient."""
        from clax.lensing import compute_cl_pp_source_limber
        _, bg, th, pt = pipeline

        def objective(params):
            cl = compute_cl_pp_source_limber(pt, params, bg, th, l_max=30)
            return jnp.sum(cl[2:])

        from clax import CosmoParams
        params = CosmoParams()
        grad = jax.grad(objective)(params)
        g_As = grad.ln10A_s
        print(f"  d(sum Cl)/d(ln10As) = {g_As:.6e}")
        assert jnp.isfinite(g_As), f"Gradient is not finite: {g_As}"
        assert abs(g_As) > 0, "Gradient is zero"
